# SPDX-FileCopyrightText: Copyright (c) 2026 Patch
# SPDX-License-Identifier: Apache-2.0

"""Many MotionBricks generations in one TensorRT call.

Upstream's engine takes one request per call. BatchGenerator builds the same ONNX model as a
TensorRT plan with a batch dimension, and runs the SOMA encode, decode and endpoint correction
for every request at once (kernels.py). Each request generates the next motion of one character,
from the four poses it just played toward up to four target poses, exactly as upstream's
MotionPlayer.generate() does for a single character.

Characters live in a motion bank on the GPU: every character's current generated motion, which
it plays from the start after each generation. Unused rows of a batch repeat the last request and
their results are dropped, so a part-filled batch costs the same as a full one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import warp as wp

from app.runtime.common.rig import RigAssets, model_to_runtime
from motionbricks.inference.builder import TensorRTBuildOptions
from motionbricks.inference.engine import InferenceEngine
from motionbricks.model.contract import (
    MODEL_MAX_TOKEN_COUNT,
    MODEL_MAX_VALID_FRAMES,
    MODEL_MIN_TOKEN_COUNT,
    MODEL_POSES_PER_TOKEN,
    ModelRig,
)
from motionbricks.model.soma.contract import (
    SOMA_N1_CONTRACT,
    SOMA_N1_FOOT_CONTACT_JOINTS,
    SOMA_N1_FP16_ONNX_PATH,
    SOMA_N1_FP32_ONNX_PATH,
    SOMA_N1_HEADING_JOINTS,
)
from motionbricks.utils.pose_utils import compute_world_poses

from . import kernels
from .kernels import FOOT_CONTACTS, FROM_GAIT, FROM_SEQUENCE, GUIDES

MODEL_JOINTS = len(SOMA_N1_CONTRACT.joint_names)
# A packed pose: root position, then a quaternion (x, y, z, w) for each model joint.
PACKED_VALUES = 3 + 4 * MODEL_JOINTS
TOKEN_COUNTS = MODEL_MAX_TOKEN_COUNT - MODEL_MIN_TOKEN_COUNT + 1


def token_bits(counts) -> int:
    """Allowed token counts as the model's mask, bit 0 for 6 tokens."""
    bits = 0
    for count in counts:
        bits |= 1 << (count - MODEL_MIN_TOKEN_COUNT)
    return bits


def ground_placement(x: float, y: float, heading_x: float, heading_y: float) -> tuple[float, ...]:
    """A transform (x, y, z, qx, qy, qz, qw) on the ground at (x, y), turned to face a heading."""
    half_yaw = 0.5 * float(np.arctan2(heading_y, heading_x))
    return (x, y, 0.0, 0.0, 0.0, float(np.sin(half_yaw)), float(np.cos(half_yaw)))


@dataclass
class Request:
    """One character's next generation.

    kind is FROM_GAIT (a gait's four targets, placed by placement) or FROM_SEQUENCE (step source
    of the sequence written for this character with write_sequence_targets()).
    """

    character: int
    kind: int
    source: int
    placement: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    num_targets: int = MODEL_POSES_PER_TOKEN
    token_bits: int = token_bits(range(8, 15))
    transition_early: int = 8


@dataclass
class Batch:
    """What a batch produced, row by row in request order."""

    characters: np.ndarray
    frames: np.ndarray  # generated frames per request, at 30 Hz
    milliseconds: float  # GPU time from target placement to the motion bank
    # With keep=True: every frame of every request's motion, packed, with its foot contacts,
    # the world position of each foot contact joint, and the request's packed target poses.
    packed: np.ndarray | None = None
    contacts: np.ndarray | None = None
    feet: np.ndarray | None = None
    targets: np.ndarray | None = None
    num_targets: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))


class BatchGenerator:
    """A batched SOMA MotionBricks model and the motion bank of up to `capacity` characters."""

    def __init__(
        self,
        assets: RigAssets,
        *,
        batch: int = 16,
        capacity: int = 256,
        fp16: bool = True,
        cache_dir: str | Path | None = None,
        max_sequence_steps: int = 4,
    ) -> None:
        self.batch = batch
        self.capacity = capacity
        self.max_sequence_steps = max_sequence_steps
        self.engine = InferenceEngine(
            SOMA_N1_FP16_ONNX_PATH if fp16 else SOMA_N1_FP32_ONNX_PATH,
            build_options=TensorRTBuildOptions(enable_fp16=fp16, batch_size=batch),
            cache_dir=cache_dir,
            logger_severity="WARNING",
        )
        self.stream = self.engine.stream
        self.device = self.engine.device
        skeleton = assets.clip_registry.skeleton
        self.skeleton = skeleton
        self.joints = skeleton.num_joints
        model_to_pose = [skeleton.joint_index(name) for name in SOMA_N1_CONTRACT.joint_names]
        pose_to_model = np.full(self.joints, -1, dtype=np.int32)
        for model_joint, pose_joint in enumerate(model_to_pose):
            parent = skeleton.joint_parent(pose_joint)
            if parent == -1 or pose_to_model[parent] >= 0:
                pose_to_model[pose_joint] = model_joint
        self.model_to_pose_list = model_to_pose
        self.foot_joints = [skeleton.joint_index(name) for name in SOMA_N1_FOOT_CONTACT_JOINTS]
        right_leg, left_leg = SOMA_N1_HEADING_JOINTS
        self.right_leg = skeleton.joint_index(right_leg)
        self.left_leg = skeleton.joint_index(left_leg)
        self.model_to_runtime = model_to_runtime(ModelRig.SOMA)
        self.runtime_to_model = wp.transform_inverse(self.model_to_runtime)
        self.reference_pose = assets.reference_pose

        Z, B, F, J = capacity, batch, MODEL_MAX_VALID_FRAMES, self.joints
        device = self.device
        with wp.ScopedStream(self.stream):
            self.model_to_pose = wp.array(model_to_pose, dtype=wp.int32, device=device)
            self.pose_to_model = wp.array(pose_to_model, dtype=wp.int32, device=device)
            # The motion bank.
            self.motion = wp.empty((Z, F, J), dtype=wp.transform, device=device)
            self.contacts = wp.zeros((Z, F, FOOT_CONTACTS), dtype=wp.float32, device=device)
            self.counts = wp.ones(Z, dtype=wp.int32, device=device)
            self.times = wp.zeros(Z, dtype=wp.float32, device=device)
            self.poses = wp.empty((Z, J), dtype=wp.transform, device=device)
            self.packed = wp.zeros((Z, PACKED_VALUES), dtype=wp.float32, device=device)
            self.sequence_targets = wp.empty((Z, max_sequence_steps * MODEL_POSES_PER_TOKEN, J), dtype=wp.transform, device=device)
            self.gait_targets = wp.empty((1, MODEL_POSES_PER_TOKEN, J), dtype=wp.transform, device=device)
            # One batch.
            self.request_character = wp.zeros(B, dtype=wp.int32, device=device)
            self.request_kind = wp.zeros(B, dtype=wp.int32, device=device)
            self.request_source = wp.zeros(B, dtype=wp.int32, device=device)
            self.placements = wp.empty(B, dtype=wp.transform, device=device)
            self.num_targets = wp.zeros(B, dtype=wp.int32, device=device)
            self.token_bits = wp.zeros(B, dtype=wp.int32, device=device)
            self.targets = wp.empty((B, MODEL_POSES_PER_TOKEN, J), dtype=wp.transform, device=device)
            self.guides = wp.empty((B, GUIDES, J), dtype=wp.transform, device=device)
            self.world = wp.empty((B, GUIDES, J), dtype=wp.transform, device=device)
            self.alignment = wp.empty(B, dtype=wp.transform, device=device)
            self.decoded = wp.empty((B, F, J), dtype=wp.transform, device=device)
            self.decoded_contacts = wp.zeros((B, F, FOOT_CONTACTS), dtype=wp.float32, device=device)
            self.decoded_counts = wp.zeros(B, dtype=wp.int32, device=device)
            self.start_deltas = wp.empty((B, J), dtype=wp.transform, device=device)
            self.end_deltas = wp.empty((B, J), dtype=wp.transform, device=device)
            # Recording, on request.
            self.packed_frames = wp.zeros((B, F, PACKED_VALUES), dtype=wp.float32, device=device)
            self.packed_targets = wp.zeros((B, MODEL_POSES_PER_TOKEN, PACKED_VALUES), dtype=wp.float32, device=device)
            self.decoded_world = wp.empty((B * F, J), dtype=wp.transform, device=device)
            self.copy_destinations = wp.zeros(Z, dtype=wp.int32, device=device)
        self.generations = 0
        self.batches = 0

    def set_gait_targets(self, targets: wp.array) -> None:
        """Every gait's four target poses, laid out as [gait, pose, joint], before placement."""
        self.gait_targets = targets

    # ------------------------------------------------------------ the motion bank

    def seed(self, character: int, pose: wp.array, placement: tuple[float, ...]) -> None:
        """Stand a character in a pose (local transforms, root in placement space) as a one-frame motion."""
        with wp.ScopedStream(self.stream):
            wp.launch(
                kernels.seed_kernel, dim=self.joints,
                inputs=[int(character), pose, wp.transform(*placement)],
                outputs=[self.motion, self.counts], stream=self.stream,
            )

    def copy(self, source: int, destinations) -> None:
        """Give other characters a copy of one character's motion."""
        destinations = np.asarray(destinations, dtype=np.int32)
        if not len(destinations):
            return
        with wp.ScopedStream(self.stream):
            self.copy_destinations.assign(np.pad(destinations, (0, self.capacity - len(destinations))))
            wp.launch(
                kernels.copy_character_kernel, dim=(len(destinations), MODEL_MAX_VALID_FRAMES, self.joints),
                inputs=[int(source), self.copy_destinations],
                outputs=[self.motion, self.counts, self.contacts], stream=self.stream,
            )

    def sequence_rows(self, character: int) -> wp.array:
        """The rows that hold a character's sequence targets, four poses per step."""
        return self.sequence_targets[int(character)]

    def sample(self, times: np.ndarray) -> np.ndarray:
        """Every character's packed pose at its playback time: (capacity, PACKED_VALUES)."""
        with wp.ScopedStream(self.stream):
            self.times.assign(times.astype(np.float32))
            wp.launch(kernels.sample_kernel, dim=(self.capacity, self.joints), inputs=[self.motion, self.counts, self.times], outputs=[self.poses], stream=self.stream)
            wp.launch(kernels.pack_kernel, dim=(self.capacity, MODEL_JOINTS), inputs=[self.poses, self.model_to_pose], outputs=[self.packed], stream=self.stream)
            return self.packed.numpy()

    # ------------------------------------------------------------ one batch

    def generate(self, requests: list[Request], times: np.ndarray, *, keep: bool = False) -> Batch:
        """Generate the next motion of every request's character, at most `batch` of them.

        times holds each character's playback time; the four poses before it are the context.
        """
        count = len(requests)
        if not 0 < count <= self.batch:
            raise ValueError(f"A batch takes 1 to {self.batch} requests, not {count}.")
        B = self.batch
        rows = list(requests) + [requests[-1]] * (B - count)
        characters = np.array([r.character for r in rows], dtype=np.int32)
        num_targets = np.array([r.num_targets for r in rows], dtype=np.int32)
        if num_targets.min() < 1 or num_targets.max() > MODEL_POSES_PER_TOKEN:
            raise ValueError("Requests give 1 to 4 target poses.")
        started = wp.Event(device=self.device, enable_timing=True)
        finished = wp.Event(device=self.device, enable_timing=True)
        engine = self.engine
        with wp.ScopedStream(self.stream):
            self.request_character.assign(characters)
            self.request_kind.assign(np.array([r.kind for r in rows], dtype=np.int32))
            self.request_source.assign(np.array([r.source for r in rows], dtype=np.int32))
            self.placements.assign(np.array([r.placement for r in rows], dtype=np.float32))
            self.num_targets.assign(num_targets)
            self.token_bits.assign(np.array([r.token_bits for r in rows], dtype=np.int32))
            self.times.assign(times.astype(np.float32))
            self.stream.record_event(started)
            wp.launch(
                kernels.targets_kernel, dim=(B, MODEL_POSES_PER_TOKEN, self.joints),
                inputs=[self.request_character, self.request_kind, self.request_source, self.placements, self.gait_targets, self.sequence_targets],
                outputs=[self.targets], stream=self.stream,
            )
            wp.launch(
                kernels.guides_kernel, dim=(B, GUIDES, self.joints),
                inputs=[self.motion, self.counts, self.times, self.request_character, self.num_targets, self.targets],
                outputs=[self.guides], stream=self.stream,
            )
            compute_world_poses(
                self.guides.reshape((B * GUIDES, self.joints)), self.skeleton.parent_indices,
                self.world.reshape((B * GUIDES, self.joints)), self.runtime_to_model, stream=self.stream,
            )
            wp.launch(kernels.alignment_kernel, dim=B, inputs=[self.right_leg, self.left_leg, wp.vec2(1.0, 0.0), self.world], outputs=[self.alignment], stream=self.stream)
            wp.launch(
                kernels.encode_kernel, dim=(B, GUIDES),
                inputs=[MODEL_JOINTS, self.right_leg, self.left_leg, self.model_to_pose, self.world, self.alignment],
                outputs=[engine.device_tensor("global_root_values"), engine.device_tensor("local_root_values"), engine.device_tensor("local_poses")],
                stream=self.stream,
            )
            wp.launch(
                kernels.conditions_kernel, dim=(B, TOKEN_COUNTS), inputs=[self.num_targets, self.token_bits],
                outputs=[
                    engine.device_tensor("has_global_root_values"),
                    engine.device_tensor("has_local_root_values"),
                    engine.device_tensor("has_local_poses"),
                    engine.device_tensor("allowed_pred_num_tokens"),
                ],
                stream=self.stream,
            )
            if not engine.cuda_graph_baked:
                engine.bake_cuda_graph()
            engine.enqueue_async()
            wp.launch(
                kernels.decode_kernel, dim=(B, MODEL_MAX_VALID_FRAMES),
                inputs=[
                    MODEL_JOINTS, self.joints, self.skeleton.parent_indices, self.pose_to_model,
                    self.skeleton.reference_local_transforms, self.alignment, self.model_to_runtime,
                    engine.device_tensor("pred_global_motions"), engine.device_tensor("pred_num_tokens"),
                ],
                outputs=[self.decoded, self.decoded_contacts, self.decoded_counts], stream=self.stream,
            )
            wp.launch(kernels.endpoint_deltas_kernel, dim=(B, self.joints), inputs=[self.decoded_counts, self.guides, self.decoded], outputs=[self.start_deltas, self.end_deltas], stream=self.stream)
            wp.launch(
                kernels.apply_endpoint_deltas_kernel, dim=(B, MODEL_MAX_VALID_FRAMES, self.joints),
                inputs=[self.decoded_counts, self.start_deltas, self.end_deltas], outputs=[self.decoded], stream=self.stream,
            )
            wp.launch(
                kernels.scatter_kernel, dim=(count, MODEL_MAX_VALID_FRAMES, self.joints),
                inputs=[self.request_character, self.decoded, self.decoded_contacts, self.decoded_counts],
                outputs=[self.motion, self.counts, self.contacts], stream=self.stream,
            )
            self.stream.record_event(finished)
            frames = self.decoded_counts.numpy()[:count].copy()
            result = Batch(characters=characters[:count].copy(), frames=frames, milliseconds=0.0, num_targets=num_targets[:count].copy())
            if keep:
                self._keep(result, count)
        result.milliseconds = wp.get_event_elapsed_time(started, finished)
        self.generations += count
        self.batches += 1
        return result

    def _keep(self, result: Batch, count: int) -> None:
        B, F = self.batch, MODEL_MAX_VALID_FRAMES
        wp.launch(kernels.pack_frames_kernel, dim=(B, F, MODEL_JOINTS), inputs=[self.decoded, self.model_to_pose], outputs=[self.packed_frames], stream=self.stream)
        wp.launch(kernels.pack_frames_kernel, dim=(B, MODEL_POSES_PER_TOKEN, MODEL_JOINTS), inputs=[self.targets, self.model_to_pose], outputs=[self.packed_targets], stream=self.stream)
        compute_world_poses(
            self.decoded.reshape((B * F, self.joints)), self.skeleton.parent_indices, self.decoded_world,
            wp.transform_identity(), stream=self.stream,
        )
        world = self.decoded_world.numpy().reshape(B, F, self.joints, 7)
        result.packed = self.packed_frames.numpy()[:count].copy()
        result.contacts = self.decoded_contacts.numpy()[:count].copy()
        result.feet = world[:count][:, :, self.foot_joints, :3].copy()
        result.targets = self.packed_targets.numpy()[:count].copy()

    def close(self) -> None:
        self.engine.synchronize()
        self.engine.close()


__all__ = [
    "Batch",
    "BatchGenerator",
    "FROM_GAIT",
    "FROM_SEQUENCE",
    "MODEL_JOINTS",
    "PACKED_VALUES",
    "Request",
    "ground_placement",
    "token_bits",
]
