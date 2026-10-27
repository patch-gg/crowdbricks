# SPDX-FileCopyrightText: Copyright (c) 2026 Patch
# SPDX-License-Identifier: Apache-2.0

"""Smart Primitives that crowd characters use: vault, hop, duck and sit.

Each Obstacle is one upstream SmartPrimitive runtime, placed in the world. A character starts an
interaction when its probe box (upstream's InteractionProbe) reaches the obstacle's interaction
box; the obstacle then picks a sequence the way upstream's VelocityMatchingSequenceBehavior does,
turns and slides its pivot to the character's approach, and writes the sequence's target poses
in world space. Characters share an obstacle: each alignment happens just before its targets are
read, so one character's approach never moves another's targets.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

from app.runtime.common.io.smart_primitive_document import SmartPrimitiveAssetPool
from app.runtime.common.rig import RigAssets
from app.runtime.soma.presets import INTERACTION_PROBE
from motionbricks.runtime.smart_primitive import CharacterProperties, InteractionBox, SmartPrimitive
from motionbricks.runtime.target_pose import TargetPoseInferenceParams

def _upstream_smart_primitives() -> Path:
    from app.runtime.common.rig import ASSETS_ROOT

    return Path(ASSETS_ROOT) / "smart_primitives" / "soma"


# Upstream's SOMA Smart Primitives by what they do.
KINDS = {
    "hop": "jump_0_5m.smart_primitive.json",
    "vault": "jump_0_75m.smart_primitive.json",
    "duck": "avoidance.smart_primitive.json",
    "sit": "sit.smart_primitive.json",
}


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by the quaternion q = (x, y, z, w)."""
    u = q[:3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + q[3] * v)


class _Box:
    """An interaction box's outline on the ground, and its height range."""

    def __init__(self, transform: np.ndarray, half: np.ndarray) -> None:
        q = transform[3:7]
        self.center = transform[:2].astype(np.float64)
        self.axis_x = _rotate(q, np.array([1.0, 0.0, 0.0]))[:2]
        self.axis_y = _rotate(q, np.array([0.0, 1.0, 0.0]))[:2]
        self.half = half[:2].astype(np.float64)
        self.z_range = (float(transform[2] - half[2]), float(transform[2] + half[2]))


def _rectangles_touch(c1, u1, v1, h1, c2, u2, v2, h2) -> bool:
    """Separating-axis test for two rectangles on the ground. Touching counts."""
    offset = c2 - c1
    for axis in (u1, v1, u2, v2):
        reach = (
            h1[0] * abs(u1 @ axis) + h1[1] * abs(v1 @ axis)
            + h2[0] * abs(u2 @ axis) + h2[1] * abs(v2 @ axis)
        )
        if abs(offset @ axis) > reach:
            return False
    return True


class Obstacle:
    """One Smart Primitive in the world."""

    def __init__(
        self,
        pool: SmartPrimitiveAssetPool,
        kind: str,
        position: tuple[float, float],
        yaw_degrees: float,
        stream: wp.Stream,
        *,
        name: str = "",
        reference: str | Path | None = None,
    ) -> None:
        self.kind = kind
        self.name = name or kind
        self.position = np.array(position, dtype=np.float64)
        self.yaw = math.radians(yaw_degrees)
        path = Path(reference) if reference is not None else _upstream_smart_primitives() / KINDS[kind]
        self.reference = path
        self.runtime: SmartPrimitive = pool.instantiate(path)
        half = 0.5 * self.yaw
        self.transform = wp.transform(wp.vec3(float(position[0]), float(position[1]), 0.0), wp.quat(0.0, 0.0, math.sin(half), math.cos(half)))
        self.runtime.set_node_world_transform(0, self.transform, stream=stream)
        self.stream = stream
        count = self.runtime.interaction_box_count
        self.boxes: list[_Box] = []
        if count:
            boxes = wp.empty(count, dtype=InteractionBox, device=stream.device)
            self.runtime.get_world_interaction_boxes(boxes, stream=stream)
            wp.synchronize_stream(stream)
            for box in boxes.numpy():
                self.boxes.append(_Box(np.array(box[0], dtype=np.float64), np.array(box[1], dtype=np.float64)))

    @property
    def sequence_names(self) -> list[str]:
        return list(self.runtime.asset.sequence_indices_by_name)

    def probe_touches(self, position: np.ndarray, heading: np.ndarray, speed: float) -> bool:
        """Whether a character's probe box, reaching ahead at this speed, touches an interaction box."""
        near, far = INTERACTION_PROBE.segment(np.array([*position, 0.0]), np.array([*heading, 0.0]), speed)
        center = 0.5 * (near[:2] + far[:2])
        direction = far[:2] - near[:2]
        depth = float(np.linalg.norm(direction))
        along = direction / max(depth, 1e-6)
        across = np.array([along[1], -along[0]])
        half = np.array([0.5 * INTERACTION_PROBE.box_width, 0.5 * depth])
        for box in self.boxes:
            if box.z_range[0] > INTERACTION_PROBE.box_height or box.z_range[1] < 0.0:
                continue
            if _rectangles_touch(center, across, along, half, box.center, box.axis_x, box.axis_y, box.half):
                return True
        return False

    def sequence_for_speed(self, speed: float) -> int:
        """The sequence whose first target is closest to this speed, as VelocityMatchingSequenceBehavior picks."""
        runtime = self.runtime
        return min(range(runtime.sequence_count), key=lambda index: abs(runtime.sequence_start_speed(index) - speed))

    def write_sequence(
        self,
        index: int,
        character: CharacterProperties,
        rows: wp.array,
    ) -> tuple[TargetPoseInferenceParams, ...]:
        """Align the sequence's pivot to the character and write its targets, four poses per step."""
        self.runtime.align_sequence_pivot(index, character, stream=self.stream)
        return tuple(self.runtime.get_world_target_poses_for_sequence(index, rows, stream=self.stream))

    def describe(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "position": [float(v) for v in self.position],
            "yaw": self.yaw,
            "boxes": [
                {"center": box.center.tolist(), "axis_x": box.axis_x.tolist(), "half": box.half.tolist(), "z": list(box.z_range)}
                for box in self.boxes
            ],
        }


def obstacle_pool(assets: RigAssets) -> SmartPrimitiveAssetPool:
    return SmartPrimitiveAssetPool(assets.clip_registry, assets.reference_pose, assets.model_to_runtime)


def character_properties(position, velocity, heading, desired_speed: float) -> CharacterProperties:
    """What a Smart Primitive needs to know about an approaching character, on the ground."""
    position3 = np.array([position[0], position[1], 0.0], dtype=np.float32)
    heading3 = np.array([heading[0], heading[1], 0.0], dtype=np.float32)
    velocity3 = np.array([velocity[0], velocity[1], 0.0], dtype=np.float32)
    speed = float(np.linalg.norm(velocity3))
    near, far = INTERACTION_PROBE.segment(position3, heading3, max(speed, desired_speed))
    return CharacterProperties(
        position=position3,
        velocity=velocity3,
        facing=heading3,
        desired_velocity=heading3 * desired_speed,
        probe_segment=(near.astype(np.float32), far.astype(np.float32)),
    )


__all__ = ["KINDS", "Obstacle", "character_properties", "obstacle_pool"]
