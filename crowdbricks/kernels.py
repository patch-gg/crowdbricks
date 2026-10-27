# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Patch
# SPDX-License-Identifier: Apache-2.0

"""MotionBricks' SOMA kernels with a batch dimension, plus the crowd's playback kernels.

The encode, conditions, decode and endpoint kernels follow motionbricks/model/soma/kernels.py and
motionbricks/model/kernels.py from NVlabs/motionbricks line for line, with the request as the
first index. Everything else (sampling, packing, seeding, forking, target placement) is new.

Layouts:
    motion[character, frame, joint]   generated local transforms at 30 Hz, root in world space
    guides[request, slot, joint]      four context poses, then four target poses
    targets[request, pose, joint]     a request's target poses, root in world space
"""

from __future__ import annotations

import warp as wp

from motionbricks.model.contract import MODEL_BOUNDARY_FRAMES, MODEL_POSES_PER_TOKEN, MODEL_SAMPLE_RATE
from motionbricks.model.soma.kernels import _aligned_position, _heading_from_hip_positions, _rotation_from_6d
from motionbricks.utils.pose_utils import (
    apply_delta_transform,
    apply_root_endpoint_correction,
    blend_transform,
    compute_delta_transform,
    compute_root_endpoint_delta,
)

GUIDES = 2 * MODEL_POSES_PER_TOKEN
FOOT_CONTACTS = 4
# A request's targets come from a gait, placed on the ground, or from a step of a sequence
# already written in world space.
FROM_GAIT = 0
FROM_SEQUENCE = 1


# ---------------------------------------------------------------- playback


@wp.func
def sample_motion(
    motion: wp.array3d(dtype=wp.transform),
    counts: wp.array(dtype=wp.int32),
    character: wp.int32,
    time: wp.float32,
    joint: wp.int32,
) -> wp.transform:
    """A character's generated motion at a playback time, holding its last frame after the end."""
    count = counts[character]
    frame = wp.max(time, 0.0) * MODEL_SAMPLE_RATE
    frame0 = wp.min(wp.int32(wp.floor(frame)), count - 1)
    frame1 = wp.min(frame0 + 1, count - 1)
    weight = wp.clamp(frame - wp.float32(frame0), 0.0, 1.0)
    return blend_transform(motion[character, frame0, joint], motion[character, frame1, joint], weight)


@wp.kernel
def sample_kernel(
    motion: wp.array3d(dtype=wp.transform),
    counts: wp.array(dtype=wp.int32),
    times: wp.array(dtype=wp.float32),
    poses: wp.array2d(dtype=wp.transform),
):
    character, joint = wp.tid()
    poses[character, joint] = sample_motion(motion, counts, character, times[character], joint)


@wp.kernel
def pack_kernel(
    poses: wp.array2d(dtype=wp.transform),
    model_to_pose: wp.array(dtype=wp.int32),
    packed: wp.array2d(dtype=wp.float32),
):
    """Root position and every model joint's local rotation: all a viewer needs to draw a pose."""
    character, model_joint = wp.tid()
    rotation = poses[character, model_to_pose[model_joint]].q
    offset = 3 + model_joint * 4
    packed[character, offset] = rotation[0]
    packed[character, offset + 1] = rotation[1]
    packed[character, offset + 2] = rotation[2]
    packed[character, offset + 3] = rotation[3]
    if model_joint == 0:
        root = poses[character, 0].p
        packed[character, 0] = root[0]
        packed[character, 1] = root[1]
        packed[character, 2] = root[2]


@wp.kernel
def pack_frames_kernel(
    request_frames: wp.array3d(dtype=wp.transform),
    model_to_pose: wp.array(dtype=wp.int32),
    packed: wp.array3d(dtype=wp.float32),
):
    """pack_kernel for every frame of every request's generated motion."""
    request, frame, model_joint = wp.tid()
    rotation = request_frames[request, frame, model_to_pose[model_joint]].q
    offset = 3 + model_joint * 4
    packed[request, frame, offset] = rotation[0]
    packed[request, frame, offset + 1] = rotation[1]
    packed[request, frame, offset + 2] = rotation[2]
    packed[request, frame, offset + 3] = rotation[3]
    if model_joint == 0:
        root = request_frames[request, frame, 0].p
        packed[request, frame, 0] = root[0]
        packed[request, frame, 1] = root[1]
        packed[request, frame, 2] = root[2]


@wp.kernel
def seed_kernel(
    character: wp.int32,
    pose: wp.array(dtype=wp.transform),
    placement: wp.transform,
    motion: wp.array3d(dtype=wp.transform),
    counts: wp.array(dtype=wp.int32),
):
    """Start a character standing in a pose, as a one-frame motion. placement moves its root."""
    joint = wp.tid()
    value = pose[joint]
    if joint == 0:
        value = wp.mul(placement, value)
        counts[character] = 1
    motion[character, 0, joint] = value


@wp.kernel
def copy_character_kernel(
    source: wp.int32,
    destinations: wp.array(dtype=wp.int32),
    motion: wp.array3d(dtype=wp.transform),
    counts: wp.array(dtype=wp.int32),
    contacts: wp.array3d(dtype=wp.float32),
):
    """Fork: copy a character's generated motion to other characters."""
    copy, frame, joint = wp.tid()
    destination = destinations[copy]
    motion[destination, frame, joint] = motion[source, frame, joint]
    if joint < FOOT_CONTACTS:
        contacts[destination, frame, joint] = contacts[source, frame, joint]
    if frame == 0 and joint == 0:
        counts[destination] = counts[source]


# ---------------------------------------------------------------- one batch of requests


@wp.kernel
def targets_kernel(
    request_character: wp.array(dtype=wp.int32),
    request_kind: wp.array(dtype=wp.int32),
    request_source: wp.array(dtype=wp.int32),
    placements: wp.array(dtype=wp.transform),
    gait_targets: wp.array3d(dtype=wp.transform),
    sequence_targets: wp.array3d(dtype=wp.transform),
    targets: wp.array3d(dtype=wp.transform),
):
    """Each request's four target poses: a gait's, placed on the ground, or one step of a sequence."""
    request, pose, joint = wp.tid()
    source = request_source[request]
    if request_kind[request] == FROM_GAIT:
        value = gait_targets[source, pose, joint]
        if joint == 0:
            value = wp.mul(placements[request], value)
        targets[request, pose, joint] = value
    else:
        character = request_character[request]
        targets[request, pose, joint] = sequence_targets[character, source * MODEL_POSES_PER_TOKEN + pose, joint]


@wp.kernel
def guides_kernel(
    motion: wp.array3d(dtype=wp.transform),
    counts: wp.array(dtype=wp.int32),
    times: wp.array(dtype=wp.float32),
    request_character: wp.array(dtype=wp.int32),
    num_targets: wp.array(dtype=wp.int32),
    targets: wp.array3d(dtype=wp.transform),
    guides: wp.array3d(dtype=wp.transform),
):
    """The context (the four poses that just played, at 30 Hz) and the targets of each request.

    A request with fewer than four target poses repeats its last one, as upstream's PoseToken does.
    """
    request, slot, joint = wp.tid()
    character = request_character[request]
    if slot < MODEL_POSES_PER_TOKEN:
        time = times[character] - wp.float32(MODEL_POSES_PER_TOKEN - 1 - slot) / MODEL_SAMPLE_RATE
        guides[request, slot, joint] = sample_motion(motion, counts, character, time, joint)
    else:
        pose = wp.min(slot - MODEL_POSES_PER_TOKEN, num_targets[request] - 1)
        guides[request, slot, joint] = targets[request, pose, joint]


@wp.kernel
def alignment_kernel(
    right_leg_index: wp.int32,
    left_leg_index: wp.int32,
    desired_heading: wp.vec2,
    world_poses: wp.array3d(dtype=wp.transform),
    alignment: wp.array(dtype=wp.transform),
):
    request = wp.tid()
    last_context_frame = MODEL_POSES_PER_TOKEN - 1
    hips_position = world_poses[request, last_context_frame, 0].p
    current_angle = _heading_from_hip_positions(
        world_poses[request, last_context_frame, right_leg_index].p,
        world_poses[request, last_context_frame, left_leg_index].p,
    )
    desired_angle = wp.atan2(desired_heading[0], desired_heading[1])
    alignment[request] = wp.transform(
        wp.vec3(hips_position[0], 0.0, hips_position[2]),
        wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), current_angle - desired_angle),
    )


@wp.kernel
def encode_kernel(
    num_model_joints: wp.int32,
    right_leg_index: wp.int32,
    left_leg_index: wp.int32,
    model_to_pose: wp.array(dtype=wp.int32),
    world_poses: wp.array3d(dtype=wp.transform),
    alignment: wp.array(dtype=wp.transform),
    global_root_values: wp.array3d(dtype=wp.float32),
    local_root_values: wp.array3d(dtype=wp.float32),
    local_poses: wp.array3d(dtype=wp.float32),
):
    request, frame = wp.tid()
    final_frame = 2 * MODEL_POSES_PER_TOKEN - 1
    next_frame = wp.where(frame == MODEL_POSES_PER_TOKEN - 1, frame, wp.min(frame + 1, final_frame))

    align = alignment[request]
    current_hips = _aligned_position(align, world_poses[request, frame, 0].p)
    next_hips = _aligned_position(align, world_poses[request, next_frame, 0].p)
    current_right = _aligned_position(align, world_poses[request, frame, right_leg_index].p)
    current_left = _aligned_position(align, world_poses[request, frame, left_leg_index].p)
    next_right = _aligned_position(align, world_poses[request, next_frame, right_leg_index].p)
    next_left = _aligned_position(align, world_poses[request, next_frame, left_leg_index].p)
    current_heading = _heading_from_hip_positions(current_right, current_left)
    next_heading = _heading_from_hip_positions(next_right, next_left)
    heading_delta = wp.atan2(wp.sin(next_heading - current_heading), wp.cos(next_heading - current_heading))

    global_root_values[request, frame, 0] = current_hips[0]
    global_root_values[request, frame, 1] = current_hips[1]
    global_root_values[request, frame, 2] = current_hips[2]
    global_root_values[request, frame, 3] = wp.cos(current_heading)
    global_root_values[request, frame, 4] = wp.sin(current_heading)

    local_root_values[request, frame, 0] = heading_delta * MODEL_SAMPLE_RATE
    local_root_values[request, frame, 1] = (next_hips[0] - current_hips[0]) * MODEL_SAMPLE_RATE
    local_root_values[request, frame, 2] = (next_hips[2] - current_hips[2]) * MODEL_SAMPLE_RATE
    local_root_values[request, frame, 3] = current_hips[1]

    root_position = wp.vec3(current_hips[0], 0.0, current_hips[2])
    for model_joint in range(num_model_joints):
        pose_joint = model_to_pose[model_joint]
        joint_position = _aligned_position(align, world_poses[request, frame, pose_joint].p)
        joint_rotation = wp.mul(wp.quat_inverse(align.q), world_poses[request, frame, pose_joint].q)
        if model_joint > 0:
            position_offset = (model_joint - 1) * 3
            centered_position = joint_position - root_position
            local_poses[request, frame, position_offset] = centered_position[0]
            local_poses[request, frame, position_offset + 1] = centered_position[1]
            local_poses[request, frame, position_offset + 2] = centered_position[2]
        rotation_offset = (num_model_joints - 1) * 3 + model_joint * 6
        x_axis = wp.quat_rotate(joint_rotation, wp.vec3(1.0, 0.0, 0.0))
        y_axis = wp.quat_rotate(joint_rotation, wp.vec3(0.0, 1.0, 0.0))
        local_poses[request, frame, rotation_offset] = x_axis[0]
        local_poses[request, frame, rotation_offset + 1] = x_axis[1]
        local_poses[request, frame, rotation_offset + 2] = x_axis[2]
        local_poses[request, frame, rotation_offset + 3] = y_axis[0]
        local_poses[request, frame, rotation_offset + 4] = y_axis[1]
        local_poses[request, frame, rotation_offset + 5] = y_axis[2]


@wp.kernel
def conditions_kernel(
    num_targets: wp.array(dtype=wp.int32),
    allowed_token_bits: wp.array(dtype=wp.int32),
    has_global_root_values: wp.array2d(dtype=wp.int32),
    has_local_root_values: wp.array2d(dtype=wp.int32),
    has_local_poses: wp.array2d(dtype=wp.int32),
    allowed_pred_num_tokens: wp.array2d(dtype=wp.int32),
):
    """Mark each request's given poses, and write its allowed token counts as a mask.

    The context poses and the first num_targets target poses count as given. Local root values stay
    unmarked at the last context pose and the last given target pose, as in upstream.
    """
    request, index = wp.tid()
    if index < GUIDES:
        given = num_targets[request]
        value = wp.where(index < MODEL_POSES_PER_TOKEN + given, wp.int32(1), wp.int32(0))
        has_global_root_values[request, index] = value
        has_local_poses[request, index] = value
        has_local_root_values[request, index] = wp.where(
            index == MODEL_POSES_PER_TOKEN - 1 or index == MODEL_POSES_PER_TOKEN + given - 1,
            wp.int32(0),
            value,
        )
    allowed_pred_num_tokens[request, index] = (allowed_token_bits[request] >> index) & 1


@wp.func
def _read_rotation(features: wp.array3d(dtype=wp.float32), request: wp.int32, frame: wp.int32, offset: wp.int32) -> wp.quat:
    x_axis = wp.vec3(features[request, frame, offset], features[request, frame, offset + 1], features[request, frame, offset + 2])
    y_axis = wp.vec3(features[request, frame, offset + 3], features[request, frame, offset + 4], features[request, frame, offset + 5])
    return _rotation_from_6d(x_axis, y_axis)


@wp.kernel
def decode_kernel(
    num_model_joints: wp.int32,
    num_pose_joints: wp.int32,
    parent_indices: wp.array(dtype=wp.int32),
    pose_to_model: wp.array(dtype=wp.int32),
    reference_local_poses: wp.array(dtype=wp.transform),
    alignment: wp.array(dtype=wp.transform),
    model_to_runtime: wp.transform,
    pred_global_motions: wp.array3d(dtype=wp.float32),
    pred_num_tokens: wp.array(dtype=wp.int32),
    output_local_poses: wp.array3d(dtype=wp.transform),
    output_foot_contacts: wp.array3d(dtype=wp.float32),
    output_frame_count: wp.array(dtype=wp.int32),
):
    request, frame = wp.tid()
    raw_count = wp.clamp(pred_num_tokens[request] * MODEL_POSES_PER_TOKEN, 0, pred_global_motions.shape[1])
    frame_count = wp.max(0, raw_count - 2 * MODEL_BOUNDARY_FRAMES)
    if frame == 0:
        output_frame_count[request] = frame_count
    if frame >= frame_count:
        return
    raw_frame = MODEL_BOUNDARY_FRAMES + frame
    position_offset = 5
    rotation_offset = position_offset + (num_model_joints - 1) * 3
    hips_position = wp.vec3(
        pred_global_motions[request, raw_frame, 0],
        pred_global_motions[request, raw_frame, 1],
        pred_global_motions[request, raw_frame, 2],
    )
    hips_rotation = _read_rotation(pred_global_motions, request, raw_frame, rotation_offset)
    hips_transform = wp.mul(alignment[request], wp.transform(hips_position, hips_rotation))
    output_local_poses[request, frame, 0] = wp.mul(model_to_runtime, hips_transform)
    first_contact = pred_global_motions.shape[2] - FOOT_CONTACTS
    for contact in range(FOOT_CONTACTS):
        output_foot_contacts[request, frame, contact] = wp.clamp(
            pred_global_motions[request, raw_frame, first_contact + contact], 0.0, 1.0,
        )
    for joint in range(1, num_pose_joints):
        model_joint = pose_to_model[joint]
        if model_joint < 0:
            output_local_poses[request, frame, joint] = reference_local_poses[joint]
            continue
        predicted_rotation = _read_rotation(pred_global_motions, request, raw_frame, rotation_offset + model_joint * 6)
        parent_model_joint = pose_to_model[parent_indices[joint]]
        parent_rotation = _read_rotation(pred_global_motions, request, raw_frame, rotation_offset + parent_model_joint * 6)
        local_rotation = wp.mul(wp.quat_inverse(parent_rotation), predicted_rotation)
        output_local_poses[request, frame, joint] = wp.transform(reference_local_poses[joint].p, wp.normalize(local_rotation))


@wp.kernel
def endpoint_deltas_kernel(
    output_frame_count: wp.array(dtype=wp.int32),
    guides: wp.array3d(dtype=wp.transform),
    output_local_poses: wp.array3d(dtype=wp.transform),
    start_deltas: wp.array2d(dtype=wp.transform),
    end_deltas: wp.array2d(dtype=wp.transform),
):
    request, joint = wp.tid()
    frame_count = output_frame_count[request]
    if frame_count < 2:
        return
    last_frame = frame_count - 1
    last_context = MODEL_POSES_PER_TOKEN - 1
    first_target = MODEL_POSES_PER_TOKEN
    if joint == 0:
        start_deltas[request, 0] = compute_root_endpoint_delta(output_local_poses[request, 0, 0], guides[request, last_context, 0])
        end_deltas[request, 0] = compute_root_endpoint_delta(output_local_poses[request, last_frame, 0], guides[request, first_target, 0])
        return
    start_deltas[request, joint] = compute_delta_transform(output_local_poses[request, 0, joint], guides[request, last_context, joint])
    end_deltas[request, joint] = compute_delta_transform(output_local_poses[request, last_frame, joint], guides[request, first_target, joint])


@wp.kernel
def apply_endpoint_deltas_kernel(
    output_frame_count: wp.array(dtype=wp.int32),
    start_deltas: wp.array2d(dtype=wp.transform),
    end_deltas: wp.array2d(dtype=wp.transform),
    output_local_poses: wp.array3d(dtype=wp.transform),
):
    request, frame, joint = wp.tid()
    frame_count = output_frame_count[request]
    if frame_count < 2 or frame >= frame_count:
        return
    end_weight = wp.float32(frame) / wp.float32(frame_count - 1)
    start_weight = 1.0 - end_weight
    if joint == 0:
        output_local_poses[request, frame, 0] = apply_root_endpoint_correction(
            output_local_poses[request, frame, 0], start_deltas[request, 0], end_deltas[request, 0], end_weight,
        )
        return
    corrected = apply_delta_transform(output_local_poses[request, frame, joint], start_deltas[request, joint], start_weight)
    output_local_poses[request, frame, joint] = apply_delta_transform(corrected, end_deltas[request, joint], end_weight)


@wp.kernel
def scatter_kernel(
    request_character: wp.array(dtype=wp.int32),
    decoded: wp.array3d(dtype=wp.transform),
    decoded_contacts: wp.array3d(dtype=wp.float32),
    decoded_counts: wp.array(dtype=wp.int32),
    motion: wp.array3d(dtype=wp.transform),
    counts: wp.array(dtype=wp.int32),
    contacts: wp.array3d(dtype=wp.float32),
):
    """Hand each request's motion to its character, which plays it from the start."""
    request, frame, joint = wp.tid()
    character = request_character[request]
    motion[character, frame, joint] = decoded[request, frame, joint]
    if joint < FOOT_CONTACTS:
        contacts[character, frame, joint] = decoded_contacts[request, frame, joint]
    if frame == 0 and joint == 0:
        counts[character] = wp.max(decoded_counts[request], 1)
