# SPDX-FileCopyrightText: Copyright (c) 2026 Patch
# SPDX-License-Identifier: Apache-2.0

"""Gaits: four locomotion target poses at one speed, from upstream's SOMA locomotion modes.

Upstream blends a mode's targets for the character's desired speed every generation. A crowd
character picks one gait at a time instead, so each gait is blended once, here: walk, jog and
run from the Forward mode, idle at rest, and each of Scene Preview's style modes at its own speed.
The targets are placed later, per request, a gait's lead ahead of the character.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from app.runtime.common.io.smart_primitive_document import SmartPrimitiveAssetPool
from app.runtime.common.motion import load_locomotion_mode
from app.runtime.common.rig import RigAssets
from app.runtime.soma.presets import LINEAR_LOCOMOTION, STYLE_LOCOMOTION_MODES
from motionbricks.runtime.blend_spaces import linear_1d_pose_blend
from motionbricks.runtime.target_pose import adjust_target_pose_speed

from .batch import token_bits

# Forward-mode gaits by speed, in m/s, as upstream's Walk, Jog and Run targets name them.
LINEAR_GAITS = (("walk", 2.3), ("jog", 3.0), ("run", 4.5))


@dataclass(frozen=True)
class Gait:
    name: str
    index: int  # row in GaitLibrary.targets
    speed: float  # m/s that the targets move at
    lead: float  # meters ahead of the character that the targets sit
    transition_early: int  # frames before the end of each motion that the next one starts
    token_bits: int  # token counts the model may choose


class GaitLibrary:
    """Every gait's target poses on the GPU, laid out as [gait, pose, joint], and the gaits by name."""

    def __init__(self, assets: RigAssets, stream: wp.Stream, *, linear=LINEAR_GAITS, styles: bool = True) -> None:
        pool = SmartPrimitiveAssetPool(assets.clip_registry, assets.reference_pose, assets.model_to_runtime)
        self.gaits: list[Gait] = []
        self.by_name: dict[str, Gait] = {}
        targets = []
        with wp.ScopedStream(stream):
            forward = load_locomotion_mode(LINEAR_LOCOMOTION, pool)
            mode = forward.definition
            for name, speed in (("idle", 0.0), *linear):
                blended = wp.empty_like(forward.target_poses[0])
                sample = forward.blend_space.evaluate(wp.vec2(speed / mode.max_speed, 0.0))
                linear_1d_pose_blend(forward.target_poses, sample, blended, stream=stream)
                if speed > 0.0:
                    adjust_target_pose_speed(blended, speed * mode.velocity_scale, stream=stream)
                targets.append(blended.numpy())
                self._add(name, speed, speed * mode.velocity_scale * mode.position_scale, mode.transition_early_by_frames, mode.allowed_token_counts)
            if styles:
                for style in STYLE_LOCOMOTION_MODES:
                    runtime = load_locomotion_mode(style, pool)
                    speed = style.max_speed * style.velocity_scale
                    # Target 1 is the style itself; target 0 is Idle.
                    poses = wp.clone(runtime.target_poses[1])
                    adjust_target_pose_speed(poses, speed, stream=stream)
                    targets.append(poses.numpy())
                    gait = self._add(style.name.removesuffix("_1d"), speed, speed * style.position_scale, style.transition_early_by_frames, style.allowed_token_counts)
                    self.by_name[style.name] = gait
            self.targets = wp.array(np.stack(targets), dtype=wp.transform, device=stream.device)

    def _add(self, name: str, speed: float, lead: float, transition_early: int, counts) -> Gait:
        gait = Gait(name, len(self.gaits), float(speed), float(lead), int(transition_early), token_bits(counts))
        self.gaits.append(gait)
        self.by_name[name] = gait
        return gait

    def __getitem__(self, name: str) -> Gait:
        return self.by_name[name]

    def __iter__(self):
        return iter(self.gaits)

    def __len__(self) -> int:
        return len(self.gaits)


__all__ = ["Gait", "GaitLibrary", "LINEAR_GAITS"]
