# SPDX-FileCopyrightText: Copyright (c) 2026 Patch
# SPDX-License-Identifier: Apache-2.0

"""Batched NVIDIA MotionBricks: generate the motion of hundreds of characters in one model call."""

from .batch import FROM_GAIT, FROM_SEQUENCE, MODEL_JOINTS, PACKED_VALUES, Batch, BatchGenerator, Request, ground_placement, token_bits
from .crowd import LOOKAHEAD, Approach, Crowd, CrowdFrame, Driver, Generation, Interact, Seated, Stand, Walk, sit
from .gaits import Gait, GaitLibrary
from .obstacles import KINDS, Obstacle, character_properties, obstacle_pool

__version__ = "0.1.0"

__all__ = [
    "Approach",
    "Batch",
    "BatchGenerator",
    "Crowd",
    "CrowdFrame",
    "Driver",
    "FROM_GAIT",
    "FROM_SEQUENCE",
    "Gait",
    "GaitLibrary",
    "Generation",
    "Interact",
    "KINDS",
    "LOOKAHEAD",
    "MODEL_JOINTS",
    "Obstacle",
    "PACKED_VALUES",
    "Request",
    "Seated",
    "Stand",
    "Walk",
    "character_properties",
    "ground_placement",
    "obstacle_pool",
    "sit",
    "token_bits",
]
