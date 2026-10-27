# SPDX-FileCopyrightText: Copyright (c) 2026 Patch
# SPDX-License-Identifier: Apache-2.0

"""Record a crowd to a take file: every character's pose each step, and every motion the model
generated (each one a plan: the frames it would play if nothing re-planned first).

A take is "CBT1", a little-endian u32 header length, a JSON header, then the arrays the header
points at ({offset, length, dtype}, offsets from the end of the header, 4-byte aligned).
Rotations are model-joint quaternions scaled to int16; viewer/take.js reads the format.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from .batch import MODEL_JOINTS, Batch
from .crowd import Crowd, CrowdFrame, Generation

MAGIC = b"CBT1"


def _quantize(rotations: np.ndarray) -> np.ndarray:
    return np.clip(np.round(rotations * 32767.0), -32767, 32767).astype(np.int16)


class Recorder:
    """Listens to a crowd's batches and captures its frames. Call capture() after each step."""

    def __init__(self, crowd: Crowd, *, plans: bool = True) -> None:
        self.crowd = crowd
        self.plans = plans
        if plans:
            crowd.keep = True
        crowd.listeners.append(self._heard)
        self.times: list[float] = []
        self.alive: list[np.ndarray] = []
        self.positions: list[np.ndarray] = []
        self.rotations: list[np.ndarray] = []
        self.generations: list[dict] = []
        self.batch_sizes: list[tuple[float, int, float]] = []

    def _heard(self, generations: list[Generation], batch: Batch) -> None:
        self.batch_sizes.append((self.crowd.clock, len(generations), float(batch.milliseconds)))
        for generation in generations:
            entry = {
                "character": generation.character,
                "clock": generation.clock,
                "frames": generation.frames,
                "transition": generation.transition,
                "label": generation.label,
                "kind": generation.request.kind,
                "num_targets": int(generation.request.num_targets),
            }
            if self.plans and batch.packed is not None:
                row = generation.row
                frames = max(generation.frames, 1)
                entry["packed"] = batch.packed[row, :frames]
                entry["contacts"] = batch.contacts[row, :frames]
                entry["feet"] = batch.feet[row, :frames]
                entry["targets"] = batch.targets[row, : generation.request.num_targets]
            self.generations.append(entry)

    def capture(self, frame: CrowdFrame) -> None:
        capacity = self.crowd.capacity
        alive = np.zeros(capacity, dtype=np.uint8)
        alive[frame.slots] = 1
        full = np.zeros((capacity, 3 + 4 * MODEL_JOINTS), dtype=np.float32)
        full[frame.slots] = frame.packed
        self.times.append(self.crowd.clock)
        self.alive.append(alive)
        self.positions.append(full[:, :3].copy())
        self.rotations.append(_quantize(full[:, 3:]))

    def save(self, path: str | Path, *, world: dict | None = None, extra: dict | None = None) -> Path:
        arrays: list[bytes] = []
        offset = 0

        def add(array) -> dict:
            nonlocal offset
            array = np.ascontiguousarray(array)
            data = array.tobytes()
            entry = {"offset": offset, "length": int(array.size), "dtype": str(array.dtype)}
            arrays.append(data)
            offset += len(data)
            pad = (-offset) % 4
            arrays.append(b"\0" * pad)
            offset += pad
            return entry

        crowd = self.crowd
        header = {
            "format": "crowdbricks.take",
            "version": 1,
            "frames": len(self.times),
            "capacity": crowd.capacity,
            "model_joints": crowd.generator.model_to_pose_list,
            "times": add(np.array(self.times, dtype=np.float32)),
            "alive": add(np.stack(self.alive)),
            "positions": add(np.stack(self.positions)),
            "rotations": add(np.stack(self.rotations)),
            "batches": [[round(c, 4), n, round(ms, 3)] for c, n, ms in self.batch_sizes],
            "world": world or {},
        }
        generations = self.generations
        if generations:
            frames = np.array([max(g["frames"], 1) for g in generations], dtype=np.int32)
            header["plans"] = {
                "count": len(generations),
                "character": add(np.array([g["character"] for g in generations], dtype=np.int16)),
                "clock": add(np.array([g["clock"] for g in generations], dtype=np.float32)),
                "frames": add(frames.astype(np.int16)),
                "transition": add(np.array([g["transition"] for g in generations], dtype=np.int16)),
                "kind": add(np.array([g["kind"] for g in generations], dtype=np.uint8)),
                "num_targets": add(np.array([g["num_targets"] for g in generations], dtype=np.uint8)),
                "labels": [g["label"] for g in generations],
            }
            if self.plans and "packed" in generations[0]:
                packed = np.concatenate([g["packed"] for g in generations])
                targets = np.concatenate([g["targets"] for g in generations])
                header["plans"].update({
                    "positions": add(packed[:, :3].astype(np.float32)),
                    "rotations": add(_quantize(packed[:, 3:])),
                    "contacts": add(np.round(np.concatenate([g["contacts"] for g in generations]) * 255).astype(np.uint8)),
                    "feet": add(np.concatenate([g["feet"] for g in generations]).astype(np.float32)),
                    "target_positions": add(targets[:, :3].astype(np.float32)),
                    "target_rotations": add(_quantize(targets[:, 3:])),
                })
        if extra:
            header.update(extra)
        header_bytes = json.dumps(header, separators=(",", ":")).encode()
        header_bytes += b" " * ((-len(header_bytes)) % 4)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + b"".join(arrays))
        return path


__all__ = ["MAGIC", "Recorder"]
