# SPDX-FileCopyrightText: Copyright (c) 2026 Patch
# SPDX-License-Identifier: Apache-2.0

"""Write viewer/soma.bin, the SOMA skinned mesh for the viewer, from your MotionBricks checkout.

The mesh is NVIDIA's (see the MotionBricks repository for its license), so this repository does
not ship it; run this once instead.

soma.bin holds the magic "MBC1", a u32 header length, a JSON header padded to 4 bytes, then the
arrays the header points at. Positions use the runtime's world space (Z up, meters).

    PYTHONPATH=<motionbricks checkout> python viewer/export_character.py
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


def _top_influences(joint_indices: np.ndarray, joint_weights: np.ndarray, count: int, keep: int = 4):
    """Keep the strongest `keep` influences of each vertex and renormalize them."""
    indices = joint_indices.reshape(count, -1)
    weights = joint_weights.reshape(count, -1)
    if indices.shape[1] < keep:
        pad = keep - indices.shape[1]
        indices = np.pad(indices, ((0, 0), (0, pad)))
        weights = np.pad(weights, ((0, 0), (0, pad)))
    order = np.argsort(-weights, axis=1)[:, :keep]
    indices = np.take_along_axis(indices, order, axis=1)
    weights = np.take_along_axis(weights, order, axis=1)
    total = weights.sum(axis=1, keepdims=True)
    weights = np.where(total > 0, weights / np.maximum(total, 1e-12), weights)
    return indices.astype(np.uint16), weights.astype(np.float32)


def export_soma_character() -> bytes:
    from app.runtime.common.rig import load_rig_assets
    from motionbricks.model.contract import ModelRig

    assets = load_rig_assets(ModelRig.SOMA)
    mesh = assets.model
    skeleton = mesh.skeleton
    arrays: list[np.ndarray] = []
    offset = 0

    def add(array: np.ndarray) -> dict:
        nonlocal offset
        array = np.ascontiguousarray(array)
        entry = {"offset": offset, "length": int(array.size), "dtype": str(array.dtype)}
        arrays.append(array)
        offset += array.nbytes
        pad = (-offset) % 4
        if pad:
            arrays.append(np.zeros(pad, dtype=np.uint8))
            offset += pad
        return entry

    parts = []
    for part in mesh.skinned_meshes:
        count = part.num_points
        skin_index, skin_weight = _top_influences(part.joint_indices.numpy(), part.joint_weights.numpy(), count)
        parts.append(
            {
                "vertices": count,
                "source_influences": part.num_influences,
                "positions": add(part.points.numpy().astype(np.float32).reshape(-1)),
                "indices": add(part.indices.numpy().astype(np.uint32).reshape(-1)),
                "skin_index": add(skin_index.reshape(-1)),
                "skin_weight": add(skin_weight.reshape(-1)),
            }
        )

    override = {name: pose.numpy().reshape(-1, 7).round(6).tolist() for name, pose in mesh.override_poses.items()}
    masks = {name: mask.numpy().round(4).tolist() for name, mask in mesh.override_pose_masks.items()}
    header = {
        "format": "motionbricks.character",
        "version": 1,
        "rig": "soma",
        "joints": skeleton.joint_names,
        "parents": [skeleton.joint_parent(i) for i in range(skeleton.num_joints)],
        "reference": skeleton.reference_local_transforms.numpy().reshape(-1, 7).round(6).tolist(),
        "inverse_bind": mesh.inverse_bind_transforms.numpy().reshape(-1, 7).round(7).tolist(),
        "override_poses": override,
        "override_masks": masks,
        "parts": parts,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    header_bytes += b" " * ((-len(header_bytes)) % 4)
    return b"".join([b"MBC1", struct.pack("<I", len(header_bytes)), header_bytes, *(a.tobytes() for a in arrays)])


if __name__ == "__main__":
    import warp as wp

    wp.init()
    target = Path(__file__).resolve().parent / "soma.bin"
    target.write_bytes(export_soma_character())
    print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")
