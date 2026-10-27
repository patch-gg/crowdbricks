# SPDX-FileCopyrightText: Copyright (c) 2026 Patch
# SPDX-License-Identifier: Apache-2.0

"""A batched generation matches upstream's one-request engine on the same request.

Each case is one request built twice: through upstream's PoseToken, place_target_poses() and
MotionBricksSOMAEngine.generate_animation(), and through a BatchGenerator.

With a batch of one, BatchGenerator builds the very TensorRT plan upstream uses, so any difference
comes from the batched encode, decode and endpoint kernels. In FP32 there is none: the root agrees
to a micrometer and every joint to the precision of the comparison (about 0.1 degree).

FP16 is another matter, for upstream as much as for us. On some requests the FP16 model is chaotic:
moving upstream's own context by 10 micrometers turns a jog's joints by 6 degrees, and turning a
run's target by a thousandth of a degree moves a joint by 15 (both ends stay pinned by the
endpoint correction). A bigger batch is also a different plan, with different kernels. So the
second test asks FP16 batches for the same length of motion and close agreement on average.

    PYTHONPATH=<motionbricks checkout> python -m pytest tests/ -s
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CACHE = os.environ.get("CROWDBRICKS_CACHE", "trt-cache")
BATCH = int(os.environ.get("CROWDBRICKS_TEST_BATCH", "4"))
# (gait, heading in degrees, target poses given)
CASES = [("run", 0.0, 4), ("walk", 50.0, 4), ("jog", -120.0, 4), ("idle", 0.0, 1)]


def _angle_degrees(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    dot = np.abs(np.sum(qa * qb, axis=-1))
    return np.degrees(2 * np.arccos(np.clip(dot, 0.0, 1.0)))


@pytest.fixture(scope="module")
def setup():
    import warp as wp

    from app.runtime.common.rig import load_rig_assets
    from motionbricks.model.contract import ModelRig

    wp.init()
    wp.set_device("cuda:0")
    return load_rig_assets(ModelRig.SOMA)


def _upstream(assets, cases, gaits_numpy, gait_index, fp16=True):
    import warp as wp

    from app.runtime.common.rig import model_to_runtime
    from motionbricks.inference.builder import TensorRTBuildOptions
    from motionbricks.model.config import MotionBricksInferenceConfig
    from motionbricks.model.contract import ModelRig
    from motionbricks.model.soma.contract import SOMA_N1_FP16_ONNX_PATH, SOMA_N1_FP32_ONNX_PATH
    from motionbricks.model.soma.engine import MotionBricksSOMAEngine
    from motionbricks.runtime.target_pose import place_target_poses

    engine = MotionBricksSOMAEngine(
        SOMA_N1_FP16_ONNX_PATH if fp16 else SOMA_N1_FP32_ONNX_PATH, model_to_runtime=model_to_runtime(ModelRig.SOMA),
        build_options=TensorRTBuildOptions(enable_fp16=fp16), cache_dir=Path(CACHE), logger_severity="WARNING",
    )
    skeleton = assets.clip_registry.skeleton
    idle = gaits_numpy[gait_index["idle"].index, 0]
    results = []
    with wp.ScopedStream(engine.stream):
        context = engine.create_input(skeleton)
        target = engine.create_input(skeleton)
        output = engine.create_output(skeleton)
        context.update_from_poses(wp.array(np.repeat(idle[None], 4, axis=0), dtype=wp.transform, device=engine.device), stream=engine.stream)
        for name, degrees, given in cases:
            gait = gait_index[name]
            heading = (math.cos(math.radians(degrees)), math.sin(math.radians(degrees)))
            blended = wp.array(gaits_numpy[gait.index], dtype=wp.transform, device=engine.device)
            placed = wp.empty_like(blended)
            ground = wp.vec3(gait.lead * heading[0], gait.lead * heading[1], 0.0)
            place_target_poses(ground, wp.vec3(heading[0], heading[1], 0.0), blended, placed, stream=engine.stream)
            target.update_from_poses(placed[:given], stream=engine.stream)
            counts = tuple(c for c in range(6, 17) if gait.token_bits >> (c - 6) & 1)
            engine.generate_animation(context, target, output, config=MotionBricksInferenceConfig(allowed_token_counts=counts))
            wp.synchronize_stream(engine.stream)
            frames = output.num_frames
            results.append(output.poses.numpy()[:frames].copy())
    engine.close()
    return results


def _batched(assets, cases, fp16=True, batch=BATCH):
    from crowdbricks import FROM_GAIT, BatchGenerator, GaitLibrary, Request, ground_placement

    BATCH = batch
    generator = BatchGenerator(assets, batch=BATCH, capacity=BATCH, fp16=fp16, cache_dir=CACHE)
    gaits = GaitLibrary(assets, generator.stream, styles=False)
    generator.set_gait_targets(gaits.targets)
    idle = gaits["idle"]
    requests = []
    for row in range(BATCH):
        name, degrees, given = cases[row % len(cases)]
        gait = gaits[name]
        heading = (math.cos(math.radians(degrees + 7 * (row // len(cases)))), math.sin(math.radians(degrees + 7 * (row // len(cases)))))
        generator.seed(row, gaits.targets[idle.index][0], ground_placement(0.0, 0.0, 1.0, 0.0))
        ground = (gait.lead * heading[0], gait.lead * heading[1])
        requests.append(Request(row, FROM_GAIT, gait.index, ground_placement(*ground, *heading), num_targets=given, token_bits=gait.token_bits))
    batch = generator.generate(requests, np.zeros(BATCH, dtype=np.float32))
    motion = generator.motion.numpy()
    results = [motion[row, : int(batch.frames[row])].copy() for row in range(len(cases))]
    gaits_numpy = gaits.targets.numpy()
    generator.close()
    return results, gaits_numpy, gaits


def _compare(assets, fp16, batch):
    batched, gaits_numpy, gaits = _batched(assets, CASES, fp16, batch)
    upstream = _upstream(assets, CASES, gaits_numpy, gaits, fp16)
    rows = []
    for (name, degrees, given), ours, theirs in zip(CASES, batched, upstream):
        assert len(ours) == len(theirs), f"{name} {degrees}: {len(ours)} frames batched, {len(theirs)} upstream"
        root = float(np.abs(ours[:, 0, :3] - theirs[:, 0, :3]).max())
        angle = _angle_degrees(ours[:, :, 3:], theirs[:, :, 3:])
        print(
            f"batch {batch} {'FP16' if fp16 else 'FP32'} {name:>5} {degrees:+6.0f} deg, {given} target poses: {len(ours)} frames, "
            f"root within {root * 100:.4f} cm, joints within {angle.max():.3f} deg (mean {angle.mean():.4f})"
        )
        rows.append((root, float(angle.max()), float(angle.mean())))
    return rows


def test_batched_kernels_match_upstream(setup):
    """One case at a time through an FP32 batch of one: upstream's own plan, so only the kernels differ."""
    for case in CASES:
        batched, gaits_numpy, gaits = _batched(setup, [case], False, 1)
        upstream = _upstream(setup, [case], gaits_numpy, gaits, False)
        ours, theirs = batched[0], upstream[0]
        assert len(ours) == len(theirs)
        root = float(np.abs(ours[:, 0, :3] - theirs[:, 0, :3]).max())
        angle = float(_angle_degrees(ours[:, :, 3:], theirs[:, :, 3:]).max())
        print(f"batch 1 FP32 {case[0]:>5} {case[1]:+6.0f} deg: {len(ours)} frames, root within {root * 100:.5f} cm, joints within {angle:.3f} deg")
        assert root < 1e-4 and angle < 0.2, (case, root, angle)


def test_bigger_batches_agree_on_average(setup):
    for root, worst, mean in _compare(setup, True, BATCH):
        assert root < 0.05 and mean < 1.0, (root, worst, mean)
