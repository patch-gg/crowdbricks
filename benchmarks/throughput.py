# SPDX-FileCopyrightText: Copyright (c) 2026 Patch
# SPDX-License-Identifier: Apache-2.0

"""Generations per second: upstream's one-request engine against batched plans.

Every request is the same kind of work a crowd asks for: four poses of context, four target poses
of a run placed ahead. Each batch size needs its own TensorRT plan, built once into --cache.

    PYTHONPATH=<motionbricks checkout> python benchmarks/throughput.py --batches 16 48
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def time_upstream(assets, fp16: bool, cache: str, runs: int) -> tuple[float, float]:
    """Median GPU and wall milliseconds of one upstream generate_animation()."""
    import warp as wp

    from app.runtime.common.rig import model_to_runtime
    from motionbricks.inference.builder import TensorRTBuildOptions
    from motionbricks.model.config import MotionBricksInferenceConfig
    from motionbricks.model.contract import ModelRig
    from motionbricks.model.soma.contract import SOMA_N1_FP16_ONNX_PATH, SOMA_N1_FP32_ONNX_PATH
    from motionbricks.model.soma.engine import MotionBricksSOMAEngine

    from crowdbricks import GaitLibrary

    engine = MotionBricksSOMAEngine(
        SOMA_N1_FP16_ONNX_PATH if fp16 else SOMA_N1_FP32_ONNX_PATH,
        model_to_runtime=model_to_runtime(ModelRig.SOMA),
        build_options=TensorRTBuildOptions(enable_fp16=fp16),
        cache_dir=Path(cache),
        logger_severity="WARNING",
    )
    skeleton = assets.clip_registry.skeleton
    gaits = GaitLibrary(assets, engine.stream, styles=False)
    targets = gaits.targets.numpy()
    idle = targets[gaits["idle"].index, 0]
    run = gaits["run"]
    context_poses = np.repeat(idle[None], 4, axis=0)
    target_poses = targets[run.index].copy()
    target_poses[:, 0, 0] += run.lead  # placed a lead ahead, along +x
    context = engine.create_input(skeleton)
    target = engine.create_input(skeleton)
    output = engine.create_output(skeleton)
    config = MotionBricksInferenceConfig(allowed_token_counts=tuple(range(8, 15)))
    with wp.ScopedStream(engine.stream):
        context.update_from_poses(wp.array(context_poses, dtype=wp.transform, device=engine.device), stream=engine.stream)
        target.update_from_poses(wp.array(target_poses, dtype=wp.transform, device=engine.device), stream=engine.stream)
    gpu, wall = [], []
    for k in range(runs + 5):
        started = wp.Event(device=engine.device, enable_timing=True)
        finished = wp.Event(device=engine.device, enable_timing=True)
        begun = time.perf_counter()
        with wp.ScopedStream(engine.stream):
            engine.stream.record_event(started)
            engine.generate_animation(context, target, output, config=config)
            engine.stream.record_event(finished)
            wp.synchronize_stream(engine.stream)
        if k >= 5:
            wall.append((time.perf_counter() - begun) * 1000)
            gpu.append(wp.get_event_elapsed_time(started, finished))
    engine.close()
    return statistics.median(gpu), statistics.median(wall)


def time_batched(assets, batch: int, fp16: bool, cache: str, runs: int) -> tuple[float, float]:
    """Median GPU and wall milliseconds of one full batch, every row a different character."""
    from crowdbricks import FROM_GAIT, BatchGenerator, GaitLibrary, Request, ground_placement

    generator = BatchGenerator(assets, batch=batch, capacity=batch, fp16=fp16, cache_dir=cache)
    gaits = GaitLibrary(assets, generator.stream, styles=False)
    generator.set_gait_targets(gaits.targets)
    idle = gaits["idle"]
    run = gaits["run"]
    requests = []
    for character in range(batch):
        angle = 2 * math.pi * character / batch
        heading = (math.cos(angle), math.sin(angle))
        generator.seed(character, gaits.targets[idle.index][0], ground_placement(0.0, 0.0, 1.0, 0.0))
        ground = (run.lead * heading[0], run.lead * heading[1])
        requests.append(Request(character, FROM_GAIT, run.index, ground_placement(*ground, *heading), token_bits=run.token_bits))
    times = np.zeros(batch, dtype=np.float32)
    gpu, wall = [], []
    for k in range(runs + 5):
        for character in range(batch):
            generator.seed(character, gaits.targets[idle.index][0], ground_placement(0.0, 0.0, 1.0, 0.0))
        begun = time.perf_counter()
        result = generator.generate(requests, times)
        if k >= 5:
            wall.append((time.perf_counter() - begun) * 1000)
            gpu.append(result.milliseconds)
    generator.close()
    return statistics.median(gpu), statistics.median(wall)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batches", type=int, nargs="+", default=[16, 48])
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--cache", default="trt-cache")
    args = parser.parse_args()

    import warp as wp

    from app.runtime.common.rig import load_rig_assets
    from motionbricks.model.contract import ModelRig

    wp.init()
    wp.set_device("cuda:0")
    assets = load_rig_assets(ModelRig.SOMA)
    fp16 = not args.fp32
    rows = [("upstream, 1 request", 1, *time_upstream(assets, fp16, args.cache, args.runs))]
    for batch in args.batches:
        rows.append((f"batched, {batch} requests", batch, *time_batched(assets, batch, fp16, args.cache, args.runs)))
    print(f"\n{wp.get_device().name}, {'FP16' if fp16 else 'FP32'}, median of {args.runs} runs\n")
    print(f"{'':24} {'GPU ms':>8} {'wall ms':>8} {'generations/s':>14} {'ms each':>8}")
    for name, size, gpu, wall in rows:
        print(f"{name:24} {gpu:8.2f} {wall:8.2f} {size / wall * 1000:14.0f} {wall / size:8.3f}")


if __name__ == "__main__":
    main()
