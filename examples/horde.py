# SPDX-FileCopyrightText: Copyright (c) 2026 Patch
# SPDX-License-Identifier: Apache-2.0

"""A horde: 192 characters in zombie, injured and combat gaits chase a point that circles the field.

Every character re-plans toward the point as it moves, keeps apart from its neighbours, and the
crowd batches whoever is due. At 60 Hz the whole horde steps in a few milliseconds on one GPU.

    PYTHONPATH=<motionbricks checkout> python examples/horde.py --record horde.take
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crowdbricks import BatchGenerator, Crowd, GaitLibrary, Walk  # noqa: E402
from crowdbricks.record import Recorder  # noqa: E402

# Upstream's style modes, and how often each shows up in the horde.
GAITS = (("zombie", 0.6), ("injured_leg", 0.15), ("injured_torso", 0.1), ("combat", 0.15))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=192)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--cache", default="trt-cache")
    parser.add_argument("--record", help="write a take here (see viewer/)")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import warp as wp

    from app.runtime.common.rig import load_rig_assets
    from motionbricks.model.contract import ModelRig

    wp.init()
    wp.set_device("cuda:0")
    assets = load_rig_assets(ModelRig.SOMA)
    generator = BatchGenerator(assets, batch=args.batch, capacity=args.count, fp16=not args.fp32, cache_dir=args.cache)
    gaits = GaitLibrary(assets, generator.stream)
    crowd = Crowd(generator, gaits, separation=1.1)
    recorder = Recorder(crowd) if args.record else None

    rng = np.random.default_rng(args.seed)
    goal = np.zeros(2)  # every zombie's Walk shares this array, so moving it moves all their goals
    names, weights = zip(*GAITS)
    for _ in range(args.count):
        angle = rng.uniform(0, 2 * math.pi)
        radius = rng.uniform(10, 22)
        spot = radius * np.array([math.cos(angle), math.sin(angle)])
        gait = gaits[str(rng.choice(names, p=np.array(weights) / sum(weights)))]
        crowd.spawn(spot, heading=-spot, driver=Walk(gait, goal=goal, arrive=0.0))

    delta_time = 1.0 / 60.0
    steps = []
    while crowd.clock < args.seconds:
        angle = 0.25 * crowd.clock
        goal[:] = (6 * math.cos(angle), 6 * math.sin(angle))
        begun = time.perf_counter()
        frame = crowd.step(delta_time)
        steps.append((time.perf_counter() - begun) * 1000)
        if recorder:
            recorder.capture(frame)

    sizes = [n for _, n, _ in crowd.batches]
    gpu = [ms for _, _, ms in crowd.batches]
    print(
        f"{args.count} characters, {args.seconds:.0f} s at 60 Hz: {generator.generations} generations "
        f"({generator.generations / args.seconds:.0f} a second) in {len(sizes)} batches of {np.mean(sizes):.1f} on average, "
        f"{np.median(gpu):.2f} ms GPU a batch; a crowd step takes {np.median(steps):.2f} ms (median), {np.percentile(steps, 99):.2f} ms (99th)."
    )
    if recorder:
        path = recorder.save(args.record, world={"cubes": [], "obstacles": []})
        print(f"recorded {path} ({path.stat().st_size / 1e6:.1f} MB)")
    generator.close()


if __name__ == "__main__":
    main()
