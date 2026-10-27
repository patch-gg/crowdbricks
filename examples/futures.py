# SPDX-FileCopyrightText: Copyright (c) 2026 Patch
# SPDX-License-Identifier: Apache-2.0

"""Every future at once: one character forks into 48, and each future takes a different way out.

The character stands in the middle of a ring of obstacles. At FORK_AT it becomes 48 copies with
the same past, and every copy heads out along its own ray: some hop a box, some duck under a
lamp, some sit on a stool, some vault a wall, at walking, jogging or running speed. Their first
motions all come from one batched model call; after that each one re-plans on its own schedule
and the crowd batches whoever is due.

    PYTHONPATH=<motionbricks checkout> python examples/futures.py --record futures.take
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crowdbricks import Approach, BatchGenerator, Crowd, GaitLibrary, Obstacle, Stand, Walk, obstacle_pool, sit  # noqa: E402
from crowdbricks.record import Recorder  # noqa: E402

FUTURES = 48
FORK_AT = 0.6
# Around the ring, each ray's obstacle in turn: kind, distance from the middle, and the gaits that
# alternate along the rays of that kind.
RING = (
    ("hop", 6.0, ("jog", "run")),
    ("duck", 7.0, ("walk", "jog")),
    ("sit", 8.5, ("walk", "walk")),
    ("vault", 10.5, ("run", "jog")),
)
# How far each future carries on after its obstacle before it stops.
CARRY_ON = {"hop": 2.5, "duck": 2.5, "vault": 2.0}
# The Solid Cube each Smart Primitive carries in upstream's soma_gym: height of its center, turn
# from the primitive, and size (width, depth, height).
CUBES = {
    "hop": (0.2, 90.0, (0.6, 0.6, 0.6)),
    "vault": (0.39313891530036926, 90.0, (0.25, 2.0, 0.8500000238418579)),
    "duck": (1.9499998092651367, 90.0, (0.5, 0.5, 1.0)),
    "sit": (0.19396841526031494, 0.0, (0.4, 0.4, 0.4)),
}


def build_ring(assets, stream, futures: int = FUTURES):
    """One obstacle per future, on rays spread evenly around the middle."""
    pool = obstacle_pool(assets)
    rays = []
    for k in range(futures):
        angle = 2 * math.pi * k / futures
        kind, radius, gaits = RING[k % len(RING)]
        heading = np.array([math.cos(angle), math.sin(angle)])
        position = radius * heading
        # Hop, duck and vault primitives face their approach turned a quarter; seats face it directly.
        yaw = math.degrees(angle) - (0.0 if kind == "sit" else 90.0)
        obstacle = Obstacle(pool, kind, tuple(position), yaw, stream, name=f"{kind}_{k}")
        rays.append({"angle": angle, "heading": heading, "kind": kind, "gait": gaits[(k // len(RING)) % 2], "obstacle": obstacle})
    return rays


def cubes_for(rays) -> list[dict]:
    cubes = []
    for ray in rays:
        obstacle = ray["obstacle"]
        z, turn, size = CUBES[obstacle.kind]
        cubes.append({
            "name": obstacle.name,
            "kind": obstacle.kind,
            "center": [float(obstacle.position[0]), float(obstacle.position[1]), z],
            "yaw": obstacle.yaw + math.radians(turn),
            "size": list(size),
        })
    return cubes


def drive_future(crowd: Crowd, character: int, ray: dict) -> None:
    gaits = crowd.gaits
    gait = gaits[ray["gait"]]
    obstacle = ray["obstacle"]
    if obstacle.kind == "sit":
        crowd.drive(character, Approach(obstacle, gait, interaction=lambda o, then: sit(o)))
        return
    carry_on = Walk(gait, heading=ray["heading"], distance=CARRY_ON[obstacle.kind], then=Stand(heading=ray["heading"]))
    crowd.drive(character, Approach(obstacle, gait, then=carry_on))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--futures", type=int, default=FUTURES)
    parser.add_argument("--batch", type=int, default=FUTURES, help="TensorRT batch size (a plan is built once per size)")
    parser.add_argument("--seconds", type=float, default=9.0)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--cache", default="trt-cache", help="where TensorRT plans are kept")
    parser.add_argument("--record", help="write a take here (see viewer/)")
    args = parser.parse_args()

    from app.runtime.common.rig import load_rig_assets
    from motionbricks.model.contract import ModelRig

    import warp as wp

    wp.init()
    wp.set_device("cuda:0")
    assets = load_rig_assets(ModelRig.SOMA)
    generator = BatchGenerator(assets, batch=args.batch, capacity=max(args.futures, args.batch), fp16=not args.fp32, cache_dir=args.cache)
    gaits = GaitLibrary(assets, generator.stream)
    crowd = Crowd(generator, gaits)
    rays = build_ring(assets, generator.stream, args.futures)
    recorder = Recorder(crowd) if args.record else None

    delta_time = 1.0 / 60.0
    first = crowd.spawn(np.zeros(2), heading=(1.0, 0.0))
    forked = False
    started = time.perf_counter()
    while crowd.clock < args.seconds:
        if not forked and crowd.clock >= FORK_AT:
            everyone = [first, *crowd.fork(first, args.futures - 1)]
            for character, ray in zip(everyone, rays):
                drive_future(crowd, character, ray)
            forked = True
        frame = crowd.step(delta_time)
        if recorder:
            recorder.capture(frame)
    elapsed = time.perf_counter() - started

    sizes = [n for _, n, _ in crowd.batches]
    fork_batch = next(((n, ms) for c, n, ms in crowd.batches if c >= FORK_AT), None)
    print(
        f"{args.futures} futures, {args.seconds:.1f} s simulated in {elapsed:.2f} s: "
        f"{generator.generations} generations in {len(sizes)} batches (mean {np.mean(sizes):.1f} a batch). "
        f"The fork's batch: {fork_batch[0]} motions in {fork_batch[1]:.2f} ms."
    )
    if recorder:
        world = {"obstacles": [ray["obstacle"].describe() for ray in rays], "cubes": cubes_for(rays), "fork_at": FORK_AT}
        path = recorder.save(args.record, world=world)
        print(f"recorded {path} ({path.stat().st_size / 1e6:.1f} MB)")
    generator.close()


if __name__ == "__main__":
    main()
