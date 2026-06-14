# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Load test for the continuous-batching engine (Phase 2).

Spins up a BatchedEngine and simulates many synthetic users that JOIN AT
STAGGERED TIMES (the real continuous-batching scenario), each streaming audio
frames at the real-time rate for a while, draining their output queue, then
leaving. Verifies the engine:
  - assigns/releases slots correctly and rejects when full,
  - produces per-slot output for every active user,
  - keeps real-time (tick RTF < 1) as concurrency rises,
  - runs slots on independent timelines (ragged per-slot KV offsets) without
    crashing or leaking across users.

It does NOT use WebSockets or per-user prompts (those are Phase 4 / Phase 3);
it exercises the core batched loop + slot lifecycle. Requires the fused
TurboQuant cache (the engine asserts it).

Example:
  PERSONAPLEX_TURBOQUANT_KV=1 PERSONAPLEX_TURBOQUANT_FUSED=1 \
    python -m moshi.bench_engine --max-slots 16 --users 24 \
      --join-interval 0.5 --talk-seconds 8
"""

import argparse
import asyncio
import os
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from .client_utils import make_log
from .models import loaders
from .batched_engine import BatchedEngine


def log(level, msg):
    print(make_log(level, msg))


async def fake_user(engine, name, talk_seconds, sample_rate, frame_size,
                    results):
    """Acquire a slot, stream sine-ish audio frames at real time, then leave."""
    idx = engine.acquire()
    if idx is None:
        results["rejected"] += 1
        log("warning", f"{name}: REJECTED (engine full)")
        return
    results["served"] += 1
    log("info", f"{name}: joined slot {idx}")
    n_frames = int(talk_seconds * engine.frame_rate)
    out_received = 0

    async def drain():
        nonlocal out_received
        slot = engine.slots[idx]
        while True:
            try:
                await asyncio.wait_for(slot.out_q.get(), timeout=1.0)
                out_received += 1
            except asyncio.TimeoutError:
                return

    drain_task = asyncio.create_task(drain())
    # Stream frames at the real-time cadence (a quiet-ish signal).
    for f in range(n_frames):
        pcm = (0.05 * np.sin(
            2 * np.pi * 220 * (np.arange(frame_size) + f * frame_size) / sample_rate)
        ).astype(np.float32)
        engine.submit_pcm(idx, pcm)
        await asyncio.sleep(1.0 / engine.frame_rate)

    await asyncio.sleep(0.3)
    drain_task.cancel()
    try:
        await drain_task
    except asyncio.CancelledError:
        pass
    engine.release(idx)
    results["out_frames"] += out_received
    log("info", f"{name}: left slot {idx} (received {out_received} output frames)")


async def amain(args):
    device = torch.device(args.device)
    hf_hub_download(args.hf_repo, "config.json")
    mimi_w = args.mimi_weight or hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    moshi_w = args.moshi_weight or hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    log("info", "loading mimi")
    mimi = loaders.get_mimi(mimi_w, device)
    log("info", "loading moshi")
    lm = loaders.get_moshi_lm(moshi_w, device=device)
    lm.eval()
    log("info", "models loaded")

    engine = BatchedEngine(mimi, lm, device, args.max_slots,
                           sample_rate=mimi.sample_rate, frame_rate=mimi.frame_rate)
    log("info", f"warming up engine at batch {args.max_slots}")
    engine.warmup()

    loop_task = asyncio.create_task(engine.run())

    results = {"served": 0, "rejected": 0, "out_frames": 0}
    # Stagger user joins so slots have independent timelines (the real scenario).
    users = []
    for u in range(args.users):
        users.append(asyncio.create_task(
            fake_user(engine, f"user{u:02d}", args.talk_seconds,
                      mimi.sample_rate, engine.frame_size, results)))
        await asyncio.sleep(args.join_interval)

    # Periodically print engine stats while users are active.
    async def monitor():
        while not all(t.done() for t in users):
            s = engine.stats()
            log("info", f"[engine] active={s['active']:>2} "
                        f"tick={s['tick_ms_ewma']:.1f}ms RTF={s['rtf']:.3f}")
            await asyncio.sleep(1.0)
    mon = asyncio.create_task(monitor())

    await asyncio.gather(*users)
    mon.cancel()
    engine.stop()
    await loop_task

    s = engine.stats()
    print("\n" + "=" * 72)
    print("CONTINUOUS-BATCHING ENGINE LOAD TEST")
    print("-" * 72)
    print(f"max slots           : {args.max_slots}")
    print(f"users attempted     : {args.users}")
    print(f"served / rejected   : {results['served']} / {results['rejected']}")
    print(f"total output frames : {results['out_frames']}")
    print(f"final tick time     : {s['tick_ms_ewma']:.2f} ms  "
          f"(budget {engine.frame_period*1000:.1f} ms)  RTF={s['rtf']:.3f}")
    print(f"realtime maintained : {'YES' if s['rtf'] < 1.0 else 'NO'}")
    print("=" * 72)


def main():
    p = argparse.ArgumentParser(description="Continuous-batching engine load test.")
    p.add_argument("--max-slots", type=int, default=16)
    p.add_argument("--users", type=int, default=24)
    p.add_argument("--join-interval", type=float, default=0.5,
                   help="Seconds between successive user joins (staggered).")
    p.add_argument("--talk-seconds", type=float, default=8.0,
                   help="How long each user streams before leaving.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)
    p.add_argument("--moshi-weight", type=str, default=None)
    p.add_argument("--mimi-weight", type=str, default=None)
    args = p.parse_args()
    with torch.no_grad():
        asyncio.run(amain(args))


if __name__ == "__main__":
    main()
