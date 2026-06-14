# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Continuous-batching engine for PersonaPlex (Phase 2).

One GPU, many simultaneous conversations. A single background loop runs the
batched model at a fixed cadence (one frame per ~80 ms tick). Each tick it
gathers one input audio frame per slot (silence if a user is quiet), stacks them
into a [N, ...] batch, runs ONE Mimi.encode -> LMGen.step -> Mimi.decode, and
scatters the per-slot audio + text outputs back to each connection's queue.

Slots have independent timelines via the per-slot KV offset (Phase 1) +
LMGen.reset_slot on join, so the shared global step counter is harmless (RoPE is
relative). The fused TurboQuant attention is required for ragged per-slot
offsets, so the engine asserts it is enabled.

This module is transport-agnostic: connections push PCM frames with submit_pcm()
and read results from a slot's output queue. Phase 4 wires WebSockets to it.
Phase 3 will add per-slot voice/text prompt injection on join.
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import torch

from .models import LMGen, MimiModel
from .utils.profiling import profile_section


class SlotState(Enum):
    FREE = 0
    ACTIVE = 1
    CLOSING = 2


@dataclass
class Slot:
    idx: int
    state: SlotState = SlotState.FREE
    in_q: "asyncio.Queue[np.ndarray]" = field(default_factory=lambda: asyncio.Queue(maxsize=64))
    out_q: "asyncio.Queue[Tuple[np.ndarray, Optional[int]]]" = field(
        default_factory=lambda: asyncio.Queue(maxsize=64))


class BatchedEngine:
    """Fixed-size batch of conversation slots driven by one async tick loop."""

    def __init__(self, mimi: MimiModel, lm, device, max_slots: int,
                 sample_rate: int, frame_rate: float):
        if os.environ.get("PERSONAPLEX_TURBOQUANT_FUSED", "0") != "1" or \
           os.environ.get("PERSONAPLEX_TURBOQUANT_KV", "0") != "1":
            raise RuntimeError(
                "BatchedEngine requires the fused TurboQuant KV cache for ragged "
                "per-slot offsets. Set PERSONAPLEX_TURBOQUANT_KV=1 and "
                "PERSONAPLEX_TURBOQUANT_FUSED=1.")
        self.device = torch.device(device) if isinstance(device, str) else device
        self.mimi = mimi
        self.max_slots = max_slots
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.frame_size = int(sample_rate / frame_rate)
        self.frame_period = 1.0 / frame_rate

        self.lm_gen = LMGen(lm, sample_rate=sample_rate, device=self.device,
                            frame_rate=frame_rate)
        self.mimi.streaming_forever(max_slots)
        self.lm_gen.streaming_forever(max_slots)

        self.slots = [Slot(i) for i in range(max_slots)]
        self._silence = torch.zeros(max_slots, 1, self.frame_size,
                                    dtype=torch.float32, device=self.device)
        self._running = False
        self._tick_count = 0
        self._tick_ms_ewma = 0.0

    # ---------------- warmup ----------------

    def warmup(self, frames: int = 8):
        for _ in range(frames + int(self.lm_gen.max_delay) + 2):
            codes = self.mimi.encode(self._silence)
            for c in range(codes.shape[-1]):
                self.lm_gen.step(codes[:, :, c:c + 1])
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    # ---------------- slot lifecycle ----------------

    def acquire(self) -> Optional[int]:
        """Assign a FREE slot to a new connection, or None if the engine is full."""
        for slot in self.slots:
            if slot.state == SlotState.FREE:
                self.lm_gen.reset_slot(slot.idx)
                # drain any stale items
                for q in (slot.in_q, slot.out_q):
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                slot.state = SlotState.ACTIVE
                return slot.idx
        return None

    def release(self, idx: int):
        self.slots[idx].state = SlotState.FREE

    def submit_pcm(self, idx: int, pcm_frame: np.ndarray) -> bool:
        """Queue one input PCM frame (length frame_size) for a slot. Drops if full."""
        try:
            self.slots[idx].in_q.put_nowait(pcm_frame)
            return True
        except asyncio.QueueFull:
            return False

    def active_count(self) -> int:
        return sum(s.state == SlotState.ACTIVE for s in self.slots)

    # ---------------- the batched tick ----------------

    @torch.no_grad()
    def _tick(self):
        # Gather one input frame per slot (silence when a slot is idle/quiet).
        batch = self._silence.clone()
        active = []
        for slot in self.slots:
            if slot.state == SlotState.ACTIVE:
                active.append(slot.idx)
                if not slot.in_q.empty():
                    try:
                        pcm = slot.in_q.get_nowait()
                        batch[slot.idx, 0] = torch.from_numpy(pcm).to(self.device)
                    except asyncio.QueueEmpty:
                        pass  # quiet this frame -> silence
        if not active:
            return  # nothing to do; skip GPU work entirely

        with profile_section("frame"):
            codes = self.mimi.encode(batch)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c:c + 1])
                if tokens is None:
                    continue
                with profile_section("mimi_decode"):
                    pcm_out = self.mimi.decode(tokens[:, 1:9])  # [N,1,frame_size]
                pcm_out_cpu = pcm_out.cpu().numpy()
                text_tokens = tokens[:, 0, 0].cpu().numpy()
                for idx in active:
                    slot = self.slots[idx]
                    if slot.state != SlotState.ACTIVE:
                        continue
                    try:
                        slot.out_q.put_nowait(
                            (pcm_out_cpu[idx, 0], int(text_tokens[idx])))
                    except asyncio.QueueFull:
                        pass  # consumer fell behind; drop oldest by skipping

    # ---------------- async run loop ----------------

    async def run(self):
        """Run the engine forever at the frame cadence. Launch as a task."""
        self._running = True
        next_t = time.monotonic()
        while self._running:
            t0 = time.monotonic()
            self._tick()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.monotonic() - t0
            self._tick_count += 1
            self._tick_ms_ewma = 0.95 * self._tick_ms_ewma + 0.05 * dt * 1000
            # Maintain real-time cadence; yield to client coroutines in the gap.
            next_t += self.frame_period
            sleep = next_t - time.monotonic()
            if sleep < 0:
                next_t = time.monotonic()  # we fell behind; resync
                sleep = 0
            await asyncio.sleep(sleep)

    def stop(self):
        self._running = False

    def stats(self) -> dict:
        return {
            "ticks": self._tick_count,
            "tick_ms_ewma": self._tick_ms_ewma,
            "active": self.active_count(),
            "rtf": self._tick_ms_ewma / (self.frame_period * 1000),
        }
