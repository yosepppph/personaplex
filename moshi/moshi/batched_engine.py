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

Concurrency model: the GPU step runs in a dedicated worker THREAD via
run_in_executor, so the asyncio event loop stays responsive to all clients'
socket I/O while the GPU is busy (a ~50 ms inline tick would otherwise stall
every client each frame). asyncio.Queue access and slot-state mutations
(join/leave/reset) happen only on the event loop, between executor calls, so
they never race with the in-flight GPU step.

This module is transport-agnostic: connections push PCM frames with submit_pcm()
and read results from a slot's output queue. Phase 4 wires WebSockets to it.
Phase 3 will add per-slot voice/text prompt injection on join.
"""

import asyncio
import concurrent.futures
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import torch

from .models import LMGen, MimiModel
from .utils.profiling import profile_section


class SlotState(Enum):
    FREE = 0       # available
    PENDING = 1    # reserved by a new connection; reset happens on the loop next tick
    ACTIVE = 2     # streaming
    CLOSING = 3


@dataclass
class Slot:
    idx: int
    state: SlotState = SlotState.FREE
    frames_since_join: int = 0
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

        # A slot joining mid-stream is at the large shared global offset and so
        # skips the normal start-of-conversation priming. For ~max_delay frames
        # its delay pipeline is unfilled and the model emits SPECIAL tokens
        # (>= card) into the audio channels -- valid for the LM but out of range
        # for Mimi's codebook. We (1) clamp audio codes to [0, card) before
        # decode so one slot's transient can't crash the shared batch, and
        # (2) suppress that slot's output until it is primed.
        self.audio_card = int(self.lm_gen.lm_model.card)
        self.prime_frames = int(self.lm_gen.max_delay) + 2

        self.slots = [Slot(i) for i in range(max_slots)]
        self._silence = torch.zeros(max_slots, 1, self.frame_size,
                                    dtype=torch.float32, device=self.device)
        # All GPU work runs on this single dedicated thread (one CUDA stream,
        # consistent CUDA-graph replay), off the asyncio event loop.
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
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

    # ---------------- slot lifecycle (event loop only) ----------------

    def acquire(self) -> Optional[int]:
        """Reserve a FREE slot for a new connection, or None if the engine is full.

        The slot is marked PENDING; the actual model reset happens on the engine
        loop (between GPU steps) so it never races with an in-flight step.
        """
        for slot in self.slots:
            if slot.state == SlotState.FREE:
                slot.state = SlotState.PENDING
                for q in (slot.in_q, slot.out_q):
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except asyncio.QueueEmpty:
                            break
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

    def _process_joins(self):
        """Promote PENDING slots to ACTIVE, resetting their model state. Runs on
        the event loop between GPU steps, so it never races with _compute."""
        for slot in self.slots:
            if slot.state == SlotState.PENDING:
                self.lm_gen.reset_slot(slot.idx)
                slot.frames_since_join = 0
                slot.state = SlotState.ACTIVE

    # ---------------- tick split: gather (loop) / compute (thread) / scatter (loop) ----------------

    def _gather_inputs(self) -> Tuple[torch.Tensor, List[int]]:
        batch = self._silence.clone()
        active: List[int] = []
        for slot in self.slots:
            if slot.state == SlotState.ACTIVE:
                active.append(slot.idx)
                if not slot.in_q.empty():
                    try:
                        pcm = slot.in_q.get_nowait()
                        batch[slot.idx, 0] = torch.from_numpy(pcm).to(self.device)
                    except asyncio.QueueEmpty:
                        pass  # quiet this frame -> silence
        return batch, active

    @torch.no_grad()
    def _compute(self, batch: torch.Tensor):
        """GPU step. Runs in the worker thread. Touches no asyncio.Queue."""
        with profile_section("frame"):
            codes = self.mimi.encode(batch)
            tokens = None
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c:c + 1])
            if tokens is None:
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                return None
            # Clamp acoustic codes into Mimi's codebook range: a slot still in its
            # join priming window can emit special tokens (>= card) that would
            # otherwise crash the shared-batch decode.
            audio_codes = tokens[:, 1:9].clamp(0, self.audio_card - 1)
            with profile_section("mimi_decode"):
                pcm_out = self.mimi.decode(audio_codes)  # [N,1,frame_size]
            pcm_out_cpu = pcm_out.cpu().numpy()
            text_tokens = tokens[:, 0, 0].cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return pcm_out_cpu, text_tokens

    def _scatter(self, active: List[int], result):
        if result is None:
            return
        pcm_out_cpu, text_tokens = result
        for idx in active:
            slot = self.slots[idx]
            if slot.state != SlotState.ACTIVE:
                continue  # released mid-step; drop its output
            slot.frames_since_join += 1
            # Suppress priming-window output (garbage until the slot's delay
            # pipeline fills with its own tokens).
            if slot.frames_since_join <= self.prime_frames:
                continue
            try:
                slot.out_q.put_nowait((pcm_out_cpu[idx, 0], int(text_tokens[idx])))
            except asyncio.QueueFull:
                pass  # consumer fell behind; drop this frame

    # ---------------- async run loop ----------------

    async def run(self):
        """Run the engine forever at the frame cadence. Launch as a task."""
        self._running = True
        loop = asyncio.get_running_loop()
        next_t = time.monotonic()
        while self._running:
            t0 = time.monotonic()
            self._process_joins()                 # event loop
            batch, active = self._gather_inputs()  # event loop
            if active:
                # GPU work off the event loop; loop services clients meanwhile.
                result = await loop.run_in_executor(self._executor, self._compute, batch)
                self._scatter(active, result)      # event loop
                dt = time.monotonic() - t0
                self._tick_count += 1
                self._tick_ms_ewma = 0.95 * self._tick_ms_ewma + 0.05 * dt * 1000
            # Maintain real-time cadence; yield to client coroutines in the gap.
            next_t += self.frame_period
            sleep = next_t - time.monotonic()
            if sleep < 0:
                next_t = time.monotonic()  # fell behind; resync
                sleep = 0
            await asyncio.sleep(sleep)

    def stop(self):
        self._running = False
        self._executor.shutdown(wait=False)

    def stats(self) -> dict:
        return {
            "ticks": self._tick_count,
            "tick_ms_ewma": self._tick_ms_ewma,
            "active": self.active_count(),
            "rtf": self._tick_ms_ewma / (self.frame_period * 1000),
        }
