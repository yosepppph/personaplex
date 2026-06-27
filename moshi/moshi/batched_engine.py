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
Phase 3a injects a per-slot recipe/text system prompt on join; Phase 3b teacher-forces
a per-slot voice prompt (acquire takes text_prompt_tokens and voice_codes). The voice
prompt must be Mimi codes from a .wav clip (see encode_voice_wav) -- the packaged .pt
voice embeddings use a different (whole-batch) code path that can't be applied per slot.
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
from .models.lm import (
    load_audio,
    normalize_audio,
    _iterate_audio,
    encode_from_sphn,
)
from .utils.profiling import profile_section


class SlotState(Enum):
    FREE = 0       # available
    PENDING = 1    # reserved by a new connection; reset happens on the loop next tick
    PRIMING = 2    # injecting the per-slot system prompt (silence + recipe text) on join
    ACTIVE = 3     # streaming
    CLOSING = 4


@dataclass
class Slot:
    idx: int
    state: SlotState = SlotState.FREE
    frames_since_join: int = 0
    # Per-slot priming script: a list of text-token ids force-fed one per tick while
    # PRIMING (silence frames use zero_text_code; recipe tokens carry the prompt).
    prime_script: List[int] = field(default_factory=list)
    # Phase 3b: the matching per-frame agent (moshi) audio codes [L, 8] force-fed
    # alongside prime_script. Voice-prompt codes for the first Tv frames, silence
    # after. None until acquire() builds it.
    prime_agent: Optional["torch.Tensor"] = None
    prime_cursor: int = 0
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

        # Match the single-user server's accent-consistency tuning: tighter audio
        # sampling reduces accent drift (was the LMGen default temp=0.8 / top_k=250).
        self.lm_gen = LMGen(lm, sample_rate=sample_rate, device=self.device,
                            frame_rate=frame_rate, temp=0.65, top_k=80)
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

        # Per-slot priming (Phase 3a): on join a slot is force-fed a short system
        # prompt -- silence frames + the recipe text tokens + silence -- to fill the
        # delay pipeline and condition the conversation, mirroring the single-user
        # step_system_prompts (minus voice, which is Phase 3b). During priming the
        # agent (moshi) stream is forced silent and the user stream is the sine frame,
        # matching _step_text_prompt_core in lm.py.
        self.zero_text_code = int(self.lm_gen.zero_text_code)
        self.silence_frames = max(1, int(0.5 * frame_rate))
        self._sine_codes = self.lm_gen._encode_sine_frame()    # [1, 8, 1] long
        self._zero_codes = self.lm_gen._encode_zero_frame()    # [1, 8, 1] long
        # Single silence frame as a [1, 8] row, used to default the per-slot agent
        # (moshi) priming stream to silence before overwriting the voice-prompt span.
        self._silence_row = self._zero_codes.reshape(1, 8)
        # Cap voice-prompt length so a long clip can't make a join wait forever
        # (the slot is silent/PRIMING for the whole voice span). 10 s at frame rate.
        self.max_voice_frames = int(10 * frame_rate)

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

    def acquire(self, text_prompt_tokens: Optional[List[int]] = None,
                voice_codes: Optional[torch.Tensor] = None) -> Optional[int]:
        """Reserve a FREE slot for a new connection, or None if the engine is full.

        `text_prompt_tokens` is the tokenized recipe/system prompt to inject on join.
        `voice_codes` ([8, Tv] Mimi audio codes) is the per-slot voice prompt (Phase
        3b); None falls back to the default voice (Phase 3a behaviour).
        The slot is marked PENDING; the actual model reset + priming start happens on
        the engine loop (between GPU steps) so it never races with an in-flight step.
        """
        for slot in self.slots:
            if slot.state == SlotState.FREE:
                slot.state = SlotState.PENDING
                slot.prime_script, slot.prime_agent = self._build_prime_script(
                    text_prompt_tokens, voice_codes)
                slot.prime_cursor = 0
                for q in (slot.in_q, slot.out_q):
                    while not q.empty():
                        try:
                            q.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                return slot.idx
        return None

    def _build_prime_script(self, text_prompt_tokens: Optional[List[int]],
                            voice_codes: Optional[torch.Tensor] = None):
        """Build the per-frame priming script: voice -> silence -> recipe text -> silence,
        mirroring the single-user `step_system_prompts`.

        Returns (text_script, agent_script):
          - text_script: list[int], one forced text token per frame (zero_text_code
            except during the recipe-text span).
          - agent_script: LongTensor [L, 8], the forced agent (moshi) audio codes per
            frame -- the voice-prompt codes for the first Tv frames, silence after.
        Both have the same length L so prime_cursor indexes both.
        """
        Tv = 0 if voice_codes is None else int(voice_codes.shape[-1])
        voice_text = [self.zero_text_code] * Tv
        silence = [self.zero_text_code] * self.silence_frames
        text = list(text_prompt_tokens) if text_prompt_tokens else []
        text_script = voice_text + silence + text + silence
        L = len(text_script)
        # Default every frame's agent stream to silence, then overwrite the voice span.
        agent_script = self._silence_row.repeat(L, 1).clone()      # [L, 8]
        if Tv > 0:
            # voice_codes is [8, Tv] -> [Tv, 8] to align with per-frame rows.
            agent_script[:Tv] = voice_codes.transpose(0, 1).to(agent_script.dtype)
        return text_script, agent_script

    @torch.no_grad()
    def encode_voice_wav(self, voice_mimi: MimiModel, wav_path: str) -> Optional[torch.Tensor]:
        """Encode a voice-prompt WAV into Mimi audio codes [8, Tv] for teacher-forcing
        during a slot's priming.

        Uses a DEDICATED `voice_mimi` (not the shared streaming engine mimi) so it can
        run on the event loop without corrupting the batch's streaming state. Mirrors
        the single-user `load_voice_prompt` (-24 LUFS mono) + `_encode_voice_prompt_frames`.
        """
        audio = load_audio(wav_path, self.sample_rate)              # (C, T)
        audio = normalize_audio(audio, self.sample_rate, -24.0)     # mono, -24 LUFS
        if audio.ndim == 1:
            audio = audio[None, :]
        voice_mimi.reset_streaming()
        frames = [chunk.reshape(1, 8, -1) for chunk in encode_from_sphn(
            voice_mimi,
            _iterate_audio(audio, sample_interval_size=self.frame_size, pad=True),
            max_batch=1,
        )]
        voice_mimi.reset_streaming()
        if not frames:
            return None
        codes = torch.cat(frames, dim=-1)[0]                       # [8, Tv]
        if codes.shape[-1] > self.max_voice_frames:
            codes = codes[:, :self.max_voice_frames]
        return codes.to(self.device)

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
        """Promote PENDING slots to PRIMING, resetting their model state. Runs on
        the event loop between GPU steps, so it never races with _compute. The slot
        stays PRIMING until its prompt script is force-fed (see _advance_priming),
        then flips to ACTIVE."""
        for slot in self.slots:
            if slot.state == SlotState.PENDING:
                self.lm_gen.reset_slot(slot.idx)
                slot.frames_since_join = 0
                slot.prime_cursor = 0
                slot.state = SlotState.PRIMING

    def _advance_priming(self):
        """Advance each PRIMING slot's cursor by one frame (run AFTER the step that
        consumed the current cursor). When the script is exhausted the slot is
        primed -> ACTIVE and its output is no longer suppressed."""
        for slot in self.slots:
            if slot.state == SlotState.PRIMING:
                slot.prime_cursor += 1
                if slot.prime_cursor >= len(slot.prime_script):
                    slot.state = SlotState.ACTIVE
                    slot.frames_since_join = 0

    # ---------------- tick split: gather (loop) / compute (thread) / scatter (loop) ----------------

    def _gather_inputs(self) -> Tuple[torch.Tensor, List[int], List[int], torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self._silence.clone()
        active: List[int] = []
        priming: List[int] = []
        # Per-slot teacher-forcing for this tick: priming slots force the scripted
        # text token + the scripted agent audio codes (voice prompt during its span,
        # silence otherwise); live slots free-run (force=False).
        force_cpu = [False] * self.max_slots
        text_cpu = [self.zero_text_code] * self.max_slots
        # Default the agent (moshi) stream to silence for every slot; priming slots
        # overwrite their row with the scripted codes (voice prompt or silence).
        moshi = self._zero_codes.expand(self.max_slots, -1, -1).clone()  # [B, 8, 1]
        for slot in self.slots:
            if slot.state == SlotState.ACTIVE:
                active.append(slot.idx)
                if not slot.in_q.empty():
                    try:
                        pcm = slot.in_q.get_nowait()
                        batch[slot.idx, 0] = torch.from_numpy(pcm).to(self.device)
                    except asyncio.QueueEmpty:
                        pass  # quiet this frame -> silence
            elif slot.state == SlotState.PRIMING:
                priming.append(slot.idx)
                force_cpu[slot.idx] = True
                c = slot.prime_cursor
                text_cpu[slot.idx] = slot.prime_script[c]
                moshi[slot.idx, :, 0] = slot.prime_agent[c]
        force = torch.tensor(force_cpu, dtype=torch.bool, device=self.device)
        text = torch.tensor(text_cpu, dtype=torch.long, device=self.device)
        return batch, active, priming, force, text, moshi

    @torch.no_grad()
    def _compute(self, batch: torch.Tensor, force: torch.Tensor, text: torch.Tensor,
                 moshi: torch.Tensor):
        """GPU step. Runs in the worker thread. Touches no asyncio.Queue.

        `force` ([B] bool) / `text` ([B] long) / `moshi` ([B, 8, 1] long) drive per-slot
        priming: where force is True the user stream is overridden with the sine frame,
        the agent stream is teacher-forced to `moshi` (the voice-prompt codes during the
        voice span, silence otherwise), and the text token is teacher-forced. Where False
        the slot free-runs (force write is a no-op), so passing these every tick is safe
        even when nothing is priming."""
        with profile_section("frame"):
            codes = self.mimi.encode(batch)                         # [B, 8, T]
            # Priming slots: user stream = sine frame (not the encoded silence).
            codes = torch.where(force.view(-1, 1, 1), self._sine_codes, codes)
            tokens = None
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(
                    input_tokens=codes[:, :, c:c + 1],
                    moshi_tokens=moshi[:, :, c:c + 1],
                    text_token=text,
                    force_mask=force,
                )
            if tokens is None:
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                return None
            # Clamp acoustic codes into Mimi's codebook range (defensive: a slot can
            # still emit special tokens >= card during early priming).
            audio_codes = tokens[:, 1:9].clamp(0, self.audio_card - 1)
            with profile_section("mimi_decode"):
                pcm_out = self.mimi.decode(audio_codes)  # [N,1,frame_size]
            pcm_out_cpu = pcm_out.cpu().numpy()
            text_tokens = tokens[:, 0, 0].cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return pcm_out_cpu, text_tokens

    def _scatter(self, active: List[int], result):
        # Only ACTIVE slots emit; PRIMING slots are stepped (to fill the pipeline)
        # but their output is suppressed.
        if result is None:
            return
        pcm_out_cpu, text_tokens = result
        for idx in active:
            slot = self.slots[idx]
            if slot.state != SlotState.ACTIVE:
                continue  # released mid-step; drop its output
            slot.frames_since_join += 1
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
            batch, active, priming, force, text, moshi = self._gather_inputs()  # event loop
            if active or priming:
                # GPU work off the event loop; loop services clients meanwhile.
                result = await loop.run_in_executor(
                    self._executor, self._compute, batch, force, text, moshi)
                self._scatter(active, result)      # event loop
                self._advance_priming()            # event loop: advance after the step
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
