# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
P0 inference profiler for PersonaPlex.

A lightweight, opt-in profiler that measures per-frame latency (via CUDA events),
peak GPU memory, and real-time factor (RTF) for the S2S inference loop. It is a
no-op unless an active profiler is installed via `set_profiler(...)`, so the hot
path stays untouched in normal runs.

Usage (driver side, e.g. offline.py):

    from .utils.profiling import Profiler, set_profiler, profile_section

    prof = Profiler(device, frame_rate=mimi.frame_rate, warmup_frames=8)
    set_profiler(prof)
    ...
    for each frame:
        prof.begin_frame()
        with profile_section("mimi_encode"):
            codes = mimi.encode(chunk)
        tokens = lm_gen.step(...)          # internally timed sections fire
        with profile_section("mimi_decode"):
            pcm = mimi.decode(...)
        prof.end_frame()
    prof.report()

Instrumented call sites (model side) just wrap work in `profile_section("name")`;
when no profiler is installed the context manager yields immediately.
"""

import time
from contextlib import contextmanager
from collections import OrderedDict

import torch

__all__ = ["Profiler", "set_profiler", "get_profiler", "profile_section"]

_ACTIVE_PROFILER = None


def get_profiler():
    """Return the currently installed profiler, or None."""
    return _ACTIVE_PROFILER


def set_profiler(profiler):
    """Install (or clear, with None) the global profiler used by profile_section."""
    global _ACTIVE_PROFILER
    _ACTIVE_PROFILER = profiler


@contextmanager
def profile_section(name: str):
    """Time the wrapped block under `name` if a profiler is installed, else no-op."""
    prof = _ACTIVE_PROFILER
    if prof is None or not prof.enabled:
        yield
        return
    with prof.section(name):
        yield


def _percentile(sorted_vals, q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list. q in [0, 100]."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (q / 100.0) * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


class Profiler:
    """Per-frame latency / memory / RTF profiler.

    Latency is measured with CUDA events on GPU (asynchronous record, drained with
    a single synchronize per frame) and with perf_counter on CPU. The first
    `warmup_frames` frames are excluded from the stats so CUDA-graph capture and
    cache allocation do not pollute the steady-state numbers.
    """

    # Section that, if present, is treated as the wall-clock per-frame cost for RTF.
    FRAME_SECTION = "frame"

    def __init__(self, device, frame_rate: float, warmup_frames: int = 8,
                 enabled: bool = True):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.use_cuda = self.device.type == "cuda"
        self.frame_rate = float(frame_rate)
        self.frame_duration_ms = 1000.0 / self.frame_rate
        self.warmup_frames = warmup_frames
        self.enabled = enabled

        self.frame_idx = 0
        self.sections: "OrderedDict[str, list]" = OrderedDict()
        self._pending = []      # cuda: list of (name, start_event, end_event)
        self._cpu_pending = []  # cpu:  list of (name, ms)
        self.peak_mem_bytes = 0

    @property
    def _counting(self) -> bool:
        return self.frame_idx >= self.warmup_frames

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        if self.use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._pending.append((name, start, end))
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self._cpu_pending.append((name, (time.perf_counter() - t0) * 1000.0))

    def _record(self, name: str, ms: float):
        self.sections.setdefault(name, []).append(ms)

    def begin_frame(self):
        """Call at the start of each frame. Resets the per-frame peak-memory counter."""
        if self.enabled and self.use_cuda and self._counting:
            torch.cuda.reset_peak_memory_stats(self.device)

    def end_frame(self):
        """Call at the end of each frame. Drains timing events and records stats."""
        if not self.enabled:
            return
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
            if self._counting:
                for name, start, end in self._pending:
                    self._record(name, start.elapsed_time(end))
                self.peak_mem_bytes = max(
                    self.peak_mem_bytes, torch.cuda.max_memory_allocated(self.device)
                )
            self._pending.clear()
        else:
            if self._counting:
                for name, ms in self._cpu_pending:
                    self._record(name, ms)
            self._cpu_pending.clear()
        self.frame_idx += 1

    def _stats(self, vals):
        s = sorted(vals)
        n = len(s)
        return {
            "count": n,
            "mean": sum(s) / n,
            "p50": _percentile(s, 50),
            "p90": _percentile(s, 90),
            "p99": _percentile(s, 99),
            "min": s[0],
            "max": s[-1],
        }

    def report(self) -> str:
        """Build and print the summary table. Returns the report string."""
        counted = max(0, self.frame_idx - self.warmup_frames)
        lines = []
        lines.append("")
        lines.append("=" * 78)
        lines.append("P0 INFERENCE PROFILE  "
                     f"(device={self.device}, frames measured={counted}, "
                     f"warmup skipped={min(self.warmup_frames, self.frame_idx)})")
        lines.append(f"frame budget = {self.frame_duration_ms:.2f} ms "
                     f"({self.frame_rate:.2f} Hz)")
        lines.append("-" * 78)
        lines.append(f"{'section':<22}{'n':>5}{'mean':>9}{'p50':>9}"
                     f"{'p90':>9}{'p99':>9}{'max':>9}")
        lines.append(f"{'':22}{'':>5}{'ms':>9}{'ms':>9}{'ms':>9}{'ms':>9}{'ms':>9}")
        lines.append("-" * 78)

        for name, vals in self.sections.items():
            if not vals:
                continue
            st = self._stats(vals)
            lines.append(
                f"{name:<22}{st['count']:>5}{st['mean']:>9.3f}{st['p50']:>9.3f}"
                f"{st['p90']:>9.3f}{st['p99']:>9.3f}{st['max']:>9.3f}"
            )
        lines.append("-" * 78)

        # RTF: prefer an explicit wall-clock "frame" section; else sum section means.
        if self.FRAME_SECTION in self.sections and self.sections[self.FRAME_SECTION]:
            frame_mean = self._stats(self.sections[self.FRAME_SECTION])["mean"]
            rtf_basis = "measured 'frame' section"
        else:
            frame_mean = sum(self._stats(v)["mean"] for v in self.sections.values() if v)
            rtf_basis = "sum of section means (approx; ignores overlap)"
        if frame_mean > 0 or self.sections:
            rtf = frame_mean / self.frame_duration_ms if self.frame_duration_ms else float("nan")
            lines.append(f"per-frame compute : {frame_mean:.3f} ms   "
                         f"RTF = {rtf:.3f}  ({rtf_basis})")
            lines.append(f"  RTF < 1.0 => faster than real time; "
                         f"theoretical max streams/GPU (compute-bound) ~ {1.0 / rtf:.1f}"
                         if rtf and rtf == rtf and rtf > 0 else "")

        if self.use_cuda:
            lines.append(f"peak GPU memory   : {self.peak_mem_bytes / (1024 ** 3):.3f} GiB")
        lines.append("=" * 78)
        report = "\n".join(l for l in lines if l is not None)
        print(report)
        return report
