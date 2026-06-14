# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Batch-scaling benchmark for PersonaPlex.

Answers the core cost question for a one-GPU / many-simultaneous-users service:
as batch size (= number of concurrent conversations stacked into one forward)
grows, how does per-frame latency and peak GPU memory scale, and where is the
ceiling -- compute (frame time hits the 80 ms real-time budget) or memory (VRAM
fills up)?

It drives the LM the same way offline.py does (mimi.encode -> lm_gen.step ->
mimi.decode) but with synthetic zero audio at batch B, so it needs no input WAV
or voice prompt. It reuses the P0 profiler for per-frame timing.

For each batch size it reports:
  - temporal_transformer / depformer / mimi_decode / frame mean latency (ms)
  - RTF (frame_ms / 80 ms); RTF < 1 => the whole batch keeps up in real time
  - peak GPU memory (GiB)
  - users/GPU implied by this batch size if RTF < 1
A final summary table makes the compute-vs-memory ceiling obvious.

Toggle TurboQuant KV with PERSONAPLEX_TURBOQUANT_KV=1 to see its effect on the
memory ceiling at high batch (where KV cache, not weights, dominates VRAM).

Example:
  python -m moshi.bench_batch --batch-sizes 1,2,4,8,16,24,32 --frames 120
"""

import argparse
import os
from typing import List, Optional

import torch
from huggingface_hub import hf_hub_download

from .client_utils import make_log
from .models import loaders, LMGen
from .utils.profiling import Profiler, set_profiler, profile_section


def log(level: str, msg: str):
    print(make_log(level, msg))


def _drive_frames(mimi, lm_gen, batch_size: int, frame_size: int, device: str,
                  n_frames: int, prof: Optional[Profiler]):
    """Feed n_frames of synthetic zero audio at the given batch size."""
    for _ in range(n_frames):
        if prof is not None:
            prof.begin_frame()
        chunk = torch.zeros(batch_size, 1, frame_size, dtype=torch.float32, device=device)
        ctx = profile_section("frame") if prof is not None else _nullctx()
        with ctx:
            codes = mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = lm_gen.step(codes[:, :, c : c + 1])
                if tokens is not None:
                    if prof is not None:
                        with profile_section("mimi_decode"):
                            _ = mimi.decode(tokens[:, 1:9])
                    else:
                        _ = mimi.decode(tokens[:, 1:9])
        if prof is not None:
            prof.end_frame()


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def run_bench(batch_sizes: List[int], frames: int, warmup_frames: int,
              device: str, hf_repo: str, moshi_weight: Optional[str],
              mimi_weight: Optional[str], cpu_offload: bool):
    turbo = os.environ.get("PERSONAPLEX_TURBOQUANT_KV", "0") == "1"
    fused = os.environ.get("PERSONAPLEX_TURBOQUANT_FUSED", "0") == "1"
    if turbo:
        log("info", f"TurboQuant KV cache: ENABLED (4-bit, "
                    f"{'fused kernel' if fused else 'phase-1 dequant'})")
    else:
        log("info", "TurboQuant KV cache: disabled (bf16)")

    # Load Mimi + LM once; only streaming state is rebuilt per batch size.
    hf_hub_download(hf_repo, "config.json")
    if mimi_weight is None:
        mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)  # type: ignore
    if moshi_weight is None:
        moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)  # type: ignore
    log("info", "loading mimi")
    mimi = loaders.get_mimi(mimi_weight, device)
    log("info", "loading moshi")
    lm = loaders.get_moshi_lm(moshi_weight, device=device, cpu_offload=cpu_offload)
    lm.eval()
    log("info", "models loaded")

    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    frame_rate = mimi.frame_rate

    results = []
    for B in batch_sizes:
        log("info", f"=== batch size {B} ===")
        lm_gen = None
        try:
            # Fresh streaming state at batch B. Inside the try so an OOM while
            # allocating the KV cache is caught and reported (and the sweep can
            # stop cleanly) instead of crashing before the summary prints.
            lm_gen = LMGen(lm, device=device, sample_rate=mimi.sample_rate,
                           frame_rate=frame_rate)
            mimi.streaming_forever(B)
            lm_gen.streaming_forever(B)

            # Warmup: prime CUDA graphs + delay buffers at this batch size.
            _drive_frames(mimi, lm_gen, B, frame_size, device,
                          warmup_frames + max(int(lm_gen.max_delay), 0) + 2, prof=None)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            prof = Profiler(device, frame_rate=frame_rate, warmup_frames=0)
            set_profiler(prof)
            _drive_frames(mimi, lm_gen, B, frame_size, device, frames, prof)
            set_profiler(None)
            prof.report()

            def _mean(name):
                v = prof.sections.get(name, [])
                return sum(v) / len(v) if v else float("nan")

            frame_ms = _mean(prof.FRAME_SECTION)
            rtf = frame_ms / prof.frame_duration_ms
            peak_gib = prof.peak_mem_bytes / (1024 ** 3)
            results.append({
                "B": B,
                "temporal": _mean("temporal_transformer"),
                "depformer": _mean("depformer"),
                "frame": frame_ms,
                "rtf": rtf,
                "peak_gib": peak_gib,
                "realtime": rtf < 1.0,
            })
            stop_sweep = False
        except RuntimeError as e:
            set_profiler(None)
            msg = str(e).split("\n")[0]
            is_oom = "out of memory" in msg.lower()
            log("error", f"batch {B} failed: {msg}")
            results.append({"B": B, "oom": is_oom, "error": msg})
            # Larger batch sizes will only need more memory, so stop on OOM.
            stop_sweep = is_oom
        finally:
            # Drop streaming state to free KV caches before the next batch size.
            # Guarded: a mid-init OOM can leave state partially set, and
            # reset_streaming() would raise on the un-set modules, masking the
            # real error. _stop_streaming just nulls state everywhere.
            try:
                if lm_gen is not None:
                    lm_gen._stop_streaming()
                mimi._stop_streaming()
            except Exception:
                pass
            del lm_gen
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if stop_sweep:
            log("info", f"stopping sweep at batch {B} (OOM); larger batches would also OOM")
            break

    # ---- scaling summary ----
    print("\n" + "=" * 86)
    print(f"BATCH-SCALING SUMMARY  (device={device}, "
          f"turboquant_kv={'on' if turbo else 'off'}, "
          f"frame budget=80.00 ms)")
    print("-" * 86)
    print(f"{'batch':>6}{'temporal':>11}{'depformer':>11}{'frame':>10}"
          f"{'RTF':>8}{'peakGiB':>10}{'realtime?':>11}{'~ms/user':>11}")
    print(f"{'(users)':>6}{'ms':>11}{'ms':>11}{'ms':>10}{'':>8}{'':>10}{'':>11}{'':>11}")
    print("-" * 86)
    best = None
    for r in results:
        if "error" in r:
            tag = "OOM" if r.get("oom") else "ERR"
            print(f"{r['B']:>6}{'':>11}{'':>11}{'':>10}{'':>8}{'':>10}{tag:>11}")
            continue
        per_user = r["frame"] / r["B"]
        flag = "yes" if r["realtime"] else "NO"
        print(f"{r['B']:>6}{r['temporal']:>11.3f}{r['depformer']:>11.3f}"
              f"{r['frame']:>10.3f}{r['rtf']:>8.3f}{r['peak_gib']:>10.3f}"
              f"{flag:>11}{per_user:>11.3f}")
        if r["realtime"]:
            best = r
    print("-" * 86)
    if best is not None:
        print(f"Max real-time batch found: {best['B']} concurrent users/GPU "
              f"(RTF={best['rtf']:.3f}, peak={best['peak_gib']:.2f} GiB). "
              f"Cost/user scales ~1/batch.")
        print("If 'realtime?' flips to NO before peakGiB nears your VRAM => "
              "COMPUTE-bound (weight quant helps). If peakGiB fills first => "
              "MEMORY-bound (TurboQuant KV helps).")
    print("=" * 86)


def main():
    p = argparse.ArgumentParser(description="PersonaPlex batch-scaling benchmark.")
    p.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,24,32",
                   help="Comma-separated batch sizes to sweep.")
    p.add_argument("--frames", type=int, default=120,
                   help="Measured steady-state frames per batch size.")
    p.add_argument("--warmup-frames", type=int, default=8,
                   help="Warmup frames before measuring (graphs + delay priming).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)
    p.add_argument("--moshi-weight", type=str, default=None)
    p.add_argument("--mimi-weight", type=str, default=None)
    p.add_argument("--cpu-offload", action="store_true")
    args = p.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    with torch.no_grad():
        run_bench(batch_sizes, args.frames, args.warmup_frames, args.device,
                  args.hf_repo, args.moshi_weight, args.mimi_weight,
                  args.cpu_offload)


if __name__ == "__main__":
    main()
