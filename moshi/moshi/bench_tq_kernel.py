# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Validation + microbenchmark gate for the TurboQuant Phase 2 fused attention kernel.

Run this on the target GPU BEFORE wiring the fused kernel into the model. It:
  1. Builds a TurboQuantRingKVCache with the temporal-transformer geometry
     (H=32, D=128, capacity=3000), fills it with synthetic post-RoPE K/V.
  2. CORRECTNESS: compares the Triton kernel against the validated torch
     reference (turboquant_attention_reference) -- and the reference against the
     true bf16 dequant+SDPA path -- so we trust the math end to end.
  3. SPEED: times, per batch size, the fused kernel vs the Phase-1
     (complete()->SDPA) path for a single T==1 decode step, so we can see the
     bandwidth win before touching the serving code.

If correctness fails, the kernel must be fixed before integration. If Triton is
unavailable, only the reference path is exercised (still validates the math).

Example:
  python -m moshi.bench_tq_kernel --batch-sizes 1,2,4,8,16 --capacity 3000 --fill 3000
"""

import argparse
import math

import torch
import torch.nn.functional as F

from .modules.turboquant_ring_kv_cache import TurboQuantRingKVCache
from .modules import turboquant_attention as tqa


def _fill_cache(cache, B, H, D, n_frames, device):
    """Stream n_frames single-token K/V into the ring via write_only."""
    for _ in range(n_frames):
        k = torch.randn(B, H, 1, D, device=device)
        v = torch.randn(B, H, 1, D, device=device)
        cache.write_only(k, v)
    return k, v  # last written, unused


def _dequant_sdpa(q, cache):
    """Ground-truth bf16 path: dequantize the ring (phase-1 style) and run SDPA
    with the same validity mask the model uses. Mirrors complete()+SDPA."""
    B, H, _, D = q.shape
    C = cache.capacity

    def unpack(codes, cb, n):
        out = torch.empty(*codes.shape[:-1], codes.shape[-1] * 2,
                          device=codes.device, dtype=torch.float32)
        out[..., 0::2] = cb[(codes & 0xF).long()]
        out[..., 1::2] = cb[(codes >> 4).long()]
        return out * n.unsqueeze(-1)

    krot = unpack(cache.codes[0], cache.cb_k, cache.norms[0].float())
    vrot = unpack(cache.codes[1], cache.cb_v, cache.norms[1].float())
    k = (krot @ cache.rot).to(q.dtype)   # back to original domain
    v = (vrot @ cache.rot).to(q.dtype)
    # per-slot validity mask (B,1,1,C): slot b attends only its first end_offset[b]
    n_valid = torch.clamp(cache.end_offset, max=C)             # (B,)
    ar = torch.arange(C, device=q.device)
    invalid = ar[None, :] >= n_valid[:, None]                  # (B, C)
    bias = torch.zeros(q.shape[0], C, device=q.device)
    bias.masked_fill_(invalid, float("-inf"))
    return F.scaled_dot_product_attention(q, k, v, bias.view(q.shape[0], 1, 1, C))


def _time_ms(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run(batch_sizes, H, D, capacity, fill, device):
    print(f"triton available: {tqa.HAS_TRITON}")
    print(f"geometry: H={H} D={D} capacity={capacity} fill={fill} device={device}")
    print("=" * 72)

    for B in batch_sizes:
        cache = TurboQuantRingKVCache(B, H, D, capacity, torch.device(device),
                                      torch.bfloat16, bits=4, rotation="haar",
                                      use_qjl_keys=False)
        _fill_cache(cache, B, H, D, fill, device)
        q = torch.randn(B, H, 1, D, device=device)

        # ---- correctness ----
        ref = tqa.turboquant_attention_reference(q, cache)
        truth = _dequant_sdpa(q, cache)
        ref_vs_truth = (ref.float() - truth.float()).abs().max().item()
        cos = F.cosine_similarity(ref.float().flatten(),
                                  truth.float().flatten(), dim=0).item()
        line = (f"B={B:>3}  reference-vs-bf16SDPA: max|d|={ref_vs_truth:.2e} "
                f"cos={cos:.4f}")
        if tqa.HAS_TRITON:
            out = tqa.turboquant_attention_triton(q, cache)
            kern_vs_ref = (out.float() - ref.float()).abs().max().item()
            line += f"   kernel-vs-reference: max|d|={kern_vs_ref:.2e}"
            ok = kern_vs_ref < 2e-3
            line += "  [PASS]" if ok else "  [FAIL]"
        print(line)

        # ---- speed (fused kernel vs phase-1 dequant+SDPA) ----
        if tqa.HAS_TRITON:
            t_fused = _time_ms(lambda: tqa.turboquant_attention_triton(q, cache))
            t_phase1 = _time_ms(lambda: _dequant_sdpa(q, cache))
            speedup = t_phase1 / t_fused if t_fused > 0 else float("nan")
            print(f"        fused={t_fused:.3f} ms  phase1(dequant+SDPA)="
                  f"{t_phase1:.3f} ms  speedup={speedup:.2f}x")
        del cache, q
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- ragged per-slot offset test (continuous-batching foundation) ----
    # Give each slot a DIFFERENT end_offset and confirm the fused kernel still
    # matches the per-slot reference. This is what lets users join/leave at
    # different times: slot b attends only its own first end_offset[b] frames.
    if len(batch_sizes) and max(batch_sizes) >= 2:
        Br = max(b for b in batch_sizes if b >= 2)
        print(f"ragged per-slot offset test (B={Br}): "
              "each slot a different timeline")
        cache = TurboQuantRingKVCache(Br, H, D, capacity, torch.device(device),
                                      torch.bfloat16, bits=4, rotation="haar",
                                      use_qjl_keys=False)
        _fill_cache(cache, Br, H, D, fill, device)
        # spread offsets: full, ~half, ~quarter, tiny, ... across slots
        ragged = [capacity, capacity // 2, capacity // 4, 7][:Br]
        while len(ragged) < Br:
            ragged.append((capacity // (len(ragged) + 1)))
        cache.end_offset = torch.tensor(ragged[:Br], device=device, dtype=torch.long)
        q = torch.randn(Br, H, 1, D, device=device)
        ref = tqa.turboquant_attention_reference(q, cache)
        truth = _dequant_sdpa(q, cache)
        rt = (ref.float() - truth.float()).abs().max().item()
        line = f"  end_offset={ragged[:Br]}  reference-vs-bf16SDPA max|d|={rt:.2e}"
        if tqa.HAS_TRITON:
            out = tqa.turboquant_attention_triton(q, cache)
            kr = (out.float() - ref.float()).abs().max().item()
            line += f"  kernel-vs-reference max|d|={kr:.2e}"
            line += "  [PASS]" if kr < 2e-3 else "  [FAIL]"
        print(line)
        del cache, q
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("=" * 72)
    print("If all rows are [PASS], the kernel math is correct on this GPU and we "
          "can wire it into StreamingMultiheadAttention. If [FAIL], the Triton "
          "kernel needs fixing before integration.")


def main():
    p = argparse.ArgumentParser(description="Validate/benchmark TurboQuant fused attention.")
    p.add_argument("--batch-sizes", type=str, default="1,2,4,8,16")
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--dim-per-head", type=int, default=128)
    p.add_argument("--capacity", type=int, default=3000)
    p.add_argument("--fill", type=int, default=3000,
                   help="Frames to write before testing (>=capacity => full ring).")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    with torch.no_grad():
        run(batch_sizes, args.heads, args.dim_per_head, args.capacity,
            args.fill, args.device)


if __name__ == "__main__":
    main()
