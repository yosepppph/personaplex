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

    # ---- B2 pinned-prefix test (prompt pinning on the fused path) ----
    # Each slot pins a DIFFERENT prefix length (incl. 0 = unpinned), the ring
    # then wraps well past capacity. Checks, per slot:
    #   1. write side: the pinned code/norm region is byte-identical after the
    #      wrap (write_only never touches [0, pinned)); the sub-ring did change.
    #   2. read side: reference == ground-truth dequant attention where ring
    #      slots j < pinned[b] are scored with the prefix query q2 (kernel too,
    #      when Triton is available).
    #   3. identity: q_prefix=q must reproduce the no-prefix output exactly.
    # Plus a RoPE sign-convention check: realigning the QUERY by -d must equal
    # realigning the KEYS by +d (the equivalence B2 is built on).
    if len(batch_sizes) and max(batch_sizes) >= 2:
        Bp = max(b for b in batch_sizes if b >= 2)
        pins = ([200, 0, 97, 400] * ((Bp + 3) // 4))[:Bp]
        print(f"B2 pinned-prefix test (B={Bp}): pinned={pins}")
        cache = TurboQuantRingKVCache(Bp, H, D, capacity, torch.device(device),
                                      torch.bfloat16, bits=4, rotation="haar",
                                      use_qjl_keys=False)
        # Stage writes so slot b pins exactly after its pins[b]-th frame.
        for i in range(max(pins)):
            k = torch.randn(Bp, H, 1, D, device=device)
            v = torch.randn(Bp, H, 1, D, device=device)
            cache.write_only(k, v)
            for b, p in enumerate(pins):
                if p == i + 1:
                    cache.pin_slot(b)
        pre_codes = cache.codes.clone()
        pre_norms = cache.norms.clone()
        _fill_cache(cache, Bp, H, D, capacity + 500, device)   # wrap the sub-rings
        ok_write = True
        for b, p in enumerate(pins):
            if p == 0:
                continue
            same_c = bool((cache.codes[:, b, :, :p].eq(pre_codes[:, b, :, :p])).all())
            same_n = bool((cache.norms[:, b, :, :p].eq(pre_norms[:, b, :, :p])).all())
            rolled = not bool((cache.codes[0, b, :, p:].eq(pre_codes[0, b, :, p:])).all())
            ok_write = ok_write and same_c and same_n and rolled
        print(f"  write side: prefix preserved + sub-ring rolled  "
              f"{'[PASS]' if ok_write else '[FAIL]'}")

        q = torch.randn(Bp, H, 1, D, device=device)
        q2 = torch.randn(Bp, H, 1, D, device=device)   # stands in for rope_realign(q, -d)
        pinned_t = cache.pinned                        # (B,)

        # ground truth in the ORIGINAL domain: unrotate, per-key query select.
        def _truth_prefix(q, q2, cache):
            def unpack(codes, cb, n):
                out = torch.empty(*codes.shape[:-1], codes.shape[-1] * 2,
                                  device=codes.device, dtype=torch.float32)
                out[..., 0::2] = cb[(codes & 0xF).long()]
                out[..., 1::2] = cb[(codes >> 4).long()]
                return out * n.unsqueeze(-1)
            C = cache.capacity
            kk = unpack(cache.codes[0], cache.cb_k, cache.norms[0].float()) @ cache.rot
            vv = unpack(cache.codes[1], cache.cb_v, cache.norms[1].float()) @ cache.rot
            sm = 1.0 / math.sqrt(q.shape[-1])
            l1 = torch.einsum('bhd,bhjd->bhj', q.float().squeeze(2), kk) * sm
            l2 = torch.einsum('bhd,bhjd->bhj', q2.float().squeeze(2), kk) * sm
            ar = torch.arange(C, device=q.device)
            logits = torch.where(ar[None, None, :] < cache.pinned[:, None, None], l2, l1)
            n_valid = torch.clamp(cache.end_offset, max=C)
            logits = logits.masked_fill(ar[None, None, :] >= n_valid[:, None, None],
                                        float('-inf'))
            attn = torch.softmax(logits, dim=-1)
            return torch.einsum('bhj,bhjd->bhd', attn, vv).unsqueeze(2)

        ref = tqa.turboquant_attention_reference(q, cache, q_prefix=q2)
        truth = _truth_prefix(q, q2, cache)
        rt = (ref.float() - truth.float()).abs().max().item()
        line = f"  read side:  reference-vs-truth max|d|={rt:.2e}"
        line += "  [PASS]" if rt < 2e-3 else "  [FAIL]"
        # identity: same query for prefix and conversation == no prefix at all
        rid = (tqa.turboquant_attention_reference(q, cache, q_prefix=q).float()
               - tqa.turboquant_attention_reference(q, cache).float()).abs().max().item()
        line += f"   identity(q_prefix=q) max|d|={rid:.2e}"
        line += "  [PASS]" if rid == 0.0 else "  [FAIL]"
        print(line)
        if tqa.HAS_TRITON:
            out = tqa.turboquant_attention_triton(q, cache, q_prefix=q2)
            kr = (out.float() - ref.float()).abs().max().item()
            # HAS_PREFIX variant with pinned=0 slots must match the plain kernel
            out_plain = tqa.turboquant_attention_triton(q, cache)
            b0 = [b for b, p in enumerate(pins) if p == 0]
            kid = (out[b0].float() - out_plain[b0].float()).abs().max().item() \
                if b0 else 0.0
            line = (f"  kernel-vs-reference max|d|={kr:.2e}"
                    + ("  [PASS]" if kr < 2e-3 else "  [FAIL]"))
            # (compiled variants may schedule float ops differently; allow eps)
            line += (f"   unpinned-slot identity max|d|={kid:.2e}"
                     + ("  [PASS]" if kid < 1e-5 else "  [FAIL]"))
            print(line)
        del cache, q, q2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- RoPE equivalence: query realigned by -d  ==  keys realigned by +d
        from .modules.rope import apply_rope, rope_realign
        Bq, Hq, Tq = 2, 4, 1
        qq = torch.randn(Bq, Hq, Tq, D, device=device)
        kkk = torch.randn(Bq, Hq, 8, D, device=device)
        S = torch.tensor([5000.0], device=device)          # query position
        kpos = torch.tensor([0.0], device=device)          # prompt keys at 0..7
        qr_, _ = apply_rope(qq, qq, S)
        _, kr_ = apply_rope(kkk, kkk, kpos)
        d = torch.tensor([3000.0, 1234.0], device=device)  # per-slot shifts
        lhs = torch.einsum('bhtd,bhjd->bhtj',
                           rope_realign(qr_, -d.view(Bq, 1)), kr_)
        rhs = torch.einsum('bhtd,bhjd->bhtj', qr_,
                           rope_realign(kr_, d.view(Bq, 1).expand(Bq, 8)))
        req = (lhs - rhs).abs().max().item()
        print(f"  rope q(-d) == k(+d) equivalence max|d|={req:.2e}  "
              + ("[PASS]" if req < 1e-2 else "[FAIL]"))
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
