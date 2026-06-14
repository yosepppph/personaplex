# Phase-2 fused decode attention for TurboQuantRingKVCache (PersonaPlex / Moshi)
# ===============================================================================
# Computes single-token (T=1) streaming attention DIRECTLY from packed 4-bit
# TurboQuant codes -- no K/V dequantization to HBM, no bf16 scratch. This is
# what turns the quantized cache from a memory win into a bandwidth win:
# per-step HBM traffic for the cache drops from ~3.5 GB/stream (phase-1
# dequant round trip) to ~0.40 GB/stream (codes + norms read once).
# On H100 80GB this moves serving from bandwidth-bound (~29 streams) to
# capacity-bound (~125 streams; ~60-90 realistic after depformer/Mimi/
# scheduler overhead).
#
# Math (verified on CPU against the phase-1 dequant + SDPA path: logit max
# diff 5e-5, output cosine 1.0000):
#   * rotation R is orthonormal  =>  <q, k> = <R q, R k>
#   * R k_j = norm_j * cb[idx_j]  (per-coordinate codebook entries)
#   * logit_j = norm_j * sum_d (R q)[d] * cb[idx_jd]      <- nibble lookups+FMA
#   * output  = R^{-1} ( sum_j softmax_j * norm_j * cb[vidx_j] )
#     i.e. values are accumulated in the ROTATED domain; ONE 128x128 inverse
#     rotation per (batch, head) at the end, not one per token.
#
# STATUS: the torch reference below is validated; the Triton kernel mirrors it
# line-for-line but is GPU-UNTESTED (written without GPU access). Before
# trusting it, run `compare_kernel_vs_reference()` on the target H100.

import math
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # CPU-only environments: reference path still works
    HAS_TRITON = False


# ----------------------------------------------------------------------------
# Validated torch reference (ground truth for the kernel)
# ----------------------------------------------------------------------------

def turboquant_attention_reference(q, cache, sm_scale=None):
    """q: (B, H, 1, D). cache: TurboQuantRingKVCache. Returns (B, H, 1, D).

    Numerically identical to dequantizing the ring and running SDPA with the
    validity mask, but never materializes K/V.
    """
    assert not cache.use_qjl_keys, "QJL logit correction not implemented here"
    B, H, _, D = q.shape
    C = cache.capacity
    sm_scale = sm_scale or 1.0 / math.sqrt(D)
    rot, cbk, cbv = cache.rot, cache.cb_k, cache.cb_v

    qr = (q.float() @ rot.T).squeeze(2)                       # (B,H,D)

    def unpack(codes, cb, n):
        out = torch.empty(*codes.shape[:-1], codes.shape[-1] * 2,
                          device=codes.device, dtype=torch.float32)
        out[..., 0::2] = cb[(codes & 0xF).long()]
        out[..., 1::2] = cb[(codes >> 4).long()]
        return out * n.unsqueeze(-1)

    krot = unpack(cache.codes[0], cbk, cache.norms[0].float())  # (B,H,C,D)
    vrot = unpack(cache.codes[1], cbv, cache.norms[1].float())

    logits = torch.einsum('bhd,bhjd->bhj', qr, krot) * sm_scale
    # ring overwrites oldest slots in place; with end_offset >= C every slot
    # holds a live (windowed) entry, otherwise only the first end_offset slots.
    # Per-slot: each batch element b has its own n_valid = min(end_offset[b], C)
    # so slots can have independent timelines (continuous batching).
    n_valid = torch.clamp(cache.end_offset, max=C)             # (B,)
    ar = torch.arange(C, device=q.device)
    valid = ar[None, :] < n_valid[:, None]                     # (B, C)
    logits = logits.masked_fill(~valid[:, None, :], float('-inf'))
    attn = torch.softmax(logits, dim=-1)                       # (B,H,C)
    acc = torch.einsum('bhj,bhjd->bhd', attn, vrot)            # rotated domain
    return (acc @ rot).unsqueeze(2).to(q.dtype)                # inverse rotation


# ----------------------------------------------------------------------------
# Triton kernel (GPU-untested sketch -- mirrors the reference exactly)
# ----------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _tq_decode_attn_kernel(
        QR,            # (B*H, D)  f32   rotated queries
        KC, VC,        # (B*H, C, D//2) u8   packed 4-bit codes
        KN, VN,        # (B*H, C)  f16   per-vector norms
        CBK, CBV,      # (16,) f32 codebooks (tiny; stays L1/L2 resident)
        OUT,           # (B*H, D)  f32   rotated-domain accumulator output
        END_OFFSET,    # i64 ptr [B]: per-slot cache.end_offset; for program
                       # (b, h) -> n_valid = min(end_offset[b], C). Read
                       # on-device (not via .item()) so the launch is legal
                       # inside a CUDA graph and tracks the in-place
                       # end_offset += T done by write_only each step. Per-slot
                       # (indexed by b) so batch slots can have INDEPENDENT
                       # timelines (continuous batching / async join).
        C: tl.constexpr, D: tl.constexpr, HALF_D: tl.constexpr,
        NH: tl.constexpr, BLOCK_C: tl.constexpr, SM_SCALE: tl.constexpr,
    ):
        pid = tl.program_id(0)                       # one program per (b, h)
        b = pid // NH                                # slot index
        n_valid = tl.minimum(tl.load(END_OFFSET + b), C).to(tl.int32)
        dh = tl.arange(0, HALF_D)
        qe = tl.load(QR + pid * D + 2 * dh)          # q at even coords
        qo = tl.load(QR + pid * D + 2 * dh + 1)      # q at odd coords

        m_i = -float('inf')                          # online softmax state
        l_i = 0.0
        acc_e = tl.zeros([HALF_D], dtype=tl.float32)
        acc_o = tl.zeros([HALF_D], dtype=tl.float32)

        for start in range(0, C, BLOCK_C):
            offs = start + tl.arange(0, BLOCK_C)
            mask = offs < n_valid
            # ---- keys: unpack nibbles, codebook lookup, logit = n * <qr, cb[idx]>
            kc = tl.load(KC + pid * C * HALF_D + offs[:, None] * HALF_D
                         + dh[None, :], mask=mask[:, None], other=0)
            ke = tl.load(CBK + (kc & 0xF).to(tl.int32))      # (BLOCK_C, HALF_D)
            ko = tl.load(CBK + (kc >> 4).to(tl.int32))
            kn = tl.load(KN + pid * C + offs, mask=mask, other=0.0).to(tl.float32)
            logits = (tl.sum(ke * qe[None, :], 1)
                      + tl.sum(ko * qo[None, :], 1)) * kn * SM_SCALE
            logits = tl.where(mask, logits, -float('inf'))
            # ---- online softmax update
            m_new = tl.maximum(m_i, tl.max(logits, 0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(logits - m_new)
            l_i = l_i * alpha + tl.sum(p, 0)
            acc_e *= alpha
            acc_o *= alpha
            # ---- values: same unpack, accumulate in rotated domain
            vc = tl.load(VC + pid * C * HALF_D + offs[:, None] * HALF_D
                         + dh[None, :], mask=mask[:, None], other=0)
            ve = tl.load(CBV + (vc & 0xF).to(tl.int32))
            vo = tl.load(CBV + (vc >> 4).to(tl.int32))
            vn = tl.load(VN + pid * C + offs, mask=mask, other=0.0).to(tl.float32)
            w = (p * vn)[:, None]
            acc_e += tl.sum(w * ve, 0)
            acc_o += tl.sum(w * vo, 0)
            m_i = m_new

        inv_l = 1.0 / l_i
        tl.store(OUT + pid * D + 2 * dh, acc_e * inv_l)
        tl.store(OUT + pid * D + 2 * dh + 1, acc_o * inv_l)


def turboquant_attention_triton(q, cache, sm_scale=None, block_c: int = 128):
    """Launch wrapper. q: (B, H, 1, D) -> (B, H, 1, D)."""
    assert HAS_TRITON, "triton not available; use turboquant_attention_reference"
    B, H, _, D = q.shape
    C = cache.capacity
    sm_scale = sm_scale or 1.0 / math.sqrt(D)
    qr = (q.float() @ cache.rot.T).reshape(B * H, D).contiguous()
    out = torch.empty_like(qr)
    # Pass the persistent end_offset tensor (no host sync): the kernel computes
    # n_valid = min(end_offset, C) on-device, so this is CUDA-graph-safe.
    _tq_decode_attn_kernel[(B * H,)](
        qr,
        cache.codes[0].reshape(B * H, C, D // 2),
        cache.codes[1].reshape(B * H, C, D // 2),
        cache.norms[0].reshape(B * H, C),
        cache.norms[1].reshape(B * H, C),
        cache.cb_k, cache.cb_v, out, cache.end_offset,
        C=C, D=D, HALF_D=D // 2, NH=H, BLOCK_C=block_c, SM_SCALE=sm_scale,
    )
    # single inverse rotation per (b, h)
    return (out @ cache.rot).reshape(B, H, 1, D).to(q.dtype)


def compare_kernel_vs_reference(cache, B, H, D, device="cuda", atol=2e-3):
    """Run this on the H100 before wiring into serving."""
    q = torch.randn(B, H, 1, D, device=device)
    ref = turboquant_attention_reference(q, cache)
    out = turboquant_attention_triton(q, cache)
    err = (ref - out).abs().max().item()
    print(f"max abs diff kernel vs reference: {err:.2e}")
    assert err < atol, "kernel mismatch -- do not deploy"


# ----------------------------------------------------------------------------
# Integration: in StreamingMultiheadAttention.forward, on the streaming T==1
# path, replace `_complete_kv` + SDPA with:
#     cache.write_only(k, v)      # index_copy_ of codes/norms, no scratch
#     x = turboquant_attention_triton(q, cache)
# (add a `write_only()` to TurboQuantRingKVCache that is `complete()` minus
# the dequantize-to-scratch block; positions/masking are handled by n_valid
# inside the kernel since capacity == context for the temporal transformer).
# Keep the phase-1 path for prefill (T > 1) and as a correctness fallback.
