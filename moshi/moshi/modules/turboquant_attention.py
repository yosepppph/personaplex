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

def turboquant_attention_reference(q, cache, sm_scale=None, q_prefix=None):
    """q: (B, H, 1, D). cache: TurboQuantRingKVCache. Returns (B, H, 1, D).

    Numerically identical to dequantizing the ring and running SDPA with the
    validity mask, but never materializes K/V.

    q_prefix (B2 pinning, optional): a REALIGNED copy of q (same shape) used for
    slot b's first cache.pinned[b] ring positions — its pinned prompt prefix.
    RoPE logits depend only on the query-key position difference, so scoring the
    prefix with a query realigned by -d is exactly equivalent to re-rotating the
    prefix keys by +d (Fix A in transformer.py), without touching the packed
    codes. The caller computes q_prefix = rope_realign(q, -d) per slot, with
    d_b = max(0, end_offset[b] - context): the prefix then sits at in-window
    distances [context - P_b, context - 1], contiguous with the conversation
    sub-ring at [0, context - P_b - 1]. With pinned == 0 the prefix branch
    selects nothing and the result is bit-identical to q_prefix=None.
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

    ar = torch.arange(C, device=q.device)
    logits = torch.einsum('bhd,bhjd->bhj', qr, krot) * sm_scale
    if q_prefix is not None:
        qr2 = (q_prefix.float() @ rot.T).squeeze(2)             # (B,H,D)
        logits2 = torch.einsum('bhd,bhjd->bhj', qr2, krot) * sm_scale
        is_pref = ar[None, None, :] < cache.pinned[:, None, None]  # (B,1,C)
        logits = torch.where(is_pref, logits2, logits)
    # ring overwrites oldest slots in place; with end_offset >= C every slot
    # holds a live (windowed) entry, otherwise only the first end_offset slots.
    # Per-slot: each batch element b has its own n_valid = min(end_offset[b], C)
    # so slots can have independent timelines (continuous batching). A pinned
    # prefix needs no extra mask handling: its slots [0, P_b) are always below
    # n_valid once written.
    n_valid = torch.clamp(cache.end_offset, max=C)             # (B,)
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
        QR2,           # (B*H, D)  f32   rotated REALIGNED queries for the pinned
                       # prefix (B2). Only read when HAS_PREFIX; callers pass QR
                       # again as a placeholder otherwise. Scoring prefix keys
                       # with a query realigned by -d equals re-rotating those
                       # keys by +d (RoPE logits are relative), so the packed
                       # codes are used as-is.
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
        PINNED,        # i64 ptr [B]: per-slot pinned-prefix length; ring slots
                       # j < pinned[b] are scored with QR2. Live tensor like
                       # END_OFFSET (CUDA-graph safe; 0 until pin_slot).
        C: tl.constexpr, D: tl.constexpr, HALF_D: tl.constexpr,
        NH: tl.constexpr, BLOCK_C: tl.constexpr, SM_SCALE: tl.constexpr,
        HAS_PREFIX: tl.constexpr,
    ):
        pid = tl.program_id(0)                       # one program per (b, h)
        b = pid // NH                                # slot index
        n_valid = tl.minimum(tl.load(END_OFFSET + b), C).to(tl.int32)
        dh = tl.arange(0, HALF_D)
        qe = tl.load(QR + pid * D + 2 * dh)          # q at even coords
        qo = tl.load(QR + pid * D + 2 * dh + 1)      # q at odd coords
        if HAS_PREFIX:
            p_b = tl.load(PINNED + b).to(tl.int32)   # prefix length for slot b
            qe2 = tl.load(QR2 + pid * D + 2 * dh)    # realigned prefix query
            qo2 = tl.load(QR2 + pid * D + 2 * dh + 1)

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
            if HAS_PREFIX:
                # Prefix slots score against the realigned query. Computed
                # unconditionally per block and selected with tl.where (no
                # runtime scalar branching) — with pinned[b] == 0 the where
                # selects nothing and this is an exact identity. The extra
                # FMAs reuse the already-loaded codes, so the kernel stays
                # bandwidth-bound; the HAS_PREFIX=False variant compiles all
                # of this out (unchanged when pinning is disabled).
                logits2 = (tl.sum(ke * qe2[None, :], 1)
                           + tl.sum(ko * qo2[None, :], 1)) * kn * SM_SCALE
                logits = tl.where(offs < p_b, logits2, logits)
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


def turboquant_attention_triton(q, cache, sm_scale=None, block_c: int = 128,
                                q_prefix=None):
    """Launch wrapper. q: (B, H, 1, D) -> (B, H, 1, D).

    q_prefix (B2 pinning, optional): realigned query for the pinned prompt
    prefix — see turboquant_attention_reference. Passing it selects the
    HAS_PREFIX kernel variant, a process-stable choice (the pin flag is read
    once at module init), so CUDA-graph capture sees one consistent kernel.
    """
    assert HAS_TRITON, "triton not available; use turboquant_attention_reference"
    B, H, _, D = q.shape
    C = cache.capacity
    sm_scale = sm_scale or 1.0 / math.sqrt(D)
    qr = (q.float() @ cache.rot.T).reshape(B * H, D).contiguous()
    has_prefix = q_prefix is not None
    qr2 = ((q_prefix.float() @ cache.rot.T).reshape(B * H, D).contiguous()
           if has_prefix else qr)  # placeholder ptr when unused
    out = torch.empty_like(qr)
    # Pass the persistent end_offset/pinned tensors (no host sync): the kernel
    # reads them on-device, so this is CUDA-graph-safe.
    _tq_decode_attn_kernel[(B * H,)](
        qr, qr2,
        cache.codes[0].reshape(B * H, C, D // 2),
        cache.codes[1].reshape(B * H, C, D // 2),
        cache.norms[0].reshape(B * H, C),
        cache.norms[1].reshape(B * H, C),
        cache.cb_k, cache.cb_v, out, cache.end_offset, cache.pinned,
        C=C, D=D, HALF_D=D // 2, NH=H, BLOCK_C=block_c, SM_SCALE=sm_scale,
        HAS_PREFIX=has_prefix,
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
