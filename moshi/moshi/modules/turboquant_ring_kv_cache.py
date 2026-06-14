# TurboQuant RingKVCache for PersonaPlex (Moshi temporal transformer)
# ====================================================================
# Drop-in replacement for `RingKVCache` in moshi/moshi/modules/transformer.py,
# implementing TurboQuant (Zandieh, Daliri, Hadian, Mirrokni 2025,
# arXiv:2504.19874) faithfully to the paper:
#
#   * Stage 1 (Q_mse, Sec 3.1): random rotation -> unit-normalize -> per-
#     coordinate optimal scalar quantization with the Lloyd-Max codebook for
#     the EXACT Beta distribution f(x) ~ (1 - x^2)^((d-3)/2) of coordinates of
#     a random unit vector (not the Gaussian limit). L2 norms stored in fp16
#     and used to rescale at dequantization, as in the paper.
#   * Rotation: the paper uses a Haar rotation (QR of an i.i.d. Gaussian
#     matrix). At D = dim_per_head = 128 this is a cheap 128x128 matmul and is
#     the default here ("haar"). A sign-flip + fast-Hadamard surrogate is
#     provided ("hadamard") for when matmul cost matters.
#   * Stage 2 (Q_prod, Secs 2.2 & 3.2, OPTIONAL, default OFF): MSE stage at
#     (b-1) bits + 1-bit QJL on the residual, where QJL uses an INDEPENDENT
#     i.i.d. Gaussian sketch S: codes = sign(S r), dequant adds
#     ||r|| * sqrt(pi/2)/D * S^T sign(S r). This yields an UNBIASED inner-
#     product estimator (paper Lemma 4: variance <= pi/(2D) ||q||^2).
#
#     Measured note (synthetic, D=128, outlier-heavy vectors): at equal TOTAL
#     bits, pure Q_mse beat Q_prod on mean logit error across 2-4 bits
#     (e.g. 4b: 3.3% vs 7.7%; 2b: 14.4% vs 26.0%); QJL's benefit is the
#     unbiasedness guarantee / worst-case bound, not average error. Hence
#     default use_qjl=False at 4-bit; revisit if you push keys very low and
#     observe systematic logit bias in the offline eval.
#
# Target geometry (loaders.py, 7B temporal transformer):
#   num_layers=32, dim=4096, num_heads=32 -> dim_per_head=128, context=3000
# Per-stream persistent KV, all layers:
#   bf16: ~1.46 GB   ->   4-bit codes + fp16 norms: ~0.38 GB  (~3.9x)
#
# Phase 1 (this file): quantize-on-write, dequantize-on-read into a SHARED
#   bf16 scratch consumed by unmodified F.scaled_dot_product_attention.
#   Concurrency win only. Phase 2 (separate): fused kernel computing logits
#   from packed codes (+ the QJL term, which corrects LOGITS, not vectors).
#   SINGLE-USE CONTRACT: the tensors in the returned KVCacheResult alias the
#   shared scratch and are only valid until the next complete() on ANY cache
#   with the same (B, H, capacity, D, dtype). This matches real serving
#   (SDPA consumes them immediately in the same layer) but do not retain them.
#
# End-to-end test vs verbatim upstream ring (100 streamed frames, wraparound,
# outlier-heavy synthetic KV, D=128, C=64): positions bit-identical to
# upstream; bits=4 MSE keys -> logit rel-err 3.4%, attention-output cosine
# 0.950; bits=4 with QJL keys (3b MSE + 1b QJL) -> 7.7% / 0.917, confirming
# the default below.
# Streaming/CUDA-graph notes: all ops fixed-shape, no data-dependent control
#   flow; k arrives post-RoPE in StreamingMultiheadAttention -- quantize as-is.

import math
import typing as tp

import torch

from .transformer import KVCacheResult  # upstream named tuple


# ----------------------------------------------------------------------------
# Codebooks and rotations
# ----------------------------------------------------------------------------

def beta_lloyd_max_codebook(bits: int, d: int, iters: int = 120) -> torch.Tensor:
    """Optimal scalar quantizer for coordinates of a random unit vector in R^d.

    Solves the continuous 1-D k-means of paper Sec 3.1 against the exact pdf
    f(x) ~ (1 - x^2)^((d-3)/2) on [-1, 1] (Lemma 1). Computed once at init.
    """
    n = 2 ** bits
    grid = torch.linspace(-1 + 1e-9, 1 - 1e-9, 400_001, dtype=torch.float64)
    pdf = (1 - grid ** 2) ** ((d - 3) / 2)
    levels = torch.linspace(-2.5 / math.sqrt(d), 2.5 / math.sqrt(d), n,
                            dtype=torch.float64)
    for _ in range(iters):
        bounds = (levels[1:] + levels[:-1]) / 2
        idx = torch.bucketize(grid, bounds)
        mass = torch.zeros(n, dtype=torch.float64).scatter_add_(0, idx, pdf)
        mean = torch.zeros(n, dtype=torch.float64).scatter_add_(0, idx, pdf * grid)
        levels = torch.where(mass > 0, mean / mass.clamp_min(1e-18), levels)
    return levels.float()  # (2**bits,), sorted, codomain ~ [-1, 1]


def haar_rotation(d: int, seed: int, device) -> torch.Tensor:
    """Random rotation as in the paper: QR of an i.i.d. Gaussian matrix."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(d, d, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))  # unique, uniform over O(d)
    return q.float().to(device)


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard over last dim (power of 2), orthonormal/self-inverse.
    Surrogate for the Haar rotation; pair with a random sign flip."""
    d = x.shape[-1]
    assert d & (d - 1) == 0
    h, y = 1, x.clone()
    while h < d:
        y = y.view(*y.shape[:-1], d // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).reshape(*y.shape[:-3], d)
        h *= 2
    return y / math.sqrt(d)


# ----------------------------------------------------------------------------
# TurboQuant ring cache
# ----------------------------------------------------------------------------

class TurboQuantRingKVCache:
    """TurboQuant replacement for RingKVCache. Same `complete(k, v)` contract.

    bits: TOTAL bit budget per coordinate for keys and values.
    use_qjl_keys: if True, keys use the paper's Q_prod = (bits-1)-bit Q_mse
        + 1-bit QJL residual (unbiased logits); values always use Q_mse at
        `bits` (values are read post-softmax; MSE is the right objective).
    """

    _scratch: tp.Dict[tuple, torch.Tensor] = {}

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        dim_per_head: int,
        capacity: int,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
        bits: int = 4,
        rotation: str = "haar",          # "haar" (paper) | "hadamard" (fast)
        use_qjl_keys: bool = False,      # see measured note in header
        seed: int = 1234,
    ):
        assert bits == 4, "packing below assumes two 4-bit codes per byte"
        D = dim_per_head
        self.capacity, self.dtype, self.bits = capacity, dtype, bits
        self.batch_size, self.num_heads = batch_size, num_heads
        self.use_qjl_keys = use_qjl_keys
        self.rotation = rotation

        if rotation == "haar":
            self.rot = haar_rotation(D, seed, device)            # (D, D)
        else:
            g = torch.Generator(device="cpu").manual_seed(seed)
            self.signs = ((torch.randint(0, 2, (D,), generator=g) * 2 - 1)
                          .float().to(device))

        k_bits = bits - 1 if use_qjl_keys else bits
        self.cb_k = beta_lloyd_max_codebook(k_bits, D).to(device)
        self.cb_v = beta_lloyd_max_codebook(bits, D).to(device)
        self.bnd_k = ((self.cb_k[1:] + self.cb_k[:-1]) / 2).contiguous()
        self.bnd_v = ((self.cb_v[1:] + self.cb_v[:-1]) / 2).contiguous()

        # ring storage: packed codes + per-vector fp16 L2 norm (k=0, v=1)
        self.codes = torch.zeros((2, batch_size, num_heads, capacity, D // 2),
                                 device=device, dtype=torch.uint8)
        self.norms = torch.zeros((2, batch_size, num_heads, capacity),
                                 device=device, dtype=torch.float16)

        if use_qjl_keys:
            # independent i.i.d. Gaussian sketch S (paper Def 1) -- NOT the
            # stage-1 rotation; unbiasedness depends on this independence.
            g2 = torch.Generator(device="cpu").manual_seed(seed + 1)
            self.S = torch.randn(D, D, generator=g2).to(device)
            self.k_res_bits = torch.zeros(
                (batch_size, num_heads, capacity, D // 8),
                device=device, dtype=torch.uint8)
            self.k_res_norm = torch.zeros((batch_size, num_heads, capacity),
                                          device=device, dtype=torch.float16)

        # Per-slot offset (B,): each batch slot has an INDEPENDENT timeline so
        # users can join/leave at different times (continuous batching). With all
        # slots synchronized (the single-user server, the batch benchmark) every
        # entry is equal and behaviour is identical to a shared scalar offset.
        self.end_offset = torch.zeros(batch_size, device=device, dtype=torch.long)

    # ---------------- core transforms ----------------

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        if self.rotation == "haar":
            return x.float() @ self.rot.T
        return hadamard_transform(x.float() * self.signs)

    def _unrotate(self, r: torch.Tensor) -> torch.Tensor:
        if self.rotation == "haar":
            return r @ self.rot
        return hadamard_transform(r) * self.signs

    def _encode(self, x, codebook, bounds, four_bit_pack=True):
        """rotate -> unit-normalize -> Beta-Lloyd-Max -> (codes, norm, residual)."""
        r = self._rotate(x)                                     # (B,H,T,D) f32
        nrm = r.norm(dim=-1)                                    # (B,H,T)
        u = r / nrm.clamp_min(1e-6).unsqueeze(-1)               # unit vectors
        idx = torch.bucketize(u, bounds).clamp_(0, len(codebook) - 1)
        deq = codebook[idx] * nrm.unsqueeze(-1)
        residual = r - deq
        if four_bit_pack:
            packed = (idx[..., 0::2] | (idx[..., 1::2] << 4)).to(torch.uint8)
        else:  # 3-bit codes stored unpacked in u8 nibbles for sketch simplicity
            packed = (idx[..., 0::2] | (idx[..., 1::2] << 4)).to(torch.uint8)
        return packed, nrm.to(torch.float16), residual

    def _decode_into(self, out, codes, norms, codebook):
        lo = (codes & 0xF).long().clamp_(0, len(codebook) - 1)
        hi = (codes >> 4).long().clamp_(0, len(codebook) - 1)
        deq = torch.empty(*codes.shape[:-1], codes.shape[-1] * 2,
                          device=codes.device, dtype=torch.float32)
        deq[..., 0::2] = codebook[lo]
        deq[..., 1::2] = codebook[hi]
        deq *= norms.float().unsqueeze(-1)
        out.copy_(self._unrotate(deq).to(self.dtype))

    # ---------------- public API (mirrors upstream RingKVCache) ----------------

    def reset(self):
        """Reset every slot's timeline (whole batch)."""
        self.end_offset.zero_()

    def reset_slot(self, b: int) -> None:
        """Reset a single slot's timeline to 0 (a new user takes slot b).

        No scrubbing needed: the per-slot n_valid = min(end_offset[b], capacity)
        masks every position >= end_offset[b], so the previous occupant's
        leftover codes are never attended to until b overwrites them.
        """
        self.end_offset[b] = 0

    def complete(self, k: torch.Tensor, v: torch.Tensor) -> KVCacheResult:
        # Phase-1 (dequant -> SDPA) fallback path. Per-slot ragged offsets are
        # only supported on the fused path (write_only); here we require the
        # batch to be synchronized so the shared-offset position math is valid.
        assert k.shape[:-1] == v.shape[:-1], (k.shape, v.shape)
        B, H, T, D = k.shape
        assert bool((self.end_offset == self.end_offset[0]).all()), (
            "complete() requires a synchronized batch; use the fused path "
            "(PERSONAPLEX_TURBOQUANT_FUSED=1) for ragged per-slot offsets."
        )
        eo = self.end_offset[:1]  # representative scalar-shaped offset

        indexes = torch.arange(T, device=self.end_offset.device,
                               dtype=self.end_offset.dtype) + eo
        indexes = indexes % self.capacity

        k_codes, k_nrm, k_res = self._encode(k, self.cb_k, self.bnd_k)
        v_codes, v_nrm, _ = self._encode(v, self.cb_v, self.bnd_v)
        self.codes[0].index_copy_(2, indexes, k_codes)
        self.codes[1].index_copy_(2, indexes, v_codes)
        self.norms[0].index_copy_(2, indexes, k_nrm)
        self.norms[1].index_copy_(2, indexes, v_nrm)

        if self.use_qjl_keys:
            z = (k_res @ self.S.T) > 0                          # sign(S r)
            packed_bits = sum((z[..., i::8].to(torch.uint8) << i)
                              for i in range(8))
            self.k_res_bits.index_copy_(2, indexes, packed_bits)
            self.k_res_norm.index_copy_(2, indexes,
                                        k_res.norm(dim=-1).to(torch.float16))

        self.end_offset.add_(T)  # all slots (synchronized) advance together

        # dequantize-on-read into shared scratch (phase 1)
        skey = (self.codes.device, B, H, self.capacity, D, self.dtype)
        scratch = self._scratch.get(skey)
        if scratch is None:
            scratch = torch.empty(2, B, H, self.capacity, D,
                                  device=self.codes.device, dtype=self.dtype)
            self._scratch[skey] = scratch
        self._decode_into(scratch[0], self.codes[0], self.norms[0], self.cb_k)
        self._decode_into(scratch[1], self.codes[1], self.norms[1], self.cb_v)
        if self.use_qjl_keys:
            # paper-exact unbiased key dequant: Q_mse^-1 + ||r|| sqrt(pi/2)/D S^T z
            bits = self.k_res_bits
            z = torch.empty(*bits.shape[:-1], D, device=bits.device,
                            dtype=torch.float32)
            for i in range(8):
                z[..., i::8] = (((bits >> i) & 1).float() * 2 - 1)
            corr = (self.k_res_norm.float().unsqueeze(-1)
                    * (math.sqrt(math.pi / 2) / D) * (z @ self.S))
            scratch[0].add_(self._unrotate(corr).to(self.dtype))

        # position bookkeeping: identical to upstream (synchronized => use eo)
        idx_all = torch.arange(self.capacity, device=self.end_offset.device,
                               dtype=torch.long)
        invalid = idx_all >= eo
        end_index = eo % self.capacity
        delta = idx_all - end_index
        positions = torch.where(delta <= 0, eo + delta,
                                eo + delta - self.capacity)
        positions = torch.where(invalid, torch.full_like(positions, -1),
                                positions)
        return KVCacheResult(scratch[0], scratch[1], positions)

    def write_only(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Phase 2: quantize-and-store k, v into the ring WITHOUT dequantizing.

        This is `complete()` minus the dequant-to-scratch block and minus the
        position bookkeeping (the fused kernel masks via n_valid = min(
        end_offset, capacity), valid because capacity == context here). Pair
        with `turboquant_attention_triton(q, self)` / `_reference`, which read
        the packed codes directly. QJL keys are not supported on this path.
        """
        assert not self.use_qjl_keys, "fused path does not support use_qjl_keys"
        assert k.shape[:-1] == v.shape[:-1], (k.shape, v.shape)
        B, H, T, D = k.shape
        assert T == 1, "streaming write_only handles one frame at a time (T==1)"
        k_codes, k_nrm, _ = self._encode(k, self.cb_k, self.bnd_k)  # (B,H,1,Dh),(B,H,1)
        v_codes, v_nrm, _ = self._encode(v, self.cb_v, self.bnd_v)
        # Per-slot write position: each slot writes at its OWN ring position
        # (end_offset[b] % C). When the batch is synchronized these are all equal
        # and this matches the previous shared index_copy_ exactly.
        write_pos = self.end_offset % self.capacity                # (B,)
        bidx = torch.arange(B, device=write_pos.device)
        self.codes[0][bidx, :, write_pos, :] = k_codes[:, :, 0, :]
        self.codes[1][bidx, :, write_pos, :] = v_codes[:, :, 0, :]
        self.norms[0][bidx, :, write_pos] = k_nrm[:, :, 0]
        self.norms[1][bidx, :, write_pos] = v_nrm[:, :, 0]
        self.end_offset += T

    def asdict(self):
        d = {"codes": self.codes, "norms": self.norms,
             "end_offset": self.end_offset}
        if self.use_qjl_keys:
            d.update({"k_res_bits": self.k_res_bits,
                      "k_res_norm": self.k_res_norm})
        return d


# ----------------------------------------------------------------------------
# Wiring (only upstream change, in StreamingMultiheadAttention._init_streaming_state):
#   kv_cache = TurboQuantRingKVCache(batch_size, self.num_heads, dim_per_head,
#                                    capacity, device, dtype,
#                                    bits=4, rotation="haar", use_qjl_keys=False)
# Gate behind a flag; enable for the temporal transformer only (depformer
# context=8 is negligible and uses the weights_per_step path). Validate with
# the repo's offline eval (WAV -> audio + transcript) A/B vs bf16 baseline.
