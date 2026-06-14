# Continuous Batching for PersonaPlex — Design

Goal: serve many simultaneous real-time conversations on one GPU by stacking each
user's current frame into a single batched `LMGen.step()`, instead of the current
single-user server (`streaming_forever(1)` + a global `asyncio.Lock`). The batch
benchmark already proved the GPU can do ~16 users at RTF 0.67 with the TurboQuant
fused KV cache; this work makes the *server* actually feed the GPU that batch.

## The core constraint (why this is multi-phase)

1. **Shared offset.** `RingKVCache.end_offset` (and `_MHAState.offset`,
   `_LMGenState.offset`) are single scalars shared across the batch. Every slot
   writes the same ring position and is assumed to be at the same timestep. Real
   users join at different times and run different lengths, so slots need
   **independent timelines** → per-slot offset `(B,)`. Without it a reused slot
   would attend to the previous occupant's KV (correctness + privacy bug).

2. **Per-user prompts.** Each user supplies a voice prompt + a text/system prompt
   (e.g. the recipe), injected over many frames. A joining slot must be
   prompt-injected *while other slots keep streaming* — never by stalling the
   batch.

## Phases

- **Phase 1 — per-slot offsets (foundation).** `end_offset`/positions/masking
  become per-slot in `RingKVCache` and `TurboQuantRingKVCache`; the fused kernel
  reads `end_offset[b]`; the bf16 SDPA path builds a per-slot `attn_bias`.
  Backward-compatible: with all slots synchronized (B=1, or the bench's
  synchronized batch) behaviour is identical. Also a per-slot `reset_slot(b)`.
  Test: extend `bench_tq_kernel` to fill slots to *different* depths and verify
  the kernel matches a per-slot dequant+SDPA reference.

- **Phase 2 — batched engine.** A `BatchedEngine` owning the batched `LMGen`/Mimi
  and a single async loop running every frame: gather one input frame per active
  slot → `[N,K,1]` → one `step()` → scatter outputs to per-slot queues. Slot
  table with states FREE/PROMPTING/ACTIVE/CLOSING. Idle slots are masked (their
  per-slot offset simply doesn't advance / their output is dropped).

- **Phase 3 — per-slot prompts.** Inject each joining user's voice + text prompt
  into their slot using the batched `provided`/forced-token machinery, advancing
  only that slot's offset, while active slots consume their live audio in the
  same `step()`.

- **Phase 4 — server wiring.** `handle_chat` registers a slot with the engine,
  pushes decoded audio frames into the slot's input queue, and forwards the
  slot's output queue to the WebSocket. Remove the global lock.

- **Phase 5 — validation.** Multi-client real-time load test; measure sustained
  concurrent users/GPU and per-user latency under load.

## Lifecycle (per connection)
connect → assign FREE slot (reject if full) → reset_slot(b) → PROMPTING (inject
voice+recipe) → ACTIVE (stream audio in/out) → disconnect → CLOSING → FREE.

## Invariants
- The batched `step()` runs once per global tick; every slot present in the batch
  advances exactly one frame.
- A slot's attention window only ever covers frames since *its* join (per-slot
  `n_valid`), so no cross-user leakage and no attending to pre-join silence.
- CUDA graphs stay intact: all per-slot offset reads are on-device (no `.item()`).
