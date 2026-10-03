# GLM-5.x port plan (`glm-dsa`: GLM-5.2 / 5.3; `glm5next`: GLM-5.3-Flash)

**Status:** not started. **Policy:** port now, prove later; not admitted, not advertised
(CLAUDE.md rule 14; [ported-families-todo](ported-families-todo.md)).

## Architecture

References:
- **`glm-dsa`:** llama.cpp `src/models/glm-dsa.cpp` (local source; the vendored b10306 binaries
  know it too). TensorSharp's note: reproducing llama.cpp's indexer top-k restored 6/6 token parity.
- **`glm5next`:** llama.cpp `src/models/glm5-next.cpp` (1,013 lines, local source since the
  2026-10-03 pull to `bed0a8566`; not in b10306) for the **trunk** (KDA, NoPE MLA, k-pool DSA, MoE,
  HC). **It is not a complete reference:** it throws
  `"GLM5-Next NextN graph not implemented yet"`, so TensorSharp (`GlmDsaModel.Glm5Next*.cs`) stays the
  only implementation reference for NextN/MTP. TensorSharp's earlier "llama.cpp is not a valid
  reference for glm5next" predates this upstream file.
- Secondary for both: TensorSharp `docs/models/glm.md`, `Models/GlmDsa/` (20 files).

**`glm-dsa`** (GLM-5.2: 744B MoE, 78 layers):

| Piece | Shape / rule |
|---|---|
| Attention | MLA with weight absorption: 64 Q heads, 1 key head, q_lora 2048, kv_lora 512, head k/v 256, n_rot 64; KV cache one 576-wide row per token per layer |
| DSA indexer | 32 heads x 128, top-k 2048, on 21 of 78 layers (0, 1, 2, then every 4th from 6); in-between layers reuse the last selection; Walsh-Hadamard rotation before the F16 key cache (needed for parity) |
| MoE | 256 experts, top-8, sigmoid gating, selection-only routing bias, renormalisation, x2.5 routed scale; first 3 layers dense SwiGLU (n_ff 12288) |
| NextN/MTP | 1 trailing block |
| RoPE | NORM, base 8e6, on the 64-wide rope slice; softmax scale 1/sqrt(256) |

**`glm5next`** (GLM-5.3-Flash, 320B):
- 45 trunk layers + 1 NextN; 288 experts, top-8, 1 shared;
- **KDA linear attention** on 34 layers, with a per-layer `head_count_kv` array (0 = KDA, 1 = MLA);
- MLA + DSA on 11 layers, NoPE;
- a pooled indexer;
- **Sinkhorn hyper-connections** (×4, the DeepSeek-V4 mHC recipe);
- a SwiGLU clamp at 10.

**Real checkpoints don't fit this PC** (64 GB RAM). The smallest published quant TensorSharp
measured: **GLM-5.3 UD-Q2_K_XL ≈ 236.4 GiB in 7 shards**. GLM-5.3-Flash (320B) is likewise far
beyond 64 GB at any usable quant. So this is port + synthetic verification only until a large-RAM
or cloud host is available.

## Reuse in Stingray

- `deepseek2` MLA with absorption (admitted, verified): attention core and KV layout.
- The `deepseek32` lightning-indexer alpha: DSA scoring + top-k mask.
- The MoE router with sigmoid gating, routing bias and renormalisation (DeepSeek-V3 family code).
- For `glm5next`: the DeepSeek-V4 alpha's Sinkhorn mHC; GDN/KDA-like recurrence pieces in
  `GdnKernels` (KDA differs: per-channel decay; read first).

## New work (phases)

- [ ] **0. Read** `glm-dsa.cpp` (primary) and TensorSharp's `GlmDsaModel*.cs`. Write the spec here.
- [ ] **1. `glm-dsa`:** `ModelGraph` branch; forward pass extending the `deepseek2`/`deepseek32`
  paths (indexer layer schedule, selection reuse, Hadamard-before-F16 key cache, routed scale).
- [ ] **2a. `glm5next` trunk:** KDA layers (short conv, l2 q/k, per-channel decay), NoPE MLA +
  pooled DSA indexer, ×4 HC streams, SwiGLU clamp, MoE. Reference: `glm5-next.cpp`. A separate class
  is likely.
- [ ] **2b. `glm5next` NextN/MTP:** TensorSharp is the primary implementation reference until an
  independent implementation exists (upstream llama.cpp doesn't have one yet).
- [ ] **3. Gate:** `// glm-dsa — NOT admitted` / `// glm5next — NOT admitted` blocks.

## Deferred (not in the initial port)

Vision (GLM-OCR ViT mmproj), NextN/MTP speculative decoding (2b only after the trunk), tensor
parallelism, and GPU paths.

## Verification (levels as in [ported-families-todo](ported-families-todo.md))

- [ ] **Specification tests (level 2):** check synthetic tiny GGUFs against test-side reimplementations.
- [ ] **Independent implementation (level 3), available now for `glm-dsa`:** run the synthetic model
   through the **vendored llama.cpp b10306**, which knows `glm-dsa`: a real independent mechanics
   check without the real checkpoint. For the `glm5next` trunk, the same needs a llama.cpp build from
   `bed0a8566` or later. For NextN, only TensorSharp is available, which is a second reading, not an
   independent check.
- [ ] **Real weights (level 4):** on a large host, verify against real weights and run `stingray admit-arch` against `llama-server`.

**Effort:** port + synthetic about 1 day (`glm-dsa`) + about 1 day (`glm5next` trunk); NextN extra.
Real-weight verification (large host), the closeout performance + DRY pass, and admission are separate.
