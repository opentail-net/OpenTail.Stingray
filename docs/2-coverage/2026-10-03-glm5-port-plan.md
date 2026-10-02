# GLM-5.x port plan (`glm-dsa`: GLM-5.2 / 5.3; `glm5next`: GLM-5.3-Flash)

**Status:** not started. **Policy:** port now, prove later; not admitted, not advertised
(CLAUDE.md rule 14; [ported-families-todo](ported-families-todo.md)).

## Architecture

Source: TensorSharp `docs/models/glm.md`, `Models/GlmDsa/` (20 files). **Primary reference for
`glm-dsa`: llama.cpp `src/models/glm-dsa.cpp`**, present locally in
`examples/llama.cpp/llama.cpp` and known to the vendored b10306. TensorSharp's note: reproducing
llama.cpp's indexer top-k restored 6/6 token parity. For `glm5next`, TensorSharp is the only
reference; it says llama.cpp is not a valid reference there.

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

**Checkpoints don't fit this machine** (GLM-5.3 GGUF about 765 GB; Flash 320B), so this is a
port + synthetic verification only, until a large-RAM or cloud host is available.

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
- [ ] **2. `glm5next`:** KDA layers (short conv, l2 q/k, per-channel decay), pooled indexer, ×4 HC
  streams, SwiGLU clamp. A separate class is likely.
- [ ] **3. Gate:** `// glm-dsa — NOT admitted` / `// glm5next — NOT admitted` blocks.

## Verification

1. Synthetic tiny GGUFs vs spec-written reference forwards. For `glm-dsa`, the synthetic model can
   also run through the **vendored llama.cpp b10306**, which knows `glm-dsa`: a real independent
   check of the mechanics even without the real checkpoint.
2. A real checkpoint needs a large host: then `stingray admit-arch` against `llama-server` for
   `glm-dsa`, and TensorSharp for `glm5next`.

**Effort:** about 1 day (`glm-dsa`) + about 1 day (`glm5next`).
