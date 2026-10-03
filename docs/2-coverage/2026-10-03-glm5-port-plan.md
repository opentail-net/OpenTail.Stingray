# GLM-5.x port plan (`glm-dsa`: GLM-5.2 / 5.3; `glm5next`: GLM-5.3-Flash)

**Status (revised 2026-10-03): PORTED, SMOKE-TESTED ONLY.** Both variants have code and a finite-logit smoke test; `glm5next` lacks K-pool sparse selection; no numeric oracle has run. **Policy:** port now, prove later; not admitted, not advertised
(CLAUDE.md rule 14; [ported-families-todo](ported-families-todo.md)).

## Architecture & Upstream References

**Primary & secondary references:**
- **`glm-dsa`** (GLM-5.2 / 5.3):
  - Primary: local llama.cpp `src/models/glm-dsa.cpp` (770 lines; vendored b10306 binaries know it too, serving as local oracle).
  - Secondary: TensorSharp `Models/GlmDsa/` (reproducing llama.cpp indexer top-k restored 6/6 token parity).
  - Hugging Face: `GlmMoeDsaForCausalLM`.
- **`glm5next`** (GLM-5.3-Flash):
  - Primary: current upstream llama.cpp `src/models/glm5-next.cpp` (1,013 lines in local source checkout `bed0a8566`, covering KDA, NoPE MLA, K-pool DSA, MoE, mHC).
  - Secondary: Hugging Face `Glm5NextForConditionalGeneration`.
  - Secondary: TensorSharp `Models/GlmDsa/` (`Glm5Next*.cs`).
  - Local oracle: vendored b10306 does not yet carry `glm5next`; an updated llama.cpp build is required for level-3 parity.

---

### 1. `glm-dsa` (GLM-5.2: 744B MoE, 78 layers)

| Piece | Exact specification |
|---|---|
| **MLA Attention** | 64 attention heads / 64 KV heads (decompressed). `q_lora_rank = 2048`, `kv_lora_rank = 512`. `qk_head_dim = 256` (`qk_nope_head_dim = 192`, `qk_rope_head_dim = 64`), `v_head_dim = 256`. Cached latent: $512 \text{ (KV)} + 64 \text{ (positional RoPE)} = 576$ values/token/layer. $W_{kv\_a}$ produces the 576-wide compressed row. |
| **DSA Indexer** | 32 heads $\times$ 128 key length, `top_k = 2048`. Present on 21 of 78 layers: default schedule is layers 0, 1, 2, then every 4th layer from 6 (or overridden by `attention.indexer_types`). In-between layers reuse the `top_k` selection from the preceding full layer. |
| **Hadamard Rotation** | In-place orthonormal Sylvester Walsh-Hadamard transform (`1/sqrt(128)`) applied to `indexer_q` and `indexer_k` before KV caching (matching `PrismHadamard` FWHT). |
| **MoE & FFN** | First 3 layers are dense SwiGLU (`n_ff = 12288`). Layers 3..77: 256 routed experts + 1 shared expert (`expert_shared_count = 1`), `top_k = 8`. Sigmoid scoring (`expert_gating_func = 2`), biased selection by `exp_probs_b`, unbiased probabilities, probability renormalization (`expert_weights_norm = true`), $\times 2.5$ routed scaling (`expert_weights_scale = 2.5`). |
| **RoPE & Scaling** | NORM RoPE (interleaved) on 64-wide slice, base 8e6. Softmax scale $\text{kq\_scale} = \text{mscale}^2 / \sqrt{256}$. YaRN scaling parameters pre-divided by 0.1 at load time. |

---

### 2. `glm5next` (GLM-5.3-Flash, 320B)

| Piece | Exact specification |
|---|---|
| **Layer Schedule** | 45 trunk layers with a strict 3:1 repeating structure (34 KDA + 11 DSA/MLA layers):<br>• `i % 4 != 3` $\to$ **KDA** (layers 0, 1, 2, 4, 5, 6, ..., 42, 44)<br>• `i % 4 == 3` $\to$ **DSA/MLA** (layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43)<br>Overrideable via GGUF `attention.head_count_kv` array (0 = KDA, 1 = MLA). |
| **KDA Recurrence** | Distinct linear attention recurrence (not a generic GDN clone): 64 heads, 128 head dimension, short conv kernel 4, gate lower bound -5. Channel-wise decay, state equations, gating, decay normalization, and dedicated state layout. |
| **NoPE MLA / Attention** | `q_lora_rank = 1536`, `qk_nope_head_dim = 256`, `qk_rope_head_dim = 0` (**NO Q/K RoPE** on main attention), `mla_use_nope = true`, `v_head_dim = 256`. Main attention carries zero rotary embedding; positional information is handled by the indexer. |
| **K-Pool Indexer** | Dedicated stateful subsystem: `index_kpool = 4` (4-token pool construction), `index_kpool_compress = true`, `index_kpool_always_select_tail = true`, `index_topk = 2048`. Pooled K layout, token $\to$ pool coordinate mapping, sequence-local pool state, and incremental updates across decode. |
| **Sinkhorn mHC** | $\times 4$ hyper-connection streams (`hc_mult = 4`), Sinkhorn iteration count `hc_sinkhorn_iters = 20`, `hc_eps = 1e-6`. Row/column normalization, pre/post residual mixing in FP32 arithmetic. |
| **MoE & Router** | First 3 layers dense SwiGLU; remaining 42 layers MoE (`mlp_layer_types`: dense $\times 3$, sparse $\times 42$). 288 routed experts + 1 shared expert, top-8 routed, sigmoid scoring, **router in FP32** (`moe_router_dtype = float32`), probability renormalization, $\times 2.5$ routed scaling. SwiGLU clamp at 10. |

---

## Hardware Constraint & Policy

**Real checkpoints do not fit this machine** (64 GB RAM):
- Smallest published quant for GLM-5.3: **UD-Q2_K_XL ≈ 236.4 GiB in 7 shards** (Q4 is ~432 GB).
- GLM-5.3-Flash (320B) is similarly far beyond 64 GB RAM.
- **Policy**: Port now, prove later. Implementation is verified via isolated synthetic specification tests and independent execution against vendored `llama.cpp` b10306 (`glm-dsa`). Code remains unadmitted and unadvertised.

---

## Reuse in Stingray

- `deepseek2` MLA with absorption (admitted, verified): attention core and KV layout.
- `DeepSeek32ForwardPass`: DSA indexer scoring and top-k selection.
- `PrismHadamard`: fast Sylvester Walsh-Hadamard transform (`FwhtBlocks`) for indexer rotation.
- `DeepSeek4Alpha`: Sinkhorn mHC primitives.

---

## New Work & Decomposed Phases (honest state, 2026-10-03)

Legend: [x] done and checked, [~] code exists but only smoke-tested (finite logits) or has a known gap, [ ] not done.
The only tests are `GlmDsaSyntheticTests` and `Glm5NextSyntheticTests`: metadata parsing, the layer pattern, and a tiny forward
pass that checks the logits are finite. **No numeric oracle has been run for either variant** (not llama.cpp b10306, not
`bed0a8566`, not an independent reimplementation).

### Phase 0: Freeze References
- [~] Shapes, tensor names and metadata keys were taken from `glm-dsa.cpp` / `glm5-next.cpp` while porting; no recorded
  reconciliation against a real GGUF inventory (the GLM-5.3 Q2_K_XL download to `E:\_models\glm-5.3` is in progress, so this
  can now be done).

### Phase 1: `glm-dsa` Trunk (`GlmDsaAlpha.cs`, `GlmDsaForwardPass.cs`)
- [~] **1a. Metadata & tensor set.** Present.
- [~] **1b. MLA & attention.** Present (absorbed Q, compressed KV, RoPE on the positional slice); unverified.
- [~] **1c. DSA indexer & refresh schedule.** Present, with top-k reuse on shared layers; the schedule is untested against
  the reference.
- [~] **1d. Hadamard indexer rotation.** Present (`PrismHadamard`); unverified.
- [~] **1e. MoE & leading dense blocks.** Present (sigmoid routing, shared expert, scale); unverified.
- [~] **1f. Full forward pass.** Present.
- [~] **1g. Not-admitted gate & synthetic test.** Gate [x]. Synthetic test is a smoke test; the planned **comparison against
  vendored llama.cpp b10306 on a synthetic GGUF was never run** [ ].

### Phase 2: `glm5next` Trunk (GLM-5.3-Flash) (`Glm5NextAlpha.cs`, `Glm5NextForwardPass.cs`)
- [~] **2a. Metadata & tensor set.** Present.
- [~] **2b. KDA recurrence.** Present; no independent numeric check.
- [~] **2c. Sinkhorn mHC.** Present. The planned isolated `Glm5NextMhcTests` **do not exist**. A bug was found and fixed
  2026-10-03: the four streams accumulated across tokens instead of being rebuilt from each token's embedding; now pinned by
  `Glm5Next_HcStreams_AreRebuiltFromTheCurrentTokenOnly` (mutation-checked).
- [ ] **2d. K-pool indexer subsystem.** The pooled keys are built and stored, but **selection does not drive attention**:
  MLA attends over all cached KV, and the forward pass throws once the context exceeds `indexer.top_k` keys (full attention
  is exact only up to that length). Tail selection and pooled-K layout are unverified.
- [~] **2e. NoPE MLA / DSA.** Dense attention only (see 2d).
- [~] **2f. MoE & SwiGLU clamp.** Present; unverified.
- [~] **2g. Full 45-layer trunk forward pass.** Present, with the 2d limit.
- [~] **2h. Not-admitted gate & synthetic test.** Gate [x]; smoke test only.

### Deferred (Explicitly Out of Scope)
- **MTP / NextN Speculative Drafter**: Text trunk first; NextN extra block is a follow-up.
- **Vision Tower**: GLM-5.3-Flash multimodal 24-block ViT (448 input, temporal patching) is deferred; initial port milestone is text-only.
- **GPU Paths & Batched Prefill**: CPU decode and synthetic verification first.

---

## Verification Plan

- **Specification Tests (Level 2)**:
  - Tiny synthetic GGUFs for `glm-dsa` and `glm5next`.
  - Check intermediate states: Hadamard rotation, KDA recurrence, mHC Sinkhorn convergence, K-pool coordinate mapping, MoE routing.
- **Independent Implementation (Level 3)**:
  - `glm-dsa`: verified against vendored **`llama.cpp` b10306** binaries using synthetic GGUFs.
  - `glm5next`: verified against upstream `bed0a8566` source / TensorSharp / HF references.
- **Real Weights (Level 4)**:
  - GLM-5.3 Q2_K_XL (236 GiB) fits the 279 GB scratch disk; a paged, hours-long, correctness-only run on this 64 GB host is acceptable (decision 2026-10-03). Download to `E:_modelsglm-5.3` in progress. Larger quants are out of reach.
- **Admission (Level 5)**:
  - Standard admission gate once level 4 evidence is captured.

---

**Effort:** the original estimate covered port + smoke tests only. Remaining: K-pool selection, the numeric oracles above, and a paged real-weight run (hours of wall-clock, dominated by disk paging).
