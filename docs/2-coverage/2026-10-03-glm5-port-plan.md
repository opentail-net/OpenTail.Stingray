# GLM-5.x port plan (`glm-dsa`: GLM-5.2 / 5.3; `glm5next`: GLM-5.3-Flash)

**Status:** not started (2026-10-03). **Policy:** port now, prove later; not admitted, not advertised
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

## New Work & Decomposed Phases

### Phase 0: Freeze References
- [ ] Freeze exact shapes, tensor naming, and metadata keys against `glm-dsa.cpp` and `glm5-next.cpp`.

### Phase 1: `glm-dsa` Trunk
- [ ] **1a. Metadata & Tensor Set (`GlmDsaAlpha.cs`)**:
  Parse `glm-dsa` / `glm_dsa` hyperparameters, load MLA tensors (`wq_a`, `wq_b`, `wkv_a_mqa`, `wk_b`, `wv_b`, `wo`), indexer tensors (`indexer_k_norm`, `indexer_k_norm_b`, `indexer_proj`, `indexer_attn_k`, `indexer_attn_q_b`), dense FFN and MoE tensors.
- [ ] **1b. MLA & Attention**:
  Compressed KV caching ($512 + 64$ per token), absorbed Q calculation, NORM RoPE on positional slice.
- [ ] **1c. DSA Indexer & Refresh Schedule**:
  Full indexer on schedule (layers 0, 1, 2, then every 4th); reuse `top_k` across shared indexer layers.
- [ ] **1d. Hadamard Indexer Rotation**:
  In-place orthonormal FWHT (`PrismHadamard`) on `indexer_q` and `indexer_k` scaled by $1/\sqrt{128}$.
- [ ] **1e. MoE & Leading Dense Blocks**:
  First 3 layers dense; subsequent layers MoE with sigmoid routing, bias, top-8, renormalization, scale 2.5, and shared expert.
- [ ] **1f. Full Forward Pass (`GlmDsaForwardPass.cs`)**:
  Assemble decode pass.
- [ ] **1g. Not-Admitted Gate & Synthetic Test**:
  Add `// glm-dsa — NOT admitted` block in `ModelCompatibility.cs`. Build synthetic specification test (`GlmDsaSyntheticTests.cs`) and verify against vendored llama.cpp b10306.

### Phase 2: `glm5next` Trunk (GLM-5.3-Flash)
- [ ] **2a. Metadata & Tensor Set (`Glm5NextAlpha.cs`)**:
  Hyperparameters (`head_count_kv`, `index_kpool`, `hc_mult`, etc.) and tensor loader.
- [ ] **2b. KDA Recurrence**:
  Independent KDA linear attention recurrence (kernel 4, gate clamp -5, channel decay).
- [ ] **2c. Sinkhorn mHC (`Glm5NextMhcTests`)**:
  4-stream hyper-connections, 20 Sinkhorn iterations, FP32 arithmetic, isolated unit tests.
- [ ] **2d. K-Pool Indexer Subsystem**:
  4-token pool construction, tail selection, pooled K layout, sequence-local state.
- [ ] **2e. NoPE MLA / DSA**:
  Main attention with `qk_rope_head_dim = 0`, `q_lora_rank = 1536`.
- [ ] **2f. MoE & SwiGLU Clamp**:
  288+1 MoE, FP32 router, top-8, scale 2.5, clamp 10.
- [ ] **2g. Full 45-Layer Trunk Forward Pass (`Glm5NextForwardPass.cs`)**:
  Assemble the 34 KDA + 11 MLA layers.
- [ ] **2h. Not-Admitted Gate & Synthetic Test**:
  Add `// glm5next — NOT admitted` block and specification tests.

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
  - Deferred until access to large-memory (>256 GB RAM) host.
- **Admission (Level 5)**:
  - Standard admission gate once level 4 evidence is captured.

---

**Effort:** ~1-2 days for `glm-dsa` port + synthetic oracle tests; ~2 days for `glm5next` trunk + mHC/K-pool/KDA tests. MTP, vision, and real-weight verification are separate.
