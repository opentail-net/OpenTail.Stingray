# DiffusionGemma port plan (`diffusion-gemma` / `diffusion_gemma`)

**Status (revised 2026-10-03): IN PROGRESS / MAJOR BACKBONE REWRITE REQUIRED.**
Earlier documentation claiming this family was "Done" or "90%" was incorrect and void; see [unverified-port-claims note](../1-correctness/2026-10-03-unverified-port-claims.md). The existing codebase represents roughly **30–40%** of the eventual system (scaffolding, component sampler math, and high-level lifecycle). The remaining **60–70%** is concentrated in building the genuine **Gemma-4 MoE dual-mode backbone** (causal prompt prefill + bidirectional canvas denoising).

**Strategic Architecture Insight:**
DiffusionGemma is **not** merely a diffusion sampler waiting for finishing touches. It is a **Gemma-4 MoE engine port plus a diffusion wrapper**. Once viewed through this lens, the implementation order becomes clear: the underlying Gemma-4 MoE backbone is the foundational blocker, and implementing it cleanly provides reusable infrastructure that directly underpins future regular Gemma-4 autoregressive model support in OpenTail.Stingray.

**Policy:** port now, prove later; not admitted, not advertised (CLAUDE.md rule 14; [ported-families-todo](ported-families-todo.md)). Real checkpoints are explicitly refused by `DiffusionGemmaRealCheckpointGuardTests`.

---

## High-Level Status & Roadmap Matrix

| Phase | Subsystem | Current State | What Remains |
| :--- | :--- | :---: | :--- |
| **0. Contract** | Upstream Modeling Spec | 🟡 Partial | Resolve open HF / diffusers / vLLM questions (self-conditioning placement, sampling details) |
| **1. Tensor Loader** | `DiffusionGemmaTensorSet` | 🔴 Rewrite | Major rewrite from guessed names to strictly-typed, per-layer full/SWA tensor inventory |
| **2. Prompt Prefill** | Causal Gemma-4 Backbone | 🔴 Rewrite | Full causal attention through all 30 layers into persistent KV cache (checkable vs llama.cpp) |
| **3. Canvas Forward** | Bidirectional Denoising | 🔴 Rewrite | Bidirectional 256-token canvas attending to prefix KV, deriving V from K on full layers |
| **4. Self-Conditioning** | Soft-Embedding & MLP | 🟡 Incomplete | Wire existing soft-embedding path into learned `self_cond` MLP, pre-norm, and step-1 seeding |
| **5. Sampler** | Denoising Math & Selection | 🟡 Plausible | Gumbel-max noise, mutual information token selection budget, linear temperature contract |
| **6. Block Lifecycle** | Auto-Regressive Canvas Loop | 🟡 Plausible | Orchestrate prompt prefill -> denoise -> commit argmax -> causal commit prefill -> next block |
| **7. Tests** | Synthetic & Parity Suite | 🟡 Outdated | Re-anchor all tests to real Gemma-4 geometry, fused MoE, V derivation, and causal parity |
| **8. Gate & CLI** | Admission & Registry | 🟡 Guarded | Guard exists; wire CLI pipeline entry point and keep unadmitted until Phase 9 passes |
| **9. Real Verification** | 16.8 GB Q4_K_M Execution | 🔴 Pending | Full end-to-end qualification on local 16.8 GB GGUF against reference generation |

---

## Hardware Constraint & Execution Reality

Unlike Qwen3.8 Flash Next (which requires a 72.5 GB paged checkpoint exceeding host memory), DiffusionGemma fits host RAM comfortably:
- **Quantization Size**: Q4_K_M is **16.8 GB** (verified on `unsloth/diffusiongemma-26B-A4B-it-GGUF`), Q5_K_M is ~19.1 GB, Q8_0 is ~26.9 GB.
- **Physical Host Context**: 64 GB host RAM + 279 GB scratch disk.
- **Feasibility**: The entire Q4_K_M checkpoint fits directly in RAM without paging acrobatics or lazy row table complications. Once the Gemma-4 backbone is built, validation can proceed directly against real weights locally.

---

## Architecture & Upstream References

**Primary & Secondary References:**
1. **llama.cpp `src/models/gemma4.cpp`** (local source checkout, `bed0a8566`):
   - Exact Gemma-4 MoE layer block contract, per-head RMS norms, router scaling, per-expert scales, fused `ffn_gate_up_exps`, and full-attention layers with no `attn_v` (V derived from K).
   - Serves as the independent reference oracle for Phase 2 (causal prefill backbone).
2. **Unsloth GGUF `unsloth/diffusiongemma-26B-A4B-it-GGUF`** (Q4_K_M, 692 tensors):
   - Ground truth for metadata keys and tensor nomenclature.
3. **Hugging Face / Google:** `google/diffusiongemma-26B-A4B-it`:
   - Configs, vocabulary, and diffusion generation parameters.
4. **TensorSharp:** `docs/models/diffusiongemma.md`, `Models/DiffusionGemma/`:
   - Dual-mode forward pass structure and block-autoregressive pipeline.

**Core Concept:**
DiffusionGemma is a **block text-diffusion** model on a Gemma-4 MoE backbone (26B total parameters, ~4B active per token). It does **not** generate tokens autoregressively token-by-token. Instead, generation proceeds in discrete blocks of 256 tokens:
1. The user prompt is processed causally to build a persistent prefix KV cache.
2. An initial 256-token canvas of pure noise / mask tokens is iteratively denoised over up to 48 steps using bidirectional attention.
3. The converged canvas is committed, causally ingested into the persistent KV cache, and the next 256-token canvas begins.

---

## Facts Verified on the Real GGUF (`unsloth/diffusiongemma-26B-A4B-it-GGUF`)

### Global Architecture Parameters
- `diffusion-gemma.block_count` = 30 layers
- `diffusion-gemma.embedding_length` = 2816 ($D$)
- `diffusion-gemma.vocab_size` = 262144
- `diffusion-gemma.feed_forward_length` = 2112 (dense FFN intermediate)
- `diffusion-gemma.expert_count` = 128 routed experts
- `diffusion-gemma.expert_used_count` = 8 selected per token
- `diffusion-gemma.expert_feed_forward_length` = 704 per expert
- `diffusion-gemma.canvas_length` = 256 tokens
- `diffusion-gemma.final_logit_softcapping` = 30.0
- Embedding scale: $\sqrt{D} = \sqrt{2816} \approx 53.0659966$ applied to token embeddings and soft self-conditioning embeddings.

### Heterogeneous Layer Schedule (30 Layers Total)
The model alternates between two fundamentally different layer types:
1. **Sliding-Window Attention (SWA) Layers (25 layers)**:
   - Layers: All layers except 5, 11, 17, 23, 29.
   - Geometry: 16 Q heads, 8 KV heads ($H_q = 16, H_{kv} = 8$).
   - Head dimension: $d_{head} = 256$.
   - Attention window: Local sliding window of 1024 tokens (`sliding_window = 1024`).
   - RoPE: Base frequency $\theta = 10{,}000$ (`rope.freq_base = 1e4`), dimension count 256.
   - Projections: Dedicated `attn_q`, `attn_k`, `attn_v`, and `attn_output`.
2. **Full-Attention (Global) Layers (5 layers)**:
   - Layers: Exactly layers **5, 11, 17, 23, 29** (0-indexed, every 6th layer starting at 5).
   - Geometry: 16 Q heads, 2 KV heads ($H_q = 16, H_{kv} = 2$).
   - Head dimension: $d_{head} = 512$ ($16 \times 512 = 8192$ Q width, $2 \times 512 = 1024$ K width).
   - Attention window: Full global context across entire prompt and canvas.
   - RoPE: Base frequency $\theta = 1{,}000{,}000$ (`rope.freq_base = 1e6`), dimension count 512, modulated by `rope_freqs.weight` [256].
   - **Crucial Ground Truth: No `attn_v` Tensor**: There are only 25 `attn_v` tensors in the checkpoint. On full-attention layers, **V is derived directly from K**, exactly as implemented in Gemma-4 (`v = k`).

### Tensors Outside the Blocks
- `token_embd.weight`: [262144, 2816] (tied input and output embeddings).
- `output_norm.weight`: [2816] final RMSNorm before logits.
- `rope_freqs.weight`: [256] rotary modulation table for full-attention layers.
- Self-conditioning MLP:
  - `self_cond_pre_norm.weight`: [2816]
  - `self_cond_gate.weight`: [2816, 2112]
  - `self_cond_up.weight`: [2816, 2112]
  - `self_cond_down.weight`: [2112, 2816]

### Per-Block Layer Tensor Inventory
Each block contains 17–18 tensors depending on layer type:
- `attn_norm.weight`: [2816]
- `attn_q.weight`, `attn_k.weight`: Q and K projections.
- `attn_v.weight`: Present **only** on the 25 SWA layers.
- `attn_q_norm.weight`, `attn_k_norm.weight`: Per-head RMS norms on Q and K.
- `attn_output.weight`: Multi-head projection back to $D = 2816$.
- `post_attention_norm.weight`: [2816]
- Dense FFN:
  - `ffn_norm.weight`: [2816]
  - `ffn_gate.weight`: [2816, 2112]
  - `ffn_up.weight`: [2816, 2112]
  - `ffn_down.weight`: [2112, 2816]
  - `post_ffw_norm_1.weight`: [2816]
- MoE Block (Parallel with Dense FFN):
  - `pre_ffw_norm_2.weight`: [2816]
  - `ffn_gate_inp.weight`: [2816, 128] router projection.
  - `ffn_gate_inp.scale`: [2816] router input scaling vector.
  - `ffn_gate_up_exps.weight`: [2816, 1408, 128] **fused** gate+up expert weights ($2 \times 704 = 1408$).
  - `ffn_down_exps.weight`: [704, 2816, 128] expert down projection.
  - `ffn_down_exps.scale`: [128] per-expert output scalar multiplier.
  - `post_ffw_norm_2.weight`: [2816]
- Post-FFN & Layer Scaling:
  - `post_ffw_norm.weight`: [2816]
  - `layer_output_scale`: Scalar multiplier applied at block output.
  - `enc_layer_output_scale`: Scalar multiplier applied during prompt prefill ("encoder") mode.

---

## Detailed Gemma-4 Block Mathematical Contract

From llama.cpp `src/models/gemma4.cpp` (MoE layers):

```text
// 1. Attention Branch
h_attn = rms_norm(x, attn_norm)

// Q, K, V projections and per-head norms:
q = per_head_rms_norm(h_attn @ W_q, attn_q_norm)
k = per_head_rms_norm(h_attn @ W_k, attn_k_norm)
if (is_full_attention_layer) {
    v = rms_norm_no_weight(k)               // V derived directly from K!
    q = apply_rope(q, theta=1e6, freqs=rope_freqs)
    k = apply_rope(k, theta=1e6, freqs=rope_freqs)
} else {
    v = rms_norm_no_weight(h_attn @ W_v)   // V from dedicated projection
    q = apply_rope(q, theta=1e4)
    k = apply_rope(k, theta=1e4)
}

// Multi-head attention (causal mask for prefill, bidirectional + prefix for canvas):
attn_out = multi_head_attention(q, k, v)
attn_res = x + rms_norm(attn_out @ W_o, post_attention_norm)

// 2. Parallel Dense FFN Branch
dense_h   = rms_norm(attn_res, ffn_norm)
dense_act = gelu(dense_h @ W_gate) * (dense_h @ W_up)
dense_out = rms_norm(dense_act @ W_down, post_ffw_norm_1)

// 3. Parallel Routed MoE Branch
router_in = (rms_norm(attn_res) / sqrt(D)) * ffn_gate_inp.scale
logits    = router_in @ ffn_gate_inp.weight
topk_exp, weights = softmax_topk(logits, k=8, renorm=true)

moe_h = rms_norm(attn_res, pre_ffw_norm_2)
moe_accum = zeros(D)
for (e, w in zip(topk_exp, weights)) {
    // Fused gate+up: gate = [0..703], up = [704..1407]
    exp_gate, exp_up = split(moe_h @ W_gate_up_exps[e])
    exp_act = gelu(exp_gate) * exp_up
    exp_out = (exp_act @ W_down_exps[e]) * ffn_down_exps.scale[e]
    moe_accum += w * exp_out
}
moe_out = rms_norm(moe_accum, post_ffw_norm_2)

// 4. Combine Dense + MoE and Scale Layer Output
combined_ffn = rms_norm(dense_out + moe_out, post_ffw_norm)
scale = is_prefill ? enc_layer_output_scale : layer_output_scale
output = (attn_res + combined_ffn) * scale
```

---

## Detailed Execution Phases

### Phase 0 — Modeling Contract & Upstream Resolution
*Objective: Resolve remaining upstream specification questions from `transformers`, `diffusers`, and `vLLM`.*

- [x] **0.1 GGUF Geometry Lock**: 30 layers, 5 full + 25 SWA, 128 experts / top-8, 256 canvas length, $\sqrt{D}$ embedding scale.
- [x] **0.2 Gemma-4 MoE Layer Contract Lock**: Full equation sequence verified from `gemma4.cpp` (Section 1b).
- [ ] **0.3 Self-Conditioning Placement & Seeding**:
  - Determine exact activation function for `self_cond_gate` / `up` (GELU vs SiLU).
  - Confirm step-1 seeding: when no previous predictions exist at step 0, is self-conditioning zeroed out, or seeded with a learned vector?
- [ ] **0.4 Sampler Contract**:
  - Confirm whether token selection strictly bounds cumulative entropy $\le 0.1$ nats.
  - Confirm Gumbel-max noise injection formula vs standard multinomial sampling.
  - Finalize temperature schedule contract (linear decay from 0.8 to 0.408 over steps $0..47$).

---

### Phase 1 — Dedicated Tensor Loader (`DiffusionGemmaTensorSet`)
*Objective: Replace the placeholder `DiffusionGemmaLayerTensors` with a strictly-typed, required-tensor loader matching the real 692-tensor GGUF layout.*

- [ ] **1.1 Design `DiffusionGemmaTensorSet`**:
  - Distinguish between `DiffusionGemmaSlidingLayerTensors` and `DiffusionGemmaFullLayerTensors`.
  - Validate that full-attention layers (5, 11, 17, 23, 29) **require no `attn_v` tensor** and enforce $d_{head} = 512, H_{kv} = 2$.
  - Validate that SWA layers **require `attn_v`** and enforce $d_{head} = 256, H_{kv} = 8$.
- [ ] **1.2 Fused MoE & Scale Tensors**:
  - Read `ffn_gate_up_exps.weight` [2816, 1408, 128] without fallback to separate tensors.
  - Load `ffn_gate_inp.scale` [2816], `ffn_down_exps.scale` [128].
  - Load `enc_layer_output_scale` and `layer_output_scale`.
- [ ] **1.3 Self-Conditioning & Rotary Tensors**:
  - Load `self_cond_pre_norm`, `self_cond_gate`, `self_cond_up`, `self_cond_down`.
  - Load `rope_freqs.weight` [256].
- [ ] **1.4 Fail-Fast Real Checkpoint Guard**:
  - Replace `DiffusionGemmaRealCheckpointGuardTests` expectation so that the new loader passes when loading real GGUF structures.

---

### Phase 2 — Causal Prompt Prefill Backbone
*Objective: Build the causal prefill path where prompt tokens actually attend through all 30 layers into a persistent KV cache.*

- [ ] **2.1 Persistent Prefix KV Cache**:
  - Allocate per-layer K and V caches for the prefix prompt.
  - Accommodate differing layer geometries (SWA: $H_{kv}=8, d=256$; Full: $H_{kv}=2, d=512$).
- [ ] **2.2 Full Causal Block Execution**:
  - For each prompt chunk, execute the full Gemma-4 block:
    `attn_norm -> Q/K/V norms -> RoPE -> Causal Attention -> post_attn_norm -> Dense FFN + MoE -> layer_scale`.
  - Use `enc_layer_output_scale` on prefill mode.
- [ ] **2.3 Independent Oracle Verification**:
  - Relabel a synthetic/real test GGUF header to `gemma4` and verify prompt logits against llama.cpp `gemma4.cpp` (`bed0a8566`). This independently proves the backbone without touching diffusion mechanics.

---

### Phase 3 — Bidirectional Canvas Denoising Backbone
*Objective: Rewrite `ExecuteCanvasPass` to execute bidirectional canvas attention with persistent prompt KV access.*

- [ ] **3.1 Canvas Input Assembly**:
  - Scale token embeddings by $\sqrt{D} = 53.0659966$.
  - Add self-conditioning representation $E_{sc}$ to canvas hidden state.
- [ ] **3.2 Dual-Context Attention Walk**:
  - Canvas tokens attend bidirectionally to all other 256 canvas tokens.
  - Canvas tokens attend back into the persistent prompt KV cache:
    - Full layers: Attend over all prompt positions $0..L_{prompt}-1$.
    - SWA layers: Attend over the active sliding window $\max(0, \text{pos} - 1024)..\text{pos}$.
- [ ] **3.3 Derived V on Full Layers**:
  - In full-attention layers, derive $V = \text{rms\_norm\_no\_weight}(K)$ on the fly.
- [ ] **3.4 Layer Output & Logit Head**:
  - Apply `layer_output_scale`.
  - Apply `output_norm` $\to$ tied embedding projection $\to$ final logit softcapping at 30.0.

---

### Phase 4 — Self-Conditioning Integration
*Objective: Connect the exact soft-embedding projection to the learned `self_cond` MLP.*

- [ ] **4.1 Soft Probability & Embedding Extraction**:
  - Existing `ComputeSoftEmbeddings` calculates $P = \text{softmax}(\text{logits} / T)$ and $E = P \cdot W_{embed} \cdot \sqrt{D}$.
- [ ] **4.2 Learned Self-Conditioning MLP**:
  - Implement $E_{sc} = \text{down}(\text{act}(\text{gate}(\text{rms}(E))) \cdot \text{up}(\text{rms}(E)))$.
  - Add $E_{sc}$ directly to canvas embeddings before layer 0.
- [ ] **4.3 Lifecycle State Management**:
  - Step 0: Zero or identity seed.
  - Step $t > 0$: Derived from step $t-1$ predicted logits.

---

### Phase 5 — Sampler Exactness
*Objective: Finalize the diffusion sampling loop, entropy filtering, and stopping criteria.*

- [ ] **5.1 Entropy Calculation & Cumulative Budget**:
  - Calculate per-token entropy $H_i = -\sum p \log p$.
  - Select lowest-entropy tokens up to the 0.1 nats mutual information bound.
- [ ] **5.2 Gumbel-Max Injection & Re-noising**:
  - Sample candidate tokens using Gumbel-max with decaying temperature $T(t)$.
  - Re-noise unaccepted canvas positions for step $t+1$.
- [ ] **5.3 Stopping & Argmax History**:
  - Check stopping criteria: mean canvas entropy $< 0.005$ AND argmax prediction history identical across two consecutive steps.

---

### Phase 6 — Block-Autoregressive Pipeline Lifecycle
*Objective: Tie together prompt prefill, iterative canvas denoising, and sequential block commitment.*

- [ ] **6.1 Outer Pipeline Loop**:
  ```text
  Prompt
    -> Causal Prefill into Persistent KV
    -> Repeat {
         Initialize Canvas (Noise / Mask)
         Loop Step 0..47 {
           Canvas Forward (Bidirectional + Prompt KV)
           Apply Self-Conditioning
           Sampler Accept / Re-noise
           Check Convergence
         }
         Commit Converged Argmax Block (256 tokens)
         Causal Prefill Committed Block into Persistent KV Cache
       } Until MaxTokens or EOS
  ```
- [ ] **6.2 Cache Continuity**:
  - Ensure prompt KV and committed block KVs remain valid and continuous across block transitions.

---

### Phase 7 — Synthetic Parity & Structural Test Suite
*Objective: Replace outdated guessed-name tests with real-geometry synthetic fixtures.*

- [ ] **7.1 Gemma-4 Layer Parity Test**:
  - Test single-layer forward pass with fused MoE, per-expert scales, router scales, and dual post-norms.
- [ ] **7.2 Full vs SWA Geometry Test**:
  - Test layer 5 (full: $d=512$, no $V$, $H_{kv}=2$, RoPE 1e6) vs layer 4 (SWA: $d=256$, with $V$, $H_{kv}=8$, RoPE 1e4).
- [ ] **7.3 Self-Conditioning Transformation Test**:
  - Verify soft embedding through `self_cond` MLP matches known numerical reference.
- [ ] **7.4 Canvas Attention Mask Test**:
  - Verify bidirectional canvas + prefix prompt attention weights.

---

### Phase 8 — Gate & CLI Entry Point
*Objective: Expose pipeline cleanly while maintaining strict model admission discipline.*

- [ ] **8.1 CLI Runner Entry Point**:
  - Add `diffusion-gemma` command / mode to the Stingray runner.
- [ ] **8.2 Admission Policy**:
  - Keep `// diffusion-gemma - NOT admitted` in `ModelCompatibility.cs` until Phase 9 verification passes.

---

### Phase 9 — Real 16.8 GB Q4_K_M Verification
*Objective: Validate output quality on the actual 16.8 GB GGUF.*

- [ ] **9.1 Load Real Checkpoint**:
  - Load `unsloth/diffusiongemma-26B-A4B-it-GGUF` directly in 64 GB host memory.
- [ ] **9.2 Single-Block Smoke Test**:
  - Run 1 prompt prefill + 1 canvas denoising block (48 steps).
  - Verify convergence and non-degraded text generation.
- [ ] **9.3 Reference Comparison**:
  - Compare generated text and token acceptance rates against upstream Google reference outputs.
- [ ] **9.4 Admit Model**:
  - Flip admission status in `ModelCompatibility.cs` and announce in `STATUS.md`.

---

## Strategic Value & Implementation Sequencing

```text
HF / Diffusers Spec Resolution (Phase 0)
                 │
                 ▼
Real Gemma-4 Tensor Set & Loader (Phase 1)
                 │
                 ▼
Causal Gemma-4 MoE Backbone (Phase 2) ◄─── Verifiable vs llama.cpp gemma4.cpp
                 │
                 ▼
Bidirectional Canvas Backbone (Phase 3)
                 │
                 ▼
Self-Conditioning & Sampler Exactness (Phases 4 & 5)
                 │
                 ▼
Block Lifecycle & Synthetic Parity (Phases 6 & 7)
                 │
                 ▼
Real 16.8 GB Q4_K_M Verification (Phase 9)
                 │
                 ▼
PRODUCTION ADMISSION (Phase 8)
```

**Key Takeaway:**
By structuring the work as a **Gemma-4 MoE backbone implementation first**, we decouple the foundational transformer mechanics (which can be independently verified against llama.cpp) from the diffusion canvas loop. This minimizes risk, provides direct reuse for regular Gemma-4 models, and ensures the implementation rests on a solid, verified foundation.
