# DiffusionGemma port plan (`diffusion-gemma` / `diffusion_gemma`)

**Status:** not started (2026-10-03). **Policy:** port now, prove later; not admitted, not advertised
(CLAUDE.md rule 14; [ported-families-todo](ported-families-todo.md)).

## Architecture & Upstream References

**Primary & secondary references:**
- **Hugging Face Reference:** `google/diffusiongemma-26B-A4B-it` (official configuration and PyTorch model).
- **llama.cpp / Unsloth:** `unsloth/diffusiongemma-26B-A4B-it-GGUF` and upstream DiffusionGemma runner.
- **TensorSharp:** `docs/models/diffusiongemma.md`, `Models/DiffusionGemma/` (8 files, CPU kernels, self-conditioning, sampler).

**DiffusionGemma** is a **block text-diffusion** model based on a Gemma-4 MoE backbone (26B total, ~4B active).
It is **not autoregressive**: `Forward(token)` is invalid; generation executes iterative denoising over a 256-token canvas.

---

### 1. Model Geometry & Layer Schedule (30 Layers)

| Parameter | Value |
|---|---|
| Hidden Dimension | $D = 2816$ |
| Layers | 30 total: **5 Full Attention** (layers 5, 11, 17, 23, 29) + **25 Sliding Attention** (all others) |
| Sliding Attention Layers | $Q = 16$ heads, $KV = 8$ heads, $\text{head\_dim} = 256$. Window size $W = 1024$. |
| Full Attention Layers | $Q = 16$ heads, $KV = 2$ heads, $\text{head\_dim} = 512$. Global attention. |
| Embedding Scaling | $\sqrt{D} = \sqrt{2816} \approx 53.0659966$, applied to token embeddings and soft self-conditioning embeddings. |
| Per-Layer MLP + MoE | Both dense MLP **and** MoE experts present in every layer: |
| Dense FFN | Intermediate dimension $= 2112$, gated GeLU activation. |
| MoE Experts | 128 routed experts, $\text{top\_k} = 8$, expert intermediate dimension $= 704$. Router evaluated in FP32. |
| Embeddings | Tied input/output embeddings (`output.weight` aliases `token_embd.weight`). Final logit softcapping. |

---

### 2. Self-Conditioning (Core Forward Contract)

DiffusionGemma conditions each denoising step on the model's own previous prediction via learned soft embeddings:
1. **Step 1 (Seed):** Initial canvas embeddings generated directly from input/noise tokens scaled by $\sqrt{2816}$.
2. **Subsequent Steps ($t > 1$):**
   - Soft probabilities: $P = \text{softmax}(\text{logits}_{t-1} / T)$.
   - Soft embedding: $E_{\text{soft}} = P \cdot W_{\text{embed}} \times \sqrt{2816}$.
   - Self-conditioning transformation: $E_{\text{sc}} = \text{MLP}_{\text{sc}}(E_{\text{soft}})$ (learned gating/FFN projection).
   - Injected into canvas: $X_{\text{canvas}} = X_{\text{canvas\_base}} + E_{\text{sc}}$.

---

### 3. Attention Mask Contract & Prefix KV Reuse

Generation is factored into a persistent prompt prefix and an iterative canvas:
- **Prompt Prefill (Persistent):**
  - Prompt tokens attend strictly causally ($j \le i$) and never see the canvas.
  - Builds persistent prefix KV cache once.
- **Canvas Denoising ($256$ tokens):**
  - **Bidirectional Canvas:** Every canvas token attends to all $256$ canvas tokens ($i_{\text{canvas}}, j_{\text{canvas}} \in [0, 255]$).
  - **Prefix Attention:** Canvas tokens attend to prompt KV:
    - *Full Layers (5, 11, 17, 23, 29):* Canvas attends to **all** prompt KV positions.
    - *Sliding Layers (all others):* Canvas attends to the **last $\min(\text{prompt\_len}, W - 1)$** prompt positions (sliding window applies to prefix).

---

### 4. Sampler Algorithm (`DiffusionGemmaSampler`)

| Parameter | Value |
|---|---|
| Canvas Length | 256 tokens |
| Max Denoising Steps | 48 steps |
| Temperature Schedule | Decays from $T_{\max} = 0.8 \to T_{\min} \approx 0.408$ across 48 steps |
| Entropy Acceptance Budget | $H_{\text{bound}} = 0.1$ nats |
| Stopping Thresholds | Mean entropy $< 0.005$ nats AND prediction stability $= 1$ |

**Step Execution:**
1. Compute Shannon entropy in nats for all 256 canvas positions: $H_i = -\sum p_{i,v} \ln p_{i,v}$.
2. Rank positions by ascending entropy (lowest entropy = highest model confidence).
3. Greedily accept positions until cumulative accepted entropy reaches $H_{\text{bound}} = 0.1$ nats.
4. Non-accepted positions are **fully re-noised** with categorical noise.
5. Check early exit criteria (mean entropy $< 0.005$ & stable predictions).

---

### 5. Block-Autoregressive Canvas Lifecycle

DiffusionGemma generates arbitrary length text through a sequence of 256-token canvas blocks:
```
Prompt -> Prefill Causal KV Cache
             │
             ▼
   [256-token Canvas Denoise Loop] <─── Self-Conditioning Loop (up to 48 steps)
             │
             ▼
   Commit Final Canvas Block
             │
             ▼
   Causal Pre-fill Committed Block into Persistent KV Cache
             │
             ▼
   Initialize Next 256-token Canvas (repeat until target length or EOS)
```

---

### 6. Component Architecture & Responsibility Separation

To maintain clean architecture, responsibilities are split across dedicated types:
- **`DiffusionGemmaForwardPass`**: Model weights, prompt prefill, canvas forward pass (bidirectional canvas + prefix KV attention, sliding/full masks), logits.
- **`DiffusionGemmaState`**: Canvas token IDs, self-conditioning vectors, committed tokens count, step index, convergence status.
- **`DiffusionGemmaSampler`**: Temperature schedule, entropy calculation, greedy acceptance budget, categorical re-noising, early exit evaluation.
- **`DiffusionGemmaPipeline`**: Block commit lifecycle, causal re-prefill orchestration, multi-block generation loop.

---

## Hardware Constraint & Fitting

- Unsloth GGUF quant sizes:
  - **Q4_K_M**: ~16.8 GB (**fits comfortably in 64 GB RAM**)
  - **Q5_K_M**: ~19.1 GB
  - **Q8_0**: ~26.9 GB
- **Policy**: Port now, prove on real weights locally (hardware capable).

---

## Decomposed Implementation Phases

- [ ] **Phase 0: Contract Freeze**
  - Freeze exact layer parameters, self-conditioning FFN tensor names, and sampler equations.
- [ ] **Phase 1: Model Graph & Tensor Loader (`DiffusionGemmaConfig.cs`, `DiffusionGemmaTensorSet.cs`)**
  - 30-layer Gemma-4 MoE backbone (16 heads, 8/2 KV heads, 256/512 head dim), tied embeddings, dense FFN (2112) + MoE (128 experts, top-8, 704 dim).
- [ ] **Phase 2: Persistent Prompt Prefill**
  - Causal prefill creating persistent prompt prefix KV cache.
- [ ] **Phase 3: Canvas Forward Pass (`DiffusionGemmaForwardPass.cs`)**
  - Bidirectional canvas attention, prefix KV cross-attention with sliding (1024) / full layer masks, $\sqrt{2816}$ scaling, logits output.
- [ ] **Phase 4: Self-Conditioning (`DiffusionGemmaSelfConditioning.cs`)**
  - Step 1 seed vs $t > 1$ soft probability weighted embeddings, self-conditioning MLP projection, canvas injection.
- [ ] **Phase 5: Sampler (`DiffusionGemmaSampler.cs`)**
  - Temperature decay ($0.8 \to 0.408$), nats entropy budget ($0.1$), greedy acceptance, categorical re-noising, confidence/stability stop.
- [ ] **Phase 6: Block Lifecycle (`DiffusionGemmaPipeline.cs`)**
  - Commit 256-token block, causal re-prefill into persistent KV cache, multi-block loop.
- [ ] **Phase 7: Synthetic End-to-End Tests (`DiffusionGemmaSyntheticTests.cs`)**
  - Tiny synthetic model (e.g. 4 layers: 3 sliding, 1 full; small canvas 16; MoE 8 experts), testing step trace parity, self-conditioning, and sampler convergence.
- [ ] **Phase 8: Gate & CLI Registry**
  - `ModelCompatibility` refusal for autoregressive `InferenceEngine`. Add pipeline CLI entry point.
- [ ] **Phase 9: Real Q4 Checkpoint Verification**
  - Step trace parity against TensorSharp and llama.cpp DiffusionGemma on pinned prompt and initial canvas IDs.

---

## Deferred (Explicitly Out of Scope)

- **Vision Tower**: Gemma-4 27-layer vision tower (1152 dim, 16 heads, 280 image tokens) is deferred; initial port is strictly text-only.
- **Server API Endpoints**: Jev `/v1/systemone` typed endpoints, structured output constraints, and GPU execution paths are follow-ups.

---

**Effort:** ~1-2 days for core model, self-conditioning, sampler, block lifecycle, and synthetic specification tests. Real Q4 checkpoint verification follows directly on local 64 GB machine.
