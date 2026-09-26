# Plan: Coverage & Resolution for Target LLM Architectures (written 2026-09-26)

Document ID: `docs/100-actionable-models-coverage-plan.md`  
Scope: Actionable resolution plan for **Phi-3.5-MoE**, **GPT-OSS 20B**, **EXAONE 4.5 33B**, and **DeepSeek V2-Lite**, alongside memory/hardware-blocking rationale for **DeepSeek V3.2** and **DeepSeek V4**.

---

## 1. Hardware Context & Scope Classification

* **Local Test Machine Baseline:** 64 GB system RAM, AMD Zen 3 CPU (AVX2/FMA, no VNNI), Vulkan iGPU (~16 GB shared) / discrete CUDA hardware when available.
* **Target Classification:**

| Model / Architecture | Size / Smallest Usable Quant | Hardware Viability (64 GB RAM) | Current Codebase Status | Priority & Scope |
|---|---|---|---|---|
| **Phi-3.5-MoE** (`phimoe`) | ~14 GB (Q2_K) / ~21 GB (Q3_K_M) | ✅ Comfortable | 🟡 Broken: missing LongRoPE factor handling (`PerformanceLeague.md:404`) | **Phase 1 (Bug Fix)** |
| **GPT-OSS 20B** (`gpt-oss`) | ~12–14 GB (MXFP4) | ✅ Comfortable | 🟡 Alpha implemented (`GptOssForwardPass.cs`), smoke-tested, unadmitted | **Phase 2 (Parity & Admission)** |
| **EXAONE 4.5 33B** (`exaone4`) | ~18–20 GB (Q4_K_M) | ✅ Comfortable | 🟡 Text CPU works; `HybridForwardPass` post-norm GPU gap; vision CLI format gap | **Phase 3 (Hybrid Offload & Vision CLI)** |
| **DeepSeek V2-Lite** (`deepseek2`) | ~6–10 GB (Q2_K / Q8_0) | ✅ Comfortable | 🟡 Runs without crash; MoE router boundary margin sensitivity (~0.002) | **Phase 4 (Targeted Investigation / Pragmatic Mitigation)** |
| **DeepSeek V4-Flash** (`deepseek4`) | ~99 GB (Q2_K_S) | ❌ Blocked (>64 GB) | 🔵 Alpha implemented (`DeepSeek4ForwardPass.cs`); untested on real weights | *Blocked by local RAM capacity* |
| **DeepSeek V3.2** (`deepseek32`) | ~149 GB (IQ1_M) | ❌ Blocked (>64 GB) | 🔵 Alpha implemented (`DeepSeek32ForwardPass.cs`); untested on real weights | *Blocked by local RAM capacity* |

---

## 2. Phase 1 — Phi-3.5-MoE (`phimoe`): LongRoPE Restoration

### Problem Statement
`Phi-3.5-MoE-instruct` loads and runs prefill/decode at reasonable speeds (18+ t/s CPU), but produces complete gibberish word-salad across both Q2_K and Q3_K_M (`PerformanceLeague.md:404`). Upstream `phimoe.cpp` confirms the model requires **LongRoPE** dynamic position frequency extension.

The GGUF carries:
* `rope_factors_long.weight` (per-dimension factors for context lengths > `original_max_position_embeddings`)
* `rope_factors_short.weight` (per-dimension factors for context lengths <= `original_max_position_embeddings`)
* Metadata: `phimoe.context_length`, `phimoe.rope.scaling.original_context_length` (or `original_max_position_embeddings`).

Currently, `ModelGraph.cs` and `ForwardPass.cs` ignore these tensors entirely for text generation, resulting in completely scrambled RoPE angles at all positions.

### Implementation Tasks
1. **Metadata & Tensor Resolution (`ModelGraph.cs` / `ForwardPass.cs`):**
   * Load `rope_factors_short.weight` and `rope_factors_long.weight` when present in the GGUF.
   * Read `original_max_position_embeddings` from metadata (fallback to 4096 if omitted).
2. **Dual-Factor Table Construction:**
   * Extend `SimdKernels.BuildRopeTable` or compute short and long RoPE tables during model initialization.
   * Ensure support for selecting between short/long factor sets depending on current context position $pos$:
     $$\text{factors} = (pos < \text{orig\_ctx}) ? \text{factors\_short} : \text{factors\_long}$$
3. **Dispatch Wiring (`ForwardPass.cs` / `PrefillCore.cs`):**
   * Thread LongRoPE selection into the attention rotary projection steps in `ForwardPass.Decode.cs` and `ForwardPass.PrefillCore.cs`.
4. **Verification & Regression Testing:**
   * Run greedy continuation on `Phi-3.5-MoE-instruct-GGUF` (Q2_K or Q3_K_M) against prompt: `"The capital of France is"`.
   * Assert output produces `"Paris"` followed by coherent factual continuation, matching `llama.cpp` oracle.
   * Create `tests/OpenTail.Stingray.Tests.ForwardPass/PhiMoeGreedyParityTests.cs`.

---

## 3. Phase 2 — GPT-OSS 20B (`gpt-oss`): Parity Verification & Model Admission

### Problem Statement
An alpha forward-pass for GPT-OSS (`GptOssForwardPass.cs`, `GptOssTensorSet.cs`, `GptOssAlpha.cs`) already exists and executes real-weight smoke tests (`GptOssRealWeightSmokeTests.cs`). However:
1. It is unadmitted in `ModelCompatibility.cs` (line 693).
2. It has not been benchmarked token-for-token against an oracle reference (`examples/llama.cpp`'s `openai_moe` class).
3. The YaRN vs. standard RoPE frequency base and per-expert MoE bias tensor layouts need empirical validation.

### Implementation Tasks
1. **Audit Hyperparameters & Tensors against GGUF:**
   * Verify sliding window attention pattern (1:1 local 128 / global attention).
   * Check attention sink behavior (first 4 tokens protected from eviction).
   * Confirm clamping and formula for OAI SwiGLU:
     $$\text{SwiGLU}_{\text{OAI}}(x, y) = \text{clamp}(x \cdot \text{sigmoid}(x) \cdot y, \dots)$$
2. **Oracle Reference Capture:**
   * Run `llama-cli` on `gpt-oss-20b-MXFP4.gguf` with `--temp 0 --top-k 1 -p "The capital of France is" -n 24`.
   * Capture greedy token IDs and logit progression.
3. **Numerics & Parity Check:**
   * Execute `GptOssForwardPass` against the same prompt and compare greedy token sequence.
   * Verify MXFP4 dequantization and matmul accuracy via `SimdKernels.MatVecMxfp4`.
4. **Admission:**
   * Add `"gpt-oss"` to `s_textGenerationArchitectures` in `ModelCompatibility.cs`.
   * Integrate with `InferenceEngineLoader` so standard CLI `stingray -m gpt-oss-20b-MXFP4.gguf` serves the model cleanly.

---

## 4. Phase 3 — EXAONE 4.5 33B (`exaone4`): Hybrid GPU Offloading & Vision CLI Alignment

### Problem Statement
`EXAONE-4.5-33B` is already admitted and verified for CPU text generation (`Exaone4VerifyTemp.cs`, `docs/PerformanceLeague-expansion-plan.md:97`). However, two distinct issues limit its full capability:
1. **GPU Layer Splitting Blocked:** `HybridForwardPass.cs` (Phase 14 of `docs/perf-sweep-plan.md`) hardcodes pre-norm tensor names (`blk.*.attn_norm.weight`, `blk.*.ffn_norm.weight`) without checking for post-norm tensors (`blk.*.post_attention_norm.weight`, `blk.*.post_ffw_norm.weight`). This forces EXAONE 4.5 onto CPU-only (`-g 0`).
2. **Vision Multimodal CLI Formatting:** `Exaone4VisionEncoder` works and emits 324 visual soft tokens (5120-dim). However, the CLI's `--image` command passes a flat string, whereas EXAONE's Jinja template expects structured multi-part message parts (`[{'type': 'image'}, {'type': 'text', ...}]`), causing the image token replacement branch to be skipped.

### Implementation Tasks
1. **Post-Norm Support in `HybridForwardPass.cs`:**
   * Update `HybridForwardPass.cs` weight upload and execution loops to recognize post-norm tensors when pre-norm tensors are absent (mirroring `ForwardPass.cs`).
   * Apply post-norm math (normalize after attention/FFN output, immediately prior to residual addition) on GPU and CPU layers.
   * Test correctness by checking that GPU-split logits match CPU-only logits on EXAONE-4.5 checkpoints.
2. **Structured Message Content in CLI (`RunCommand.cs`):**
   * Support structured conversation message representation in `ChatTemplate.cs` / `RunCommand.cs` when `--image` is provided.
   * Ensure Jinja templates inspecting `content.type == 'image'` or list-of-dicts format execute their visual formatting blocks correctly.
3. **Benchmarking:**
   * Measure EXAONE-4.5-33B throughput with GPU offload (e.g. 12–16 layers on GPU) vs. current 1.6–1.7 t/s CPU baseline. Record in `PerformanceLeague.md`.

---

## 5. Phase 4 — DeepSeek V2-Lite (`deepseek2`): Pragmatic Routing & Precision Assessment

### Problem Statement
`DeepSeek-V2-Lite-Chat` loads and runs without crashing on CPU (`ForwardPass.cs`), but diverges from expected greedy output. Deep-dive investigation (`docs/done/032-deepseek2-mla-yarn-moe-routing-investigation.md`) demonstrated that the trained MoE router decisions sit on razor-thin boundary margins (~0.002 margin between rank 6 and rank 7), which any floating-point summation order delta between .NET SIMD and `ggml` flips, compounding into divergent residuals over 27 layers.

### Implementation Tasks & Assessment Options
1. **High-Precision Kernel Baseline Check:**
   * Run `DeepSeek-V2-Lite` using unquantized / FP32 activation paths and evaluate whether divergence is delayable beyond layer 15.
2. **MoE Routing Margin Stabilization Probe:**
   * Test whether introducing a slight router temperature scaling or boundary epsilon reduces near-tied flips without degrading response coherence.
3. **Documentation & Gating Clarity:**
   * Keep `deepseek2` guarded behind `--allow-unverified-arch` unless exact parity is reached.
   * Update CLI diagnostic messaging to clearly explain the router margin sensitivity caveat when users run `deepseek2` models.

---

## 6. Out-of-Scope Hardware Justification: DeepSeek V3.2 & V4

* **DeepSeek V4-Flash (284B total / 13B active):**
  * Smallest quant: `Q2_K_S` is **~99 GB**.
  * Execution requirement: Requires >110 GB system RAM to load without paging crash.
  * Verdict: Strictly out of scope on 64 GB workstation. Retain `DeepSeek4ForwardPass.cs` as structural reference.
* **DeepSeek V3.2 (671B MoE):**
  * Smallest quant: `IQ1_M` is **~149 GB**.
  * Execution requirement: Requires >160 GB system RAM.
  * Verdict: Strictly out of scope on 64 GB workstation. Retain `DeepSeek32ForwardPass.cs` as structural reference.

---

## 7. Recommended Execution Order

```
[Phase 1: Phi-3.5-MoE LongRoPE] ───► [Phase 2: GPT-OSS 20B Parity] ───► [Phase 3: EXAONE Hybrid & Vision]
           │                                    │                                    │
           ▼                                    ▼                                    ▼
Coherent greedy output              Admit "gpt-oss" into               GPU layer-offloading unlocked;
(Tokens: Paris...)                  ModelCompatibility.cs              Multi-part vision CLI working
```

* **Step 1:** Implement LongRoPE factors in `ForwardPass` & verify `Phi-3.5-MoE`.
* **Step 2:** Run oracle parity against `llama.cpp` for `GPT-OSS 20B` and formally admit to `ModelCompatibility.cs`.
* **Step 3:** Extend `HybridForwardPass.cs` with post-norm support for `EXAONE 4.5 33B` and test GPU speedup.
