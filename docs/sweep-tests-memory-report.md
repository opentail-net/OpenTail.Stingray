# Test Memory Sweep Report

**Generated:** 2026-09-28  
**Script:** `pwsh scripts/sweep-tests.ps1`  
**Scope:** top-10 most memory-consuming tests across the suite

---

## Summary table

| Rank | Peak RAM | Wall time | Domain | Test class |
|------|----------|-----------|--------|------------|
| 1 | **44.96 GB** | 32.8 s | Audio | `PersonaPlexLiveDuplexRealWeightsTests` |
| 2 | **44.78 GB** | 59.1 s | Diffusion | `HunyuanVideoGpuParityTests` |
| 3 | **44.01 GB** | 94.1 s | Audio | `PersonaPlexDelayedPipelineRealWeightsTests` |
| 4 | **43.1 GB** | 134.45 s | Audio | `PersonaPlexGenerateWavDebugTest` |
| 5 | **42.65 GB** | 73.29 s | Audio | `PersonaPlexVoicePromptRealWeightsTests` |
| 6 | **42.57 GB** | 31.36 s | Audio | `PersonaPlexFullPipelineRealWeightsTests` |
| 7 | **42.25 GB** | 28.02 s | Audio | `PersonaPlexGeneratorRealWeightsTests` |
| 8 | **37.9 GB** | 49.97 s | Diffusion | `ZImageGpuRealScaleBisectTests` |
| 9 | **36.99 GB** | 28.14 s | Audio | `PersonaPlexLmTensorSourceRealWeightsTests` |
| 10 | **36.82 GB** | 59.48 s | Diffusion | `Sd3PerStepTrajectoryParityTests` |

The PersonaPlex cluster (ranks 1, 3-7, 9) dominates the list. All PersonaPlex tests share the same `personaplex-7b-v1-q8_0.gguf` checkpoint; the GGUF is loaded fresh per test-process, so the base weight cost is paid every time.

---

## Per-test analysis

### 1 - `PersonaPlexLiveDuplexRealWeightsTests` -- 44.96 GB / 32.8 s

**File:** `tests/OpenTail.Stingray.Tests.Audio/PersonaPlexLiveDuplexRealWeightsTests.cs`

**What it tests:** The live-duplex path -- real user audio in, real generated audio out.
Specifically: encodes a decoder-generated waveform via `MimiCodecEncoder`, feeds the per-frame codes through `PersonaPlexGenerator.GenerateWithUserAudio` (the `run_user_frame`-driven duplex loop), then decodes the model output back via `MimiCodecDecoder`.

**Why it is the heaviest:**
On top of the 7B Q8_0 LM weights (~7 GB raw, expanded to float32 activations during inference) and the Depformer weights, this test additionally holds **both** the Mimi encoder **and** the Mimi decoder simultaneously for the audio roundtrip -- the only test that needs the full codec in both directions at once. That pushes it to the highest peak in the list.

**Notes:** Not a golden/parity check -- no reference duplex run exists yet. The 32.8 s wall time is the shortest of the full-pipeline tests, suggesting the generation is kept to a small number of frames.

---

### 2 - `HunyuanVideoGpuParityTests` -- 44.78 GB / 59.1 s

**File:** `tests/OpenTail.Stingray.Tests.Diffusion/HunyuanVideoGpuParityTests.cs`

**What it tests:** GPU-vs-CPU parity for the HunyuanVideo transformer using the real `hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors` checkpoint (~13 GB on disk, fp8).
Runs one full forward at a small synthetic latent (1 frame, 16x16 latent = 64 image tokens) through **both** a `VulkanBackend` (GPU) and a CPU path, then compares cosine similarity (threshold > 0.9999) and relative L2 (threshold < 1%).

**Why it is the second heaviest:**
The 13 GB fp8 checkpoint is expanded to fp32 for CPU comparison, and the test holds **two** independent model instances in memory at once (`model.UseGpu=true` + `model.UseGpu=false` via the same instance). HunyuanVideo''s 60-block transformer is architecturally large even before expansion. The GPU upload adds a separate VRAM allocation on top of system RAM.

**Notes:** Gated by `STINGRAY_RUN_HEAVY_TESTS=1` **and** checkpoint presence; will skip visibly in CI if either is absent. Wall time includes Vulkan device init and GPU weight upload on the first call.

---

### 3 - `PersonaPlexDelayedPipelineRealWeightsTests` -- 44.01 GB / 94.1 s

**File:** `tests/OpenTail.Stingray.Tests.Audio/PersonaPlexDelayedPipelineRealWeightsTests.cs`

**What it tests:** The delay-correct `PersonaPlexGenerator.GenerateDelayed` path -- the real `PersonaPlexDelayState` ring-buffer state machine ported from `session.cpp`. Two facts: the argmax path (basic) and a temperature/top-k sampling path (`HfSampler`) are each exercised in a separate `[Fact]`.

**Why it is large:**
Same checkpoint as all other PersonaPlex tests; the extra wall time (94 s vs ~30 s for the argmax-only tests) comes from running **two** full generation passes sequentially within the same process. The longer resident time means the KV-cache and depformer activations accumulate across both facts before GC can reclaim them.

**Notes:** The sequential two-fact layout is the reason this class runs longer than `PersonaPlexFullPipelineRealWeightsTests` despite both generating only 4 output frames each.

---

### 4 - `PersonaPlexGenerateWavDebugTest` -- 43.1 GB / 134.45 s

**File:** `tests/OpenTail.Stingray.Tests.Audio/PersonaPlexGenerateWavDebugTest.cs`

**What it tests:** Marked as a _temporary debug test_ for informal listening. Runs `PersonaPlexGenerator.GenerateWithVoicePrompt` with a real voice-id-conditioned bootstrap (NATF0 safetensors embedded in the GGUF), generates **50 output frames** by default (configurable via `PP_FRAMES`), decodes them to a WAV, and writes both a WAV and a decoded-text `.txt` to `docs/audio-samples`.

**Why it is the longest and ~3rd heaviest:**
50 frames is 4x-12x more frames than the other PersonaPlex tests, so the KV-cache grows substantially longer, keeping both LM activations and the MimiCodecDecoder weights live for the full generation pass. Additionally, the test deserializes and decodes the full embedded voice-prompt safetensors blob in-process.

**Notes:** Writes output files to disk (`docs/audio-samples/`, gitignored). The `PP_FRAMES`, `PP_SEED`, and `PP_OUT` env-var knobs make this useful as a manual listening harness. The 134 s wall time is the longest in the top 10 by a wide margin. Consider whether this should be excluded from automated sweep runs or capped to fewer frames.

---

### 5 - `PersonaPlexVoicePromptRealWeightsTests` -- 42.65 GB / 73.29 s

**File:** `tests/OpenTail.Stingray.Tests.Audio/PersonaPlexVoicePromptRealWeightsTests.cs`

**What it tests:** End-to-end smoke test for the voice-id-conditioned bootstrap (`GenerateWithVoicePrompt`): extracts the `NATF0.safetensors` file embedded in the GGUF, replays its real embeddings and delay-cache snapshot, pads with silence frames, then continues into the generation loop. Generates 4 output frames and decodes via the Mimi decoder.

**Why it is large:**
Standard PersonaPlex checkpoint weight cost, plus the full embedded-asset extraction (decoding the entire GGUF metadata blob to find the safetensors boundary). The higher wall time vs. `PersonaPlexGeneratorRealWeightsTests` (no Mimi + no voice asset extraction) is explained by the decoder decode pass and metadata scan.

**Notes:** The 73 s vs 28 s delta relative to `PersonaPlexGeneratorRealWeightsTests` is a rough upper bound on the cost of (a) the voice-prompt safetensors extraction + `PersonaPlexVoicePrompt.Load` and (b) the `MimiCodecDecoder.Decode` pass.

---

### 6 - `PersonaPlexFullPipelineRealWeightsTests` -- 42.57 GB / 31.36 s

**File:** `tests/OpenTail.Stingray.Tests.Audio/PersonaPlexFullPipelineRealWeightsTests.cs`

**What it tests:** The "first full text-to-waveform run" for PersonaPlex: LM generates 4 frames (each conditioning a Depformer 16-codebook step), then the first 8 codes per frame are fed into the real MimiCodecDecoder to produce actual audio samples. The 8-codebook mapping to Mimi''s `ActiveCodebooks` is documented as a tested assumption in the code.

**Why it is large:**
Standard checkpoint cost + Depformer weights + MimiCodecDecoder weights, all in memory simultaneously. The short wall time (31 s) relative to the similar `PersonaPlexDelayedPipelineRealWeightsTests` (94 s) is because this test runs only one fact and uses the simpler non-delay path.

**Notes:** The code explicitly acknowledges the "two 8-codebook streams?" open question about the correct LM-to-Mimi code mapping. If that mapping changes, this test''s assertions need revisiting.

---

### 7 - `PersonaPlexGeneratorRealWeightsTests` -- 42.25 GB / 28.02 s

**File:** `tests/OpenTail.Stingray.Tests.Audio/PersonaPlexGeneratorRealWeightsTests.cs`

**What it tests:** The cheapest full PersonaPlex test that still exercises both autoregressive decoders together: `PersonaPlexGenerator.Generate` chains the temporal LM and the Depformer for 3 frames, checking code ranges only. No Mimi decoder, no voice prompt extraction -- the lowest overhead of all real-weight PersonaPlex tests that go beyond LM-only.

**Why it is large:**
Unavoidably pays the same 7B LM checkpoint load cost as all other PersonaPlex tests. The Depformer weights add another tier. No Mimi codec means there''s nothing substantial pulling it above the others -- it''s the floor cost of running PersonaPlex with real weights.

**Notes:** This is the best test to use as a "checkpoint load cost" baseline when comparing marginal costs of the other PersonaPlex tests. The ~0.3 GB and ~4 s gap vs. `PersonaPlexFullPipelineRealWeightsTests` is roughly the cost of adding the Mimi decoder.

---

### 8 - `ZImageGpuRealScaleBisectTests` -- 37.9 GB / 49.97 s

**File:** `tests/OpenTail.Stingray.Tests.Diffusion/ZImageGpuRealScaleBisectTests.cs`

**What it tests:** A real-weight, real-scale (dim=3840, nHeads=30) GPU-residency bisection for Z-Image-Turbo. Loads `z_image_turbo-Q4_0.gguf` **twice** (once for `cpuDit`, once for `gpuDit`), runs the 30-block main loop through both the CPU path (`RunMainLayersCpuForTest`) and the Vulkan GPU path (`RunMainLayersGpuForTest`), and captures per-block activations to identify the first block where cosine similarity < 0.999.

**Why it is large:**
The checkpoint is loaded as two separate `GgufWeightLoader` instances to keep weight tensors truly independent. At real dimensions (dim=3840, nTok=320) the per-block activation snapshots (`cpuBlocks`/`gpuBlocks` dictionaries) accumulate up to 30 x nTok x dim x 4 bytes = ~150 MB of captured float arrays on top of the two model instances. Adding VRAM mirroring via `VulkanBackend` makes this the heaviest Diffusion test after HunyuanVideo.

**Notes:** The diagnostic was added to track down the GPU residency noise bug (docs/094 Phase 7). Once the bug is confirmed fixed and the test passes reliably, consider whether the per-block capture hooks can be compiled out or the test demoted to a lighter scale for routine sweeps.

---

### 9 - `PersonaPlexLmTensorSourceRealWeightsTests` -- 36.99 GB / 28.14 s

**File:** `tests/OpenTail.Stingray.Tests.Audio/PersonaPlexLmTensorSourceRealWeightsTests.cs`

**What it tests:** Two facts:
(a) `FindTensor_OnRealCheckpoint_ResolvesRealShapes` -- shape validation only (cheap).
(b) `Prefill_OnRealCheckpoint_ProducesFiniteLogits` -- a real `ForwardPass.Prefill` on a 4-token prompt, checking that all 32 000 logits are finite.

**Why it is large:**
No Depformer, no Mimi -- the lowest-overhead PersonaPlex test that runs an actual forward pass. The ~6 GB gap below `PersonaPlexGeneratorRealWeightsTests` (no Depformer) confirms the Depformer weights account for roughly that headroom. The Prefill runs in 28 s, most of which is the 32-layer transformer forward over 4 tokens.

**Notes:** This is the right test to check when suspecting a regression in the LM-only path (tensor wiring, embedding lookup, RMSNorm, RoPE) without involving the Depformer or codec noise. The shape-check fact (`FindTensor`) is essentially free and could be extracted to a separate lightweight class if the memory sweep target is tightened.

---

### 10 - `Sd3PerStepTrajectoryParityTests` -- 36.82 GB / 59.48 s

**File:** `tests/OpenTail.Stingray.Tests.Diffusion/Sd3PerStepTrajectoryParityTests.cs`

**What it tests:** Per-step CPU-vs-GPU latent trajectory comparison for SD3.5 Medium (docs/094 Phase 1). Loads the full pipeline (`Sd3Pipeline.LoadSeparate`) **twice** -- once CPU-only, once with a `VulkanBackend` -- then runs 6 Euler steps with real CLIP-L / CLIP-G / T5 text conditioning, logging mean/std/maxAbs of the latent and the inter-trajectory max-abs-diff after each step.

**Why it is large:**
Two full SD3.5 Medium pipeline instances simultaneously: the MMDiT `sd3.5_medium-Q4_K_M.gguf`, two CLIP text encoders, a T5 encoder, and the VAE -- all doubly resident. This is a diagnostic test written to find the multi-step GPU accumulation bug (identified after the single-forward cosine hit 1.000000 but the full run still produced noise). The 6-step limit was a deliberate cost-saving concession.

**Notes:** Six steps at real 32x32 latent resolution means 6 x 2 (cond + uncond) x 2 (CPU + GPU) = 24 MMDiT forward calls per test run. If the trajectory bug is confirmed resolved, this test''s scope and resource cost should be revisited -- it may be reducible to a single-step residual check or retired in favour of the established `Sd3BaselineTests`.

---

## Observations

1. **The PersonaPlex LM is the dominant cost.** Ranks 1, 3-7, 9 all come from the same `personaplex-7b-v1-q8_0.gguf` checkpoint. The 7B Q8 model is inherently expensive; there is no lightweight substitute for real-weight inference.

2. **Double-load patterns are expensive.** `ZImageGpuRealScaleBisectTests` and `Sd3PerStepTrajectoryParityTests` each load the checkpoint twice (CPU + GPU instances). This is deliberate for isolation but doubles the base cost.

3. **Debug/diagnostic tests carry runtime costs.** `PersonaPlexGenerateWavDebugTest` (50 frames, writes to disk) and `Sd3PerStepTrajectoryParityTests` (6-step full trajectory) are not designed for speed -- they were written for observability. Consider excluding them from the automated sweep or gating them behind a dedicated env-var.

4. **Wall time and memory are not strongly correlated.** `PersonaPlexLiveDuplexRealWeightsTests` is both the heaviest (44.96 GB) and one of the fastest (32.8 s), because it runs a small number of frames but holds extra objects simultaneously (both codec directions). Conversely, `PersonaPlexGenerateWavDebugTest` runs longest (134 s) at lower peak RAM because it sequentially streams 50 frames rather than holding many objects at once.
