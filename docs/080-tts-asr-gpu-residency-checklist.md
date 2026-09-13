# TTS/ASR Transformer Stack — GPU Residency Checklist (2026-09-13)

## Context and how to use this doc

This is a **checklist**, not a per-model deep plan like `docs/069`/`docs/071`-`079` — those cover
diffusion/video DiTs where the GPU-residency playbook is now proven (real, committed wins: FLUX
`53839a3`, Wan `0132aa7`/`d877353`). This doc is the equivalent starting point for the **TTS/ASR
transformer stack** in `src/OpenTail.Stingray.Audio/`, which is a much larger set of smaller
models rather than a few large DiTs — so the right approach is: work through this list one model
at a time, check ☐ off as GPU residency is confirmed real and measured, and write a short
model-specific note (not necessarily a full separate doc) once each one is done. Use the **same
methodology** as the diffusion docs: real playbook is
`FluxGpuWeights`/`FluxGpuWorkspace`/`ForwardGpu` (`docs/069`) — persistent VRAM-resident weights +
workspace, a real per-block/per-layer GPU forward, batched one `BeginBatch()`/`EndBatch()` per
block, verified with a real numeric parity test before trusting it, and an honest
`PerformanceLeague.md` update whether the result is a win or not (Wan's own GPU path was
initially *slower* than CPU before kernel tuning, `docs/072`/`docs/073` — a real, non-alarming
possible outcome here too, not something to hide).

## Real audit performed before writing this list (don't re-derive)

Grepped every model directory under `src/OpenTail.Stingray.Audio/` for any reference to
`IComputeBackend` — a cheap, real signal for "has any GPU code at all" (not a proof of real
residency; even a hit could be `MatQ`-style ping-pong, or in HunyuanVideo's case (`docs/078`) even
a genuinely *dead, unused* field — check for real, not just grep-and-assume):

| Model | `IComputeBackend` hits | Status |
|---|---|---|
| **F5TTS** | 8/16 files | Some GPU code exists — **check depth first** (could be real residency, `MatQ`, or partial/dead like HunyuanVideo) before assuming it needs this playbook from scratch |
| **Chatterbox** | 3/12 files | Some GPU code exists — check depth first |
| **CosyVoice** | 2/26 files | Some GPU code exists — check depth first (26 files is a large model family, likely CosyVoice v1/v2/v3 variants; the 2 hits may only cover one variant) |
| **Parler** | 2/10 files | Some GPU code exists — check depth first |
| AudioGen | 0/6 | **Confirmed 100% CPU** |
| FishSpeech | 0/7 | **Confirmed 100% CPU** |
| HiggsAudio | 0/10 | **Confirmed 100% CPU** |
| Kokoro | 0/13 | **Confirmed 100% CPU** |
| MeloTTS | 0/11 | **Confirmed 100% CPU** |
| MossTts | 0/9 | **Confirmed 100% CPU** |
| MusicGen | 0/8 | **Confirmed 100% CPU** |
| NeuTts | 0/5 | **Confirmed 100% CPU** |
| OmniVoice | 0/14 | **Confirmed 100% CPU** |
| Orpheus | 0/3 | **Confirmed 100% CPU** |
| PersonaPlex | 0/11 | **Confirmed 100% CPU** (see caveat below — this one already got a real CPU-side residency-style fix) |
| Piper | 0/9 | **Confirmed 100% CPU** |
| QwenTTS | 0/21 | **Confirmed 100% CPU** |
| Rvc | 0/10 | **Confirmed 100% CPU** |
| VibeVoice | 0/21 | **Confirmed 100% CPU** |
| VoxCpm2 | 0/15 | **Confirmed 100% CPU** |
| Xtts | 0/21 | **Confirmed 100% CPU** |

**Update, same day: the remaining 12 models have now been audited too** (same grep). All 12 are
**confirmed 100% CPU** as well — Citrinet (0/4), FunASR (0/12), MarbleNet (0/3), MmsTts (0/8),
NemotronAsr (0/5), ParaformerOnnx (0/4), Parakeet (0/6), QwenASR (0/11), SenseVoice (0/4), Vad
(0/4), VoxtralRealtime (0/6), Whisper (0/10). **Correction to this doc's own earlier speculation**:
`ParaformerOnnx` is NOT ONNX-Runtime-backed despite the name — confirmed by checking its actual
source (`ParaformerOnnxModel.cs`/`ParaformerOnnxDecoder.cs`/`ParaformerOnnxPipeline.cs`): no
`OnnxRuntime`/`InferenceSession` references anywhere, it's a native port like everything else in
this codebase, "Onnx" in the name refers only to the original weight-export format it was
converted from. So it's a normal candidate for this playbook like any other entry here, not a
special case to skip.

So: **all 30 audited models in `src/OpenTail.Stingray.Audio/` are either 100% CPU (26 of them) or
have some partial GPU code worth auditing first (4: F5TTS, Chatterbox, CosyVoice, Parler)** — no
model in this stack has confirmed real GPU residency yet. Several of the ASR models (Whisper,
Parakeet, SenseVoice, FunASR, NemotronAsr, ParaformerOnnx, QwenASR, MarbleNet, Citrinet, MmsTts,
VoxtralRealtime) are recognition models, typically smaller/faster than TTS generation models —
check real size/RTF before assuming GPU residency is worth the effort there (see prioritization
note below); Vad (Silero VAD) in particular is almost certainly too tiny to be worth it at all.

**Important caveat on `PersonaPlex`**: `PerformanceLeague.md` already documents a **real GPU win**
for this model (`PersonaPlex, PERF-SWEPT (Horizontal Pass A)`: 1099s → 135.140s, 8.1×) — but that
was a `SimdKernels`/`DenseKernels` **CPU-side** SIMD fix (per that row's own text: "never wired to
this engine's own `SimdKernels`/`DenseKernels` SIMD+parallel matvec"), not Vulkan/CUDA GPU
residency. The `0/11 IComputeBackend` grep result is real and consistent with that — this model's
big win was a different, already-completed class of fix (naive-scalar-loop → SIMD), not the
GPU-residency playbook this doc is about. Genuine Vulkan/CUDA GPU residency is still a separate,
unexplored opportunity for this model, same as everything else marked "Confirmed 100% CPU" above.

## How to prioritize this list

Not all of these are equally worth the effort. Before picking one, check (real data, not
assumption):

1. **Real model size** (parameter count / hidden dim) — bigger transformers benefit more from GPU
   residency's core win (eliminating per-matmul host round-trips); tiny models may not be worth it.
2. **Existing CPU RTF** (real time factor) in `PerformanceLeague.md` — a model already close to
   real-time on CPU has less headroom to gain; one that's many multiples slower than real-time is
   a better GPU-residency candidate, mirroring why FLUX/Wan (both large, slow-on-CPU DiTs) were
   worth doing first.
3. **Known correctness status** — same lesson as LTX-Video (`docs/077`): don't build GPU residency
   on top of a model with an unresolved correctness bug; check `PerformanceLeague.md` and
   `docs/audio-review-progress.md` for each candidate's real, current status before starting.

## Checklist — work through as capacity allows, check off once GPU residency is real and measured

- [ ] **F5TTS** — has partial GPU code (8/16 files); audit depth first, may be further along than
      the rest of this list.
- [ ] **Chatterbox** — has partial GPU code (3/12 files); audit depth first.
- [ ] **CosyVoice** — has partial GPU code (2/26 files); audit depth first, and note this is a
      multi-variant family (v1/v2/v3) — check which variant(s) the existing GPU code covers.
- [ ] **Parler** — has partial GPU code (2/10 files); audit depth first.
- [ ] **AudioGen** — confirmed 100% CPU.
- [ ] **FishSpeech** — confirmed 100% CPU.
- [ ] **HiggsAudio** — confirmed 100% CPU. Real CPU baseline already exists (`PerformanceLeague.md`
      36.01s / 13.64× RTF for text→2.64s audio) plus a Horizontal-Pass-A SIMD fix already landed
      for its codec decoder — check that entry before assuming a from-scratch GPU story.
- [ ] **Kokoro** — confirmed 100% CPU.
- [ ] **MeloTTS** — confirmed 100% CPU. Already had a Horizontal-Pass-A SIMD parallelization fix
      for its speaker-embedding `LinearVec` (per `PerformanceLeague.md`) — check that entry.
- [ ] **MossTts** — confirmed 100% CPU. Already has a `SimdKernels.MatVecF32`-based `Linear`/
      `LinearBatched` path per source inspection earlier this session — check whether that's
      already close to what CPU SIMD can offer before assuming GPU residency is the next lever.
- [ ] **MusicGen** — confirmed 100% CPU.
- [ ] **NeuTts** — confirmed 100% CPU.
- [ ] **OmniVoice** — confirmed 100% CPU.
- [ ] **Orpheus** — confirmed 100% CPU.
- [ ] **PersonaPlex** — confirmed 100% CPU for real Vulkan/CUDA residency (see caveat above — its
      existing 8.1× win was a CPU SIMD fix, a different, already-completed class of work).
- [ ] **Piper** — confirmed 100% CPU.
- [ ] **QwenTTS** — confirmed 100% CPU. Per this session's own memory notes, QwenTTS has a known,
      separate, larger issue (partially fake/procedural weights per `reference_qwentts_completion_plan`
      memory) — **check that this model's port is actually real and complete before investing in
      GPU residency for it; a from-scratch correctness pass may be the real prerequisite here**,
      the same class of warning `docs/077` gives for LTX-Video.
- [ ] **Rvc** — confirmed 100% CPU.
- [ ] **VibeVoice** — confirmed 100% CPU. Already had a Horizontal-Pass-A SIMD parallelization fix
      for its ConvNeXt block's `LinearVec` (per `PerformanceLeague.md`) — check that entry.
- [ ] **VoxCpm2** — confirmed 100% CPU.
- [ ] **Xtts** — confirmed 100% CPU. Already had two Horizontal-Pass-A SIMD fixes (conditioning
      encoder, vocoder FiLM projection) plus its own earlier `XttsResNetEncoder` GPU-SIMD-wiring
      win (2.774× RTF, per `PerformanceLeague.md`) — this one already has real perf-pass history,
      check it before assuming a from-scratch story.
- [ ] **Citrinet** — confirmed 100% CPU. ASR model, check real size/RTF before prioritizing.
- [ ] **FunASR** — confirmed 100% CPU. ASR model, check real size/RTF before prioritizing.
- [ ] **MarbleNet** — confirmed 100% CPU. ASR (VAD-adjacent) model, likely small — check size first.
- [ ] **MmsTts** — confirmed 100% CPU. A TTS model despite living near the ASR cluster in this
      list — check real size/RTF like the other TTS candidates above.
- [ ] **NemotronAsr** — confirmed 100% CPU. ASR model, check real size/RTF before prioritizing.
- [ ] **ParaformerOnnx** — confirmed 100% CPU, and confirmed NOT ONNX-Runtime-backed (checked the
      real source directly — no `OnnxRuntime`/`InferenceSession` references anywhere; "Onnx" in
      the name is just the original weight-export format, this is a normal native port like every
      other model here, not a special case).
- [ ] **Parakeet** — confirmed 100% CPU. ASR model, check real size/RTF before prioritizing.
- [ ] **QwenASR** — confirmed 100% CPU. ASR model, check real size/RTF before prioritizing.
- [ ] **SenseVoice** — confirmed 100% CPU. ASR model, check real size/RTF before prioritizing.
- [ ] **Vad** — confirmed 100% CPU. Silero VAD is typically tiny; likely not worth GPU residency
      at all — check real size before spending any effort here.
- [ ] **VoxtralRealtime** — confirmed 100% CPU. Check real size/RTF and current correctness status
      before prioritizing — this project's own memory notes flag a prior "vibevoice_asr corruption
      finding" cross-reference worth checking isn't related before assuming this one is solid.
- [ ] **Whisper** — confirmed 100% CPU. This project's own most-cited ASR reference point in
      `PerformanceLeague.md` (base/small/medium/large-v3 RTF rows already exist) — likely worth
      checking first among the ASR candidates given how much real baseline data already exists to
      compare against.

## Practical constraints (same as every prior handoff this session)

- **No subagents** — do all work directly in the main session (`CLAUDE.md` rule 6).
- **Never remove or revert existing correct code.**
- **Ask before committing** — confirm current expectations before committing, don't assume
  standing permission carries forward indefinitely.
- **Run one thing at a time, alone** — check `Get-CimInstance Win32_OperatingSystem | Select
  FreePhysicalMemory` (PowerShell) before a benchmark; a real incident earlier this session came
  from an unrelated concurrent `dotnet test` run contaminating a timing measurement.
- **A green "RealWeights" test does not mean it ran against real weights** — per this project's own
  `CLAUDE.md` rule 12, many `*RealWeights*` tests silently no-op (`if (modelPath is null) return;`)
  when a checkpoint isn't present locally, reporting "passed" in ~0.1-0.4s. Check the real per-test
  timing before trusting any existing test as evidence a model's CPU baseline is solid.

## Success criterion, per model

Real, measured GPU-vs-CPU timing (win or not, report honestly), a real numeric parity test against
the existing CPU path, and (where relevant) a real audio sample re-listened-to for correctness —
not just "the run completed without error." Update `PerformanceLeague.md` per model as each is
picked up, following this session's own established honesty bar throughout.
