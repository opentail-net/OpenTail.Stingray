# TTS/ASR Transformer Stack — GPU Residency Checklist (2026-09-13)

> Completed items moved to [done/080-tts-asr-gpu-residency-checklist-completed-items-2026-10.md](../../done/080-tts-asr-gpu-residency-checklist-completed-items-2026-10.md) on 2026-10-30. This file keeps only open work.

> **Status (checked 2026-10-30, file-level only):** 28 open items; the items were not re-run. **Thread:** [audio GPU residency](080-tts-asr-gpu-residency-checklist.md).

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
| Orpheus | 0/3 | **CORRECTED 2026-09-14: real GPU support already exists and is already measured (see checklist entry below) — the `0/3 IComputeBackend` grep is a false negative, same root cause as the broader `IForwardPass` correction elsewhere in this doc** |
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

## 2026-09-14 major correction: the "0 IComputeBackend hits = 100% CPU" audit criterion has a real,
## broadly-applicable blind spot — check `IForwardPass` usage before assuming any of these need a
## from-scratch GPU-residency port

The audit above greps for direct `IComputeBackend` usage. That misses any model whose LLM-style
backbone is built on this project's own shared, backend-agnostic `IForwardPass`/`ForwardPass` engine
(the same flagship infrastructure every normal GGUF text model in this codebase runs on, which
already has real, mature Vulkan/CUDA support — `VulkanBackend`, `CudaForwardPass`) instead of calling
`IComputeBackend` directly. A model built this way can ALREADY run on GPU with **zero new
architecture code** — no `GpuWeights`/`GpuWorkspace`/`ForwardGpu` port needed at all — simply by
constructing its `ForwardPass` with a GPU backend instead of `CpuBackend`.

**Checked directly**: 8 of the 26 models marked "100% CPU" above actually reference `IForwardPass`:
`HiggsAudio` (`HiggsGenerator`/`HiggsArStepper`), `VibeVoice` (both `VibeVoiceGenerator` and the
separate `VibeVoiceAsrGenerator`), `PersonaPlex` (`PersonaPlexGenerator`), `OmniVoice`
(`OmniVoiceMaskGitForward`/`Weights`), `NeuTts` (`NeuTtsGenerator`), `VoxCpm2`
(`VoxCpm2Generator`), `QwenTTS` (`QwenTtsTalkerGeneration`), `Orpheus` (`OrpheusPipeline`).
**And the pattern holds where checked**: both `HiggsAudioPerfBaselineDebugTest.cs` (HiggsAudio's own
`PerformanceLeague.md` baseline, 36.01s/13.64× RTF) and `VibeVoiceAsrPerfBaselineDebugTest.cs`
explicitly construct `new CpuBackend()` — meaning **their existing CPU baselines never even tried
the GPU path these models can already use for free.**

**This does NOT mean GPU residency is automatically a big win for all 8** — it needs to actually be
measured, same honesty bar as everything else in this doc — but it DOES mean the right next action
for each of these 8 specifically is "swap `CpuBackend` for `VulkanBackend` in the existing benchmark
and re-measure" (cheap, ~one line, no new files) BEFORE spending effort on a from-scratch residency
port (expensive, many new files, the playbook the rest of this checklist assumes). **Only the
non-LLM parts of these 8 models** (codec decoders, vocoders, audio-specific post-processing — e.g.
HiggsAudio's own `HiggsCodecDecoder`/`HiggsCodecEncoder`) still have no GPU path at all and would
still need the standard from-scratch playbook if pursued. See HiggsAudio's own entry below for the
first fully-worked-through example of this correction; the other 7 have not yet been individually
re-audited under this same lens — doing so for each (confirm real `IForwardPass` usage in its actual
generation path, find or write a perf-baseline test, swap the backend, measure) is real, low-risk,
high-value work for whoever picks up this checklist next, likely higher-value-per-hour than most of
the from-scratch candidates below given the near-zero implementation cost.

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


**Correction to this doc's own earlier framing**: of the original 4 "partial GPU" entries, only
**F5TTS** and **CosyVoice3** have any real (if minimal, per-op) GPU dispatch at all -- Chatterbox
and Parler are dead-plumbing false positives from the original grep-only pass, effectively
identical to the "confirmed 100% CPU" tier below. Real state, all 30 models: **28 are effectively
100% CPU** (26 original + Chatterbox + Parler), and only **F5TTS and CosyVoice3** have any real
(pre-residency, MatQ-style) GPU code to build on -- no model in this entire stack has confirmed
real GPU residency yet, unchanged from the original conclusion, but the "which 4 are worth auditing
first" framing was half wrong.
- [ ] **AudioGen** — confirmed 100% CPU.
- [ ] **FishSpeech** — confirmed 100% CPU.
- [ ] **HiggsAudio** — **2026-09-14 correction: the "confirmed 100% CPU" grep finding above is
      MISLEADING for this specific model, and it's a much cheaper GPU-residency candidate than this
      checklist's from-scratch playbook implies.** `HiggsGenerator`/`HiggsArStepper` (the 4B-param,
      36-layer autoregressive LLM backbone that dominates this model's cost) don't call
      `IComputeBackend` directly — they route through this project's own shared, backend-agnostic
      `IForwardPass`/`ForwardPass` engine (the same flagship infrastructure every normal GGUF LLM in
      this codebase already runs on, which DOES have real, mature Vulkan/CUDA support via
      `VulkanBackend`/`CudaForwardPass`). The `docs/080` audit's "0/10 IComputeBackend hits" grep
      criterion structurally cannot see this pattern — it only catches models calling
      `IComputeBackend` directly, not ones going through the shared engine abstraction. **Checked the
      exact benchmark that produced the existing `PerformanceLeague.md` baseline** (36.01s / 13.64×
      RTF, `HiggsAudioPerfBaselineDebugTest.cs`): it explicitly constructs `new CpuBackend()` and
      passes that into `ForwardPass` — i.e. **the existing baseline never tried the GPU path this
      model can already use with ZERO new code**, unlike every other item on this list which
      genuinely needs a from-scratch `GpuWeights`/`GpuWorkspace`/`ForwardGpu` port. The real, cheap
      next step for whoever has room: re-run that exact same benchmark with `new VulkanBackend()` (or
      a CUDA backend, if built) in place of `CpuBackend()` and compare — this could be a large, close
      to free win given how mature and already-tuned this project's general LLM GPU engine is
      (see the LLM inference benchmarks throughout the rest of `PerformanceLeague.md`), with no
      residency architecture work needed at all. **Not run this pass**: the real checkpoint
      (`models/_models/higgs_audio_tts/Higgs-Audio-v3-TTS-4B-GGUF/higgs-audio-v3-tts-4b-q8_0.gguf`,
      several GB) is not present locally, and this machine's disk was at only ~11GB free when
      checked — a speculative multi-GB download for an unconfirmed (if promising) win wasn't judged
      worth the space risk this pass; flagging as the concrete next action instead. The remaining,
      separate piece of this model (`HiggsCodecDecoder`/`HiggsCodecEncoder`/
      `HiggsSemanticPostEncoder` — the audio codec, not the LLM) genuinely has no `IComputeBackend`
      or `IForwardPass` usage and would still need the standard from-scratch GPU-residency playbook
      if pursued, same as everything else on this list; only the LLM-backbone portion benefits from
      this shortcut. A Horizontal-Pass-A SIMD fix already landed for the codec decoder specifically —
      check that `PerformanceLeague.md` entry before assuming more CPU-side headroom remains there.
- [ ] **Kokoro** — confirmed 100% CPU.
- [ ] **MeloTTS** — confirmed 100% CPU. Already had a Horizontal-Pass-A SIMD parallelization fix
      for its speaker-embedding `LinearVec` (per `PerformanceLeague.md`) — check that entry.
- [ ] **MossTts** — confirmed 100% CPU. Already has a `SimdKernels.MatVecF32`-based `Linear`/
      `LinearBatched` path per source inspection earlier this session — check whether that's
      already close to what CPU SIMD can offer before assuming GPU residency is the next lever.
- [ ] **MusicGen** — confirmed 100% CPU.
- [ ] **NeuTts** — confirmed 100% CPU.
- [ ] **OmniVoice** — confirmed 100% CPU.
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
- **Measurement rules:** see [../../reference/performance-measurement-rules.md](../../reference/performance-measurement-rules.md) (one thing at a time, memory check before benchmarks).
- **A green "RealWeights" test does not mean it ran against real weights** — per this project's own
  `CLAUDE.md` rule 12, many `*RealWeights*` tests silently no-op (`if (modelPath is null) return;`)
  when a checkpoint isn't present locally, reporting "passed" in ~0.1-0.4s. Check the real per-test
  timing before trusting any existing test as evidence a model's CPU baseline is solid.

## Closure rule and claim caveat (ported from the retired audio-GPU-residency workstream, 2026-10-31)

- Close the checklist when each engine is either resident and measured, or written off with the reason in this file.
- Any residency speed claim needs a discrete-GPU run, or an explicit iGPU-only caveat (CLAUDE.md rule 13).

## Success criterion, per model

Real, measured GPU-vs-CPU timing (win or not, report honestly), a real numeric parity test against
the existing CPU path, and (where relevant) a real audio sample re-listened-to for correctness —
not just "the run completed without error." Update `PerformanceLeague.md` per model as each is
picked up, following this session's own established honesty bar throughout.

---

## Research findings (2026-10-30)

Checked in code:

- **The `IForwardPass` correction for HiggsAudio is confirmed.** `HiggsGenerator.cs` (line 31) and `HiggsArStepper.cs` (line 76) take an `IForwardPass`, so the LLM backbone can run on any backend the engine supports.
- **The baseline never used a GPU.** `tests/OpenTail.Stingray.Tests.Audio/HiggsAudioPerfBaselineDebugTest.cs` line 44 constructs `new CpuBackend()`. Swapping in `VulkanBackend` is a one-line change for the backbone.
- **The checkpoint is absent on this machine.** `find models -iname "higgs-audio-v3-tts-4b*"` returns nothing, and the test skips without it (`Assert.SkipUnless`, line 37). **The next step is therefore blocked on a multi-GB download,** and disk space needs checking first. I did not download it.

Not checked: the other seven `IForwardPass` models the doc names (VibeVoice, PersonaPlex, OmniVoice, NeuTts, VoxCpm2, QwenTTS, Orpheus). Their perf tests were not read.

The Vulkan backend's behaviour on this iGPU is covered in [GR_performance.md](../GR_performance.md) and the leads plan. Dispatch overhead is measured there, not here.

---

## Research findings (2026-10-31, deep pass)

Web research was not applicable to the performance question. The HiggsAudio perf path is internal (the `IForwardPass` and `CpuBackend` facts were checked on 2026-10-30). The Vulkan and CUDA constraints that would govern a GPU run are in the [GR doc](../GR_performance.md): the CUDA module needs per-architecture guards and the Vulkan pin needs the subgroup bound.

**Still open, unchanged:** the Higgs checkpoint is not on disk, so the one-line backend swap cannot be measured yet.

---

## Thread state (from the retired workstreams folder, 2026-10-31: audio-gpu-residency.md)

### Correct state

- **The goal:** keep the transformer stacks resident on the GPU, one engine at a time.
- **Hardware limit:** this machine has an integrated Radeon only. CLAUDE.md rule 13 applies: an iGPU result cannot show whether a discrete GPU would be faster. Residency results here describe this APU.
- **Measurement:** both plans carry the "one thing at a time" rule, which now lives in [../reference/performance-measurement-rules.md](../../reference/performance-measurement-rules.md).
- **079's own status (2026-09-27):** the goal is still open. The 2026-10-30 check did not re-run its items.

### Open items

- [ ] 080: 28 open items, phase by phase (see the file)
- [ ] 079: 7 open items (see the file)
- [ ] Any residency speed claim needs a discrete-GPU run or an explicit iGPU-only caveat

### Cadence

- Review when an audio engine's residency path changes, or when a discrete GPU is available.
- Re-read at least every two weeks while any item is open.
- Close when each engine in 080 is either resident and measured, or written off with the reason.
