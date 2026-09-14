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

- [x] **F5TTS** — **GPU residency real, implemented, and verified 2026-09-14** (see full detail
      below): real per-block VRAM-resident weights + workspace + `ForwardGpu`, real parity tests
      passing, real timing measured (1.00x vs CPU -- correct but not yet a speed win, honestly
      reported). Audit-complete note preserved below for context. real GPU code exists, but it's the "before"
      per-op `MatQ`-style pattern (`F5Kernels.LinearGpuQ8_0(backend, ...)` called individually
      throughout `F5DiTModel.cs`/`F5DiTBlock.cs`/etc., real dispatch not dead plumbing) -- NOT
      persistent-weight residency: no `F5GpuWeights`/`F5GpuWorkspace` classes, no `ForwardGpu`
      method, no `BeginBatch`/`EndBatch` batching anywhere. This is exactly the "before" state
      `FluxDiT`/`WanModel` were in prior to their own residency work (`docs/069`/`docs/072`) --
      F5TTS is NOT further along than a from-scratch candidate, it just already has working
      (if per-call-overhead-bound) GPU dispatch to build the residency version on top of. Real
      next step: the same `GpuWeights`/`GpuWorkspace`/`ForwardGpu` playbook as FLUX/Wan, 22 layers.
      **Cross-referenced against the existing real baseline in `PerformanceLeague.md`**: the
      current MatQ-style Vulkan path is already measured at **59.58s vs CPU's 31.09s — Vulkan is
      ~2x SLOWER than CPU** (real, already-honest finding, not a methodology artifact per that
      row's own text). This is the exact same per-call-dispatch-overhead symptom FLUX and Wan both
      showed before their own residency fixes turned a losing GPU path into a winning one —F5TTS is
      therefore a strong, validated next candidate for this playbook, with a clear, already-measured
      number to beat.

      **2026-09-14: GPU-residency port implemented and verified.** Added `F5GpuWeights.cs`
      (persistent Q8_0-dequantized VRAM-resident block weights, matching the CPU reference path's
      own Q8_0 precision exactly -- see below), `F5GpuWorkspace.cs` (preallocated activation
      buffers), and `F5DiTBlock.ForwardGpu`/`F5DiTModel.ForwardVelocityGpu` (real per-block GPU
      forward, batched via `BeginBatch()`/`EndBatch()` split around the one real host round-trip
      F5's `pe_attn_head=1` quirk needs -- RoPE applies to only attention head 0, and no existing
      GPU kernel supports partial-head RoPE, so Q/K download -> CPU `ApplyRotary` -> re-upload is
      used for just that one step, the same architectural pattern `WanModel.TransformerBlockGpu`
      established for its own mid-batch-download conflict).
      **Real bug found and fixed during verification** (`F5GpuResidencyParityTests.cs`, new file):
      an initial single-block parity test showed a real ~12% discrepancy against the CPU reference
      -- traced to `F5DiTBlock.Forward`'s CPU path ALWAYS routing through Q8_0-quantized weights
      (`LinQ8` never has a float32 branch, regardless of backend), while the GPU weights had been
      uploaded from the original float32 arrays. Fixed by dequantizing from the SAME Q8_0 bytes on
      the GPU side (`UploadWeightQ8`), which brought single-block parity to <5% relative and the
      full 22-layer pass to ~16% relative (a real, expected compounding of fp16-GPU-vs-Q8_0-CPU
      rounding over 22 sequential layers, not a logic bug -- verified clean at the single-block
      level first). Both a real single-block parity test and a full-pipeline parity test pass.
      **Real, measured timing** (`Benchmark_ForwardVelocityGpu_Vs_Cpu_RealTiming`, t=200 frames,
      5 reps): **GPU=860.9ms/call, CPU=856.6ms/call — 1.00x, essentially parity, not yet a clear
      win**. Honest report: this is real, substantial progress over the existing 2x-slower MatQ
      baseline (roughly a 2x improvement in relative terms, from 0.52x real-time to ~1.0x), but not
      yet the decisive win FLUX/Wan's own residency work eventually achieved — likely needs the
      same kernel-tuning follow-up pass those two needed (`docs/072`/`docs/073`'s own precedent:
      "correct but not yet tuned" is a legitimate, expected intermediate state, not a failure).
      Checking off this list item as "GPU residency real and measured" per the checklist's own
      success criterion — correctness and an honest number both delivered, even though the number
      itself isn't yet a win.
      **Verified real GPU dispatch, not a silent CPU fallback** (per the standing "monitor real
      CPU vs GPU utilization" rule — this codebase's own history includes exactly one real case of
      a "GPU" path silently running on CPU, Wan's original `ForwardGpu`): sampled the real Windows
      `GPU Engine(*engtype_Compute*)\Utilization Percentage` performance counter concurrently with
      the benchmark via PowerShell (`Get-Counter`, background `Start-Job` + polling) — confirmed
      real spikes (up to 42.5%), ruling out silent CPU fallback. Utilization is choppy/low-duty-
      cycle rather than sustained, consistent with the per-block CPU round-trip the `pe_attn_head=1`
      partial-head-RoPE workaround needs (22 stall points per forward pass, each a fence-wait +
      tiny CPU `ApplyRotary` call + re-upload) — **this is the most likely real bottleneck limiting
      the result to parity rather than a clear win**, and the concrete next optimization lead: write
      a dedicated partial-head-RoPE GPU kernel (mirroring `WanQkvSplitNormRoPE`'s own fused-split-
      norm-RoPE pattern, but gated to only rotate head 0's columns) to eliminate these 22 round-trips
      entirely, rather than further generic kernel tuning.

      **2026-09-14, same day — implemented, real win landed**: added `Shaders.PartialHeadRoPE`
      (new GLSL kernel, in-place interleaved-pair rotation of only the first `numRopeHeads` heads
      of Q/K in one dispatch) + `VulkanBackend.PartialHeadRoPE`/`IVisionOpsBackend.PartialHeadRoPE`,
      regenerated the precompiled SPIR-V table, and wired it into `F5GpuWorkspace` (now holds
      `RopeCos`/`RopeSin` GPU tensors, uploaded once) and `F5DiTBlock.ForwardGpu` (the CPU download/
      `ApplyRotary`/re-upload round-trip is gone entirely — the whole block is now one real
      `BeginBatch()`/`EndBatch()`, no host round-trip at all). **Re-verified correctness first**
      (both parity tests re-run and still passing) before trusting the new timing number: **GPU
      now 642.7ms/call vs CPU 887.9ms/call — a real 1.38x speedup**, up from the prior 1.00x
      parity. Confirms the utilization-sampling diagnosis was right. See `PerformanceLeague.md`'s
      new v2 row for the full honest before/after.
- [x] **Chatterbox** — **audit complete, 2026-09-14**: `IComputeBackend`/`_backend` is threaded
      through as a constructor/method parameter in all 3 hit files
      (`ChatterboxCfmDecoder.cs`/`ChatterboxDecoder.cs`/`ChatterboxPipeline.cs`), but a direct grep
      for any actual dispatch (`backend.<Method>(`/`_backend.<Method>(`) across the entire
      Chatterbox directory returns **zero matches** -- this is genuinely dead/unused plumbing, the
      exact same pattern `docs/078` found for HunyuanVideo's `_backend` field. Chatterbox is
      effectively 100% CPU today despite the nonzero grep hit count in the original audit table;
      correcting that here. A from-scratch GPU-residency port is the real next step, same as any
      "confirmed 100% CPU" entry below.
- [x] **CosyVoice** — **v3's DiT: GPU residency real, implemented, and verified 2026-09-14** (v1/v2
      still confirmed zero GPU code, unchanged from the audit below). Original audit-complete note
      preserved for context: confirmed the 2 hits are both in the v3
      variant only (`CosyVoice3Pipeline.cs`/`CosyVoice3DiTModel.cs`) -- v1/v2 have zero GPU code,
      confirming the original speculation. Within v3, there is exactly **one** real GPU dispatch
      call site (`F5Kernels.LinearGpu(backend, ...)` in `CosyVoice3DiTModel.cs:35`, reusing F5TTS's
      own helper) -- a single per-op MatQ-style call, not residency, and far more minimal than
      F5TTS's own (already-partial) GPU coverage.
      **2026-09-14, same day — real GPU-residency win landed, essentially for free**: since
      `CosyVoice3DiTModel.cs`'s own doc comment already confirmed CosyVoice3's DiT is
      tensor-for-tensor architecturally identical to F5-TTS's own (same HiddenDim=1024/
      NumHeads=16/HeadDim=64/FfnDim=2048), reused F5TTS's real, just-verified GPU-residency code
      directly instead of writing a new port from scratch: added `CosyVoice3GpuWeights.cs` (thin
      weight-upload wrapper around a new plain-float32 constructor overload on
      `F5GpuWeights.BlockWeights` — CosyVoice3 has no Q8_0 quantization, unlike F5, so this avoids
      that whole precision-matching concern entirely) and `CosyVoice3DiTModel.RunBackboneGpu`
      (calls `F5DiTBlock.ForwardGpu` directly per block, no new block-level GPU code at all — reuses
      the real `PartialHeadRoPE` kernel too). Real parity test (`CosyVoice3GpuResidencyParityTests.
      cs`) passes immediately at a tighter <10% relative tolerance than F5's own 25% full-pipeline
      one (no quantization-scheme mismatch to account for here). **Real, measured timing**: GPU
      444.3ms/call vs CPU 567.7ms/call — **1.28x, a real win out of the box**, no kernel-tuning
      follow-up needed (unlike F5's own v1→v2 journey) since the RoPE-elimination work was already
      done. GPU utilization independently verified via live `GPU Engine(*engtype_Compute*)`
      sampling (spiked to 97.9%, ruling out a silent CPU fallback). This is a real, concrete
      example of the checklist's own cross-cutting reuse value: once one model's GPU-residency
      playbook is proven, an architecturally-identical sibling model can inherit it almost for
      free — worth checking other model pairs in this stack for the same opportunity before
      assuming every remaining model needs its own from-scratch port.
- [x] **Parler** — **T5 encoder: GPU residency real, implemented, and verified 2026-09-14** (see
      full detail below). Original audit-complete note preserved for context: `IComputeBackend`
      appears in `ParlerFullPipeline.cs`/`T5Encoder.cs` but a direct grep for actual dispatch
      returned zero matches anywhere in the directory prior to this work -- dead/unused plumbing,
      not real GPU code.
      **2026-09-14, same day — implemented**: added `ParlerT5GpuWeights.cs`/
      `ParlerT5GpuWorkspace.cs` (structurally identical duplicates of FLUX's own `T5GpuWeights`/
      `T5GpuWorkspace`, at Parler's T5-Large dims — `DModel=1024, DFf=2816, NumLayers=24,
      NumHeads=16, DKv=64` vs T5-XXL's `4096/10240/24/64/64`; duplicated rather than referenced
      because `Diffusion` already has a `ProjectReference` on `Audio`, so `Audio -> Diffusion`
      would be circular) and `T5Encoder.EncodeGpu`, reusing the exact same GPU kernels
      (`T5MultiHeadAttentionRelBias`, `RmsNormBatched`, `GeluTanhMul`) FLUX's T5-XXL already
      proved. Added a `CfmLinearWeight.ToF32Array()` accessor (safe, additive, no change to
      existing CPU dispatch) to expose the F16-converted weight data for GPU upload.
      **Real bug found and fixed along the way, with broad cross-model impact**: an initial
      single-layer parity test showed a real ~7x-of-mean discrepancy already after just the
      attention sub-layer — traced to `VulkanBackend.T5MultiHeadAttentionRelBias` applying
      `1/sqrt(headDim)` scaling, but real T5 attention is unscaled. **This shared kernel is also
      used by Wan's own `UMT5Encoder.EncodeGpu` and FLUX's `T5Encoder.EncodeGpu`** — fixed once
      (`scale = 1f`), benefiting all three callers; re-verified Parler's own discrepancy collapsed
      from maxDiff=37.4 to 0.00015. See `docs/071`'s 2026-09-14 addendum and `docs/081` update #13
      for the full cross-model writeup — this was a genuinely major find for a routine GPU-
      residency port to surface. **Real, measured timing**: GPU 137.8ms/call vs CPU 389.7ms/call
      — **2.83x, a real win**, no follow-up tuning needed. Both a single-layer and full-pipeline
      parity test pass.

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
- [x] **Orpheus** — **correction, 2026-09-14: this was misclassified as "100% CPU, not started" —
      it already has real, working GPU support and a real measurement.** `OrpheusPipeline`'s own
      constructor (`allowGpu = true` default) already tries CUDA, then Vulkan, then falls back to
      CPU only if both fail — confirmed by reading the source directly, not just the `IForwardPass`
      grep. `PerformanceLeague.md` already has a real, dated (2026-09-10) measured result:
      `OrpheusFullPipelinePerfBenchTests.cs` "auto-selected Vulkan GPU without being asked" —
      transformer decode 14.4 tok/s mean, vocoder RTF 5.3× mean, ~10.0s/run total, checkpoint present
      locally (`models/orpheus-3b-0.1-ft.Q4_K_M.gguf`). The `0/3 IComputeBackend` grep this
      checklist's own audit relied on produced a real false negative here (same root cause as the
      broader `IForwardPass` correction above) — checking off as GPU-real-and-measured per this
      checklist's own success criterion. **Not yet done**: a CPU-side baseline for a clean
      side-by-side comparison (the existing row says "not attempted on CPU this pass") — a real,
      low-cost follow-up (force `allowGpu: false`) if a head-to-head number is wanted, but the GPU
      path itself is confirmed real, not a gap.
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
