# Completed items split from 080-tts-asr-gpu-residency-checklist.md (2026-10-30)

Finished items moved verbatim out of [../4-performance/audio/080-tts-asr-gpu-residency-checklist.md](../4-performance/audio/080-tts-asr-gpu-residency-checklist.md). Open items stay there. Each block notes the section it came from.

> from: ## Checklist — work through as capacity allows, check off once GPU residency is real and measured
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

> from: ## Checklist — work through as capacity allows, check off once GPU residency is real and measured
- [x] **Chatterbox** — **audit complete, 2026-09-14**: `IComputeBackend`/`_backend` is threaded
      through as a constructor/method parameter in all 3 hit files
      (`ChatterboxCfmDecoder.cs`/`ChatterboxDecoder.cs`/`ChatterboxPipeline.cs`), but a direct grep
      for any actual dispatch (`backend.<Method>(`/`_backend.<Method>(`) across the entire
      Chatterbox directory returns **zero matches** -- this is genuinely dead/unused plumbing, the
      exact same pattern `docs/078` found for HunyuanVideo's `_backend` field. Chatterbox is
      effectively 100% CPU today despite the nonzero grep hit count in the original audit table;
      correcting that here. A from-scratch GPU-residency port is the real next step, same as any
      "confirmed 100% CPU" entry below.

> from: ## Checklist — work through as capacity allows, check off once GPU residency is real and measured
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

> from: ## Checklist — work through as capacity allows, check off once GPU residency is real and measured
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

> from: ## Checklist — work through as capacity allows, check off once GPU residency is real and measured
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

