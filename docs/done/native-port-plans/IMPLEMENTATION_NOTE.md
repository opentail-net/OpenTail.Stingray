Corrective slice against 033-native-stable-diffusion-family-port-plan.md
=========================================================================

The original scaffold (added while waiting on a slower model) invented a
disconnected `SdTensor`/`ITextEncoder`/`IUNet`/`IVaeDecoder`/`IScheduler` type
system and jumped straight to SDXL stub components, skipping SD1.5 and
ignoring the existing FLUX/Z-Image pipelines entirely. That violated 033
Phase 1's actual requirement — adapt the *existing* pipelines to a shared
abstraction, not build a fantasy one next to them — and Phase 4's sequencing
(SDXL extends SD1.5's UNet; SD1.5 doesn't exist yet, so there was nothing to
extend).

What this slice did
--------------------
- Deleted the SDXL-premature files (`ClipGEncoder`, `UNetSDXL`,
  `VaeDecoderSDXL`, `SDXLPipeline` + their tests) and the wrong-interface
  scaffold (`SdTensor`-based `Interfaces.cs`, `DiffusionPipeline.cs`,
  `ModelRegistry.cs`, the placeholder `Loading/WeightLoader.cs` +
  `TensorAssignment.cs`). All were untracked/uncommitted, so nothing was lost
  from history.
- Added `src/OpenTail.Stingray.Diffusion/IDiffusionPipeline.cs` at the
  project root (not nested under `StableDiffusion/`) since these are meant
  to be shared across FLUX, Z-Image, and every SD family per 033 section 1's
  end-state tree. Defines `IDiffusionPipeline`, `ImageGenerationRequest`,
  `IVaeDecoder`, `ITextConditioner`/`IConditioning`, `IDiffusionModel`,
  `IDiffusionScheduler`, `IDiffusionSampler` — named per the plan's Phase 1
  list, typed against plain `float[]` + explicit dims (matching how
  `VaeDecoder`/`FluxDiT`/`ZImageDiT` already represent tensors at the
  orchestration layer — none of them use `Core.Tensor` there; that type is
  reserved for backend-resident GPU handles inside individual components).
- **Actually implemented, not just declared:** `VaeDecoder : IVaeDecoder`
  (its existing `Decode(float[], int, int)` already matched — zero-diff),
  and `ImagePipeline`/`ZImagePipeline` both now explicitly implement
  `IDiffusionPipeline.Generate(ImageGenerationRequest)` as a thin adapter
  over their existing public `Generate(...)` overloads. Their internals are
  untouched, so Phase 1's acceptance criterion ("existing FLUX and Z-Image
  output remains numerically/conceptually unchanged") holds by construction.
- **Declared but not yet implemented by real code:** `ITextConditioner`,
  `IDiffusionModel`, `IDiffusionScheduler`, `IDiffusionSampler`. Retrofitting
  `ClipLEncoder`/`FluxDiT`/`EulerFlowScheduler` onto these now would mean
  guessing the right shape from a single implementation (FLUX's, which has
  RoPE position ids and pooled embeddings baked into its forward signature).
  These interfaces are sized for SD1.5's upcoming CLIP/UNet/scheduler
  (033 Phase 2.2-2.5) — validate/adjust them once that second real
  implementation exists, then decide whether adapting FLUX/Z-Image's
  internals to match is worth the risk to proven, working numeric code.
- Added `tests/.../DiffusionPipelineInterfaceTests.cs` (assignability +
  `ImageGenerationRequest` default checks — can't exercise real generation
  without model weights) and registered
  `tests/OpenTail.Stingray.Tests.Diffusion` in `OpenTail.Stingray.slnx`,
  which it was missing from — `dotnet test` at the repo root was silently
  skipping this project before.

What this slice deliberately did NOT do (still open, per 033 sequencing)
--------------------------------------------------------------------------
- Phase 0 (reference freeze + operator audit): a local clone of
  `stable-diffusion.cpp` already exists at
  `examples/stable-diffusion.cpp` — record its commit SHA and produce the
  Phase 0.2 operator matrix (grep `IComputeBackend`/`IImageOpsBackend` +
  existing Diffusion classes) before starting SD1.5 UNet work. The
  `s1`-`s9`/`zzz_*` `.txt` files in this folder and `files of interest.txt`
  are untested draft notes/pointers for that phase and Phase 2 — treat as
  input to verify, not as finished design.
- Phase 2 (SD1.5): model loading (reuse `IWeightLoader`/`SafetensorsLoader`/
  `GgufWeightLoader` directly — no new loader class, per 2.1's "no best
  effort tensor mapping"), CLIP text encoder, the SD UNet (the actual major
  new component), scheduler + CFG, VAE reuse/extension. None of this exists
  yet; this slice only fixed the foundation it will sit on.
