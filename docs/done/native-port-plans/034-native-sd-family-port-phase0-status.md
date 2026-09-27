# Phase 0/1 Status — Native Stable Diffusion Family Port

**Parent plan:** `033-native-stable-diffusion-family-port-plan.md`
**This doc:** the Phase 0 audit (0.1 reference freeze, 0.2 operator matrix) plus the current
stable pause point and exactly what to do first when work resumes.

## Status

**Phase 0 done. Phase 1 done (corrective slice).** Nothing in Phase 2+ has started. This is a
deliberate pause point — see "Resuming" at the bottom before writing any SD1.5 code.

---

## Phase 0.1 — Reference freeze

Reference clone already exists locally at `examples/stable-diffusion.cpp` (not a build
dependency — dev/reference only, per 033's Phase 0.1/13 rules).

- **Commit:** `de298c225bed97c3f9026b73cd7b71e7879bd41b`
- **Tag:** `master-820-de298c2`
- **Remote:** `https://github.com/leejet/stable-diffusion.cpp.git`
- **Date:** 2026-08-12

Key reference files (verified present at these paths in the frozen commit — `files of interest.txt`
in this folder is a matching, accurate draft file guide, written before this freeze but checks out):

| Concern | Reference path |
|---|---|
| Orchestration (model loading, backend selection, conditioning, sampling, generation) | `src/stable-diffusion.cpp` |
| SD1.x / SDXL UNet | `src/model/diffusion/unet.hpp` |
| SD3/3.5 MMDiT | `src/model/diffusion/mmdit.hpp` |
| FLUX | `src/model/diffusion/flux.hpp` |
| CLIP text encoder | `src/model/te/clip.hpp` |
| T5 text encoder | `src/model/te/t5.hpp` |
| VAE (family-specific variants) | `src/model/vae/*.hpp` |
| CLIP tokenizer | `src/tokenizers/clip_tokenizer.{h,cpp}` |
| T5 tokenizer | `src/tokenizers/t5_unigram_tokenizer.{h,cpp}` |
| Model/weight loading | `src/model_loader.{h,cpp}`, `src/model_manager.{h,cpp}`, `src/weight_manager.h` |
| Upscaling | `src/upscaler.{h,cpp}` |

The `s1`–`s9` and `zzz_*` `.txt` files in this folder are earlier, **untested** draft notes
against this same reference — treat as a first pass to verify against the frozen commit above,
not as settled design.

---

## Phase 0.2 — Operator audit

Grounded in the actual current codebase (not aspirational) — checked by reading
`IComputeBackend.cs`, `IImageOpsBackend.cs`, `DiffusionOps.cs`, and the FLUX/Z-Image
implementations directly.

**Headline finding: the CPU operator library is far more complete than the deleted scaffold
assumed.** `DiffusionOps.cs` already has `Conv2D`, `GroupNorm`, `LayerNorm`, `SiLU`/`GeLU`,
`Linear`, `Softmax`, `RmsNorm`, `Upsample2x`/`Bilinear`/`Bicubic`, `PixelShuffle`/`Unshuffle`,
`LeakyReLU` — essentially every primitive SD1.5's UNet needs already exists as a CPU function.
SD1.5 Phase 2.3 is mostly **block topology + weight-name mapping + cross-attention wiring**
on top of existing primitives, not "build numeric kernels from scratch."

| Operator | Stingray | SD1.5 need | SDXL need | SD3.5 need | Notes |
|---|---|---|---|---|---|
| MatMul / GEMM | 🟢 | reuse | reuse | reuse | `IComputeBackend.MatMul`/`Sgemm` (GPU), `DiffusionOps.Linear` (CPU) |
| Conv2D | 🟢 | reuse | reuse | n/a (DiT, no conv) | `DiffusionOps.Conv2D` (CPU), `IImageOpsBackend.Conv2d` (GPU, im2col) |
| GroupNorm | 🟢 | reuse | reuse | n/a (uses LayerNorm/AdaLN) | `DiffusionOps.GroupNorm` — used today by `VaeDecoder` |
| LayerNorm | 🟢 | reuse | reuse | reuse | `DiffusionOps.LayerNorm` exists but unused by FLUX/Z-Image today (they use RmsNorm-style AdaLN) — first real consumer will be SD1.5 CLIP/UNet |
| SiLU / GeLU | 🟢 | reuse | reuse | reuse | Both CPU (`DiffusionOps`) and GPU (`IComputeBackend.SiLU`, QuickGELU in `ClipLEncoder`) |
| Self-attention | 🟢 | reuse (UNet mid-block) | reuse | reuse | `FluxDiT.SelfAttention`/`JointAttention` (CPU, concatenated-sequence) |
| **Cross-attention (Q len ≠ K/V len)** | 🆕 | **required** | required | required | Nothing today does asymmetric Q/KV lengths — FLUX's "joint attention" concatenates img+txt into one sequence rather than image-queries-attend-to-fixed-77-text-tokens. This is SD1.5's one genuinely new attention shape; built from existing `Softmax`/`Linear` primitives, not a new kernel family. |
| RoPE | 🟢 | not used by SD1.x/SDXL (learned pos + sinusoidal timestep only) | not used | n/a for SD3.5 MMDiT (uses learned pos) | Present (`Flux2DRoPE`, `ZImageRoPE`, `IComputeBackend.RoPE`) but not on SD's critical path |
| VAE | 🟡 | **extend** | extend (shares SD1.5's) | new (16-ch, different) | `VaeDecoder` is FLUX/Z-Image-shaped: 16-channel latent, fixed 512→512→256→128 channel progression, `post_quant_conv` optional. SD1.5/SDXL VAE is 4-channel with a different progression — same block *primitives* (GroupNorm+SiLU+Conv, ResBlock, attention, upsample), different *topology* → extend `VaeDecoder`, don't duplicate, per 033 §2.6. |
| CLIP-L (SD1.5, SDXL secondary) | 🟢 | **likely direct reuse** | reuse (as secondary encoder) | reuse | `ClipLEncoder` is exactly ViT-L/14, 768-dim, 12 layers, QuickGELU — the same CLIP-L SD1.5/SDXL/SD3.5 all use. Validate tokenizer/weight-key compatibility against a real SD1.5 checkpoint before assuming zero-work reuse. |
| CLIP-G (SDXL, SD3.5) | 🆕 | n/a | **required** | required | Nothing today implements the larger (2048-dim) OpenCLIP ViT-bigG encoder FLUX doesn't use. |
| T5-XXL | 🟢/🟡 | n/a | n/a | reuse | `T5Encoder` exists (used by FLUX) — SD3.5 needs the same model, verify config matches (T5-XXL is standard across FLUX/SD3.5/Z-Image-adjacent stacks). |
| GGUF / Safetensors loading | 🟢 | reuse | reuse | reuse | `GgufWeightLoader`, `SafetensorsLoader`, `IWeightLoader` — directory- and single-file-shard aware already (see `ZImagePipeline.Load`). Phase 2.1's "strict tensor validation, no best-effort mapping" is a policy to apply when writing the SD1.5 loader, not new infra. |
| UNet (ResBlock+SpatialTransformer+cross-attn+skip connections) | 🆕 | **the major new component** | extend SD1.5's | n/a | Nothing today — see cross-attention note above. This is genuinely the biggest single chunk of Phase 2. |
| MMDiT (dual-stream + single-stream transformer) | 🟢/🟡 | n/a | n/a | **strong reuse candidate** | `FluxDiT` already implements `DoubleBlock`/`SingleBlock` dual→single-stream transformer blocks — architecturally the same family as SD3's MMDiT. SD3.5 Phase 5 should mine `FluxDiT` directly rather than reading `mmdit.hpp` cold. |
| Scheduler (Euler flow-matching) | 🟢 | 🟡 different family | 🟡 different family | reuse | `EulerFlowScheduler` is flow-matching (FLUX/Z-Image). SD1.5/SDXL use the DDPM-derived samplers (DDIM, PNDM, Euler-discrete, DPM++) — different noise schedule math, not flow-matching. New scheduler(s) needed per 033 §2.4, though the `Denoise(...)` loop *shape* (callback-driven step loop + pack/unpack latent) is directly reusable as a pattern. |
| Classifier-free guidance | 🟢/🟡 | reuse pattern | reuse pattern | varies | FLUX (`guidance` param, distilled — no explicit uncond pass in the same way) and SD1.5 (explicit uncond+cond dual forward pass, classic CFG) differ in *mechanism*; SD1.5 needs the textbook two-pass CFG combine, worth its own small `IDiffusionSampler` implementation (interface already declared, see `IDiffusionPipeline.cs`). |
| PNG output | 🟢 | reuse | reuse | reuse | `PngWriter` — family-agnostic already |
| Upscaling (RRDBNet) | 🟢 | reuse | reuse | reuse | Already family-agnostic, takes raw RGB float[] |

---

## Phase 1 — Common diffusion abstraction (corrective slice, done)

Full writeup: `IMPLEMENTATION_NOTE.md` in this folder. Summary:

- `IDiffusionPipeline`, `IVaeDecoder`, `ITextConditioner`/`IConditioning`, `IDiffusionModel`,
  `IDiffusionScheduler`, `IDiffusionSampler` defined in
  `src/OpenTail.Stingray.Diffusion/IDiffusionPipeline.cs` (project root, shared across families).
- `VaeDecoder` implements `IVaeDecoder`; `ImagePipeline`/`ZImagePipeline` implement
  `IDiffusionPipeline` via adapters. Zero behavior change (Phase 1's acceptance criterion) —
  verified: full solution build green, existing pipelines' `Generate(...)` bodies untouched.
  Regression guard: `tests/OpenTail.Stingray.Tests.Diffusion/DiffusionPipelineInterfaceTests.cs`
  (4 tests, passing).
- `ITextConditioner`/`IDiffusionModel`/`IDiffusionScheduler`/`IDiffusionSampler` are declared
  but **not** retrofitted onto `ClipLEncoder`/`FluxDiT`/`EulerFlowScheduler` — deliberately
  deferred until SD1.5 gives a second real implementation to validate the interface shape
  against, rather than guessing from FLUX alone and risking a rewrite.
- Removed the earlier scaffold's mistakes: bespoke disconnected `SdTensor` type, premature
  SDXL stubs (built before SD1.5 existed, contradicting 033 §4's "extend, don't duplicate"),
  placeholder `WeightLoader`/`TensorAssignment` that duplicated real loaders already in the
  project. `OpenTail.Stingray.Tests.Diffusion` is now registered in `OpenTail.Stingray.slnx`
  (was silently skipped by root `dotnet test` before).

---

## Resuming — do this first

Next unit of work is **033 Phase 2.1 + 2.2**: SD1.5 model loading + CLIP text encoder. In order:

1. Get one real SD1.5 checkpoint (safetensors) locally for validation — e.g.
   `runwayml/stable-diffusion-v1-5` or a well-known Q-quant GGUF if one exists for SD1.5 in the
   wild. Needed before any tensor-name mapping can be verified, not just designed on paper.
2. Confirm the CLIP-L reuse hypothesis from the operator matrix above: diff SD1.5's
   `cond_stage_model.transformer.*` tensor names (or equivalent) against what `ClipLEncoder`
   already expects from `clip_l.safetensors`. If the weight keys/shapes line up, Phase 2.2 may
   be near-zero new code (a loader shim, not a new encoder). If they don't, scope a proper
   `IWeightLoader`-based key remapper before touching UNet work.
3. Write the SD1.5 tensor-mapping loader per §2.1 (strict validation — fail loud on
   unrecognized/mismatched tensors, no best-effort guessing).
4. Only then start the UNet (§2.3) — cross-attention is the one genuinely new op (see matrix
   above); everything else it needs already exists in `DiffusionOps`.
5. Re-verify against `examples/stable-diffusion.cpp` at the frozen commit above (or re-freeze
   deliberately, recording the new SHA here, if resuming after the upstream repo has moved).

Do not re-attempt an SDXL or SD3.5 stub before step 4 is real and passing conformance
(033 Phase 3) — that ordering mistake is exactly what this corrective slice undid.
