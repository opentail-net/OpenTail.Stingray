# Plan — Native Stable Diffusion Family for OpenTail.Stingray

**Reference:** `leejet/stable-diffusion.cpp`  
**Target:** `opentail-net/OpenTail.Stingray`  
**Execution:** **100% local/native C# — no cloud, Python, P/Invoke, or external inference process**

## 1. Objective

Add native C# support for the three major Stable Diffusion generations:

1. **Stable Diffusion 1.x / 1.5**
2. **Stable Diffusion XL**
3. **Stable Diffusion 3 / 3.5**

Reuse Stingray's existing tensor, model-loading, CPU/GPU, VAE, tokenizer and diffusion infrastructure wherever possible.

The end state should be:

```text
OpenTail.Stingray.Diffusion
│
├── FLUX
├── Z-Image
│
├── StableDiffusion
│   ├── SD 1.x / 1.5
│   └── SDXL
│
└── StableDiffusion3
    └── SD3 / SD3.5
```

**Do not copy the stable-diffusion.cpp architecture into C#.** Use it as the reference/oracle:

```text
stable-diffusion.cpp
        │
        │ reference implementation
        ▼
architecture / tensor / numerical specification
        │
        ▼
Stingray native C# implementation
        │
        ▼
golden-output conformance tests
```

---

# Phase 0 — Repository and reference freeze

Before writing implementation code:

## 0.1 Clone the reference locally

```bash
git clone https://github.com/leejet/stable-diffusion.cpp.git
```

Record:

- commit SHA
- version/tag
- supported SD architectures
- model formats
- scheduler implementations
- sampler implementations
- VAE implementation
- text encoders
- UNet implementation
- MMDiT implementation
- LoRA/ControlNet handling

**Do not depend on the repository at runtime.** It is purely a development/reference dependency.

## 0.2 Audit the current Stingray diffusion subsystem

Inventory:

- tensor primitives
- convolution
- normalization
- activation functions
- attention
- cross-attention
- linear layers
- embeddings
- VAE
- tokenizer
- scheduler
- sampler
- image tensor representation
- image output
- GGUF loader
- SafeTensors loader
- quantized operators
- CPU kernels
- Vulkan kernels
- CUDA kernels

Produce an operator matrix:

| Operator | Stingray | SD1.5 | SDXL | SD3.5 |
|---|---|---|---|---|
| MatMul | 🟢 | | | |
| Conv2D | ? | | | |
| GroupNorm | ? | | | |
| LayerNorm | | | | |
| SiLU | | | | |
| Cross-attention | 🟢/🟡 | | | |
| Self-attention | 🟢 | | | |
| RoPE | 🟢 | | | |
| VAE | 🟢/🟡 | | | |
| CLIP | | | | |
| T5 | 🟢/🟡 | | | |
| UNet | | 🆕 | 🆕 | |
| MMDiT | 🟢/🟡 | | | |

This phase determines the **real** implementation size.

---

# Phase 1 — Common diffusion abstraction

Before implementing SD1.5, extract the reusable pieces from the existing FLUX/Z-Image pipelines.

Introduce/standardise concepts such as:

```csharp
IDiffusionPipeline
IDiffusionModel
ITextConditioner
IVaeDecoder
IDiffusionScheduler
IDiffusionSampler
IConditioning
```

with a common execution flow:

```text
Prompt
 ↓
Tokenizer
 ↓
Text Encoder(s)
 ↓
Conditioning
 ↓
Noise / latent
 ↓
Scheduler
 ↓
Denoising loop
 ↓
VAE
 ↓
Image
```

Adapt existing FLUX and Z-Image pipelines to these abstractions rather than duplicating them.

**Acceptance criterion:** existing FLUX and Z-Image output remains numerically/conceptually unchanged.

---

# Phase 2 — SD 1.5

This is the first implementation target.

## 2.1 Model loading

Support the actual SD1.x model structures required by Stingray:

- SafeTensors
- GGUF where supported by the reference implementation
- configuration/metadata
- checkpoint tensor mapping
- strict tensor validation

No "best effort" tensor mapping. Unknown or mismatched models should fail explicitly.

## 2.2 CLIP text encoder

Implement/reuse the CLIP text encoder required by SD1.5.

Validate against reference intermediate tensors:

```text
tokens
 ↓
embeddings
 ↓
transformer layers
 ↓
final conditioning
```

## 2.3 SD UNet

Implement the SD1.x UNet architecture:

```text
Input latent
   │
   ├── timestep embedding
   │
   ├── input blocks
   │     ├── ResNet
   │     ├── attention
   │     └── cross-attention
   │
   ├── middle block
   │
   └── output blocks
         ├── skip connections
         ├── ResNet
         └── attention
```

This is the **major new architectural component**. Do not optimise it initially. First make it numerically correct.

## 2.4 Scheduler/sampler

Implement the minimum SD1.5-compatible set required for baseline generation.

Start with the scheduler used by the reference test configuration.

Then add:

- Euler
- Euler ancestral
- DDIM
- DPM variants

only after the baseline works.

## 2.5 Classifier-free guidance

Implement classifier-free guidance and make guidance scale explicit in the public API.

## 2.6 VAE

Reuse the existing VAE implementation where mathematically compatible. Otherwise extend it rather than introducing another VAE implementation.

## 2.7 First milestone

Provide a fully local command such as:

```bash
stingray image \
    --model sd15.gguf \
    --prompt "A lighthouse on Mars"
```

It must produce a valid image **entirely inside the Stingray process**.

No Python. No subprocess. No P/Invoke.

---

# Phase 3 — SD1.5 conformance

This phase is mandatory.

Generate deterministic test vectors from `stable-diffusion.cpp`.

Compare:

```text
token IDs
text embeddings
initial latent
timestep embeddings
UNet outputs
scheduler outputs
final latent
VAE output
final pixels
```

Use tolerances appropriate to the precision/backend.

Tests should cover:

- 512×512
- different seeds
- CFG
- negative prompts
- multiple inference steps
- batch size 1
- deterministic generation

Verify that Stingray and the reference produce materially equivalent images. Do not require pixel equality where floating-point/backend differences make that inappropriate.

---

# Phase 4 — SDXL

Once SD1.5 is stable, extend the same architecture.

Add:

- SDXL dual text encoders
- pooled text embeddings
- additional conditioning
- SDXL timestep/additional embeddings
- larger UNet
- SDXL VAE behaviour
- SDXL model metadata/tensor mapping

Do **not** duplicate SD1.5's UNet infrastructure. Extend it.

---

# Phase 5 — SD3 / SD3.5

Treat this as a separate architectural layer.

SD3.5 uses the modern diffusion-transformer/MMDiT approach. Reuse concepts from the existing FLUX/Z-Image implementations where appropriate.

Implement:

- SD3 text-conditioning pipeline
- CLIP-L
- CLIP-G
- T5-XXL
- MMDiT
- SD3 conditioning
- SD3 timestep handling
- SD3 VAE
- SD3/3.5 scheduler
- model-specific tensor mapping

Target architecture:

```text
                     ┌── CLIP-L
                     │
Prompt ──────────────┼── CLIP-G
                     │
                     └── T5-XXL
                           │
                           ▼
                    Multi-modal conditioning
                           │
                           ▼
                         MMDiT
                           │
                           ▼
                         VAE
                           │
                           ▼
                         Image
```

Mine the existing FLUX/Z-Image transformer infrastructure for reusable components rather than copying it.

---

# Phase 6 — Quantisation

Do **not** make quantisation the first acceptance criterion.

First establish F32/F16/BF16 correctness. Then add:

- GGUF
- Q4 variants
- Q5
- Q6
- Q8
- applicable low-bit formats

Only expose a quantisation profile once it has numerical validation, image-quality validation, and benchmark evidence.

---

# Phase 7 — LoRA

Once baseline inference works, implement a generic adapter mechanism that can eventually be shared between:

- SD1.5
- SDXL
- SD3.5
- FLUX
- Z-Image

Do **not** build SD-specific LoRA machinery if the underlying operation can be generalized.

---

# Phase 8 — ControlNet / image conditioning

After text-to-image is stable, add:

- ControlNet
- img2img
- inpainting
- image conditioning

These should become additional pipeline capabilities rather than separate engines.

---

# Phase 9 — Performance

Only after correctness.

## CPU

Optimise:

- Conv2D
- GEMM
- attention
- normalization
- tensor transforms
- VAE
- UNet blocks

for AVX2 and AVX-512 where available.

## GPU

Reuse Stingray's existing Vulkan/CUDA execution infrastructure. Do not create a separate diffusion GPU runtime.

---

# Phase 10 — Memory engineering

Explicitly measure:

- model resident memory
- intermediate activation memory
- latent memory
- VAE memory
- text encoder memory
- peak UNet memory
- peak SDXL memory
- peak SD3.5 memory

Add appropriate lifetime/reuse mechanisms.

The objective should be:

> Never keep a tensor alive merely because the implementation happened to allocate it there.

---

# Phase 11 — Public API

Expose a clean API such as:

```csharp
var image = await runtime.GenerateImageAsync(
    new ImageGenerationRequest
    {
        Model = "sd15",
        Prompt = "A lighthouse on Mars",
        Width = 512,
        Height = 512,
        Steps = 20,
        GuidanceScale = 7.5f,
        Seed = 12345
    });
```

Eventually expose:

```csharp
ModelFamily.StableDiffusion15
ModelFamily.StableDiffusionXL
ModelFamily.StableDiffusion35
ModelFamily.Flux
ModelFamily.ZImage
```

The caller should not need to know which internal pipeline is being used.

---

# Phase 12 — Capability reporting

Extend Stingray's capability system and report capabilities **per model/profile**, not merely "SD supported".

Example:

```text
SD1.5
  F16        ✓
  GGUF       ✓
  CPU        ✓
  Vulkan     ✓
  LoRA       ✓
  ControlNet ✗
```

---

# Phase 13 — NativeAOT

Every new component must remain:

- NativeAOT compatible
- reflection-light
- dependency-light
- deterministic
- Windows/Linux compatible

No:

```text
Python
PyTorch
ONNX Runtime
P/Invoke
stable-diffusion.cpp DLL
external inference server
```

The reference repository is **development-time only**.

---

# Phase 14 — Test matrix

Minimum final matrix:

| Test | SD1.5 | SDXL | SD3.5 |
|---|---:|---:|---:|
| Model loading | ✅ | ✅ | ✅ |
| Tensor mapping | ✅ | ✅ | ✅ |
| Tokenization | ✅ | ✅ | ✅ |
| Text conditioning | ✅ | ✅ | ✅ |
| Denoising | ✅ | ✅ | ✅ |
| Scheduler | ✅ | ✅ | ✅ |
| CFG | ✅ | ✅ | varies |
| VAE | ✅ | ✅ | ✅ |
| Deterministic seed | ✅ | ✅ | ✅ |
| Golden intermediates | ✅ | ✅ | ✅ |
| Golden image | ✅ | ✅ | ✅ |
| CPU | ✅ | ✅ | ✅ |
| Vulkan | later | later | later |
| CUDA | later | later | later |
| GGUF | later | later | later |
| LoRA | later | later | later |
| ControlNet | later | later | later |

---

# Phase 15 — Performance acceptance

Do not declare success merely because an image was generated.

For each family record:

- model size
- precision
- peak RAM
- text encoding time
- denoising time
- VAE time
- total generation time
- images/hour
- CPU utilisation
- GPU utilisation where applicable

Compare against `stable-diffusion.cpp` on **the same machine and same model/settings**.

---

# Implementation order

```text
0. Reference freeze
        ↓
1. Stingray diffusion operator audit
        ↓
2. Common diffusion abstractions
        ↓
3. SD1.5 model loader
        ↓
4. CLIP
        ↓
5. UNet
        ↓
6. Scheduler + CFG
        ↓
7. VAE integration
        ↓
8. SD1.5 end-to-end
        ↓
9. SD1.5 golden conformance
        ↓
10. SDXL
        ↓
11. SDXL conformance
        ↓
12. SD3.5 conditioning
        ↓
13. MMDiT
        ↓
14. SD3.5 end-to-end
        ↓
15. SD3.5 conformance
        ↓
16. Quantisation
        ↓
17. LoRA
        ↓
18. ControlNet / img2img / inpainting
        ↓
19. CPU optimisation
        ↓
20. Vulkan/CUDA optimisation
```

# Critical implementation rule

**Do not port stable-diffusion.cpp line-by-line.**

Port **behaviour and architecture**, not its implementation.

`stable-diffusion.cpp` is the **reference implementation**, not a runtime dependency.

The Stingray implementation must remain:

- native C#
- managed .NET 10
- in-process
- NativeAOT compatible
- independent of Python
- independent of PyTorch
- independent of llama.cpp
- independent of stable-diffusion.cpp binaries
- capable of running fully offline/local

The desired result is not merely "Stable Diffusion works in Stingray".

It is a **unified native C# diffusion runtime** in which SD1.5, SDXL, SD3.5, FLUX and Z-Image share Stingray's common tensor, model, memory, scheduling and hardware infrastructure.
