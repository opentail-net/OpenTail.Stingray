# Plan — Native Qwen Image Support for OpenTail.Stingray

**Reference:** `leejet/stable-diffusion.cpp`  
**Target:** `opentail-net/OpenTail.Stingray`  
**Execution:** **100% local/native C# — no cloud, Python, P/Invoke, or external inference process**

---

## Status

**PLANNED**

OpenTail.Stingray already contains substantial native diffusion infrastructure, including:

- Stable Diffusion 1.5 / SDXL
- Stable Diffusion 3 / 3.5
- FLUX
- Z-Image
- VAE encode/decode infrastructure
- Qwen3 text encoding infrastructure
- GGUF and SafeTensors loading
- LoRA support
- CPU/GPU execution infrastructure
- multiple diffusion schedulers

This plan adds **Qwen Image text-to-image generation** as the next major image-family implementation.

The implementation should reuse the existing Stingray transformer, Qwen, VAE, scheduler, model-loading and hardware infrastructure rather than creating a parallel runtime.

Qwen Image Edit is deliberately treated as a **follow-on capability**, not a prerequisite for the first Qwen Image milestone.

---

# 1. Objective

Add native C# support for the **Qwen Image** diffusion family.

The first target is:

> **Qwen Image text-to-image**

The desired end state is:

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
├── StableDiffusion3
│   └── SD3 / SD3.5
│
└── QwenImage
    └── Qwen Image
```

The architecture should leave room for:

```text
QwenImage
├── Qwen Image
├── Qwen Image Edit
├── Qwen Image Edit 2509
├── Qwen Image Edit 2511
└── Qwen Image Layered
```

without prematurely implementing those variants.

The reference implementation currently supports Qwen Image and the Qwen Image Edit series, with Qwen2.5-VL used as the conditioning model. citeturn1search5turn1search7

---

# 2. Architectural position

Qwen Image should **not** be implemented as another generic SD/UNet pipeline.

Treat it as a distinct transformer-diffusion family which reuses common Stingray primitives.

Conceptually:

```text
Prompt
  │
  ▼
Qwen2.5-VL
  │
  │ text conditioning
  ▼
Qwen Image conditioning
  │
  ▼
Qwen Image Transformer
  │
  │ denoising / flow matching
  ▼
Qwen Image latent
  │
  ▼
Qwen Image / Wan-family VAE
  │
  ▼
Image
```

The reference command uses:

- Qwen Image diffusion weights
- Qwen Image VAE
- Qwen2.5-VL 7B
- Euler sampling
- CFG scale around 2.5
- flow shift around 3

as the baseline configuration. citeturn1search7

---

# 3. Design principle

**Do not port `stable-diffusion.cpp` line-by-line.**

Use it as the reference/oracle:

```text
stable-diffusion.cpp
        │
        │ reference implementation
        ▼
architecture / tensor / numerical specification
        │
        ▼
OpenTail.Stingray native C#
        │
        ▼
golden-output conformance tests
```

The reference repository is a development-time dependency only.

The Stingray implementation remains:

- native C#
- managed .NET 10
- in-process
- NativeAOT compatible
- offline/local
- independent of Python
- independent of PyTorch
- independent of llama.cpp binaries
- independent of stable-diffusion.cpp binaries

---

# Phase 0 — Reference and repository freeze

## 0.1 Freeze the reference

Clone the reference locally:

```bash
git clone https://github.com/leejet/stable-diffusion.cpp.git
```

Record:

- commit SHA
- version/tag
- Qwen Image implementation
- Qwen Image Edit implementation
- Qwen Image VAE implementation
- Qwen2.5-VL conditioning implementation
- tensor-name conversion
- scheduler configuration
- flow-shift handling
- CFG implementation
- quantisation support
- CPU/GPU backend behaviour

The current reference contains a dedicated `qwen_image.hpp` implementation and identifies Qwen Image as a first-class diffusion model. citeturn1search2

## 0.2 Freeze the baseline model

Use a known Qwen Image checkpoint and record:

```text
Diffusion:
  format
  precision / quantisation
  tensor names
  model configuration

Conditioner:
  Qwen2.5-VL 7B
  format
  precision / quantisation

VAE:
  Qwen Image VAE
  format
  precision

Generation:
  width
  height
  seed
  steps
  CFG scale
  flow shift
  scheduler
```

The reference documentation provides both SafeTensors and GGUF model paths for Qwen Image, plus a Qwen2.5-VL 7B conditioner and Qwen Image VAE. citeturn1search7

## 0.3 Audit existing Stingray components

Before implementing anything new, inventory:

- existing transformer blocks
- attention
- QKV projection
- RoPE
- timestep embedding
- MLP / gated MLP
- normalization
- conditioning projection
- image latent packing/unpacking
- positional IDs
- Qwen tokenizer
- Qwen3/Qwen-family text encoder
- VAE
- scheduler
- flow-matching scheduler
- CFG
- GGUF loader
- SafeTensors loader
- quantised linear layers
- CPU kernels
- Vulkan kernels
- CUDA kernels
- LoRA hooks
- image tensor representation

Produce a Qwen-specific operator matrix:

| Operator / component | Existing Stingray | Qwen Image |
|---|---:|---:|
| MatMul | 🟢 | required |
| Linear | 🟢 | required |
| Attention | 🟢 | required |
| RoPE / positional encoding | 🟢 | required |
| LayerNorm / RMSNorm | 🟢 | verify exact variant |
| SiLU / activation | 🟢 | verify |
| Gated MLP | 🟢 | verify |
| Timestep embedding | 🟢 | required |
| Transformer block | 🟢 | extend |
| Qwen2.5-VL | 🟢/🟡 | required |
| VAE | 🟢 | verify latent layout |
| Flow scheduler | 🟢 | verify flow shift |
| CFG | 🟢 | required |
| GGUF | 🟢 | required |
| SafeTensors | 🟢 | required |

**Acceptance criterion:** identify every genuinely new primitive before writing the Qwen model.

---

# Phase 1 — Qwen Image architecture extraction

## 1.1 Establish the model contract

Create a dedicated model type:

```csharp
QwenImageModel
```

or equivalent family-specific abstraction.

It should own:

- model configuration
- transformer blocks
- input projection
- conditioning projection
- timestep embedding
- positional encoding
- attention
- MLP
- output projection
- weight loading
- tensor validation

Do not put Qwen-specific logic into `FluxDiT` merely because both are transformer diffusion models.

---

## 1.2 Determine exact architecture

Use the reference implementation and official model configuration to establish:

- number of transformer layers
- hidden dimension
- attention heads
- head dimension
- MLP dimension
- input channels
- output channels
- conditioning dimension
- timestep embedding dimension
- positional encoding scheme
- normalization
- activation
- bias usage
- image token layout
- text token layout
- latent packing
- output projection

The reference implementation currently reports **60 Qwen Image diffusion layers** for the standard family. citeturn1search3

Do not hard-code architecture dimensions until they have been validated against the checkpoint metadata.

---

# Phase 2 — Qwen Image transformer

## 2.1 Input projection

Implement the mapping from packed latent representation into transformer hidden space.

Validate:

```text
latent
  ↓
latent packing
  ↓
input projection
  ↓
hidden sequence
```

Compare intermediate tensors with the reference.

## 2.2 Timestep embedding

Implement the Qwen Image timestep path exactly.

Validate:

```text
timestep
   ↓
timestep projection
   ↓
embedding
   ↓
transformer conditioning
```

Pay particular attention to:

- timestep scaling
- flow shift interaction
- embedding frequency
- projection dimensions
- dtype

Qwen Image Layered is documented as adding an `addition_t_embedding` path to the otherwise similar transformer; this is evidence that timestep conditioning is a meaningful family-level extension point. Do not implement Layered now, but avoid designing the timestep path so that this extension becomes invasive. citeturn1search0

## 2.3 Positional encoding

Implement the Qwen Image positional representation exactly.

Validate separately for:

```text
text positions
image positions
packed latent positions
```

Do not assume FLUX positional IDs are interchangeable with Qwen Image merely because both are transformer diffusion models.

## 2.4 Attention

Reuse Stingray attention infrastructure where mathematically identical.

Validate:

- Q projection
- K projection
- V projection
- head reshaping
- positional transformation
- attention scaling
- attention mask
- output projection
- dtype

If Qwen Image requires an attention layout not already supported, add it as a reusable primitive rather than embedding it inside `QwenImageModel`.

## 2.5 Transformer blocks

Implement the complete Qwen Image transformer block.

Conceptually:

```text
hidden
  │
  ├── normalization
  │
  ├── attention
  │
  ├── residual
  │
  ├── normalization / modulation
  │
  ├── gated MLP
  │
  └── residual
```

The exact ordering must come from the reference implementation and checkpoint architecture, not from this conceptual diagram.

## 2.6 Output projection

Implement the final projection from transformer hidden states back into the diffusion latent representation.

Validate:

```text
transformer output
       ↓
output projection
       ↓
latent prediction
```

---

# Phase 3 — Qwen2.5-VL conditioning

This phase is deliberately separate from the diffusion transformer.

## 3.1 Reuse existing Qwen infrastructure

Stingray already has Qwen-family text encoding infrastructure.

Determine whether the existing implementation can load the required **Qwen2.5-VL 7B** text/vision model directly.

The reference Qwen Image pipeline uses Qwen2.5-VL 7B as its conditioner. citeturn1search7

Do not create another Qwen implementation if the existing model runtime can be extended cleanly.

## 3.2 Text-only baseline

The first Qwen Image milestone requires:

```text
prompt
 ↓
tokenizer
 ↓
Qwen2.5-VL text path
 ↓
conditioning embeddings
```

No image-conditioning input is required for the initial text-to-image implementation.

## 3.3 Validate conditioning

Capture:

- token IDs
- attention mask
- hidden states
- selected conditioning output
- pooled/auxiliary outputs where applicable

Compare against the reference.

## 3.4 Vision path

Do **not** make the Qwen2.5-VL vision encoder a prerequisite for text-to-image.

However, design the conditioning API so that:

```csharp
QwenImageConditioning
{
    Text
    ImageFeatures?
    ReferenceLatents?
}
```

can later support Qwen Image Edit.

---

# Phase 4 — Qwen Image VAE

## 4.1 Identify the exact VAE

Do not automatically assume the existing SD/FLUX VAE implementation is compatible.

The reference loads a dedicated Qwen Image VAE, and the current implementation treats the relevant VAE as a Wan-family VAE. citeturn1search3turn1search10

Audit:

- latent channels
- spatial compression
- temporal dimension
- scaling factor
- normalization
- decoder block structure
- output range
- tensor naming

## 4.2 Reuse or extend

Preferred order:

```text
existing Stingray VAE
      │
      ├── mathematically compatible → reuse
      │
      └── close but different → generalise
                              │
                              └── Qwen-specific implementation
```

Do not create a duplicate VAE if the Wan/Qwen VAE can be represented by a shared abstraction.

## 4.3 VAE conformance

Compare:

```text
latent
 ↓
VAE decode
 ↓
pixel tensor
```

against the reference independently of the diffusion model.

This isolates VAE errors from transformer errors.

---

# Phase 5 — Flow scheduler and sampling

## 5.1 Establish the baseline

Use the exact reference baseline:

```text
Scheduler:
  Euler

CFG:
  2.5

Flow shift:
  3

Steps:
  reference baseline
```

The reference Qwen Image documentation uses Euler with CFG 2.5 and flow shift 3. citeturn1search7

## 5.2 Verify existing scheduler

Determine whether Stingray's existing `EulerFlowScheduler` is mathematically equivalent.

Compare timestep/sigma sequences:

```text
reference
   │
   ├── timesteps
   ├── sigmas
   ├── flow shift
   └── update equations
        │
        ▼
Stingray
```

If there is any difference, fix the scheduler at the shared abstraction rather than adding `QwenEulerScheduler` unless the equations are genuinely different.

## 5.3 CFG

Implement the Qwen Image CFG path using the existing guidance infrastructure where possible.

Validate:

```text
positive conditioning
negative/unconditional conditioning
        │
        ▼
conditional prediction
unconditional prediction
        │
        ▼
CFG combination
```

The public API must make CFG scale explicit.

---

# Phase 6 — Qwen Image pipeline

Create:

```csharp
QwenImagePipeline
```

with the following conceptual structure:

```text
QwenImagePipeline
│
├── QwenImageModel
│
├── Qwen2_5VLConditioner
│
├── QwenImageVae
│
├── EulerFlowScheduler
│
└── sampler / guidance
```

Execution:

```text
Prompt
  ↓
Qwen2.5-VL
  ↓
Qwen conditioning
  ↓
Random latent
  ↓
Flow timestep schedule
  ↓
Qwen Image transformer
  ↓
CFG
  ↓
Denoising loop
  ↓
Qwen VAE
  ↓
Image
```

---

# Phase 7 — Model loading

## 7.1 SafeTensors

Support the native Qwen Image SafeTensors layout.

Implement strict tensor mapping.

Unknown or missing tensors should fail clearly.

## 7.2 GGUF

Support the Qwen Image GGUF format through Stingray's existing GGUF infrastructure.

Do not create a Qwen-specific GGUF loader.

Validate:

- F16/BF16
- Q8
- Q6
- Q5
- Q4
- other formats actually supported by Stingray's kernels

The reference supports Qwen Image GGUF models and uses quantised diffusion and conditioner weights in practical configurations. citeturn1search7turn1search4

## 7.3 Tensor-name mapping

Create:

```text
QwenImageWeightLoader
```

or equivalent mapping layer.

Responsibilities:

- checkpoint-name → Stingray tensor
- shape validation
- dtype validation
- optional transpose/reshape
- architecture metadata
- clear error reporting

Do not hide shape mismatches with permissive fallback mapping.

---

# Phase 8 — First end-to-end milestone

Provide:

```bash
stingray image \
    --model qwen-image \
    --prompt "A lighthouse on Mars" \
    --width 1024 \
    --height 1024 \
    --steps 20 \
    --guidance 2.5 \
    --seed 12345
```

The exact CLI syntax should follow the existing Stingray image command rather than introducing a separate command if the current command already supports model-family dispatch.

Acceptance:

- fully local
- single Stingray process
- no Python
- no subprocess
- no P/Invoke
- valid generated image
- deterministic seed
- repeatable output

---

# Phase 9 — Qwen Image golden conformance

This phase is mandatory.

Generate deterministic reference vectors from `stable-diffusion.cpp`.

Capture:

```text
1. token IDs
2. attention mask
3. Qwen2.5-VL conditioning
4. initial latent
5. timestep sequence
6. flow-shifted schedule
7. timestep embeddings
8. first transformer block output
9. selected attention outputs
10. transformer output
11. scheduler output
12. final latent
13. VAE output
14. final pixels
```

Compare Stingray against the reference at each stage.

## Test matrix

At minimum:

| Test | Baseline |
|---|---:|
| 512×512 | ✅ |
| 1024×1024 | ✅ |
| fixed seed | ✅ |
| multiple seeds | ✅ |
| CFG 1.0 | ✅ |
| CFG 2.5 | ✅ |
| negative prompt | ✅ |
| different step counts | ✅ |
| batch 1 | ✅ |
| deterministic CPU | ✅ |
| F16/BF16 | ✅ |
| GGUF | later |
| GPU | later |

Use numerical tolerances appropriate to precision/backend.

Do not demand pixel equality where backend/precision differences make that inappropriate.

---

# Phase 10 — Quantisation

Do not make low-bit inference the first acceptance criterion.

Order:

```text
F32
 ↓
F16 / BF16
 ↓
Q8
 ↓
Q6
 ↓
Q5
 ↓
Q4
```

For every supported quantisation:

- numerical validation
- generation-quality validation
- memory measurement
- speed measurement

Record:

```text
model size
peak RAM
peak VRAM
generation time
tokens / conditioning time
denoising time
VAE time
total time
```

---

# Phase 11 — Hardware execution

## CPU

First establish correctness on CPU.

Then optimise:

- linear layers
- attention
- normalization
- RoPE/positional operations
- MLP
- latent packing
- VAE

Target existing Stingray CPU infrastructure.

## Vulkan

Reuse the existing Stingray Vulkan backend.

Pay particular attention to Qwen Image's large model memory footprint.

The reference can require roughly 20 GB combined model memory for a Q5 diffusion model plus Qwen2.5-VL 7B conditioner, and has practical CPU offload behaviour for constrained GPUs. citeturn1search4

Therefore:

```text
Qwen Image
    │
    ├── diffusion model
    ├── Qwen2.5-VL
    └── VAE
```

must be compatible with Stingray's residency/offload architecture.

## CUDA

Reuse Stingray CUDA execution.

Do not introduce a Qwen-specific GPU runtime.

---

# Phase 12 — Memory and residency

Qwen Image is a useful stress test for Stingray's existing model residency work.

Measure independently:

```text
Qwen2.5-VL resident memory
Qwen Image transformer memory
VAE memory
conditioning activations
transformer activations
latent memory
peak generation memory
```

Support:

```text
GPU resident
GPU + CPU offload
CPU-only
```

where existing Stingray infrastructure permits it.

The runtime should be able to avoid keeping all three major components resident on a constrained GPU when unnecessary.

This should integrate with Stingray's existing model residency/admission system rather than inventing a diffusion-specific memory manager.

---

# Phase 13 — Public API integration

Extend the existing image-generation request rather than creating a Qwen-specific public API.

Conceptually:

```csharp
var image = await runtime.GenerateImageAsync(
    new ImageGenerationRequest
    {
        Model = "qwen-image",
        Prompt = "A lighthouse on Mars",
        Width = 1024,
        Height = 1024,
        Steps = 20,
        GuidanceScale = 2.5f,
        Seed = 12345
    });
```

Model-family dispatch:

```text
Model
 │
 ├── sd15
 ├── sdxl
 ├── sd35
 ├── flux
 ├── z-image
 └── qwen-image
```

The caller should not need to know the internal pipeline implementation.

---

# Phase 14 — Capability reporting

Expose Qwen Image capabilities explicitly.

Example:

```text
Qwen Image
  Text-to-image       ✓
  1024×1024           ✓
  F16/BF16            ✓
  GGUF                ✓
  SafeTensors         ✓
  CPU                 ✓
  Vulkan              ✓
  CUDA                ✓
  LoRA                ?
  Image editing       ✗
  ControlNet          ✗
```

Do not report unsupported capabilities merely because the underlying model family can theoretically support them.

---

# Phase 15 — LoRA integration

Stingray already has a generic diffusion LoRA mechanism.

Validate whether Qwen Image LoRA weight naming and injection points are compatible.

If compatible:

```text
existing DiffusionLoraApplier
          │
          └── QwenImageModel
```

If not:

```text
DiffusionLoraApplier
        │
        ├── SD
        ├── FLUX
        ├── Z-Image
        └── Qwen Image
```

generalise the adapter targeting rather than creating `QwenImageLoraApplier`.

LoRA is **not required for the first end-to-end milestone**.

---

# Phase 16 — Qwen Image Edit

This is a separate phase.

Do not block text-to-image support on it.

The current reference supports Qwen Image Edit and later 2509/2511 variants. citeturn1search8

## 16.1 Reference-image pipeline

Target:

```text
Input image
    │
    ├── Qwen2.5-VL vision encoder
    │
    └── VAE / reference latent path
              │
              ▼
        Qwen conditioning
              │
Prompt ───────┤
              ▼
       Qwen Image transformer
              │
              ▼
             VAE
              │
              ▼
           output
```

The reference uses both `--llm` and, for later edit variants, `--llm_vision` with a Qwen2.5-VL vision projection model. citeturn1search8

## 16.2 Do not contaminate the base pipeline

Represent reference-image conditioning as an optional capability:

```csharp
ImageConditioning?
```

rather than making every `QwenImagePipeline` invocation image-aware.

---

# Phase 17 — Qwen Image Edit 2509 / 2511

Only after base Qwen Image Edit works.

The reference documents additional requirements for later edit variants, including:

- Qwen2.5-VL vision projection
- model-specific reference-image handling
- `qwen_image_zero_cond_t` for Qwen Image Edit 2511

The 2511 mode specifically requires the zero-condition timestep option for correct quality according to the reference documentation. citeturn1search8

Implement these as model capabilities/configuration, not arbitrary command-line flags buried in the pipeline.

---

# Phase 18 — Qwen Image Layered

**Optional / later.**

Do not implement during the initial Qwen Image port.

The reference indicates that Qwen Image Layered is architecturally close to Qwen Image, with an additional timestep-conditioning embedding and a VAE/layout difference involving transparency/layers. citeturn1search0

This should therefore be evaluated after the base Qwen Image implementation is stable.

Potential abstraction:

```text
QwenImageModel
      │
      ├── Qwen Image
      ├── Qwen Image Edit
      └── Qwen Image Layered
```

---

# Phase 19 — Performance

Only after numerical correctness.

Benchmark against `stable-diffusion.cpp` on the same machine and model.

Measure:

```text
Model load
Qwen2.5-VL conditioning
Initial latent
Each denoising step
VAE decode
Total generation
Peak RAM
Peak VRAM
```

Benchmark at:

```text
512×512
1024×1024
```

and with:

```text
F16/BF16
Q8
Q6
Q5
Q4
```

where supported.

The objective is not initially to beat the reference.

The first goal is:

> **Equivalent output and predictable performance.**

Optimisation follows.

---

# Phase 20 — NativeAOT and packaging

Every new component must remain:

- NativeAOT compatible
- reflection-light
- dependency-light
- Windows/Linux compatible
- fully offline

No:

```text
Python
PyTorch
ONNX Runtime
P/Invoke
stable-diffusion.cpp DLL
external inference server
```

The Qwen2.5-VL implementation should reuse Stingray's existing native Qwen runtime rather than embedding a Python/Transformers implementation.

---

# Phase 21 — Test matrix

Minimum final matrix:

| Test | Qwen Image |
|---|---:|
| Model loading | ✅ |
| Tensor mapping | ✅ |
| Qwen2.5-VL loading | ✅ |
| Tokenization | ✅ |
| Text conditioning | ✅ |
| Initial latent | ✅ |
| Timestep schedule | ✅ |
| Flow shift | ✅ |
| Transformer | ✅ |
| CFG | ✅ |
| Euler sampling | ✅ |
| VAE | ✅ |
| Deterministic seed | ✅ |
| Golden intermediates | ✅ |
| Golden image | ✅ |
| CPU | ✅ |
| Vulkan | later |
| CUDA | later |
| GGUF | later |
| Quantisation | later |
| LoRA | later |
| Image Edit | later |
| Layered | optional |

---

# Phase 22 — Performance acceptance

Do not declare success merely because an image was generated.

For each supported configuration record:

- model size
- precision
- quantisation
- peak RAM
- peak VRAM
- Qwen2.5-VL conditioning time
- denoising time
- VAE time
- total generation time
- images/hour
- CPU utilisation
- GPU utilisation
- model load time

Compare against `stable-diffusion.cpp` using:

- same model
- same seed
- same resolution
- same step count
- same CFG
- same flow shift
- same precision/quantisation
- same hardware/backend

---

# Implementation order

```text
0. Reference freeze
        ↓
1. Stingray Qwen/operator audit
        ↓
2. Confirm Qwen2.5-VL reuse
        ↓
3. Confirm VAE compatibility
        ↓
4. Qwen Image model configuration
        ↓
5. Weight-name mapping
        ↓
6. Input / latent packing
        ↓
7. Timestep embedding
        ↓
8. Positional encoding
        ↓
9. Attention
        ↓
10. Transformer blocks
        ↓
11. Output projection
        ↓
12. Qwen2.5-VL conditioning
        ↓
13. Qwen VAE integration
        ↓
14. Euler / flow-shift validation
        ↓
15. CFG
        ↓
16. Qwen Image end-to-end
        ↓
17. Golden intermediate conformance
        ↓
18. Golden image conformance
        ↓
19. GGUF / SafeTensors validation
        ↓
20. Quantisation
        ↓
21. CPU optimisation
        ↓
22. Vulkan / CUDA
        ↓
23. Memory / residency optimisation
        ↓
24. LoRA
        ↓
25. Qwen Image Edit
        ↓
26. Qwen Image Edit 2509/2511
        ↓
27. Qwen Image Layered
```

---

# Critical implementation rules

## Rule 1 — Reuse Stingray's existing Qwen implementation

Do not create a second general-purpose Qwen language model.

The diffusion project should consume the existing Qwen runtime wherever possible.

## Rule 2 — Do not make Qwen Image a FLUX subclass

FLUX and Qwen Image are both transformer diffusion architectures, but their conditioning and tensor contracts differ.

Reuse transformer **primitives**, not family-specific model classes.

## Rule 3 — Do not duplicate the VAE

First determine whether the existing VAE or a shared Wan/Qwen VAE implementation can represent the required Qwen Image latent layout.

## Rule 4 — Keep image editing separate

Text-to-image is the first milestone.

Reference-image/VLM conditioning is a later capability.

## Rule 5 — Validate intermediates before optimising

The debugging sequence should be:

```text
Weights
  ↓
Conditioning
  ↓
Latent
  ↓
Timestep
  ↓
Transformer block
  ↓
Transformer output
  ↓
Scheduler
  ↓
VAE
  ↓
Pixels
```

Do not debug the entire pipeline from final images alone.

## Rule 6 — Use the reference as an oracle

The reference implementation is useful precisely because it provides a working numerical specification.

Do not make its internal abstractions the architecture of Stingray.

---

# Expected reuse

The implementation should aim to reuse:

```text
Existing Stingray
        │
        ├── Tensor primitives
        ├── Transformer infrastructure
        ├── Attention
        ├── RoPE / positional encoding
        ├── Qwen runtime
        ├── GGUF loader
        ├── SafeTensors loader
        ├── VAE infrastructure
        ├── Flow scheduler
        ├── CFG
        ├── LoRA
        ├── CPU backend
        ├── Vulkan backend
        ├── CUDA backend
        └── model residency / memory infrastructure
```

New Qwen-specific code should ideally be limited to:

```text
QwenImage
├── model configuration
├── weight mapping
├── latent packing
├── Qwen-specific positional handling
├── transformer-specific conditioning
└── pipeline wiring
```

The exact amount must be established by Phase 0 rather than assumed.

---

# Success criteria

Qwen Image support is complete when Stingray can:

1. Load a known Qwen Image checkpoint.
2. Load the required Qwen2.5-VL conditioner.
3. Load the compatible Qwen Image VAE.
4. Generate deterministic images from text.
5. Match reference intermediate tensors within defined numerical tolerances.
6. Produce materially equivalent images to `stable-diffusion.cpp`.
7. Run fully inside the Stingray process.
8. Run without Python, PyTorch or external inference processes.
9. Use Stingray's existing memory/residency infrastructure.
10. Use the existing image-generation public API.
11. Report Qwen Image capabilities through Stingray's capability system.
12. Preserve NativeAOT compatibility.

---

# Definition of done

```text
Qwen Image
    │
    ├── Native C#                       ✅
    ├── In-process                      ✅
    ├── Qwen2.5-VL conditioning        ✅
    ├── Transformer inference          ✅
    ├── Flow matching                  ✅
    ├── Euler baseline                 ✅
    ├── CFG                            ✅
    ├── Qwen VAE                       ✅
    ├── GGUF                           ✅
    ├── SafeTensors                    ✅
    ├── deterministic generation       ✅
    ├── golden conformance             ✅
    ├── CPU                            ✅
    ├── GPU                            later/validated
    ├── quantisation                   later/validated
    ├── LoRA                           later/validated
    └── Image Edit                     follow-on
```

The strategic objective is **not** merely to make Qwen Image work.

It is to prove that Stingray's existing native diffusion architecture can absorb another major modern transformer-diffusion family with **small, reusable additions rather than another bespoke inference stack**.
