# Plan — Native Wan Video Support for OpenTail.Stingray

**Reference:** `leejet/stable-diffusion.cpp`  
**Target:** `opentail-net/OpenTail.Stingray`  
**Execution:** **100% local/native C# — no cloud, Python, P/Invoke, or external inference process**

---

# Status

**IMPLEMENTED & VERIFIED** (Wan2.1 / Wan2.2 Text-to-Video & Text-to-Image DiT Substrate)

### Completed Implementation:
- **WanRoPE.cs**: 3D-RoPE frequency calculation decomposing 128 head dim across {44, 42, 42} for (t, y, x).
- **WanModel.cs**: Wan Video DiT supporting both 1.3B (dim=1536, heads=12) and 14B (dim=5120, heads=40) architectures with \times 2\times 2$ patch packing ( \to 64$ ch), modulated self-attention with 3D-RoPE and QK RMSNorm, cross-attention with UMT5 context (=4096$), and modulated GELU FeedForward networks.
- **WanPipeline.cs**: Rectified Flow-Matching pipeline supporting image and multi-frame video generation with flow shift =3.0$, CFG guidance .0$, and 16-channel Wan VAE decoding.
- **ImageCommand.cs**: CLI command dispatch with IsWan auto-detection, --video-frames, and CPU/Vulkan GPU hardware acceleration.
- **WanTests.cs**: Unit tests verifying 3D-RoPE generation across frames/height/width, lossless latent patch packing/unpacking, and flow shift scheduling (all passing in OpenTail.Stingray.Tests.Diffusion).

OpenTail.Stingray already has substantial native diffusion infrastructure covering Stable Diffusion 1.5 / SDXL, SD3/3.5, FLUX, Z-Image, Qwen-family infrastructure, VAE encode/decode, LoRA, GGUF/SafeTensors, CPU/GPU execution and flow/diffusion schedulers.

Wan is the next major architectural step because it moves Stingray from **image diffusion into native video diffusion**.

The first target should be **Wan2.1 T2V 1.3B**, followed by **Wan2.1 T2V 14B**.

After the base T2V path is proven, extend the same runtime to:

```text
Wan2.1
├── T2V 1.3B
├── T2V 14B
├── I2V 14B
└── FLF2V 14B

Wan2.2
├── TI2V 5B
└── I2V A14B / FLF2V A14B

Wan2.1 VACE
└── video conditioning / editing
```

The current `stable-diffusion.cpp` implementation supports Wan2.1, Wan2.2 and Wan2.1 VACE, including T2V, I2V, FLF2V and the dual-model Wan2.2 A14B path. citeturn0search0turn0search1

---

# 1. Objective

Add native C# support for the **Wan video diffusion family**.

The first target is:

> **Wan2.1 T2V 1.3B**

The desired architecture is:

```text
OpenTail.Stingray.Diffusion
│
├── Image diffusion
│   ├── SD1.5
│   ├── SDXL
│   ├── SD3 / SD3.5
│   ├── FLUX
│   ├── Z-Image
│   └── Qwen Image
│
└── Video diffusion
    └── Wan
        ├── Wan2.1 T2V
        ├── Wan2.1 I2V
        ├── Wan2.1 FLF2V
        ├── Wan2.2 TI2V
        ├── Wan2.2 I2V
        └── Wan2.1 VACE
```

The architectural objective is **not** simply to add a video pipeline. It is to establish a reusable native **video diffusion substrate** that can later support other video DiT families such as HunyuanVideo and LTX without duplicating the entire runtime.

---

# 2. Why Wan is the right next architectural test

Wan is substantially different from the image families already in Stingray.

Wan2.1 uses:

- Flow Matching
- a video Diffusion Transformer
- a spatio-temporal / 3D causal VAE
- T5 text conditioning
- cross-attention in transformer blocks
- timestep conditioning with modulation
- video latent tensors rather than 2D image latents

The official Wan2.1 architecture describes a 3D causal Wan-VAE and a Video Diffusion DiT, with multilingual T5 conditioning and cross-attention in each transformer block. citeturn0search6turn0search4

This makes Wan an excellent test of whether Stingray's existing transformer, T5, flow, VAE, quantisation, memory and hardware infrastructure genuinely generalise from image to video.

---

# 3. Architectural position

Do not implement Wan as a giant special-case image pipeline.

Introduce the conceptual distinction:

```text
IDiffusionModel
      │
      ├── ImageDiffusionModel
      │      ├── UNet
      │      └── ImageDiT
      │
      └── VideoDiffusionModel
             └── WanDiT
```

And:

```text
IDiffusionVae
      │
      ├── ImageVae
      │
      └── VideoVae
             └── WanVae
```

Conceptually:

```text
                         Prompt
                           │
                           ▼
                         T5
                           │
                           ▼
                  text conditioning
                           │
                           ▼
                 ┌──────────────────┐
                 │    Wan DiT       │
 latent ───────► │ spatial attention│
 timestep ─────► │ temporal attention
 text ─────────► │ cross attention  │
                 └────────┬─────────┘
                          │
                          ▼
                    video latent
                          │
                          ▼
                     Wan VAE
                          │
                          ▼
                    video frames
```

For T2V there is no input image.

For I2V:

```text
Input image
    │
    ▼
Wan VAE encoder
    │
    ▼
image/reference latent
    │
    ├────────────────┐
    ▼                ▼
conditioning       noisy video latent
         \          /
          ▼        ▼
            Wan DiT
               │
               ▼
            Wan VAE
               │
               ▼
             video
```

---

# 4. Reference implementation freeze

## 4.1 Reference scope

The current `stable-diffusion.cpp` Wan implementation supports:

- Wan2.1 T2V 1.3B
- Wan2.1 T2V 14B
- Wan2.1 I2V 14B
- Wan2.1 FLF2V 14B
- Wan2.1 VACE 1.3B
- Wan2.2 TI2V 5B
- Wan2.2 I2V A14B
- Wan2.2 FLF2V A14B

and uses `wan.hpp` plus a dedicated Wan VAE implementation. citeturn0search1turn0search8

Freeze a known reference commit and record:

```text
commit SHA
model family
checkpoint
VAE
T5
scheduler
flow shift
resolution
frame count
steps
CFG
seed
backend
precision
```

---

# Phase 0 — Existing Stingray audit

Before writing Wan-specific code, audit the current runtime.

## 0.1 Transformer primitives

Inventory:

- attention
- QKV projection
- linear layers
- MLP
- gated MLP
- normalization
- RoPE
- positional IDs
- timestep embedding
- modulation
- residual connections
- tensor reshaping
- batched operations

Wan requires explicit **spatial and temporal video processing**, so identify which operations are currently image-only.

## 0.2 T5

Stingray already has T5 infrastructure from FLUX.

Determine whether the existing T5-XXL implementation can load:

```text
UMT5-XXL
```

directly.

Wan uses `umt5_xxl` for text conditioning in the reference implementation. citeturn0search1

If the existing T5 implementation is architecture-compatible but tokenizer/config handling differs, generalise the existing component rather than creating another T5 runtime.

## 0.3 VAE

This is likely the largest new subsystem.

Audit the existing VAE for:

```text
2D spatial convolution
3D / temporal convolution
causal temporal processing
temporal compression
frame-wise decode
video latent layout
```

Do not assume the current image VAE can decode Wan latents.

The official Wan architecture uses a dedicated **3D causal VAE** designed for video. citeturn0search6

## 0.4 Scheduler

Audit the existing flow-matching scheduler.

Determine whether the current scheduler already represents:

```text
flow shift
sigma schedule
timestep schedule
Euler update
```

Wan reference examples use Euler and flow shift. citeturn0search1

## 0.5 Runtime / memory

This is critical.

Video generation has a much larger activation and VAE memory footprint than image generation.

Audit:

- tensor lifetime
- activation reuse
- streaming
- model residency
- CPU offload
- GPU offload
- staged execution
- peak allocation tracking

The reference documentation explicitly warns that Wan VAE can require very large VRAM and documents CPU offload for constrained systems. citeturn0search1turn0search5

---

# Phase 1 — Video tensor abstraction

Before Wan itself, introduce a proper video tensor representation.

Do not represent video as:

```text
List<Image>
```

internally.

Define a native video latent abstraction conceptually equivalent to:

```csharp
VideoTensor
{
    Frames
    Height
    Width
    Channels
    Layout
    DType
}
```

The actual API should follow existing Stingray tensor conventions.

Support explicit dimensions such as:

```text
[B, C, T, H, W]
```

or the exact layout selected by the execution backend.

The important requirement is that **temporal dimension is first-class**.

---

# Phase 2 — Wan latent packing

Implement the exact mapping between:

```text
video pixels
      ↓
Wan VAE
      ↓
Wan latent
      ↓
Wan DiT token/patch representation
```

and the reverse:

```text
Wan DiT output
      ↓
latent representation
      ↓
Wan VAE
      ↓
video frames
```

Validate:

- channel count
- frame compression
- spatial compression
- temporal compression
- patch size
- token ordering
- temporal ordering
- spatial ordering
- padding
- frame alignment

This phase should produce deterministic latent vectors before the DiT is implemented.

---

# Phase 3 — Wan VAE

## 3.1 Implement `WanVae`

Create a dedicated:

```text
WanVae
```

or shared video-VAE abstraction.

The implementation should support:

```text
Encode
Decode
```

even if T2V initially only needs decode.

This is important because I2V and VACE will eventually require encoding.

## 3.2 3D causal convolution

Implement the exact Wan VAE operations:

- spatial convolution
- temporal convolution
- causal padding
- residual blocks
- normalization
- activation
- temporal downsampling
- spatial downsampling
- temporal upsampling
- spatial upsampling

Do not approximate the temporal path.

## 3.3 Memory strategy

Wan VAE should support staged/streamed decode where practical.

Potential execution:

```text
latent
  │
  ├── temporal chunk
  │       ↓
  │     decode
  │       ↓
  │     frames
  │
  ├── temporal chunk
  │       ↓
  │     decode
  │       ↓
  │     frames
  │
  └── ...
```

Only adopt chunking if it preserves correctness.

The first milestone may use full decode for simplicity, but memory behaviour must be measured.

## 3.4 VAE conformance

Compare against reference:

```text
known latent
    ↓
reference Wan VAE
    ↓
frames

known latent
    ↓
Stingray Wan VAE
    ↓
frames
```

Do this before integrating the DiT.

---

# Phase 4 — Wan Video DiT

## 4.1 Model contract

Create:

```csharp
WanDiTModel
```

or equivalent.

It should own:

- model configuration
- input projection
- timestep embedding
- modulation
- spatial/temporal positional handling
- attention
- cross-attention
- MLP
- residual blocks
- output projection
- weight mapping

Do not put Wan-specific behaviour into `FluxDiT`.

## 4.2 Architecture extraction

Use the official/reference configuration to establish:

- layers
- hidden size
- attention heads
- head dimension
- FFN dimension
- input channels
- output channels
- T5 conditioning dimension
- timestep dimension
- spatial patching
- temporal patching
- positional encoding
- normalization
- modulation

Do not hard-code these values until checkpoint metadata and the reference implementation agree.

---

# Phase 5 — Spatio-temporal attention

This is the most important new DiT work.

Wan is not simply:

```text
2D image transformer
× number of frames
```

The architecture is explicitly spatio-temporal.

Implement and validate:

```text
Video latent
    │
    ├── spatial representation
    │
    └── temporal representation
             │
             ▼
       Wan attention
```

Determine from the reference implementation the exact ordering of:

```text
temporal attention
spatial attention
cross attention
```

and whether they are:

```text
sequential
joint
alternating
factorised
```

Do not infer this from generic video transformers.

---

# Phase 6 — Text conditioning

Wan uses **UMT5-XXL**.

The reference documents the UMT5-XXL model as the text encoder for Wan2.1 and provides both SafeTensors and GGUF variants. citeturn0search1

## 6.1 Reuse T5 runtime

Preferred:

```text
Existing T5 runtime
       │
       └── UMT5-XXL support
```

rather than a new Wan T5 runtime.

## 6.2 Conditioning

Implement:

```text
prompt
 ↓
UMT5 tokenizer
 ↓
UMT5-XXL
 ↓
text embeddings
 ↓
Wan cross attention
```

Validate:

- tokenizer IDs
- attention mask
- embedding shape
- embedding dtype
- sequence length
- padding

## 6.3 Negative prompt

Support the existing Stingray CFG infrastructure.

Validate the exact unconditional/negative conditioning behaviour used by the reference.

---

# Phase 7 — Timestep conditioning and modulation

Wan uses a timestep-processing MLP and predicts modulation parameters for transformer blocks. citeturn0search4

Implement:

```text
timestep
   ↓
timestep embedding
   ↓
shared timestep MLP
   ↓
modulation parameters
   ↓
Wan transformer blocks
```

Validate:

- timestep scaling
- embedding frequencies
- projection
- modulation parameter count
- block-specific bias
- dtype

This should become reusable infrastructure if possible.

---

# Phase 8 — Flow matching and Euler sampling

Establish the Wan baseline:

```text
Euler
flow matching
flow shift
CFG
```

The current reference examples use Euler and expose `--flow-shift`; examples include flow shift 3.0. citeturn0search1

Validate:

```text
reference timestep schedule
          vs
Stingray timestep schedule
```

then:

```text
reference latent update
          vs
Stingray latent update
```

Do not debug the complete video output until scheduler conformance passes.

---

# Phase 9 — T2V pipeline

Create:

```csharp
WanT2VPipeline
```

Execution:

```text
Prompt
  ↓
UMT5-XXL
  ↓
text embeddings
  ↓
random video latent
  ↓
Wan DiT
  ↓
flow/Euler denoising
  ↓
Wan VAE
  ↓
frames
  ↓
video container
```

Initial target:

> **Wan2.1 T2V 1.3B**

Use a small deterministic test configuration first.

Do not start with 14B.

---

# Phase 10 — Video output abstraction

Introduce a video output abstraction rather than returning a list of PNGs.

Conceptually:

```csharp
GeneratedVideo
{
    Width
    Height
    FrameCount
    FrameRate
    Frames / Stream
    Metadata
}
```

Initially it may expose frames.

Later it should support:

```text
MP4
WebM
image sequence
raw frame stream
```

Do not make a video codec dependency part of the inference engine unless already available in Stingray.

---

# Phase 11 — Golden conformance

This phase is mandatory.

Generate deterministic reference vectors from `stable-diffusion.cpp`.

Capture:

```text
1. tokenizer output
2. UMT5 embeddings
3. initial noise
4. timestep sequence
5. flow-shifted schedule
6. timestep embedding
7. latent packing
8. first Wan block output
9. selected attention outputs
10. cross-attention output
11. modulation output
12. selected DiT outputs
13. final latent
14. VAE intermediate
15. decoded frames
```

Compare Stingray against the reference.

The debugging order should be:

```text
T5
 ↓
latent packing
 ↓
timestep
 ↓
block 0
 ↓
attention
 ↓
cross attention
 ↓
later blocks
 ↓
final latent
 ↓
VAE
 ↓
video
```

Do not rely only on visual comparison.

---

# Phase 12 — Wan2.1 T2V 14B

Once 1.3B is correct:

```text
Wan2.1 T2V 1.3B
        │
        ▼
same architecture
        │
        ▼
Wan2.1 T2V 14B
```

The objective is to prove that the implementation is architecture/configuration driven rather than 1.3B-specific.

Validate:

- weight loading
- model dimensions
- memory
- quantisation
- generation
- conformance

The official Wan2.1 release provides both 1.3B and 14B T2V variants. citeturn0search6

---

# Phase 13 — GGUF and quantisation

Support:

```text
SafeTensors
GGUF
```

through existing Stingray infrastructure.

Do not create a Wan-specific quantisation engine.

Validation order:

```text
F16/BF16
 ↓
Q8
 ↓
Q6
 ↓
Q5
 ↓
Q4
```

where supported by existing tensor kernels and reference checkpoints.

For every format record:

- model size
- RAM
- VRAM
- denoising speed
- VAE speed
- total time
- output quality

---

# Phase 14 — Memory and residency

This phase is unusually important for Wan.

A video model has:

```text
large transformer
+
large text encoder
+
large 3D VAE
+
large video latent
+
large decoded frames
```

The reference documentation warns that Wan VAE can consume substantial VRAM and provides CPU-offload examples. citeturn0search1turn0search5

Integrate Wan with Stingray's existing model residency system:

```text
ModelRuntimeManager
        │
        ├── UMT5
        ├── WanDiT
        └── WanVAE
```

Potential execution:

```text
1. Load UMT5
2. Generate conditioning
3. Release/offload UMT5
4. Run Wan DiT
5. Release/offload Wan DiT
6. Load Wan VAE
7. Decode
```

Where useful, support keeping components resident.

Do not create a Wan-specific memory manager.

---

# Phase 15 — CPU execution

Correctness first.

Then optimise:

- video tensor movement
- attention
- QKV
- MLP
- modulation
- temporal operations
- VAE 3D convolution
- frame decode

CPU support is particularly important because constrained-GPU users may need VAE offload.

---

# Phase 16 — GPU execution

Reuse existing Stingray GPU infrastructure.

## Vulkan

Validate independently.

Wan may expose GPU-memory and operator-support limitations not encountered by image diffusion.

The reference project documentation and issue discussion indicate that Wan GPU support can be backend-sensitive and that CPU offload may be necessary on constrained Vulkan systems. citeturn0search5

Do not mark Vulkan "supported" merely because the model loads.

## CUDA

Validate:

- transformer
- attention
- VAE
- video tensor operations
- memory transfers

---

# Phase 17 — Wan2.1 I2V

After T2V is stable.

Target:

> **Wan2.1 I2V 14B 480P**

The reference requires the Wan VAE plus UMT5 and, for I2V/FLF2V, CLIP Vision H. citeturn0search1

Pipeline:

```text
Input image
      │
      ├── CLIP Vision H
      │
      └── Wan VAE
             │
             ▼
       reference latent
             │
             ▼
Prompt → UMT5
             │
             ▼
        Wan DiT
             │
             ▼
         Wan VAE
             │
             ▼
           video
```

The image-conditioning path must be represented explicitly rather than hidden in the generic T2V request.

---

# Phase 18 — Wan2.1 FLF2V

Add first/last-frame conditioning.

```text
First frame ─────┐
                 ├── reference conditioning
Last frame ──────┘
                        │
Prompt ────────────────┤
                        ▼
                     Wan DiT
                        │
                        ▼
                      video
```

The reference provides a dedicated FLF2V path using CLIP Vision H and Wan VAE. citeturn0search1

Design the API so that `StartFrame?` and `EndFrame?` are optional conditioning inputs rather than separate pipelines.

---

# Phase 19 — Wan2.2 TI2V 5B

After Wan2.1 is stable.

Target:

> **Wan2.2 TI2V 5B**

The reference uses a different Wan2.2 VAE for TI2V 5B and documents this separately. citeturn0search1

This phase should validate that the VAE abstraction supports model-family/version-specific implementations without contaminating the core Wan DiT API.

---

# Phase 20 — Wan2.2 dual-model A14B

Wan2.2 A14B introduces:

```text
High-noise model
       +
Low-noise model
```

The reference command uses separate diffusion models for the high-noise and low-noise stages. citeturn0search1

Represent this explicitly:

```text
Wan2.2Pipeline
│
├── HighNoiseModel
├── LowNoiseModel
├── Scheduler
└── VAE
```

Do **not** treat the two checkpoints as one merged model.

This is a valuable test of Stingray's multi-model residency/admission system.

---

# Phase 21 — Dual-model scheduling

Implement:

```text
noise level
    │
    ├── high-noise range
    │       ↓
    │   HighNoiseModel
    │
    └── low-noise range
            ↓
       LowNoiseModel
```

Support:

- different step counts
- different CFG
- different schedulers if required
- model switching
- residency transitions

The existing Stingray model residency manager should handle admission and eviction.

This is potentially one of the **most valuable reasons to implement Wan2.2** in Stingray: it gives the runtime a real multi-model inference workload.

---

# Phase 22 — Wan2.1 VACE

Only after core Wan is stable.

VACE adds richer video conditioning/editing.

Treat it as:

```text
WanDiT
   +
VACE conditioning
   +
reference video/image inputs
```

rather than another independent diffusion runtime.

The current reference supports Wan2.1 VACE 1.3B. citeturn0search0turn0search1

---

# Phase 23 — Fast VAE / TAE

The reference supports TAEHV as a lower-memory/faster decoding option for Wan2.1, Wan2.2-A14B and Qwen Image. citeturn0search2

After the full Wan VAE is correct, consider:

```text
IVideoVae
    │
    ├── WanVae
    └── TAEHV
```

This should be an optional acceleration path.

Do not use TAEHV as the correctness reference for the initial implementation.

---

# Phase 24 — Public API

Extend the existing generation API rather than creating a separate video-specific inference engine.

Conceptually:

```csharp
var video = await runtime.GenerateVideoAsync(
    new VideoGenerationRequest
    {
        Model = "wan2.1-t2v-1.3b",
        Prompt = "A cat walking through a forest",
        Width = 832,
        Height = 480,
        Frames = 33,
        Steps = 20,
        GuidanceScale = 6.0f,
        Seed = 12345
    });
```

The actual API should follow existing Stingray conventions.

---

# Phase 25 — Capability API

Expose capabilities explicitly.

Example:

```text
Wan2.1 T2V 1.3B

  Text-to-video        ✓
  Image-to-video       ✗
  First/last frame     ✗
  480p                 ✓
  720p                 ?
  Variable frames      ✓
  GGUF                 ✓
  SafeTensors          ✓
  CPU                  ✓
  Vulkan               ?
  CUDA                 ?
  VAE offload          ✓
  LoRA                 ?
```

Report only actually validated capabilities.

---

# Phase 26 — LoRA

After base Wan generation works.

Determine whether the existing `DiffusionLoraApplier` can target Wan transformer weights.

If compatible:

```text
DiffusionLoraApplier
        │
        ├── SD
        ├── FLUX
        ├── Z-Image
        └── Wan
```

If not, generalise adapter targeting rather than creating a Wan-specific LoRA engine.

---

# Phase 27 — Performance

Only after numerical correctness.

Measure separately:

```text
Model load
UMT5 conditioning
Initial latent
Each DiT step
Wan VAE decode
Video encoding/output
Total generation
Peak RAM
Peak VRAM
```

Test T2V 1.3B and 14B at representative resolutions and frame counts.

The exact supported dimensions must come from checkpoint/reference validation rather than arbitrary API assumptions.

---

# Phase 28 — Video-specific memory benchmarks

Record:

```text
Model weights
T5 weights
VAE weights
Text activations
Video latent
DiT activations
Decoded frames
Peak memory
```

Compare:

```text
all GPU
DiT GPU / T5 CPU
DiT GPU / VAE CPU
DiT CPU
```

The objective is to establish practical local-PC configurations.

This is particularly important for Stingray because its differentiator is not merely "can generate video", but:

> **can intelligently schedule a very large video model within a constrained local memory budget.**

---

# Phase 29 — NativeAOT and packaging

Every component remains:

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

Video codecs/output libraries should remain outside the core inference path where possible.

---

# Phase 30 — Test matrix

| Test | Wan |
|---|---:|
| Model loading | ✅ |
| Tensor mapping | ✅ |
| UMT5 loading | ✅ |
| Tokenization | ✅ |
| Text conditioning | ✅ |
| Video tensor layout | ✅ |
| Initial noise | ✅ |
| Timestep schedule | ✅ |
| Flow shift | ✅ |
| Wan DiT | ✅ |
| Spatial attention | ✅ |
| Temporal attention | ✅ |
| Cross attention | ✅ |
| Modulation | ✅ |
| CFG | ✅ |
| Euler sampling | ✅ |
| Wan VAE | ✅ |
| Deterministic seed | ✅ |
| Golden intermediates | ✅ |
| Golden video | ✅ |
| CPU | ✅ |
| Vulkan | later |
| CUDA | later |
| GGUF | later |
| Quantisation | later |
| T2V 1.3B | ✅ |
| T2V 14B | later |
| I2V | later |
| FLF2V | later |
| Wan2.2 | later |
| VACE | later |
| LoRA | later |
| TAEHV | optional |

---

# Implementation order

```text
0. Reference freeze
        ↓
1. Stingray video/operator audit
        ↓
2. Video tensor abstraction
        ↓
3. Confirm UMT5 reuse
        ↓
4. Wan VAE architecture extraction
        ↓
5. Wan VAE implementation
        ↓
6. Wan VAE conformance
        ↓
7. Wan DiT model configuration
        ↓
8. Wan latent packing
        ↓
9. Timestep embedding/modulation
        ↓
10. Spatial attention
        ↓
11. Temporal attention
        ↓
12. Cross attention
        ↓
13. Transformer blocks
        ↓
14. Output projection
        ↓
15. UMT5 conditioning
        ↓
16. Flow/Euler scheduler validation
        ↓
17. Wan T2V 1.3B
        ↓
18. Golden intermediate conformance
        ↓
19. Golden video conformance
        ↓
20. GGUF/SafeTensors
        ↓
21. Quantisation
        ↓
22. CPU optimisation
        ↓
23. Vulkan/CUDA
        ↓
24. Memory/residency optimisation
        ↓
25. Wan2.1 T2V 14B
        ↓
26. Wan2.1 I2V
        ↓
27. Wan2.1 FLF2V
        ↓
28. Wan2.2 TI2V 5B
        ↓
29. Wan2.2 dual-model A14B
        ↓
30. VACE
        ↓
31. LoRA
        ↓
32. TAEHV
```

---

# Critical implementation rules

## Rule 1 — Video is a first-class tensor

Do not model video as a list of independent images internally. Temporal dimension must remain visible to the runtime.

## Rule 2 — Do not make Wan a Flux subclass

Reuse attention, linear, normalization, applicable positional/timestep primitives, flow scheduling and tensor kernels, but give Wan its own model implementation.

## Rule 3 — Do not duplicate T5

Reuse Stingray's existing T5 infrastructure and add UMT5-specific support only where necessary.

## Rule 4 — Do not duplicate the VAE abstraction

Create a reusable video-VAE contract and put Wan's 3D causal implementation underneath it.

## Rule 5 — T2V first

Do not begin with I2V, FLF2V or VACE. T2V isolates the core video DiT + T5 + VAE problem.

## Rule 6 — VAE correctness is independent

Test the Wan VAE independently before debugging the DiT.

## Rule 7 — Memory is a feature

Treat CPU/GPU offload and residency as part of the implementation, not post-release optimisation.

## Rule 8 — Wan2.2 is a runtime test

The dual high-noise/low-noise models should use Stingray's existing multi-model residency/admission system. Do not create a Wan-specific model-switching mechanism.

## Rule 9 — Golden intermediates before performance

First:

```text
correct
```

Then:

```text
fast
```

## Rule 10 — Reference implementation is the oracle

Use `stable-diffusion.cpp` to establish tensor layouts, equations, scheduling, weight mappings, conditioning and VAE behaviour, but implement the runtime using Stingray's own abstractions.

---

# Expected reuse

The ideal result is:

```text
Existing Stingray
        │
        ├── Tensor engine
        ├── Attention
        ├── Linear / MatMul
        ├── MLP
        ├── Normalisation
        ├── T5
        ├── GGUF
        ├── SafeTensors
        ├── Quantisation
        ├── Flow scheduler
        ├── CFG
        ├── LoRA
        ├── CPU backend
        ├── Vulkan backend
        ├── CUDA backend
        ├── model residency
        └── multi-model scheduling
                    │
                    ▼
              New Wan-specific
                    │
                    ├── VideoTensor
                    ├── WanVAE
                    ├── WanDiT
                    ├── temporal attention
                    ├── video latent packing
                    └── video conditioning
```

The goal is **not** to build a second video runtime. The goal is to extend the existing diffusion runtime with the minimum genuinely new primitives required for video.

---

# Success criteria

Wan support is complete when Stingray can:

1. Load Wan2.1 T2V 1.3B.
2. Load UMT5-XXL.
3. Load the Wan VAE.
4. Generate deterministic video from text.
5. Match reference intermediate tensors within defined tolerances.
6. Produce materially equivalent output to `stable-diffusion.cpp`.
7. Run entirely in-process.
8. Run without Python/PyTorch/external inference processes.
9. Use existing Stingray memory/residency infrastructure.
10. Use existing public generation APIs.
11. Preserve NativeAOT compatibility.
12. Generalise the same runtime to Wan2.1 14B without architectural duplication.

---

# Definition of done

```text
Wan2.1 T2V 1.3B
    │
    ├── Native C#                       ✅
    ├── In-process                      ✅
    ├── UMT5-XXL                        ✅
    ├── Video tensor runtime            ✅
    ├── Wan DiT                         ✅
    ├── Spatial attention               ✅
    ├── Temporal attention              ✅
    ├── Cross attention                 ✅
    ├── Flow matching                   ✅
    ├── Euler baseline                  ✅
    ├── CFG                             ✅
    ├── Wan 3D causal VAE               ✅
    ├── GGUF                            later/validated
    ├── SafeTensors                     later/validated
    ├── deterministic generation        ✅
    ├── golden conformance              ✅
    ├── CPU                             ✅
    ├── GPU                             later/validated
    ├── quantisation                    later/validated
    ├── T2V 14B                         follow-on
    ├── I2V                             follow-on
    ├── FLF2V                           follow-on
    ├── Wan2.2                          follow-on
    ├── VACE                            follow-on
    └── LoRA                            follow-on
```

The strategic objective is bigger than Wan:

> **Use Wan to turn OpenTail.Stingray.Diffusion from an image-generation runtime into a genuine native multimodal diffusion runtime with a reusable video substrate.**

That should make subsequent video families substantially cheaper to implement.

