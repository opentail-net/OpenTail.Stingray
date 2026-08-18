# OpenTail.Stingray.Diffusion

Native, high-performance diffusion inference runtime for .NET 10 — 100% managed C#, running in-process with zero Python, zero subprocesses, zero P/Invoke, and full NativeAOT compatibility.

Supports the entire modern diffusion landscape: **Stable Diffusion 1.5, SDXL, Stable Diffusion 3 / 3.5, FLUX.1, and Z-Image-Turbo**, accelerated natively on CPU (AVX2/AVX-512), Vulkan, or CUDA.

[![.NET 10](https://img.shields.io/badge/.NET-10-blue)]()
[![NativeAOT](https://img.shields.io/badge/NativeAOT-ready-green)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> **Part of the [OpenTail.Stingray](https://github.com/opentail-net/OpenTail.Stingray) ecosystem by [opentail.net](https://opentail.net)**

---

## Supported Architectures & Models

| Model Family | Core Backbone | Text Conditioning | Schedulers | Latent Channels | VAE |
|---|---|---|---|:---:|:---:|
| **Stable Diffusion 1.5** | 4-Stage UNet + Spatial Cross-Attn | CLIP-L (768d, ViT-L/14) | Euler, Euler-A, DDIM, DPM++ 2M, DPM++ 2M Karras | 4 | 4-ch AutoencoderKL |
| **Stable Diffusion XL (SDXL)** | 3-Stage UNet + 2048d Cross-Attn | Dual: CLIP-L (768d) + OpenCLIP-bigG (1280d) + Pooled (1280d) + Micro-Coords (6×256d) | Euler, Euler-A, DDIM, DPM++ 2M, DPM++ 2M Karras | 4 | 4-ch AutoencoderKL |
| **Stable Diffusion 3 / 3.5** | MMDiT Dual-Stream + Single-Stream DiT | Triple: CLIP-L (768d) + OpenCLIP-bigG (1280d) + T5-XXL + Pooled (2048d) | Rectified Flow-Matching | 16 | 16-ch AutoencoderKL |
| **FLUX.1 (schnell / dev)** | MM-DiT Dual-Stream + Single-Stream DiT | Dual: CLIP-L + T5-XXL | Rectified Flow-Matching | 16 | 16-ch AutoencoderKL |
| **Z-Image-Turbo** | S3-DiT Scaled Transformer | Qwen3-4B LLM Text Encoder | Rectified Flow-Matching (4 steps) | 16 | 16-ch AutoencoderKL |

---

## Key Capabilities

* **100% Native C# Engine:** Every tensor operation, cross-attention layer, normalization kernel, scheduler step, and VAE decode runs directly inside the .NET 10 CLR / NativeAOT runtime.
* **Unified Abstractions (`IDiffusionPipeline`):** Consistent API across UNet, DiT, and MMDiT model families with standardized generation requests and progress reporting.
* **Universal Multi-Scheduler:**
  * **Euler & Euler Ancestral:** Discrete 1000-step linear beta schedule.
  * **DPM++ 2M & DPM++ 2M Karras:** 2nd-order Adams-Bashforth ODE solver with Karras $\sigma$-distribution ($\rho = 7.0$) for high quality in 10–15 steps.
  * **DDIM:** Deterministic inversion and sampling trajectory.
  * **Rectified Flow-Matching:** Euler flow trajectory for MMDiT and S3-DiT architectures.
* **LoRA Runtime Engine (`DiffusionLoraApplier`):** Load `.safetensors` LoRA weights and apply low-rank parameter deltas $\Delta W = \alpha \cdot \frac{1}{r}(A \times B)$ in-memory to base weights.
* **Universal VAE Subsystem:**
  * `VaeDecoder`: Decodes 4-channel and 16-channel latents to RGB pixels $[3, H, W]$.
  * `VaeEncoder`: Encodes input images to Gaussian latent distributions for **img2img** and inpainting workflows.
* **Integrated Super-Resolution (`RRDBNet`):** Built-in ×2 and ×4 Real-ESRGAN upscaling with bicubic blending support.

---

## Quick Start (C# API)

### 1. Generating with Stable Diffusion 1.5

```csharp
using OpenTail.Stingray.Diffusion.StableDiffusion;

// Load checkpoint (safetensors) and CLIP tokenizer
using var pipeline = StableDiffusionPipeline.Load("models/v1-5-pruned-emaonly.safetensors");

// Generate 512x512 image using 20 Euler steps with CFG = 7.5
pipeline.Generate(
    prompt: "A majestic lion standing on a cliff at sunset, cinematic lighting, 8k",
    negativePrompt: "blurry, low quality, distorted",
    width: 512,
    height: 512,
    steps: 20,
    guidance: 7.5f,
    seed: 42,
    schedulerType: DiffusionSchedulerType.DpmPlusPlus2MKarras,
    outputPath: "lion.png",
    progress: (step, total) => Console.WriteLine($"Step {step}/{total}"));
```

### 2. Generating with Stable Diffusion XL (SDXL)

```csharp
using OpenTail.Stingray.Diffusion.SDXL;

using var pipeline = SdxlPipeline.Load("models/sd_xl_turbo_1.0_fp16.safetensors");

pipeline.Generate(
    prompt: "An astronaut exploring a vibrant alien jungle, photorealistic, 4k",
    width: 1024,
    height: 1024,
    steps: 4,
    guidance: 1.0f,
    seed: 12345,
    outputPath: "sdxl_output.png");
```

### 3. Generating with Stable Diffusion 3 (SD3 MMDiT)

```csharp
using OpenTail.Stingray.Diffusion.SD3;

using var pipeline = Sd3Pipeline.Load("models/sd3_medium.safetensors");

pipeline.Generate(
    prompt: "A modern glass villa surrounded by autumn forest, hyper-detailed architectural render",
    width: 1024,
    height: 1024,
    steps: 20,
    guidance: 4.5f,
    outputPath: "sd3_villa.png");
```

### 4. Applying Runtime LoRA Adapters

```csharp
using OpenTail.Stingray.Diffusion;

// Load LoRA weights and merge into pipeline weights
var loraLayers = DiffusionLoraApplier.Load("models/lora/detail_enhancer.safetensors");
// Target linear/conv weights are updated in-place with scaled low-rank deltas
DiffusionLoraApplier.ApplyToWeights(modelWeights, loraLayers, multiplier: 0.8f);
```

### 5. Encoding Latents for Image-to-Image (img2img)

```csharp
using OpenTail.Stingray.Diffusion;

using var vae = new VaeEncoder("models/v1-5-pruned-emaonly.safetensors");

// Encode 512x512 RGB float array -> [4, 64, 64] latent representation
float[] latent = vae.Encode(inputRgb, height: 512, width: 512, latentChannels: 4);
```

---

## Command-Line Usage (CLI)

The `stingray image` CLI command automatically detects model family and architecture from the checkpoint file:

```bash
# Stable Diffusion 1.5
stingray image -m models/v1-5-pruned-emaonly.safetensors \
               -p "A medieval castle in the misty mountains" \
               -W 512 -H 512 --steps 20 --cfg-scale 7.5 -o castle.png

# Stable Diffusion XL (SDXL Turbo)
stingray image -m models/sd_xl_turbo_1.0_fp16.safetensors \
               -p "A futuristic hovercraft racing through Neo-Tokyo" \
               -W 1024 -H 1024 --steps 4 -o hovercraft.png

# Z-Image-Turbo (GGUF DiT + Qwen3-4B text encoder)
stingray image -m models/z_image_turbo-Q5_K_M.gguf \
               --qwen-encoder models/Z-Image-AbliteratedV1.Q5_K_M.gguf \
               --vae models/z-image-turbo/vae/ \
               -p "A cinematic photograph of a snow leopard" \
               -W 1024 -H 1024 -o leopard.png

# Enable Super-Resolution 4x upscaling
stingray image -m models/v1-5-pruned-emaonly.safetensors \
               -p "Macro shot of a dew drop on a leaf" \
               --upscaler models/RealESRGAN_x4plus.safetensors \
               -o macro_4k.png
```

---

## Testing & Conformance

Run the test suite using `dotnet test`:

```bash
dotnet test tests/OpenTail.Stingray.Tests.Diffusion/OpenTail.Stingray.Tests.Diffusion.csproj
```

The test project validates:
* **`DiffusionPipelineInterfaceTests`:** Common abstraction contracts across all pipeline implementations.
* **`SchedulerTests`:** Discrete $\sigma$-schedules, Karras distributions, CFG combinations, and DPM++ 2M ODE trajectory steps.
* **`SdxlConformanceTests`:** SDXL context concatenation (`[77, 2048]`), pooled text representations, and 2816d micro-conditioning.
* **`Sd3ConformanceTests`:** MMDiT triple-conditioning context (`[77, 4096]`), pooled projection vector $y$ (`[2048]`), and rectified flow-matching.
* **`DiffusionLoraTests`:** Accurate rank-1 / rank-$r$ outer product matrix expansion and in-place delta updates.
* **`VaeEncoderTests`:** Resolution constraints and dimension pyramid validation.
* **`Sd15PipelineTests`:** Full end-to-end image generation through UNet and VAE decoder.

