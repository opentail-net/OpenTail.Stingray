# OpenTail Stingray — Diffusion Generation Sample

A clean, standalone C# console sample demonstrating high-performance text-to-image diffusion generation using the [OpenTail Stingray](https://github.com/opentail-net/OpenTail.Stingray) public APIs.

This sample demonstrates:
* Loading unified diffusion checkpoints (such as Stable Diffusion 1.5 in `.safetensors` or `.gguf` format) via `StableDiffusionPipeline.Load(...)`.
* Using the high-level `IDiffusionPipeline` interface and `ImageGenerationRequest`.
* Enabling automatic hardware acceleration (Vulkan GPU compute with graceful fallback to managed SIMD CPU kernels) via `DiffusionBackendResolver`.
* Real-time denoising step progress callbacks and latency reporting.
* Saving synthesized images to disk as PNG files.

---

## Prerequisites

1. **.NET 10.0 SDK** or later installed.
2. **Model Weights**: A standard Stable Diffusion 1.5 checkpoint (e.g. `v1-5-pruned-emaonly.safetensors`) and the CLIP BPE tokenizer (`clip_tokenizer.json`).

### Downloading Assets

Download model weights into the local `models/` directory or your user cache (`%LOCALAPPDATA%\stingray\models` on Windows, `~/.cache/stingray/models` on Linux/macOS):

```bash
# 1. Download SD 1.5 safetensors checkpoint (~4.27 GB)
curl -L -o models/v1-5-pruned-emaonly.safetensors \
  https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors

# 2. Download CLIP tokenizer (~1.7 MB)
curl -L -o models/clip_tokenizer.json \
  https://huggingface.co/openai/clip-vit-base-patch32/raw/main/tokenizer.json
```

---

## Running the Sample

### Basic Text-to-Image Generation

```bash
dotnet run --project samples/OpenTail.Stingray.Sample.Diffusion -c Release -- \
  --prompt "A serene alpine mountain lake at sunrise, highly detailed digital art" \
  --output lake.png
```

### Specifying Explicit Model and Generation Parameters

```bash
dotnet run --project samples/OpenTail.Stingray.Sample.Diffusion -c Release -- \
  --model models/v1-5-pruned-emaonly.safetensors \
  --prompt "Retro futuristic synthwave sports car driving towards a neon city" \
  --steps 25 \
  --guidance 7.5 \
  --width 512 \
  --height 512 \
  --seed 42 \
  --output synthwave.png
```

---

## Command-Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `-m, --model <path>` | Path to `.safetensors` or `.gguf` checkpoint | Auto-discovered from ModelHome / `models/` |
| `-p, --prompt <text>` | Text prompt describing the target image | `"A serene alpine mountain lake at sunrise, highly detailed digital art"` |
| `-o, --output <path>` | Destination path for output PNG image | `output.png` |
| `--steps <int>` | Number of Euler discrete denoising steps | `20` |
| `--guidance <float>` | Classifier-free guidance scale | `7.5` |
| `--seed <int>` | RNG seed for deterministic generation (`-1` = random) | `-1` |
| `--width <int>` | Image width in pixels (must be divisible by 8) | `512` |
| `--height <int>` | Image height in pixels (must be divisible by 8) | `512` |
| `--tokenizer <path>` | Explicit path to `clip_tokenizer.json` | Auto-discovered |
| `-h, --help` | Show usage options | |

---

## Public API Reference

The sample code shows how to integrate diffusion generation into your own .NET application:

```csharp
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Diffusion.StableDiffusion;

// 1. Resolve compute backend (Vulkan GPU or managed SIMD CPU)
var (backend, ownsBackend) = DiffusionBackendResolver.Resolve(explicitBackend: null);

// 2. Load the pipeline from a single checkpoint
using IDiffusionPipeline pipeline = StableDiffusionPipeline.Load(modelPath, backend: backend);

// 3. Configure the generation request
var request = new ImageGenerationRequest
{
    Prompt = "A serene alpine mountain lake at sunrise",
    Width = 512,
    Height = 512,
    Steps = 20,
    Guidance = 7.5f,
    Seed = 42,
    OutputPath = "output.png",
    Progress = (step, total) => Console.Write($"\rStep {step}/{total}...")
};

// 4. Generate the image
pipeline.Generate(request);
```

---

## Canonical Documentation

* [Diffusion Model Pipelines](../../docs/models/diffusion-pipelines.md) — Architectural coverage across UNet (SD 1.5, SDXL), DiT (FLUX.1, Z-Image-Turbo), and video diffusion models.
* [Hardware Backends](../../docs/guides/hardware-backends.md) — Backend selection, Vulkan GPU compute, and SIMD CPU vectorization.
* [Feature Index](../../docs/WHAT-YOU-CAN-DO.md) — Complete capabilities overview.
