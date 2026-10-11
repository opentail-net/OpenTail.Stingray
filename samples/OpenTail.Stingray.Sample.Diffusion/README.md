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

---

## Expected Output

A progress line per denoising step, then the path of the PNG and the denoising latency. The run behind this README (Radeon integrated graphics through Vulkan, SD 1.5, `--width 256 --height 256 --steps 8 --seed 42`, prompt "a red apple on a wooden table"; 31 s in total):

```text
[Denoising] Step 8/8 (100%)...
[Success] Image written to: ...\sample_diffusion.png
Denoising Latency: 28567 ms (3570.9 ms/step)
```

The image was a recognisable red apple on a wooden surface (inspected by eye). At the default 512x512 and 20 steps expect several times longer; on CPU, minutes.

## Known Limitations

- Stable Diffusion 1.5 only in this sample. FLUX, SD 3.5, Z-Image, video and audio diffusion need several model files each and have their own setup: see [docs/STATUS.md](../../docs/STATUS.md) and the [image and video samples](../../docs/diffusion-samples/README.md).
- A per-step `[UNet.ForwardGpu] record=...ms, submitWait=...ms` line is printed on the GPU path.
- The checkpoint is large (about 4.27 GB) and its source repository has moved on Hugging Face; both `runwayml/stable-diffusion-v1-5` and `stable-diffusion-v1-5/stable-diffusion-v1-5` resolved when checked on 2026-10-11, and the checkpoint licence (CreativeML Open RAIL-M) applies.

## Verification

- **Run against a real checkpoint, 2026-10-11:** `v1-5-pruned-emaonly.safetensors` and the CLIP tokenizer, on the GPU, as shown. One run; the output was judged by eye, not compared against a reference image. [docs/STATUS.md](../../docs/STATUS.md) records Stable Diffusion 1.5 as checked by human judgement of the output (the 👂 tier), not by a numeric comparison with a reference implementation.
- **Package contract:** builds against the packed `OpenTail.Stingray` 1.0.8 NuGet package with no project references (`scripts/verify-package-consumers.ps1`, 2026-10-11).
