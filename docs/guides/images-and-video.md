# Guide: generate images and short video

**You get:** a PNG from a text prompt, plus image-to-image, inpainting, ControlNet, upscaling, and
short video clips with Wan.

**Be ready for:** these models need **several files each**, and on a CPU a full-size image takes
minutes. Start with a tiny preview (256×256, few steps) to prove your setup, then raise the size.

## The easiest first run: Stable Diffusion 1.5

```bash
stingray image -m models/sd15/v1-5-pruned-emaonly.safetensors \
  -p "a photograph of an astronaut riding a horse" \
  -W 256 -H 256 --steps 4 -o output.png
```

On the reference CPU that 256×256 preview takes about 13 s. Add `--device none` to force the CPU, or
leave it off to let the program use the GPU if it finds one.

## Pick a model

| Model | Files needed | Notes |
|---|---|---|
| **Stable Diffusion 1.5** | one `.safetensors` | The simplest. Also supports ControlNet |
| **Z-Image-Turbo** | the model `.gguf`, a VAE, a Qwen text encoder `.gguf` and its tokenizer | Quick (2 to 4 steps) |
| **FLUX.1-schnell** | the model `.gguf`, VAE, CLIP-L, T5-XXL and both tokenizers | 4 steps; many files |
| **Stable Diffusion 3.5 Medium** | the model `.gguf`, CLIP-L, CLIP-G, a VAE and a tokenizer | |
| **Wan 2.1 / 2.2 (video)** | the model `.gguf`, a VAE, a UMT5 encoder and tokenizer | `--video-frames N` |

The **exact command with every file for each model** is in the diffusion tables of
[RUNNING.md](../RUNNING.md). Copy it and change the prompt. Option names that apply to only some
models are marked in `stingray image --help` (for example `--clip-l` is for FLUX, `--clip-g` for
SD3/3.5, `--qwen-encoder` for Z-Image, `--umt5-encoder` for Wan).

## More things to do

| You want to | Options |
|---|---|
| Change a picture | `-i start.png --strength 0.6` (image-to-image) |
| Repaint part of a picture | `-i start.png --mask mask.png` (white is repainted) |
| Follow a shape or edge map | `--control-net <weights> --control-image edges.png --control-strength 1.0` |
| Make it larger | `--upscaler <Real-ESRGAN weights>` and `--upscale-blend` |
| Fix a result | `--seed 42` (same seed, same picture), `--negative-prompt "blurry"` |
| Make a short video | Wan: `--video-frames 16` (see the Wan row in RUNNING.md) |

Sizes must be a multiple of 16. `-v` shows per-step timing, which is the quickest way to see
whether the GPU is helping.

## How long will it take?

On the reference machine (a CPU and an integrated GPU, 64 GB of RAM), a small preview is seconds to
a minute, and Wan 14B takes over six minutes for one 256×256 frame. A real GPU helps a lot. The
measured timings are in [RUNNING.md](../RUNNING.md); the verification status of each model is in
[STATUS.md](../STATUS.md).

## Other models

FLUX.2, Qwen Image, SDXL-Turbo, LTX-Video and HunyuanVideo work and are listed in STATUS.md, but
the command line covers the models in the table above. For the others use the C# library
(`OpenTail.Stingray.Diffusion`, see its [README](../../src/OpenTail.Stingray.Diffusion/README.md)).

## From C#

To integrate text-to-image generation directly into a .NET application, see the [Diffusion sample (`samples/OpenTail.Stingray.Sample.Diffusion`)](../../samples/OpenTail.Stingray.Sample.Diffusion/) demonstrating `IDiffusionPipeline` and `StableDiffusionPipeline.Load`.

## Over HTTP

`POST /v1/images/generations`, `/v1/images/edits` and `/v1/images/variations`. See the
[serving guide](serve-an-api.md).
