# Diffusion proof samples

Real, weight-driven generations from OpenTail Stingray's native C# pipelines. No placeholders, no mock latents, no Python.
Every image uses the **same prompt and seed** so the models can be compared directly:

> **Prompt:** `a red apple on a wooden table` &nbsp;·&nbsp; **Seed:** 42 &nbsp;·&nbsp; **Hardware:** Ryzen 5700G with integrated Radeon graphics (no discrete GPU); each row says which backend produced it &nbsp;·&nbsp; generated 2026-10-10

Times are wall-clock for the whole run, one model at a time. "GPU (Vulkan)" means the integrated Radeon, confirmed from the run log; a discrete GPU would be much faster. Wan 2.1 ran on CPU (it only uses the GPU when given `--device 0`).

| Model | Image | Size / steps | Time |
|---|---|---|---|
| Stable Diffusion 1.5 | ![SD1.5](sd15-apple-512.png) | 512×512, 20 | 162s, GPU (Vulkan) |
| SDXL-Turbo | ![SDXL-Turbo](sdxl-turbo-apple-512.png) | 512×512, 4 | 42s, GPU (Vulkan) |
| SDXL base 1.0 (fp16 safetensors) | ![SDXL base](sdxl-base-apple-1024.png) | 1024×1024, 20 | 690s, GPU (Vulkan) |
| FLUX.1-schnell (Q4_K_S) | ![FLUX.1](flux1-schnell-apple-512.png) | 512×512, 4 | 156s, GPU (Vulkan) |
| Qwen Image (Q3_K_S + Qwen2.5-VL-7B encoder) | ![Qwen Image](qwen-image-apple-256.png) | 256×256, 8, CFG 4 | 151s, GPU (Vulkan), 60/60 blocks resident |
| Z-Image-Turbo (Q4_0) | ![Z-Image](zimage-turbo-apple-512.png) | 512×512, 4 | 129s, GPU (Vulkan) |
| Stable Diffusion 3.5 Medium (Q4_K_M) | ![SD3.5](sd35-medium-apple-512.png) | 512×512, 20 | 203s, GPU (Vulkan) |
| Wan 2.1 T2V 1.3B (Q4_0), one video frame | ![Wan 2.1](wan21-apple-256.png) | 256×256, 20 | 100s, **CPU** |
| LTX-Video 2B v0.9.1, one video frame | ![LTX-Video](ltx-video-apple-512.png) | 512×512, 20 | 155s, GPU requested (`--device 0`); the run log prints no backend line |

SDXL base is shown at its native 1024×1024: at 512×512 it produced a two-panel collage, which is expected for a model trained at 1024. The Q4_0 SDXL-base GGUF uses sd.cpp-style tensor names that our loader does not read yet, so the safetensors file was used.

LTX-Video's single frame is abstract colour blobs with a few red round shapes, not a recognisable apple: the pipeline runs end to end, but this output is weak, and one frame is not what a video model is designed for.

Wan 2.1 is a video model; this is a single frame at low resolution, so it is softer than the image models. It is a smooth red apple-like object on wood: it proves the pipeline runs end to end, not that it matches the image models for quality.

## Commands

Run from the repo root (`stingray` is the built CLI). Add `-o <file>.png`; `--seed 42` and `-p "a red apple on a wooden table"` are shared.

```powershell
# SD 1.5
stingray image -m models/sd15/v1-5-pruned-emaonly.safetensors -W 512 -H 512 --steps 20 ...
# SDXL-Turbo
stingray image -m models/sd_xl_turbo_1.0_fp16.safetensors -W 512 -H 512 --steps 4 ...
# FLUX.1-schnell
stingray image -m models/flux1-schnell/flux1-schnell-Q4_K_S.gguf --vae models/flux1-schnell/ae.safetensors --clip-l models/flux1-schnell/clip_l.safetensors --clip-tokenizer models/flux1-schnell/tokenizer_clip/tokenizer.json --t5xxl models/flux1-schnell/t5xxl_fp8_e4m3fn.safetensors --t5-tokenizer models/flux1-schnell/tokenizer_t5/tokenizer.json -W 512 -H 512 --steps 4 ...
# Z-Image-Turbo
stingray image -m models/_models/z_image_turbo-Q4_0.gguf --vae models/z-image-turbo/vae/diffusion_pytorch_model.safetensors --qwen-encoder models/_models/Z-Image-AbliteratedV1.Q5_K_M.gguf --qwen-tokenizer models/z-image-turbo/tokenizer/tokenizer.json -W 512 -H 512 --steps 4 ...
# SD 3.5 Medium
stingray image -m models/_models/sd3.5_medium-Q4_K_M.gguf --clip-l models/sd35-medium-aux/text_encoder/model.fp16.safetensors --clip-g models/sd35-medium-aux/text_encoder_2/model.fp16.safetensors --vae models/sd35-medium-aux/vae/diffusion_pytorch_model.safetensors --clip-tokenizer models/clip_tokenizer.json -W 512 -H 512 --steps 20 ...
# Wan 2.1 (single frame)
stingray image -m models/_models/Wan2.1-T2V-1.3B-Q4_0.gguf --vae models/wan2.1/Wan2.1_VAE.safetensors --umt5-encoder models/wan2.1/models_t5_umt5-xxl-enc-bf16.safetensors --umt5-tokenizer models/wan2.1/umt5-tokenizer.json -W 256 -H 256 --steps 20 --video-frames 1 ...
```

More models (FLUX.2, Qwen Image, LTX-Video, HunyuanVideo) and their verification status are in [STATUS.md](../STATUS.md). The engineering history of early bring-up is archived in [docs/done/diffusion-spot-check-history-2026-09.md](../done/diffusion-spot-check-history-2026-09.md).

For audio samples (speech, **music** and **sound effects**) see [audio-samples](../audio-samples/README.md).
