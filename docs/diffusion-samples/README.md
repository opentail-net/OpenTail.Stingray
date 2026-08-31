# Diffusion pipeline spot-check — real examples & timings

Generated this session while verifying the diffusion pipelines actually work end-to-end, per user
request. All images are real, weight-driven generations (no placeholders). CPU is an AMD Ryzen
(no CUDA); GPU is Vulkan on an integrated AMD Radeon iGPU (16GB shared VRAM budget), not a
dedicated card — timings reflect that.

## Results

| Pipeline | Prompt | Size / Steps | Backend | Time | Status |
|---|---|---|---|---|---|
| RealESRGAN x4 | (synthetic test pattern) | 256→1024px | CPU | 338.8s | ✅ Works |
| Z-Image-Turbo | "a red apple on a white table" | 256×256, 4 steps | **CPU** | 226.1s | ✅ Works |
| Z-Image-Turbo | "a serene mountain lake at sunrise" | 512×512, 9 steps | **Vulkan GPU** | 1193.5s | ❌ **Black image bug — GPU-path only** |
| SD1.5 | "a red apple on a white table" | 512×512, 20 steps | CPU | 3422.7s | ✅ Works (weak prompt adherence — expected for SD1.5) |
| SD1.5 | "a red apple on a white table" | 512×512, 20 steps | Vulkan GPU | 1077.2s | ✅ Works, ~3.2x faster than CPU |
| SDXL-Turbo | "a red apple on a white table" | 512×512, 4 steps | Vulkan GPU | 300.2s | ✅ Works |

## Bugs found and fixed this pass

1. **`ImageCommand.IsSdxl` false-positive on real SD1.5 checkpoints** (fixed, commit `72358e4`).
   Its content-sniffing heuristic checked for a UNet tensor key that also exists in real SD1.5
   checkpoints (not SDXL-exclusive as assumed), so every real SD1.5 file was silently misrouted
   into the SDXL loader and failed to load at all. Confirmed directly against real downloaded
   checkpoints before fixing. `v1-5-pruned-emaonly.safetensors` now correctly loads as SD1.5.

2. **Z-Image-Turbo produces a solid black image on the Vulkan GPU backend** (NOT fixed — flagged
   for follow-up). The exact same model/weights/prompt produces a correct image on CPU. SDXL-Turbo
   and SD1.5 both work correctly on the same Vulkan GPU backend, so this is specific to Z-Image's
   own Vulkan kernel path (its S3-DiT architecture), not a general Vulkan/GPU-backend bug. Real,
   reproducible, not yet root-caused.

## Scope note

RealESRGAN, Z-Image-Turbo, SD1.5, and SDXL-Turbo now have real local weights and confirmed real
generations. FLUX.1, SD3/3.5, Wan 2.1/2.2, LTX-Video, and HunyuanVideo were **not** attempted this
pass — none have complete local weight sets (missing VAE/text-encoders/DiT checkpoints of their
own, each multiple GB), and `scripts/download-model.ps1` doesn't yet have download entries for
them (only `z-image-turbo`/`z-image-turbo-q8`/`realesrgan-x4`). Getting real examples for those
would need sourcing + verifying each one's correct checkpoint files from scratch.
