# Wan Video Family (Wan 2.1 & 2.2 3D Video DiT)

[← Back to Architecture Cards](README.md)

Related: [flux](flux.md), [stable-diffusion](stable-diffusion.md). Verification: [STATUS.md](../STATUS.md); measured commands: [RUNNING.md](../RUNNING.md).

| Property | Value |
|---|---|
| **Provider** | Wan Video Team |
| **Model Formats** | GGUF (`wan2.1-t2v-1.3B-Q4_K_M.gguf`) & SafeTensors |
| **Engine Implementations** | `OpenTail.Stingray.Diffusion.Wan.WanPipeline`, `WanVaeDecoder3D` |
| **Modalities** | Text-to-Video, Image-to-Video |
| **Architecture** | 3D Spatio-Temporal Diffusion Transformer (3D DiT) |
| **Video Exporter** | Pure C# animated GIF & animated PNG sequence writer with Floyd-Steinberg dithering |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (`Wan 2.1 1.3B & 14B`, VAE 3D and 3D DiT components Level 2 Proven) |

---

## 1. Overview & Architectural Highlights

**Backends:** CPU, Vulkan and CUDA (see [STATUS](../STATUS.md)). Measured 2026-10-10 on a Ryzen 5700G iGPU (AMD Radeon, shared memory): single frame, 256×256, 20 steps, Vulkan: 100 s (soft output at that size).

Wan 2.1 and 2.2 are cutting-edge open video generation models based on a 3D Spatio-Temporal Diffusion Transformer. The architecture synthesizes temporally coherent video clips directly from text descriptions or reference starter images.

### Key Architectural Characteristics
* **3D Spatio-Temporal Patchification:**
  Breaks temporal video frames ($T \times H \times W$) into 3D voxel patches ($2 \times 8 \times 8$). Temporal attention tracks motion flow across frames, while spatial attention maintains per-frame photorealism.
* **3D Causal VAE Decoder (`WanVaeDecoder3D`):**
  Decodes compressed 16-channel spatio-temporal latents back to video frames using 3D causal convolutions that prevent future-frame leakage.
* **Self-Contained Video Animation Exporter:**
  Stingray writes generated video frame sequences directly into compressed `.gif` and `.png` sequence files with Floyd-Steinberg dithering without calling `ffmpeg` or external binaries.

---

## 2. Checkpoints & Recommended GGUFs

| Model | Parameters | Resolution | Frames | Recommended Quantization | Typical Size | Hugging Face Repository |
|---|---|---|---|---|---|---|
| **Wan2.1-T2V-1.3B** | 1.3B | 480p / 720p | 16–81 frames | `Q4_K_M` | ~1.1 GB | [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) *(no verified public GGUF repo)* |
| **Wan2.1-T2V-14B** | 14.2B | 720p | 16–81 frames | `Q4_K_M` | ~9.2 GB | [Wan-AI/Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) *(GGUF: [city96/Wan2.1-T2V-14B-gguf](https://huggingface.co/city96/Wan2.1-T2V-14B-gguf))* |

---

## 3. Usage & Code Examples

### CLI Video Generation

```bash
# Generate a 16-frame animated GIF from prompt
stingray image \
  -m models/wan2.1-t2v-1.3B-Q4_K_M.gguf \
  -p "Ocean waves crashing against jagged rocks at sunset, dramatic spray, slow motion" \
  --video-frames 16 \
  -W 480 -H 480 \
  -o ocean_waves.gif
```
