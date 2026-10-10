# Z-Image-Turbo Family (Ultra-Fast 4-Step Distilled DiT)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | Community / Distilled DiT Foundation |
| **Model Formats** | GGUF (`z_image_turbo-Q5_K_M.gguf`) |
| **Engine Implementations** | `OpenTail.Stingray.Diffusion.ZImageTurboPipeline` |
| **Synthesis Speed** | **4 Steps** to high-detail 512×512 synthesis (~28s on CPU, ~1.2s on GPU) |
| **Text Conditioning** | Qwen-based text encoder (`Z-Image-AbliteratedV1.Q5_K_M.gguf`) |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (Tested with real checkpoints, documented in CLI README) |

---

## 1. Overview & Architectural Highlights

Z-Image-Turbo is an ultra-fast, adversarial/distilled diffusion transformer designed for real-time and near-instantaneous local image generation. By distilling multi-step flow trajectories into 4 discrete integration steps, it bypasses the compute overhead of traditional 20–50 step diffusion models.

### Key Architectural Characteristics
* **4-Step Adversarial Distillation:**
  Achieves sharp, artifact-free image generation in only 4 Euler steps with fixed timestep schedules ($t \in [1.0, 0.75, 0.5, 0.25]$).
* **Qwen Text Conditioning:**
  Uses a dedicated Qwen language model encoder rather than legacy CLIP, enabling rich linguistic prompt understanding and complex multi-clause sentence parsing.
* **GGUF-Native Execution:**
  Both the DiT backbone and the Qwen text conditioning model are quantized into GGUF (`Q5_K_M`), requiring less than 4 GB of total system memory.

---

## 2. Checkpoints & Recommended GGUFs

| Component | Recommended File | Quantization | Size | Hugging Face Repository |
|---|---|---|---|---|
| **DiT Model** | `z_image_turbo-Q5_K_M.gguf` | `Q5_K_M` | ~1.4 GB | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) |
| **Qwen Encoder** | `Z-Image-AbliteratedV1.Q5_K_M.gguf` | `Q5_K_M` | ~1.2 GB | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) |
| **Tokenizer** | `tokenizer.json` | — | ~2 MB | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) |
| **VAE** | `vae/` | FP16 | ~335 MB | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) |

---

## 3. Usage & Code Examples

### CLI Command

```bash
# Generate high-detail image in 4 steps
stingray image \
  -m models/z_image_turbo-Q5_K_M.gguf \
  --vae models/z-image-turbo/vae \
  --qwen-encoder models/Z-Image-AbliteratedV1.Q5_K_M.gguf \
  --qwen-tokenizer models/z-image-turbo/tokenizer/tokenizer.json \
  -p "A serene mountain lake at sunrise, crystal clear water, highly detailed" \
  -W 512 -H 512 --steps 4 -o out.png
```
