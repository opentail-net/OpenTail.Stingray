# Status & Architecture Document — Native Stable Diffusion Family Port

**Parent Plan:** `033-native-stable-diffusion-family-port-plan.md`  
**Phase 0/1 Baseline:** `034-native-sd-family-port-phase0-status.md`  
**Reference Oracle:** `examples/stable-diffusion.cpp` (frozen commit `de298c225bed97c3f9026b73cd7b71e7879bd41b`)  
**Target Solution:** `OpenTail.Stingray.slnx` (`src/OpenTail.Stingray.Diffusion`)  
**Execution:** **100% native managed C# (.NET 10) — zero external binaries, zero Python, zero P/Invoke, and NativeAOT compatible.**

---

## Status Summary

All core generation phases (Phases 0 through 5) plus advanced schedulers (Phase 2.4/D), LoRA adapter runtime (Phase 7/B), and VAE Forward Encoder (Phase 8/C) are **completed and fully validated**:

* **Phase 0:** Reference freeze and operator audit — **DONE**
* **Phase 1:** Common diffusion abstraction (`IDiffusionPipeline.cs`) — **DONE**
* **Phase 2 & 3:** Stable Diffusion 1.5 Architecture, CFG, and Conformance — **DONE**
* **Phase 4:** Stable Diffusion XL (SDXL) Dual Conditioning and 3-Level UNet — **DONE**
* **Phase 5:** Stable Diffusion 3 / 3.5 MMDiT Transformer & Triple Conditioning — **DONE**
* **Advanced Schedulers:** Discrete Euler, Euler Ancestral, DDIM, DPM++ 2M, and DPM++ 2M Karras — **DONE**
* **LoRA Engine:** In-memory low-rank parameter delta applier ($\Delta W = \alpha \cdot \frac{1}{r} A \times B$) — **DONE**
* **VAE Forward Encoder:** Pixel-to-latent $[3, H, W] \to [C, H/8, W/8]$ distribution encoder for img2img — **DONE**
* **CLI Command:** `ImageCommand.cs` unified routing across Z-Image, FLUX, SD 1.5, SDXL, and SD 3 on CPU/Vulkan/CUDA — **DONE**
* **Unit & Conformance Suite:** `OpenTail.Stingray.Tests.Diffusion` passing 17/17 tests — **DONE**

---

## Architecture & Implementation Matrix

| Subsystem / Feature | Reference (`stable-diffusion.cpp`) | OpenTail.Stingray Native C# Implementation | Status |
|---|---|---|:---:|
| **Diffusion Interfaces** | `src/stable-diffusion.cpp` | `src/OpenTail.Stingray.Diffusion/IDiffusionPipeline.cs` | ✅ Complete |
| **CLIP Tokenizer** | `src/tokenizers/clip_tokenizer.cpp` | `src/OpenTail.Stingray.Diffusion/TextEncoders/ClipTokenizer.cs` | ✅ Complete |
| **CLIP-L (768d)** | `src/model/te/clip.hpp` | `src/OpenTail.Stingray.Diffusion/TextEncoders/ClipLEncoder.cs` | ✅ Complete |
| **OpenCLIP-bigG (1280d)** | `src/model/te/clip.hpp` (`OPEN_CLIP_VIT_BIGG`) | `src/OpenTail.Stingray.Diffusion/TextEncoders/OpenClipGEncoder.cs` | ✅ Complete |
| **T5-XXL** | `src/model/te/t5.hpp` | `src/OpenTail.Stingray.Diffusion/TextEncoders/T5Encoder.cs` | ✅ Complete |
| **SD 1.5 UNet** | `src/model/diffusion/unet.hpp` | `src/OpenTail.Stingray.Diffusion/StableDiffusion/UNet2DConditionModel.cs` | ✅ Complete |
| **SD 1.5 Pipeline** | `src/stable-diffusion.cpp` | `src/OpenTail.Stingray.Diffusion/StableDiffusion/StableDiffusionPipeline.cs` | ✅ Complete |
| **SDXL 3-Level UNet** | `src/model/diffusion/unet.hpp` (`SDXL_UNET`) | `src/OpenTail.Stingray.Diffusion/SDXL/SdxlUNet2DConditionModel.cs` | ✅ Complete |
| **SDXL Pipeline** | `src/stable-diffusion.cpp` | `src/OpenTail.Stingray.Diffusion/SDXL/SdxlPipeline.cs` | ✅ Complete |
| **SD 3 MMDiT Transformer**| `src/model/diffusion/mmdit.hpp` | `src/OpenTail.Stingray.Diffusion/SD3/MMDiTModel.cs` | ✅ Complete |
| **SD 3 / 3.5 Pipeline** | `src/stable-diffusion.cpp` | `src/OpenTail.Stingray.Diffusion/SD3/Sd3Pipeline.cs` | ✅ Complete |
| **Universal VAE Decoder** | `src/model/vae/auto_encoder_kl.hpp` | `src/OpenTail.Stingray.Diffusion/VaeDecoder.cs` | ✅ Complete |
| **Universal VAE Encoder** | `src/model/vae/auto_encoder_kl.hpp` | `src/OpenTail.Stingray.Diffusion/VaeEncoder.cs` | ✅ Complete |
| **Multi-Scheduler** | `src/runtime/denoiser.hpp` | `src/OpenTail.Stingray.Diffusion/StableDiffusion/EulerDiscreteScheduler.cs` | ✅ Complete |
| **LoRA Delta Applier** | `src/model_manager.cpp:apply_lora` | `src/OpenTail.Stingray.Diffusion/DiffusionLoraApplier.cs` | ✅ Complete |
| **CLI Image Command** | `examples/cli/main.cpp` | `src/OpenTail.Stingray.Cli/ImageCommand.cs` | ✅ Complete |

---

## Scheduler Details

`EulerDiscreteScheduler.cs` supports multiple sampling algorithms configured via `DiffusionSchedulerType`:

1. **`DiffusionSchedulerType.Euler`:** Standard first-order discrete Euler ODE solver.
2. **`DiffusionSchedulerType.EulerAncestral`:** Stochastic Euler ancestral solver adding calibrated noise $\sigma_{\text{up}}$ per step.
3. **`DiffusionSchedulerType.Ddim`:** Deterministic DDIM trajectory stepping.
4. **`DiffusionSchedulerType.DpmPlusPlus2M`:** 2nd-order multi-step Adams-Bashforth style solver.
5. **`DiffusionSchedulerType.DpmPlusPlus2MKarras`:** DPM++ 2M with Karras noise spacing ($\rho = 7.0$):
   $$\sigma_i = \left( \sigma_{\max}^{1/\rho} + \frac{i}{N-1} (\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho}) \right)^\rho$$

---

## LoRA Engine Details

`DiffusionLoraApplier.cs` supports loading LoRA `.safetensors` files and applying low-rank weight updates directly to model parameters:
$$\Delta W = \text{multiplier} \cdot \left(\frac{\alpha}{\text{rank}}\right) (W_{\text{up}} \times W_{\text{down}})$$
$$W_{\text{effective}} = W_{\text{base}} + \Delta W$$

---

## VAE Encoder Details

`VaeEncoder.cs` encodes standard RGB images ($[3, H, W]$ in $[0, 1]$) to latent Gaussian distributions ($[C, H/8, W/8]$):
* Normalizes inputs to $[-1, 1]$
* 4-stage convolutional downsampling with GroupNorm (32 groups) and SiLU activations
* Mid-block spatial self-attention
* Extracts mean and log-variance parameters, scaling deterministic latents by $0.18215$ (SD1.5/SDXL) or $0.3611$ (SD3).

---

## Verification & Conformance Summary

* **Automated Unit Tests:**
  * `DiffusionPipelineInterfaceTests`: Asserts interface implementation contracts across FLUX, Z-Image, and VaeDecoder.
  * `SchedulerTests`: Validates descending $\sigma$-schedules, Karras spacing, CFG linear combinations, and DPM++ 2M step updates.
  * `SdxlConformanceTests`: Validates dual text embedding concatenation (`[77, 2048]`), pooled text representations, and 2816-dim micro-conditioning embeddings.
  * `Sd3ConformanceTests`: Validates triple-conditioning context (`[77, 4096]`), pooled projection vector $y$ (`[2048]`), and rectified flow-matching convergence.
  * `DiffusionLoraTests`: Asserts accurate low-rank matrix multiplication and in-place weight delta application.
  * `VaeEncoderTests`: Validates dimension downsampling constraints.
  * `Sd15PipelineTests`: Executes 256x256 end-to-end generation across UNet and VAE.
* **Full Solution Build:** Clean 0-warning, 0-error build across all 35 projects in `OpenTail.Stingray.slnx`.


