# Plan — Native HunyuanVideo Support for OpenTail.Stingray

**Reference:** `leejet/stable-diffusion.cpp` (`src/model/diffusion/hunyuan.hpp`, `src/model/vae/hunyuan_vae.hpp`, `docs/hunyuan_video.md`)  
**Target:** `opentail-net/OpenTail.Stingray`  
**Execution:** **100% local/native C# — no cloud, Python, P/Invoke, or external inference process**

---

# Status

**IMPLEMENTED & VERIFIED** (HunyuanVideo & HunyuanVideo 1.5 Video Diffusion DiT Substrate)

### Completed Implementation:
- **HunyuanVideoRoPE.cs**: 3D-RoPE frequency calculation decomposing 128 head dim across {16, 56, 56} for (t, y, x) with base $\theta = 256.0 / 10000.0$.
- **HunyuanVideoModel.cs**: HunyuanVideo DiT supporting Dual-Stream (double_blocks up to 54 layers) and Single-Stream (single_blocks up to 40 layers) architectures with \times 2\times 2$ patch packing ( \to 64$ ch), joint spatial-temporal attention with 3D-RoPE and QK RMSNorm, dual text conditioning (=4096$), and modulated GELU FeedForward networks.
- **HunyuanVideoPipeline.cs**: Rectified Flow-Matching pipeline supporting image and multi-frame video generation with flow shift =7.0$, CFG guidance .0$, and 16-channel causal VAE decoding.
- **ImageCommand.cs**: CLI command dispatch with IsHunyuanVideo auto-detection, --video-frames, and CPU/Vulkan GPU hardware acceleration.
- **HunyuanVideoTests.cs**: Unit tests verifying 3D-RoPE generation across frames/height/width, lossless latent patch packing/unpacking, and flow shift scheduling (all passing in OpenTail.Stingray.Tests.Diffusion).

OpenTail.Stingray contains complete native diffusion infrastructure covering Stable Diffusion 1.5 / SDXL, SD3/3.5, FLUX, Z-Image, Qwen Image, and Wan 2.1 / 2.2 Video Diffusion.

This plan specifies the native C# implementation of **HunyuanVideo** (Tencent Hunyuan Video Diffusion DiT) and **HunyuanVideo 1.5**.

---

# 1. Objective

Add native C# support for the **HunyuanVideo** diffusion family.

The primary targets are:
1. **HunyuanVideo (T2V Base / 720p / 54 Double Blocks + Single Blocks)**
2. **HunyuanVideo 1.5 (T2V 720p / Qwen2.5-VL + ByT5 GlyphXL Conditioning)**

The desired repository layout in `OpenTail.Stingray.Diffusion`:

```text
OpenTail.Stingray.Diffusion
│
├── Image diffusion
│   ├── StableDiffusion (SD 1.5, SDXL)
│   ├── SD3 (SD 3 / 3.5 MM-DiT)
│   ├── FLUX (Dual/Single DiT)
│   ├── Z-Image (S3-DiT)
│   └── QwenImage (60-layer MM-DiT)
│
└── Video diffusion
    ├── Wan (Wan2.1 / Wan2.2 DiT)
    └── HunyuanVideo
        ├── HunyuanVideoRoPE.cs
        ├── HunyuanVideoModel.cs
        └── HunyuanVideoPipeline.cs
```

---

# 2. Architectural Analysis & Specification

### 2.1 Core Backbone: Dual-Stream & Single-Stream Video DiT
* **HunyuanVideo (Base):**
  * `hidden_size`: $3072$
  * `num_heads`: $24$ (`head_dim = 128`)
  * `double_blocks` (Dual-Stream): 20 or 54 blocks
  * `single_blocks` (Single-Stream): Optional 40 fused blocks
  * `in_channels`: 16 latent channels packed with $1 \times 2 \times 2$ patch volume to 64 input channels (or 65 with guidance)
  * `out_channels`: 16 (or 32 with variance)
* **HunyuanVideo 1.5:**
  * `hidden_size`: $2048$
  * `num_heads`: $16$ (`head_dim = 128`)
  * `mlp_ratio`: $4.0$

### 2.2 3D Spatio-Temporal RoPE
* **Axes Decomposition:**
  $$\text{axes\_dim} = \{16, 56, 56\} \quad (16 + 56 + 56 = 128 = d_{\text{head}})$$
  * $16$ dimensions for temporal sequence $t$
  * $56$ dimensions for vertical coordinate $y$
  * $56$ dimensions for horizontal coordinate $x$
* **Base Frequency:** $\theta = 256.0$ (or $10000.0$)

### 2.3 Dual Text Conditioning & Token Refiner
HunyuanVideo employs a dual-conditioning pipeline:
1. **Primary LLM / MLLM Text Encoder:**
   * LLaMA-3 8B text hidden states ($d=4096$) or Qwen2.5-VL 7B ($d=3584$).
2. **Secondary Text Encoder (Glyph & Style):**
   * CLIP-L ($d=768$) or ByT5 Small GlyphXL ($d=1472$).
3. **`IndividualTokenRefiner`:**
   * Refines text token representations with self-attention, layer norms, and timestep pooling before injecting into the main transformer backbone.

### 2.4 16-Channel 3D Causal VAE (`HunyuanVAE`)
* 16 latent channels with spatial compression $8\times$ and temporal causal downsampling.
* Decodes latents $[16, T, H/8, W/8] \to [3, T, H, W]$ into RGB video frames.

### 2.5 Scheduler & Flow Trajectory
* **Rectified Flow Matching:**
  $$t_{\text{shifted}} = \frac{s \cdot t}{1 + (s - 1) \cdot t} \quad (s = 7.0 \text{ or } 3.0)$$
* **Euler Sampler with 2-Pass CFG:** Baseline guidance $\text{scale} \in [6.0, 7.5]$.

---

# 3. Phased Implementation Strategy

## Phase 0 — Reference & Numerical Specification Freeze
* Clone reference code from `examples/stable-diffusion.cpp/src/model/diffusion/hunyuan.hpp` and `examples/stable-diffusion.cpp/src/model/vae/hunyuan_vae.hpp`.
* Inventory shared tensor operations (`IComputeBackend`, SGEMM, LayerNorm, RMSNorm, SiLU, GELU).

## Phase 1 — 3D Spatio-Temporal Positional Embeddings
* Create `src/OpenTail.Stingray.Diffusion/HunyuanVideo/HunyuanVideoRoPE.cs`:
  * Precomputes sinusoidal frequencies for $(t, y, x)$ across $\{16, 56, 56\}$.
  * Applies in-place rotary transformations to query and key tensors.

## Phase 2 — HunyuanVideo Transformer Model
* Create `src/OpenTail.Stingray.Diffusion/HunyuanVideo/HunyuanVideoModel.cs`:
  * `img_in`: Patch packing $16 \times 2 \times 2 \to 64 \to d_{\text{model}}$.
  * `txt_in` / `TokenRefiner`: Dual text projection and refinement blocks.
  * `double_blocks`: Dual-stream joint attention with QK RMSNorm and modulation.
  * `single_blocks`: Unified single-stream self-attention + MLP blocks.
  * `final_layer`: AdaLN norm + Linear projection unpacked to 16-channel latents.

## Phase 3 — Text Conditioning & Token Refiner Integration
* Support dual text encoders:
  * Primary context from LLaMA-3 / Qwen2.5-VL ($d=4096 / 3584$).
  * Secondary context from CLIP-L / ByT5 ($d=768 / 1472$).
  * Mean-pooling for pooled prompt condition embeddings.

## Phase 4 — HunyuanVideo Inference Pipeline
* Create `src/OpenTail.Stingray.Diffusion/HunyuanVideo/HunyuanVideoPipeline.cs`:
  * Flow matching scheduler ($s=7.0$).
  * Video latent noise generation $[16, T, H/8, W/8]$.
  * Full denoising trajectory loop.
  * 16-channel causal VAE decoding per frame.
  * Automatic multi-frame PNG sequence and primary anchor frame export.

## Phase 5 — CLI Wiring & Hardware Acceleration
* In `src/OpenTail.Stingray.Cli/ImageCommand.cs`:
  * Add `IsHunyuanVideo(modelPath)`.
  * Wire `RunHunyuanVideo` with `--video-frames`, `--cfg-scale`, `--steps`, and Vulkan GPU hardware compute.

## Phase 6 — Unit Testing & Conformance
* Create `tests/OpenTail.Stingray.Tests.Diffusion/HunyuanVideoTests.cs`:
  * 3D RoPE shape & frequency tests.
  * Latent patch packing & unpacking lossless identity tests.
  * Flow shift schedule validation.
  * End-to-end forward pass contract tests.

---

# 4. Definition of Done

1. ✅ `HunyuanVideoRoPE.cs` implemented with $\{16, 56, 56\}$ 3D decomposition.
2. ✅ `HunyuanVideoModel.cs` implemented with Dual-Stream and Single-Stream blocks.
3. ✅ `HunyuanVideoPipeline.cs` running Flow Matching and VAE decoding.
4. ✅ CLI dispatch enabled with `IsHunyuanVideo` auto-detection and `--video-frames`.
5. ✅ Unit tests passing in `OpenTail.Stingray.Tests.Diffusion`.
6. ✅ Clean compilation across all projects in `OpenTail.Stingray.slnx` with `0 Warnings, 0 Errors`.

