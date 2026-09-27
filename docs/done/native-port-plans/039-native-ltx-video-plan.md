# Plan — Native LTX-Video Support for OpenTail.Stingray

**Reference:** `leejet/stable-diffusion.cpp` (`src/model/diffusion/ltxv.hpp`, `src/model/vae/ltx_vae.hpp`, `src/stable-diffusion.cpp`)  
**Target:** `opentail-net/OpenTail.Stingray`  
**Execution:** **100% local/native C# (.NET 10) — no cloud, Python, P/Invoke, or external inference process**

---

# Status

**IMPLEMENTED & VERIFIED** (LTX-Video Fast Video Diffusion DiT Substrate)

### Completed Implementation:
- **LtxVideoRoPE.cs**: 3D Continuous fractional positional grid generating rotary embeddings across (t, h, w) coordinate spaces with $\theta = 10000.0$.
- **LtxVideoModel.cs**: 128-channel video DiT backbone (=2048$, 32 attention heads, 28 transformer blocks) with AdaLN modulation, self-attention, T5-XXL cross-attention (=4096$), and modulated GeLU FFN.
- **LtxVideoPipeline.cs**: Flow-matching inference pipeline supporting text-to-video, image-to-video, multi-frame video sequence export, and 128-channel VAE decoding.
- **ImageCommand.cs**: Integrated IsLtxVideo model signature detection and RunLtxVideo execution.
- **LtxVideoTests.cs**: Unit tests in OpenTail.Stingray.Tests.Diffusion verifying continuous RoPE ranges, velocity prediction, and video frame generation (all 3 passing).

OpenTail.Stingray contains complete native diffusion infrastructure covering Stable Diffusion 1.5 / SDXL, SD3/3.5, FLUX, Z-Image, Qwen Image & Edit, Wan 2.1 / 2.2 Video, and HunyuanVideo / 1.5 Diffusion.

This plan specifies the native C# implementation of **LTX-Video** (Lightricks High-Speed Video Diffusion DiT).

---

# 1. Objective

Add native C# support for the **LTX-Video** diffusion family.

The primary targets are:
1. **LTX-Video (Base 0.9.x / 24–28 Transformer Blocks / 720p 24fps T2V & I2V)**
2. **LTX-Video Continuous 3D Positional Grid & 128-Channel VAE Decoding**

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
    ├── HunyuanVideo (54-layer Dual/Single DiT)
    └── LTXVideo
        ├── LtxVideoRoPE.cs
        ├── LtxVideoModel.cs
        └── LtxVideoPipeline.cs
```

---

# 2. Architectural Analysis & Specification

### 2.1 Core Backbone: LTX-Video Diffusion Transformer (`ltxv.hpp`)
* **Model Dimensions:**
  * `hidden_size`: $2048$ (or $3840$ for large configuration)
  * `num_attention_heads`: $32$ (or $30$) with `attention_head_dim`: $64$ (or $128$)
  * `num_layers`: $28$ transformer blocks
  * `in_channels`: $128$ (direct 128-channel latent representations)
  * `out_channels`: $128$
  * `cross_attention_dim`: $4096$ (T5-XXL text conditioning)
  * `caption_channels`: $3840$ / $4096$

### 2.2 Continuous 3D Spatio-Temporal Positional Grid & 3D RoPE
Unlike integer-grid RoPE, LTX-Video computes continuous fractional bounding coordinates for each latent token:
* Temporal coordinates: $[t_{\text{start}}, t_{\text{end}}]$ mapped from pixel frames:
  $$t_{\text{start}} = \frac{\text{corner}(t, \text{scale}=8)}{\text{fps}}, \quad t_{\text{end}} = \frac{\text{corner}(t+1, \text{scale}=8)}{\text{fps}}$$
* Spatial coordinates: $[h_{\text{start}}, h_{\text{end}}, w_{\text{start}}, w_{\text{end}}]$ with spatial downsampling $32\times$.
* RoPE frequency modulation with base frequency $\theta = 10000.0$.

### 2.3 AdaLN Modulation & Transformer Blocks
Each LTX-Video block contains:
1. **Self-Attention (`attn1`):** Spatial-temporal full self-attention with 3D RoPE and AdaLN scale/shift modulation ($mod_{\text{msa}}$).
2. **Cross-Attention (`attn2`):** Text cross-attention with T5-XXL sequence projections ($d=4096$).
3. **Feed-Forward Network (`ff`):** Modulated GeLU Feed-Forward Network ($mod_{\text{mlp}}$) with gating.
4. **Timestep Embedder:** SiLU-modulated MLP expanding scalar $t \in [0, 1000]$ to `hidden_size`.

### 2.4 128-Channel Spatial-Temporal VAE (`LTXVideoVAE`)
* Latents have shape $[128, T/8, H/32, W/32]$.
* Fast 3D causal spatial-temporal decoding to $[3, T, H, W]$ RGB frames.

### 2.5 Scheduler & Flow Trajectory
* **Rectified Flow-Matching:**
  $$t_{\text{shifted}} = \frac{s \cdot t}{1 + (s - 1) \cdot t} \quad (s = 3.0 \text{ or } 5.0)$$
* Fast sampling in 20–30 steps with linear or shifted Euler flow.

---

# 3. Phased Implementation Strategy

### Phase 1: 3D Continuous Positional Grid & RoPE (`LtxVideoRoPE.cs`)
* Implement `LtxVideoRoPE.cs`:
  * Continuous coordinate calculation $[t_{\text{start}}, t_{\text{end}}, h_{\text{start}}, h_{\text{end}}, w_{\text{start}}, w_{\text{end}}]$.
  * 3D rotary frequency generation across temporal and spatial axes.

### Phase 2: Native LTX-Video DiT Model (`LtxVideoModel.cs`)
* Implement `LtxVideoModel.cs`:
  * `patchify_proj` ($128 \to 2048$).
  * 28 Transformer blocks with AdaLN modulation, self-attention, T5 cross-attention, and GeLU FFN.
  * `proj_out` ($2048 \to 128$).

### Phase 3: Flow-Matching Pipeline & Sequence Exporter (`LtxVideoPipeline.cs`)
* Implement `LtxVideoPipeline.cs`:
  * 128-channel latent initialization for text-to-video (T2V) and image-to-video (I2V).
  * Flow matching ODE integration.
  * 128-channel VAE decoding and video frame sequence export (`_frame_000.png`).

### Phase 4: CLI Integration & Automated Verification
* Wire `IsLtxVideo` and `RunLtxVideo` into `ImageCommand.cs`.
* Create `LtxVideoTests.cs` in `OpenTail.Stingray.Tests.Diffusion`.
* Build and verify all 35 projects in `OpenTail.Stingray.slnx`.

