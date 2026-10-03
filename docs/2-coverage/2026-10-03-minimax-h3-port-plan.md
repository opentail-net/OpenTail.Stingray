# MiniMax-H3 port plan (video + native 32 kHz stereo audio)

**Status:** not started. **Policy:** port now, prove later; kept off the CLI's model list and the
diffusion docs until verified (CLAUDE.md rule 14; [ported-families-todo](ported-families-todo.md)).

## Licensing & Distribution Note

The MiniMax H3 Community License applies downstream conditions and territorial restrictions to the
reproduction and distribution of H3 model weights / works. OpenTail does **not** distribute model
weights or checkpoints; OpenTail only provides inference and execution runtime capabilities.
In accordance with CLAUDE.md Rule 14, all code remains unadvertised and internal until verified,
and no proprietary weights are bundled or distributed.

## Architecture

Source: TensorSharp `docs/models/minimax-h3.md`, `Models/MiniMaxH3/` (20 files, including a 100%
pure-C# backend), upstream HF code (`MiniMaxAI/MiniMax-H3`).

One diffusion transformer denoises a **packed video+audio latent** in a single token sequence:
up to 15 s at 24 fps, plus a 32 kHz stereo soundtrack written as a sidecar `.wav`.

### Graphs vs Weight Sets
Seven execution graphs / passes over four principal weight sets:
1. **H3-Encoder (text & visual semantic encode):** Qwen3-VL-32B derivative (its own GGUF, no tokenizer embedded; `vocab.json` + `merges.txt` loaded externally).
2. **DiT Denoiser:** single omni-modal DiT operating over packed video + audio sequence.
3. **Video VAE (3D causal):** encode + decode (5.2 GB fp16 safetensors).
4. **Audio VAE:** encode + decode (0.6 GB fp32 safetensors).

### DiT Architecture Contract
- **Depth:** 50 transformer blocks.
- **Hidden dimension:** 5,376.
- **Attention:** 56 heads × 128 head dim = 7,168 attention dim. Full bidirectional attention, no cross-attention.
- **FFN:** SwiGLU.
- **Patch size:** `(1, 2, 2)` (1 temporal, 2 height, 2 width).
- **Patch ordering:** Video patch values are **channel-major, patch-minor**. (Critical: getting this wrong silently scrambles every token).
- **Latent channels:** Video = 24 latent channels; Audio = 32 latent channels.
- **Single packed sequence:** `[text tokens | conditioning visual tokens | target video tokens | target audio tokens]`.

### Dual Flow Schedules
Unlike single-modality diffusion models, H3 uses independent flow shift parameters and dual sigma schedules:
- **Video flow shift:** 12.
- **Audio flow shift:** 3.
- Two independent sigma trajectories driving the unified denoiser step.

### Guidance & CFG
- Distilled model: standard inference uses `cfg_scale = 1.0`.
- No unconditional branch / negative-prompt pass in the base path.

### DiT Conditioning (AdaLN Table)
- Not a standard scalar timestep MLP: uses a learned timestep curve table `adaln_t_table = [8, 1025]`.
- Continuous interpolation at timestep $t$.
- Per-block projection from 8 -> 96,768 (yielding 6 modulation vectors × 3 modalities: $6 \times 5376 = 32256$ per modality, $32256 \times 3 = 96768$).

### Video VAE Temporal Chunking
- 3D causal VAE encoder/decoder with spatial tiling.
- **Temporal chunking is correctness-sensitive:** clips past 22 frames are decoded in chunks of 5 latent frames with overlap/look-ahead and seam cross-fading.

### Conditioning Modes (Staged)
Two denoiser checkpoints support different conditioning:
- `fl2va`: text + first/last keyframes; also t2v and i2v.
- `ref2va`: text + multimodal references (up to 9 images, 3 video clips, 3 audio clips).

**Sequencing:**
- M1: T2V (Text to Video + Audio)
- M2: I2V (Image to Video + Audio)
- M3: FL2V (First & Last Frame to Video + Audio)
- M4: Ref2VA images
- M5: Ref2VA video/audio references

**Files** (about 35.5 GB per set):

| File | Size |
|---|---|
| denoiser GGUF (Q4_K) | ~10.6 GiB |
| `qwen3vl_32b_minimax_h3` text encoder | 17 GiB Q4_K_M, or 12.2 GiB Q2_K |
| video VAE (fp16 safetensors) | 5.2 GB |
| audio VAE (fp32) | 0.6 GB |

**Memory:** Peak working-set target is approximately the largest active network plus its conditioning/output buffers (~17 GiB); sequential loading allows fitting comfortably within a 64 GB host without keeping all networks resident simultaneously.

**Out of Scope for Initial Port:**
- Initial scope excludes MiniMax's proprietary hosted `H3-Context-IR` orchestration system. Local port accepts already-authored prompts and references directly.

## Packed Sequence Layout (`MiniMaxH3Layout`)

The central engineering abstraction is `MiniMaxH3Layout`:
- Modality token partition and range indexing: `[text | visual_cond | video_target | audio_target]`.
- Modality-specific spatial, temporal, and frequency RoPE coordinate indexing.
- Mask generation: conditioning masks vs target/noise masks.
- Extraction of predicted velocity outputs split into video latent and audio latent slices.

## Implementation Phases

- [x] **0. Mathematical Primitives & Dual Schedulers:**
  - `MiniMaxH3Scheduler`: dual flow shifts (video=12, audio=3), independent sigma generation, Euler solver step.
  - `MiniMaxH3AdaLN`: `adaln_t_table = [8, 1025]` interpolation and 8 -> 96768 block modulation vector extraction.
  - `MiniMaxH3Layout`: token layout, coordinate packing, RoPE coordinate generation, post-denoise slice extraction.
- [ ] **1. Text Encoder Integration:**
  - Hidden-state extraction from Qwen3-VL-32B GGUF with external `vocab.json` + `merges.txt`.
- [ ] **2. DiT Denoiser:**
  - 2a. Tiny DiT with video only.
  - 2b. Tiny DiT with audio only.
  - [x] 2c. Packed video+audio sequence (`MiniMaxH3Layout` + `MiniMaxH3DiTBlock`).
  - [ ] 2d. Multimodal RoPE / positional layout.
  - [x] 2e. AdaLN modulation ($8 \to 96768$).
  - [ ] 2f. Full 50-block DiT graph with real weight loader.
- [ ] **3. VAEs:**
  - Audio VAE decode / encode (fp32).
  - Video VAE 3D decode with spatial tiling and 5-latent-frame temporal chunking.
- [ ] **4. Pipeline & Denoising Loop:**
  - Denoising orchestration (t2v first).
- [ ] **5. Host Output:**
  - WAV export for audio latent decode.
  - Frame export / video container writer for decoded video frames.
- [ ] **6. Experimental Gate:**
  - Gated under experimental flag; unadvertised in CLI/STATUS until verified (Rule 14).

## Verification Ladder

1. **Exact Shape & Layout Validation:** Synthetic tests for `MiniMaxH3Layout`, coordinate bounds, and patch channel ordering.
2. **Synthetic Component Tests:**
   - Dual flow schedule values for video ($shift=12$) and audio ($shift=3$).
   - AdaLN curve table interpolation and modulation projections.
   - Isolated single DiT block forward pass with packed video+audio tokens.
   - Audio VAE decode test.
   - Video VAE 3D temporal chunking test (<=22 frames vs >22 frames).
3. **Synthetic End-to-End Tiny H3:** Video + audio packed denoising step with dummy weights.
4. **Real Checkpoint Fixture Parity (Level 3):**
   - Intermediate text embedding against upstream PyTorch / TensorSharp fixture.
   - Single DiT velocity step comparison on fixed seed.
   - Decoded video and audio latents.
5. **Long-clip VAE verification:** Seam blending on >22 frame latent sequence.
6. **Conditioning Extensions:** I2V, FL2V, Ref2VA.

**Target:** 3–5 days for a structurally complete CPU T2V implementation, assuming existing VAE/DiT primitives map cleanly. I2V/FL2V/Ref2VA and Vulkan are subsequent work.
