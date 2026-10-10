# Model Architecture Cards

> **Purpose:** Detailed architectural reference for supported model families in OpenTail.Stingray.  
> **Audience:** Contributors, performance engineers, and developers integrating specific model families.  
> **Governing Rule:** Every card documents how Stingray's managed C# engine *actually executes* the architecture, including tensor layouts, custom SIMD kernels, RoPE parameters, memory footprints, and verified golden status.

---

## 1. Flagship Core LLM Families

| Family | Architectures | Key Highlights | Architecture Card |
|---|---|---|---|
| **Llama** | Llama 3, 3.1, 3.2, 3.3, Llama 4 | GQA, RMSNorm, 128k YaRN RoPE, Llama 4 L2 QK-Norm & NoPE 4-step | [llama.md](llama.md) |
| **Gemma** | Gemma 1, 2, 3, Gemma 4 / E4B | RMSNorm unit offset, dual residual norms, logit soft-capping, SigLIP vision | [gemma.md](gemma.md) |
| **Mistral & Mixtral** | Mistral 7B, Mistral 3, Mixtral 8x7B/8x22B, Ministral | Sliding Window Attention (SWA), Tekken tokenizer, top-2 sparse MoE | [mistral.md](mistral.md) |
| **Microsoft Phi** | Phi-2, Phi-3, Phi-3.5, PhiMoE | LongRoPE dynamic factors, RMSNorm+bias, 16x3.8B MoE with 2 active experts | [phi.md](phi.md) |
| **IBM Granite** | Granite 3.x, Granite MoE, Granite 4.0-H | Residual/logit scale trio, Mamba-2 SSM hybrid recurrence without KV cache | [granite.md](granite.md) |
| **SmolLM** | SmolLM2 (135M, 360M, 1.7B), SmolLM3 | 2-stage digit tokenizer, SpinParkWorkerPool SIMD tuning, SmolLM3 NoPE | [smollm.md](smollm.md) |

---

## 2. Recurrent, MoE & Hybrid Families

| Family | Architectures | Key Highlights | Architecture Card |
|---|---|---|---|
| **Qwen Series** | Qwen 2.5, Qwen 3, Qwen 3.5 MoE, Qwen-Coder | Default catalogue model (`0.5b`), weighted QK-norm, Gated DeltaNet MoE | [qwen.md](qwen.md) / [qwen-series.md](qwen-series.md) |
| **DeepSeek** | DeepSeek-V2, V2.5, V3, R1 | Multi-Head Latent Attention (MLA, 3.5× KV reduction), DeepSeekMoE, thinking traces | [deepseek.md](deepseek.md) |
| **RWKV** | RWKV-6 Finch, RWKV-7 Goose | Zero KV cache overhead, $O(1)$ constant RAM, matrix decay state evolution | [rwkv.md](rwkv.md) |
| **Liquid LFM2** | LFM2-1.2B, LFM2-MoE 8B-A1B | Gated short-convolution mixer layers + causal depthwise 1D conv | [lfm2.md](lfm2.md) |
| **Allen AI OLMoE** | OLMoE-1B-7B | 64 fine-grained experts (8 active), whole-vector per-channel QK-norm | [olmoe.md](olmoe.md) |

---

## 3. Multimodal Vision & Document OCR

| Family | Projector Type | Vision Backbone & Features | Architecture Card |
|---|---|---|---|
| **Overview (11+ Towers)** | Multi-Architecture | Unified Vision Pipeline overview, token injection, dynamic patch grids | [multimodal-vision.md](multimodal-vision.md) |
| **Qwen-VL** | `qwen2vl`, `qwen3vl` | 3D Conv stem, 3D M-RoPE, $2\times 2$ spatial merge, intermediate DeepStack slices | [qwen-vl.md](qwen-vl.md) |
| **Pixtral** | `pixtral` | 2D continuous RoPE, SwiGLU vision MLP, arbitrary resolution budgeting | [pixtral.md](pixtral.md) |
| **DeepSeek-OCR** | `deepseek2_ocr` | Dual SAM ViT + CLIP ViT fusion, $1024\times 1024$ multi-scale high-density grid | [deepseek-ocr.md](deepseek-ocr.md) |
| **LLaVA** | `mlp` (LLaVA-1.5, NeXT, OneVision) | CLIP ViT-L/14 & SigLIP SO400M, 2-layer GELU MLP, AnyRes dynamic multi-crop | [llava.md](llava.md) |
| **InternVL** | `internvl` (2.5, 3, 4) | InternViT-6B, PixelShuffle $2\times 2$ downsampling, $N \times 448\times 448$ patch slicing | [internvl.md](internvl.md) |
| **Gemma 4 Vision** | `gemma4uv`, `gemma4v`, `gemma3` | Unified Vision (UV) shared scale embedding, SigLIP SO400M alignment | [gemma4-vision.md](gemma4-vision.md) |

---

## 4. Studio Audio & Speech Pipelines

| Pipeline | Model Architecture | Key Highlights | Architecture Card |
|---|---|---|---|
| **Overview (Studio Audio)** | Full Audio Stack | Piper, Whisper, Kokoro, Paraformer, ATSC/ITU DSP, rational resamplers | [audio-pipelines.md](audio-pipelines.md) |
| **OpenAI Whisper** | Transformer Enc-Dec | Tiny through Large-v3/Turbo, Log-Mel FFTs, word-level timestamps | [whisper.md](whisper.md) |
| **Piper VITS** | Neural VITS + HiFi-GAN | 🥇 0.129× RTF (~320 ms per sentence), default speech catalogue bundle | [piper.md](piper.md) |
| **Kokoro-82M** | Compact StyleTTS 2 | 24 kHz studio voice synthesis, 82M parameters, swappable speaker vectors | [kokoro.md](kokoro.md) |
| **Coqui XTTS-v2** | GPT-2 Codec + HiFi-GAN | Zero-shot voice cloning from 3–6s reference audio, 13 golden stages | [xtts2.md](xtts2.md) |
| **Qwen Audio** | Qwen3-TTS & Qwen3-ASR | 12Hz discrete audio codec, ERes2NetV2 192-dim speaker cloning, ForcedAligner | [qwen-audio.md](qwen-audio.md) |

---

## 5. Diffusion, Video & Embeddings

| Pipeline | Model Architecture | Key Highlights | Architecture Card |
|---|---|---|---|
| **Overview (Diffusion)** | Multi-Modality Media | Tier classification, Euler flow-matching schedulers, tiled VAE decoding | [diffusion-pipelines.md](diffusion-pipelines.md) |
| **FLUX** | FLUX.1 schnell/dev, FLUX.2 | 12B Rectified Flow DiT, joint text-image attention (MMDiT), 4-step generation | [flux.md](flux.md) |
| **Stable Diffusion** | SD 1.5, SDXL, SD 3/3.5 | Classic UNet 2D, Dual-CLIP micro-conditioning, MMDiT triple encoders | [stable-diffusion.md](stable-diffusion.md) |
| **Z-Image-Turbo** | Distilled DiT | Ultra-fast 4-step high-detail 512×512 generation, Qwen text conditioning | [z-image-turbo.md](z-image-turbo.md) |
| **Wan Video** | 3D Spatio-Temporal DiT | Text-to-video / image-to-video, 3D causal VAE, native animated GIF writer | [wan-video.md](wan-video.md) |
| **Text Encoders & Embeddings** | Bidirectional Transformers | BERT, MiniLM, BGE, E5, NomicBERT, BGE-M3 (dense/sparse/ColBERT), T5 | [embeddings-encoders.md](embeddings-encoders.md) |

---

## 6. Verification Criteria

Each architecture card cross-references:
1. Active entry and test evidence in [docs/STATUS.md](../STATUS.md).
2. Measured memory and speed in [docs/RUNNING.md](../RUNNING.md).
3. Associated forward-pass test fixtures in `tests/OpenTail.Stingray.Tests.ForwardPass.Fast`.
4. Standing rules in [CLAUDE.md](../../CLAUDE.md) (Rule 14: ported, unverified families stay internal).
