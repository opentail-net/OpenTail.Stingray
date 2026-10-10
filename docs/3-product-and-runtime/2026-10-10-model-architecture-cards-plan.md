# Model Architecture Cards Expansion Plan

**Status:** In Progress (2026-10-10)  
**Goal:** Expand and elevate `docs/models/` into a first-class, exhaustive library of architecture cards covering every model family admitted in OpenTail.Stingray across Text LLMs, Sparse MoE, Multimodal Vision, Neural Speech/Audio, and Diffusion/Video.

---

## 1. Motivation & Benchmark

Previously, `docs/models/` contained only 5 consolidated rollup summaries (`qwen-series.md`, `deepseek.md`, `multimodal-vision.md`, `audio-pipelines.md`, `diffusion-pipelines.md`). While informative, this forced distinct architectures (e.g., Llama vs Mistral vs SmolLM; Whisper vs Kokoro vs XTTS; SDXL vs FLUX vs Wan) into single generic pages.

In contrast, leading engines provide standalone, deep-dive architecture cards per model family. For OpenTail.Stingray, these cards serve as the definitive technical guide for developers, detailing:
- Exact GGUF keys, tensor naming patterns, and relabelling recognition.
- Pure managed C# source classes in `OpenTail.Stingray`.
- RoPE scaling, attention quirks, and tokenizer specifications.
- Memory budgets, CPU/GPU placement, and verified benchmarks.
- Runnable C# and CLI quickstarts.

---

## 2. Standard Architecture Card Schema

Each document in `docs/models/*.md` follows this standardized specification:

1. **Header & Metadata Matrix**:
   - Provider / Organization
   - GGUF architecture key (`ArchitectureRegistry`) and recognized aliases
   - C# engine implementation classes (in `OpenTail.Stingray.Engine`, `Vision`, `Audio`, or `Diffusion`)
   - Supported modalities (Text, Vision, Audio, Image generation)
   - Feature capabilities (Thinking mode, Tool calling, JSON schema BNF masks, Speculative decoding)
   - Verification status & confidence tier (cross-referenced to `docs/STATUS.md`)
2. **Verified Checkpoints & Hugging Face Pointers**:
   - Canonical repo, revision, file names, quant tiers (e.g. Q4_K_M, Q8_0)
   - Shard handling (multi-file GGUF resolution)
3. **Architecture & Tensor Graph Deep Dive**:
   - Normalization (RMSNorm, LayerNorm, Norm eps)
   - Activation function (SwiGLU, GeLU, Squared ReLU)
   - Position embeddings (RoPE, M-RoPE, YaRN, linear scaling)
   - Attention variant (MHA, GQA, MLA, Sliding-Window Attention)
4. **Hardware, Memory & Performance Profile**:
   - Working-set formula (weights + repacked copy + KV cache + context headroom)
   - Measured speed on reference hardware (from `docs/RUNNING.md`)
   - GPU offload recommendations (Vulkan vs CUDA vs CPU SIMD)
5. **Code & CLI Quickstart**:
   - High-level C# snippet (`Model.Load` / `ChatSession` / `Pipeline`)
   - CLI execution command (`stingray chat`, `stingray -m`, `stingray speak`)

---

## 3. Coverage Roadmap

### Phase 1: Flagship Core LLM Families
- [x] `docs/models/llama.md`: Llama 3 / 3.1 / 3.2 / 3.3 / Llama 4
- [x] `docs/models/gemma.md`: Gemma 2 / 3 / 4
- [x] `docs/models/mistral.md`: Mistral 7B / Mistral 3 / Mixtral MoE / Ministral
- [x] `docs/models/phi.md`: Microsoft Phi-2 / Phi-3 / Phi-3.5 / PhiMoE
- [x] `docs/models/granite.md`: IBM Granite 3.x / Granite 4.0-H (hybrid recurrent)
- [x] `docs/models/smollm.md`: SmolLM2 (135M, 360M, 1.7B) / SmolLM3

### Phase 2: Recurrent, MoE & Specialized Families
- [x] `docs/models/qwen.md`: Qwen 2.5 / Qwen 3 / Qwen 3.5 MoE / Coder
- [x] `docs/models/deepseek.md`: DeepSeek-V2 / V3 / R1 (MLA latent attention & MoE)
- [x] `docs/models/rwkv.md`: RWKV-6 / RWKV-7 linear attention / RNN
- [x] `docs/models/lfm2.md`: Liquid LFM2 Conv/SSM hybrid
- [x] `docs/models/olmoe.md`: OLMoE 1B-7B 64-expert sparse MoE

### Phase 3: Multimodal Vision & Document OCR
- [x] `docs/models/qwen-vl.md`: Qwen2.5-VL / Qwen3-VL
- [x] `docs/models/pixtral.md`: Mistral Pixtral 12B
- [x] `docs/models/deepseek-ocr.md`: DeepSeek-OCR & DeepSeek-OCR2 SAM+CLIP ViT fusion
- [x] `docs/models/llava.md`: LLaVA 1.5 / NeXT / OneVision
- [x] `docs/models/internvl.md`: InternVL 2.5 / 3 / 4
- [x] `docs/models/gemma4-vision.md`: Gemma 4 UV & ViT

### Phase 4: Speech & Audio Pipelines
- [x] `docs/models/whisper.md`: OpenAI Whisper Tiny to Large-v3 / Turbo
- [x] `docs/models/piper.md`: Piper VITS neural text-to-speech
- [x] `docs/models/kokoro.md`: Kokoro-82M neural TTS
- [x] `docs/models/xtts2.md`: Coqui XTTS-v2 zero-shot voice cloning
- [x] `docs/models/qwen-audio.md`: Qwen3-TTS & Qwen3-ASR

### Phase 5: Diffusion, Video & Embeddings
- [x] `docs/models/flux.md`: FLUX.1 (schnell/dev) & FLUX.2
- [x] `docs/models/stable-diffusion.md`: Stable Diffusion 1.5 / SDXL / SD 3.5 MMDiT
- [x] `docs/models/z-image-turbo.md`: Z-Image-Turbo fast DiT
- [x] `docs/models/wan-video.md`: Wan 2.1 / 2.2 Video DiT
- [x] `docs/models/embeddings-encoders.md`: BGE, MiniLM, E5, NomicBERT, BGE-M3

### Phase 6: Index Overhaul & Cross-Linking
- [x] `docs/models/README.md`: Overhaul model index table with direct links, status badges, and quick summaries.
