# Model Provenance & Real-Weights Verification Plan

**Target:** `opentail-net/OpenTail.Stingray`  
**Purpose:** Honest, rigorous inventory, tracking checklist, and architecture map of all model formats, weight files, and pipelines in OpenTail.Stingray.

---

## Verification Levels Explained

- **Level 1: Container & Metadata Integrity (Smoke Test)**  
  The real binary file exists on disk, magic bytes (`GGUF` / `Safetensors` / `GGML` / `ONNX`) parse correctly, tensor dictionaries are populated, and quantization formats (Q4_K, Q8_0, FP16) are validated without throwing exceptions.
- **Level 2: End-to-End Real-Weight Inference (Functional)**  
  Tensors are mapped directly into the model's neural layers (Attention, Projections, Conformer, Vocoder, LSTM, FeedForward, UNet, DiT, Causal LM), inputs are processed, real matrix operations execute, and genuine text, audio, image, or embeddings are generated.

---

## 1. Executive Summary & Verification Matrix

| Domain | Architecture / Model Family | Format | Real Weights File on Disk | Level 1: Container Validated | Level 2: End-to-End Wired & Inferenced | Current Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Vision-Language** | Alibaba Qwen2.5-VL (3B / 7B ViT) | GGUF | ✅ `mmproj-qwen2.5-vl-7b-f16.gguf` (1.29GB) | ✅ Passed | ✅ **PROVEN (Real M-RoPE ViT + Spatial 2x2 Merge + UnifiedVisionPipeline API)** | **Level 2 (Fully Proven)** |
| **LLM (Text)** | SmolLM2 (135M / 360M / 1.7B) | GGUF | ✅ `SmolLM2-135M/360M/1.7B-Instruct-Q4_K_M.gguf` | ✅ Passed | ✅ **PROVEN (Real Prefill + Greedy Decode Loop)** | **Level 2 (Fully Proven)** |
| **LLM (Text)** | Qwen2.5 (0.5B / 1.5B / 3B) | GGUF | ✅ `qwen2.5-0.5b/1.5b/3b-instruct-q4_k_m.gguf` | ✅ Passed | ✅ **PROVEN (Real Prefill + Greedy Decode Loop)** | **Level 2 (Fully Proven)** |
| **LLM (Text)** | Qwen2.5-Coder (0.5B / 1.5B / 3B) | GGUF | ✅ `qwen2.5-coder-0.5b/1.5b/3b-instruct-q4_k_m.gguf` | ✅ Passed | ✅ **PROVEN (Real Prefill + Greedy Code Generation)** | **Level 2 (Fully Proven)** |
| **ASR (STT)** | OpenAI Whisper (Tiny/Base/Small/Med) | GGML | ✅ `ggml-tiny.bin`, `base`, `small`, `medium` | ✅ Passed | ✅ **PROVEN (Real ASR Output + WhisperPipeline.Load API)** | **Level 2 (Fully Proven)** |
| **ASR (STT)** | Alibaba Qwen3-ASR (0.6B / 1.7B) | GGUF | ✅ `qwen3-asr-0.6b-q4_k.gguf` | ✅ Passed | ✅ **PROVEN (Real ASR Output + QwenAsrPipeline.Load API)** | **Level 2 (Fully Proven)** |
| **ASR (STT)** | Alibaba Qwen3-ForcedAligner (0.6B) | Safetensors | ✅ `qwen3-forcedaligner-0.6b.safetensors` | ✅ Passed | ✅ **PROVEN (Real DTW Word Alignment + QwenAsrForcedAligner.Load API)** | **Level 2 (Fully Proven)** |
| **ASR (STT)** | Alibaba FunASR / Paraformer (0.2B) | GGUF / ONNX | ✅ `paraformer-q8.gguf` (225MB), `paraformer-zh-small.int8.onnx` | ✅ Passed | ✅ **PROVEN (Real ASR Output)** | **Level 2 (Fully Proven)** |
| **ASR (STT)** | Alibaba SenseVoice | ONNX | ✅ `sensevoice-small.int8.onnx` (239MB) | ✅ Passed | ✅ **PROVEN (Real ASR Output)** | **Level 2 (Fully Proven)** |
| **ASR (STT)** | NVIDIA Parakeet FastConformer CTC (0.6B) | GGUF | ✅ `parakeet-ctc-0.6b-q4_k.gguf` (366MB) | ✅ Passed | ✅ **PROVEN (Real ASR Output + ParakeetPipeline.Load API)** | **Level 2 (Fully Proven)** |
| **VAD** | Silero VAD (v4 / v5 RNN) | GGUF / ONNX | ✅ `silero_vad.gguf` (2.2MB), `silero_vad.onnx` | ✅ Passed | ✅ **PROVEN (Real Speech/Silence Gating + SileroVad.Load API)** | **Level 2 (Fully Proven)** |
| **TTS** | Kokoro-82M | GGUF | ✅ `kokoro-82m-q8_0.gguf` (135MB), `kokoro-voice-af_heart.gguf` | ✅ Passed | ✅ **PROVEN (Real 24kHz Audio Generated)** | **Level 2 (Fully Proven)** |
| **TTS** | Chatterbox-Turbo (T3 & S3Gen) | GGUF | ✅ `chatterbox-turbo-t3-q4_k.gguf` (457MB), `chatterbox-turbo-s3gen-q4_k.gguf` (244MB) | ✅ Passed | ✅ **PROVEN (Real 24kHz Audio Generated + ChatterboxPipeline.Load API)** | **Level 2 (Fully Proven)** |
| **TTS** | F5-TTS (Non-Autoregressive DiT) | Safetensors | ✅ `f5tts_base.safetensors` (336MB) | ✅ Passed | ✅ **PROVEN (Real 24kHz Audio Generated + F5TtsPipeline.Load API)** | **Level 2 (Fully Proven)** |
| **TTS** | CosyVoice 300M / 2.0 | Safetensors / ONNX | ✅ `cosyvoice2_0.5b.safetensors`, `cosyvoice_speech_tokenizer.onnx` | ✅ Passed | ✅ **PROVEN (Real 24kHz Audio Generated + CosyVoicePipeline.Load API)** | **Level 2 (Fully Proven)** |
| **TTS** | Piper VITS | ONNX | ✅ `en_US-lessac-medium.onnx` | ✅ Passed | ✅ **PROVEN (Real 22kHz Audio Generated + PiperPipeline.FromConfigFile API)** | **Level 2 (Fully Proven)** |
| **TTS** | MeloTTS | ONNX | ✅ `melotts-zh_en.onnx` (162MB) | ✅ Passed | ✅ **PROVEN (Real 44.1kHz Audio Generated + MeloPipeline.Load API)** | **Level 2 (Fully Proven)** |
| **Embeddings** | all-MiniLM-L6-v2 | GGUF | ✅ `all-MiniLM-L6-v2-Q8_0.gguf` (23.8MB) | ✅ Passed | ✅ **PROVEN (Dense Vectors Generated)** | **Level 2 (Fully Proven)** |
| **Embeddings** | BGE-Small-EN-v1.5 | GGUF | ✅ `bge-small-en-v1.5-q8_0.gguf` (34.9MB) | ✅ Passed | ✅ **PROVEN (Dense Vectors Generated)** | **Level 2 (Fully Proven)** |
| **Vision** | Gemma 4 E4B ViT Projector | GGUF | ✅ `gemma-4-E4B-it-mmproj.gguf` (110MB) | ✅ Passed | ✅ **PROVEN (Real ViT Embeddings + UnifiedVisionPipeline.Open API)** | **Level 2 (Fully Proven)** |
| **Vision** | Gemma 4 12B UV Projector | GGUF | ✅ `mmproj-gemma-4-12b-it-qat-q4_0.gguf` (230MB) | ✅ Passed | ✅ **PROVEN (Real Pixel Unroll + GemmaUvVisionEmbedder API)** | **Level 2 (Fully Proven)** |
| **Vision** | Gemma 3 SigLIP ViT Projector | GGUF | ✅ `mmproj-gemma-3-4b-it-f16.gguf` (800MB) | ✅ Passed | ✅ **PROVEN (Real 896x896 ViT Embeddings + UnifiedVisionPipeline.Open API)** | **Level 2 (Fully Proven)** |
| **Diffusion** | Stable Diffusion 1.5 | Safetensors | ✅ `v1-5-pruned-emaonly.safetensors` (4.06GB) | ✅ Passed | ✅ **PROVEN (Real UNet + VAE + CLIP + StableDiffusionPipeline.Load API)** | **Level 2 (Fully Proven)** |
| **Diffusion** | SDXL Turbo (1.0 FP16) | Safetensors | ✅ `sd_xl_turbo_1.0_fp16.safetensors` (6.61GB) | ✅ Passed | ✅ **PROVEN (Real SDXL UNet + VAE + Dual-CLIP + SdxlPipeline.Load API)** | **Level 2 (Fully Proven)** |
| **Embeddings** | BGE-Base / Large | ONNX | ✅ `bge-base/large-en-v1.5_quantized.onnx` | ✅ Passed | 🔲 Scaffolding | Level 1 (Weights on disk) |
| **LLM** | DSpark Block7 Speculative | Safetensors | ✅ `dspark_qwen3_4b_block7` | ✅ Passed | 🔲 Scaffolding | Level 1 (Weights on disk) |
| **Vision** | MiniCPM-V 2.6 Vision Projector | GGUF | ✅ `mmproj-minicpm-v-2_6-f16.gguf` | ✅ Passed | 🔲 Vision test suite | Level 1 (Weights on disk) |
| **Vision** | Llama 4 Scout ViT Projector | GGUF | ✅ `mmproj-llama-4-scout-17b-16e-instruct-f16.gguf` | ✅ Passed | 🔲 Vision test suite | Level 1 (Weights on disk) |
| **Video Gen** | LTX-Video (2B DiT) | Safetensors | ✅ `ltx-video-2b-v0.9.1.safetensors` | ✅ Passed | 🔲 Scaffolding | Level 1 (Weights on disk) |
| **Video Gen** | Wan Video 2.1 (1.3B DiT) | GGUF | ✅ `Wan2.1-T2V-1.3B-Q4_0.gguf` | ✅ Passed | 🔲 Scaffolding | Level 1 (Weights on disk) |
| **Video Gen** | Hunyuan Video (FP8 DiT) | Safetensors | ✅ `hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors` | ✅ Passed | 🔲 Scaffolding | Level 1 (Weights on disk) |
| **Upscaling** | Real-ESRGAN x4plus | Safetensors | ✅ `RealESRGAN_x4plus.safetensors` | ✅ Passed | 🔲 Scaffolding | Level 1 (Weights on disk) |

---

## 2. Detailed Technical Breakdown: Level 2 Proven Architectures

### A. Alibaba Qwen2.5-VL (Vision-Language)
- **Files Tested:** `mmproj-qwen2.5-vl-7b-f16.gguf` (1.29 GB).
- **Public API:** `UnifiedVisionPipeline.Open(string mmprojPath)`.
- **Implementation:** `src/OpenTail.Stingray.Vision/` (`QwenVlVisionModel.cs`, `QwenVlImagePreprocessor.cs`, `QwenVlVisionEncoder.cs`, `UnifiedVisionPipeline.cs`).
- **Verified In:** `OpenTail.Stingray.Tests.Vision/QwenVlVisionRealWeightsTests.cs`.
- **Details:** 14×14 Conv2D patch embedding projection, dynamic aspect ratio resolution snapped to multiples of 28, 4-section M-RoPE (multimodal rotary position embeddings), ViT transformer blocks with RMSNorm, and 2×2 spatial pixel merging with 2-layer MLP projection into 3584-dim Qwen text context.

### B. SmolLM2, Qwen2.5 & Qwen2.5-Coder (Causal LLM Text Engines)
- **Files Tested:** 
  - `SmolLM2-135M-Instruct-Q4_K_M.gguf` (100.57 MB), `SmolLM2-360M-Instruct-Q4_K_M.gguf` (258.06 MB), `SmolLM2-1.7B-Instruct-Q4_K_M.gguf` (1006.71 MB).
  - `qwen2.5-0.5b-instruct-q4_k_m.gguf` (468.64 MB), `qwen2.5-1.5b-instruct-q4_k_m.gguf` (1065.56 MB), `qwen2.5-3b-instruct-q4_k_m.gguf` (2007.42 MB).
  - `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` (468.64 MB), `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` (1065.56 MB), `qwen2.5-coder-3b-instruct-q4_k_m.gguf` (2007.42 MB).
- **Public API:** `InferenceEngine.GenerateAsync(string prompt, SamplingParams params)`.
- **Implementation:** `src/OpenTail.Stingray.Engine/ForwardPass.cs`, `InferenceEngine.cs`, `CpuBackend.cs`, `GgufTokenizer.cs`.
- **Verified In:** `OpenTail.Stingray.Tests.ForwardPass.Fast/SmolLm2RealWeightsTests.cs`, `Qwen25RealWeightsTests.cs`, `QwenCoderRealWeightsTests.cs`.
- **Details:** Full GGUF tensor ingestion, RoPE embeddings, Multi-Query / Grouped-Query Attention (GQA), SwiGLU feed-forward networks, RMSNorm pre- and post-attention, KV cache state persistence, prompt prefill matrix-vector multiplications, and autoregressive greedy decoding loops.

### C. Gemma Vision Family (Gemma 4 UV, Gemma 4 E4B ViT, Gemma 3 SigLIP)
- **Files Tested:** `mmproj-gemma-4-12b-it-qat-q4_0.gguf` (230MB), `gemma-4-E4B-it-mmproj.gguf` (110MB), `mmproj-gemma-3-4b-it-f16.gguf` (800MB).
- **Public API:** `UnifiedVisionPipeline.Open(string mmprojPath)`.
- **Implementation:** `src/OpenTail.Stingray.Vision/` (`UnifiedVisionPipeline.cs`, `Gemma4VVisionEncoder.cs`, `GemmaUvVisionEmbedder.cs`, `Gemma3VisionEncoder.cs`).
- **Verified In:** `OpenTail.Stingray.Tests.Vision/UnifiedVisionPipelineTests.cs`, `Gemma4VVisionEncoderTests.cs`.
- **Details:** 2D RoPE, per-head QK/V-norms, sandwich RMSNorms, spatial pooling, and linear projection into multimodal LLM embedding space.

### D. Stable Diffusion 1.5 & SDXL Turbo (Image Generation)
- **Files Tested:** `v1-5-pruned-emaonly.safetensors` (4.06 GB), `sd_xl_turbo_1.0_fp16.safetensors` (6.61 GB), `clip_tokenizer.json`.
- **Public API:** `StableDiffusionPipeline.Load(string modelPath)`, `SdxlPipeline.Load(string modelPath)`.
- **Implementation:** `src/OpenTail.Stingray.Diffusion/StableDiffusion/`, `src/OpenTail.Stingray.Diffusion/SDXL/`.
- **Verified In:** `OpenTail.Stingray.Tests.Diffusion/Sd15PipelineTests.cs`, `SdxlRealWeightsTests.cs`, `SdxlConformanceTests.cs`.
- **Details:** 
  - SD 1.5: 860M parameter Cross-Attention UNet (`model.diffusion_model.*`), OpenAI CLIP-L/14 text encoder (`cond_stage_model.transformer.*`), 4-channel VAE encoder/decoder (`first_stage_model.*`), Euler/Euler-A/DDIM samplers.
  - SDXL Turbo: Dual text conditioning (OpenAI CLIP ViT-L/14 + OpenCLIP ViT-bigG/14 concatenated to 2048 dim), 2816-dim micro-conditioning coordinate embeddings (original size, crop coordinates, target size), SDXL cross-attention UNet, 1-step rectified Euler flow trajectory.
