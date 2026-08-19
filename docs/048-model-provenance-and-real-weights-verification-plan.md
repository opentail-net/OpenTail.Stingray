# Model Provenance & Real-Weights Verification Plan

**Target:** `opentail-net/OpenTail.Stingray`  
**Purpose:** Honest, rigorous inventory and tracking checklist of all model architectures, formats, and pipelines claimed by OpenTail.Stingray.

---

## Verification Levels Explained

- **Level 1: Container & Metadata Integrity (Smoke Test)**  
  The real binary file exists on disk, magic bytes (`GGUF` / `Safetensors` / `GGML`) parse correctly, tensor dictionaries are populated, and quantization formats (Q4_K, Q8_0, FP16) are validated without throwing exceptions.
- **Level 2: End-to-End Real-Weight Inference (Functional)**  
  Tensors are mapped directly into the model's neural layers (Attention, Projections, Conformer, Vocoder), inputs are processed, real matrix operations execute, and genuine text, audio, or embeddings are generated.

---

## 1. Executive Summary & Verification Matrix

| Domain | Architecture / Model Family | Format | Real Weights File on Disk | Level 1: Container Validated | Level 2: End-to-End Wired & Inferenced | Current Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **ASR (STT)** | OpenAI Whisper (Tiny/Base/Small/Med) | GGML | ✅ `ggml-tiny.bin`, `base`, `small`, `medium` | ✅ Passed | ✅ **PROVEN (Real ASR Output + WhisperPipeline.Load API)** | **Level 2 (Fully Proven)** |
| **ASR (STT)** | Alibaba FunASR / Paraformer (0.2B) | GGUF / ONNX | ✅ `paraformer-q8.gguf` (225MB), `paraformer-zh-small.int8.onnx` | ✅ Passed | ✅ **PROVEN (Real ASR Output)** | **Level 2 (Fully Proven)** |
| **ASR (STT)** | Alibaba SenseVoice | ONNX | ✅ `sensevoice-small.int8.onnx` (239MB) | ✅ Passed | ✅ **PROVEN (Real ASR Output)** | **Level 2 (Fully Proven)** |
| **TTS** | Kokoro-82M | GGUF | ✅ `kokoro-82m-q8_0.gguf` (135MB), `kokoro-voice-af_heart.gguf` | ✅ Passed | ✅ **PROVEN (Real 24kHz Audio Generated)** | **Level 2 (Fully Proven)** |
| **Embeddings** | all-MiniLM-L6-v2 | GGUF | ✅ `all-MiniLM-L6-v2-Q8_0.gguf` (23.8MB) | ✅ Passed | ✅ **PROVEN (Dense Vectors Generated)** | **Level 2 (Fully Proven)** |
| **Embeddings** | BGE-Small-EN-v1.5 | GGUF | ✅ `bge-small-en-v1.5-q8_0.gguf` (34.9MB) | ✅ Passed | ✅ **PROVEN (Dense Vectors Generated)** | **Level 2 (Fully Proven)** |
| **ASR (STT)** | Alibaba Qwen3-ASR (0.6B) | GGUF | ✅ `qwen3-asr-0.6b-q4_k.gguf` | ✅ Passed | 🔲 Scaffolding | Level 1 (Weights on disk) |
| **ASR (STT)** | Alibaba Qwen3-ForcedAligner (0.6B) | Safetensors | ✅ `qwen3-forcedaligner-0.6b.safetensors` | ✅ Passed | 🔲 Scaffolding | Level 1 (Weights on disk) |
| **ASR (STT)** | NVIDIA Parakeet CTC (0.6B) | GGUF | ✅ `parakeet-ctc-0.6b-q4_k.gguf` | ✅ Passed | 🔲 Scaffolding | Level 1 (Weights on disk) |
| **VAD** | Silero VAD (v4 / v5 RNN) | ONNX | ✅ `silero_vad.onnx` | ✅ Passed | 🔲 Scaffolding | Level 1 (Weights on disk) |
| **TTS** | Piper VITS | ONNX | ✅ `en_US-lessac-medium.onnx` | ✅ Passed | 🔲 Scaffolding | Level 1 (Weights on disk) |
| **TTS** | MeloTTS | ONNX | ✅ `melotts-zh_en.onnx` | ✅ Passed | 🔲 Scaffolding | Level 1 (Weights on disk) |
| **TTS** | Chatterbox TTS | ONNX | ✅ `chatterbox-speech_encoder.onnx` | ✅ Passed | 🔲 Scaffolding | Level 1 (Weights on disk) |
| **TTS** | CosyVoice 300M / 2.0 | Safetensors / ONNX | ✅ `cosyvoice2_0.5b.safetensors`, ONNX flow models | ✅ Passed | 🔲 Scaffolding | Level 1 (Weights on disk) |
| **TTS** | F5-TTS | Safetensors | ✅ `f5tts_base.safetensors` | ✅ Passed | 🔲 Scaffolding | Level 1 (Weights on disk) |
| **Embeddings** | BGE-Base / Large | ONNX | ✅ `bge-base/large-en-v1.5_quantized.onnx` | ✅ Passed | 🔲 Scaffolding | Level 1 (Weights on disk) |
| **LLM** | SmolLM2 (135M / 360M / 1.7B) | GGUF | ✅ `SmolLM2-135M/360M/1.7B-Instruct-Q4_K_M.gguf` | ✅ Passed | 🔲 Core test suite | Level 1 / Partial 2 |
| **LLM** | Qwen2.5 (0.5B / 1.5B / 3B) | GGUF | ✅ `qwen2.5-0.5b/1.5b/3b-instruct-q4_k_m.gguf` | ✅ Passed | 🔲 Core test suite | Level 1 / Partial 2 |
| **LLM** | Qwen2.5-Coder (0.5B / 1.5B / 3B) | GGUF | ✅ `qwen2.5-coder-0.5b/1.5b/3b-instruct-q4_k_m.gguf` | ✅ Passed | 🔲 Core test suite | Level 1 / Partial 2 |
| **LLM** | DSpark Block7 Speculative | Safetensors | ✅ `dspark_qwen3_4b_block7` | ✅ Passed | 🔲 Scaffolding | Level 1 (Weights on disk) |
| **Vision** | MiniCPM-V 2.6 Vision Projector | GGUF | ✅ `mmproj-minicpm-v-2_6-f16.gguf` | ✅ Passed | 🔲 Vision test suite | Level 1 (Weights on disk) |
| **Vision** | Llama 4 Scout ViT Projector | GGUF | ✅ `mmproj-llama-4-scout-17b-16e-instruct-f16.gguf` | ✅ Passed | 🔲 Vision test suite | Level 1 (Weights on disk) |
| **Vision** | Gemma 4 (12B & E4B) Projectors | GGUF | ✅ `mmproj-gemma-4-12b-it-qat-q4_0.gguf`, `E4B` | ✅ Passed | 🔲 Vision test suite | Level 1 (Weights on disk) |
| **Video Gen** | LTX-Video (2B DiT) | Safetensors | ✅ `ltx-video-2b-v0.9.1.safetensors` | ✅ Passed | 🔲 Scaffolding | Level 1 (Weights on disk) |
| **Video Gen** | Wan Video 2.1 (1.3B DiT) | GGUF | ✅ `Wan2.1-T2V-1.3B-Q4_0.gguf` | ✅ Passed | 🔲 Scaffolding | Level 1 (Weights on disk) |
| **Video Gen** | Hunyuan Video (FP8 DiT) | Safetensors | ✅ `hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors` | ✅ Passed | 🔲 Scaffolding | Level 1 (Weights on disk) |
| **Diffusion** | Stable Diffusion 1.5 & SDXL Turbo | Safetensors | ✅ `v1-5-pruned-emaonly.safetensors`, `sd_xl_turbo` | ✅ Passed | 🔲 Vulkan pipeline | Level 1 (Weights on disk) |
| **Upscaling** | Real-ESRGAN x4plus | Safetensors | ✅ `RealESRGAN_x4plus.safetensors` | ✅ Passed | 🔲 Scaffolding | Level 1 (Weights on disk) |
