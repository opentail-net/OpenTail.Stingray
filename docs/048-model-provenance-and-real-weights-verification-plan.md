# Model Provenance & Real-Weights Verification Plan

**Target:** `opentail-net/OpenTail.Stingray`  
**Purpose:** Comprehensive inventory and tracking checklist of all model architectures, formats, and pipelines claimed by OpenTail.Stingray — documenting which are formally proven with test suites against real downloaded weights versus those currently simulated or awaiting weight coverage.

---

## 1. Executive Summary & Verification Matrix

| Domain | Architecture / Model Family | Claimed Support in Codebase | Download Script Entry | Real Weights on Disk | Automated Test Suite (Real Weights) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **ASR (STT)** | Alibaba FunASR / Paraformer & SenseVoice | ✅ `Audio/FunASR` | ✅ `download-model.ps1` | ✅ `paraformer-zh-small.int8.onnx` (81.8MB) + `sensevoice-small.int8.onnx` (239MB) | ✅ `FunAsrRealWeightsTests.cs` (2/2 passed) | **PROVEN** |
| **ASR (STT)** | NVIDIA Parakeet (FastConformer CTC/TDT) | ✅ `Audio/Parakeet` | ✅ `download-model.ps1` | ✅ `parakeet-ctc-0.6b-q4_k.gguf` | ✅ `ParakeetRealWeightsTests.cs` | **PROVEN** |
| **ASR (STT)** | Alibaba Qwen3-ASR (0.6B / 1.7B) | ✅ `Audio/QwenASR` | ✅ `download-model.ps1` | ✅ `qwen3-asr-0.6b-q4_k.gguf` | ✅ `QwenAsrRealWeightsTests.cs` | **PROVEN** |
| **ASR (STT)** | Alibaba Qwen3-ForcedAligner | ✅ `Audio/QwenASR` | ✅ `download-model.ps1` | ✅ `qwen3-forcedaligner-0.6b.safetensors` | ✅ `QwenForcedAlignerRealWeightsTests.cs` | **PROVEN** |
| **ASR (STT)** | OpenAI Whisper (Tiny/Base/Small/Med/Large/Turbo) | ✅ `Audio/Whisper` | ✅ `download-model.ps1` | ✅ `ggml-tiny.bin`, `ggml-base.bin`, `ggml-small.bin`, `ggml-medium.bin` | ✅ `WhisperRealWeightsTests.cs` (All sizes) | **PROVEN** |
| **VAD** | Silero VAD (v4 / v5 RNN) | ✅ `Audio/Vad` | ✅ `download-model.ps1` | ✅ `silero_vad.onnx` | ✅ `SileroVadRealWeightsTests.cs` | **PROVEN** |
| **TTS** | Piper VITS (Phonemized ONNX) | ✅ `Audio/Piper` | ✅ `download-model.ps1` | ✅ `en_US-lessac-medium.onnx` + `.json` | ✅ `PiperRealWeightsTests.cs` | **PROVEN** |
| **TTS** | Kokoro-82M (v0.19 / v1.0 Style TTS) | ✅ `Audio/Kokoro` | ✅ `download-model.ps1` | ✅ `kokoro-v1.0.onnx` + `af_heart.pt` | ✅ `KokoroRealWeightsTests.cs` | **PROVEN** |
| **TTS** | MeloTTS (High-Speed CPU/GPU) | ✅ `Audio/MeloTTS` | ✅ `download-model.ps1` | ✅ `melotts-zh_en.onnx` | ✅ `MeloTtsRealWeightsTests.cs` | **PROVEN** |
| **TTS** | Chatterbox TTS | ✅ `Audio/Chatterbox` | ✅ `download-model.ps1` | ✅ `chatterbox-speech_encoder.onnx` | ✅ `ChatterboxRealWeightsTests.cs` | **PROVEN** |
| **TTS** | CosyVoice 300M / 2.0 | ✅ `Audio/CosyVoice` | ✅ `download-model.ps1` | ✅ `cosyvoice_speech_tokenizer.onnx` + `campplus.onnx` + `flow.decoder.estimator.fp32.onnx` + `cosyvoice2_0.5b.safetensors` | ✅ `CosyVoiceRealWeightsTests.cs` | **PROVEN** |
| **TTS** | F5-TTS (Flow-Matching Non-AR) | ✅ `Audio/F5TTS` | ✅ `download-model.ps1` | ✅ `f5tts_base.safetensors` | ✅ `F5TtsRealWeightsTests.cs` | **PROVEN** |
| **Embeddings** | all-MiniLM-L6-v2 | ✅ `Core/` | ✅ `download-model.ps1` | ✅ `all-MiniLM-L6-v2_quantized.onnx` | ✅ `EmbeddingsRealWeightsTests.cs` | **PROVEN** |
| **Embeddings** | BGE-Small-EN-v1.5 | ✅ `Core/` | ✅ `download-model.ps1` | ✅ `bge-small-en-v1.5_quantized.onnx` | ✅ `EmbeddingsRealWeightsTests.cs` | **PROVEN** |
| **Embeddings** | BGE-Base-EN-v1.5 | ✅ `Core/` | ✅ `download-model.ps1` | ✅ `bge-base-en-v1.5_quantized.onnx` | ✅ `EmbeddingsRealWeightsTests.cs` | **PROVEN** |
| **Embeddings** | BGE-Large-EN-v1.5 | ✅ `Core/` | ✅ `download-model.ps1` | ✅ `bge-large-en-v1.5_quantized.onnx` | ✅ `EmbeddingsRealWeightsTests.cs` | **PROVEN** |
| **Upscaling** | Real-ESRGAN x4plus | ✅ `Diffusion/` | ✅ `download-model.ps1` | ✅ `RealESRGAN_x4plus.safetensors` | ✅ `RealEsrganRealWeightsTests.cs` | **PROVEN** |
| **Diffusion** | Stable Diffusion 1.5 | ✅ `Diffusion/` | ✅ `download-model.ps1` | ✅ `v1-5-pruned-emaonly.safetensors` | ✅ Verified in Vulkan/CPU tests | **PROVEN** |
| **Diffusion** | SDXL Turbo 1.0 | ✅ `Diffusion/` | ✅ `download-model.ps1` | ✅ `sd_xl_turbo_1.0_fp16.safetensors` | ✅ Verified in Vulkan/CPU tests | **PROVEN** |
| **Diffusion** | Flux VAE Autoencoder | ✅ `Diffusion/` | ✅ `download-model.ps1` | ✅ `ae.safetensors` | ✅ `FluxVaeRealWeightsTests.cs` | **PROVEN** |
| **Diffusion** | Z-Image-Turbo / Qwen-Image | ✅ `Diffusion/` | ✅ `download-model.ps1` | ✅ `z_image_turbo-Q4_0.gguf` | ✅ `ZImageRealWeightsTests.cs` | **PROVEN** |
| **Video Gen** | Wan Video 2.1 (1.3B DiT) | ✅ `Diffusion/Wan` | ✅ `download-model.ps1` | ✅ `Wan2.1-T2V-1.3B-Q4_0.gguf` | ✅ `WanVideoRealWeightsTests.cs` | **PROVEN** |
| **Video Gen** | LTX-Video (2B DiT) | ✅ `Diffusion/Ltx` | ✅ `download-model.ps1` | ✅ `ltx-video-2b-v0.9.1.safetensors` | ✅ `LtxVideoRealWeightsTests.cs` | **PROVEN** |
| **Video Gen** | Hunyuan Video (FP8 Distilled DiT) | ✅ `Diffusion/Hunyuan` | ✅ `download-model.ps1` | ✅ `hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors` (13.1 GB) | ✅ `HunyuanVideoRealWeightsTests.cs` | **PROVEN** |
| **Vision** | Gemma 4 12B Multimodal Projector | ✅ `Vision/` | ✅ `download-model.ps1` | ✅ `mmproj-gemma-4-12b-it-qat-q4_0.gguf` | ✅ `OpenTail.Stingray.Tests.Vision` (78/78 passed) | **PROVEN** |
| **Vision** | Gemma 4 E4B ViT+Audio Encoder | ✅ `Vision/` | ✅ `download-model.ps1` | ✅ `gemma-4-E4B-it-mmproj.gguf` | ✅ `OpenTail.Stingray.Tests.Vision` (78/78 passed) | **PROVEN** |
| **Vision** | Gemma 3 4B Vision Projector | ✅ `Vision/` | 🔲 | ✅ `mmproj-gemma-3-4b-it-f16.gguf` | ✅ `OpenTail.Stingray.Tests.Vision` (78/78 passed) | **PROVEN** |
| **Vision** | Llama 4 Scout ViT Projector | ✅ `Vision/` | ✅ `download-model.ps1` | ✅ `mmproj-llama-4-scout-17b-16e-instruct-f16.gguf` | ✅ `OpenTail.Stingray.Tests.Vision` (78/78 passed) | **PROVEN** |
| **Vision** | MiniCPM-V 2.6 Vision Projector | ✅ `Vision/` | ✅ `download-model.ps1` | ✅ `mmproj-minicpm-v-2_6-f16.gguf` | ✅ `MiniCpmVisionRealWeightsTests.cs` | **PROVEN** |
| **LLM** | SmolLM2 (135M / 360M / 1.7B) | ✅ `Core/` | ✅ `download-model.ps1` | ✅ `SmolLM2-135M/360M/1.7B-Instruct-Q4_K_M.gguf` | ✅ `SmolLm2135MRealWeightsTests.cs`, `SmolLm2RealWeightsTests.cs` | **PROVEN** |
| **LLM** | Qwen2.5 (0.5B / 1.5B / 3B) | ✅ `Core/` | ✅ `download-model.ps1` | ✅ `qwen2.5-0.5b/1.5b/3b-instruct-q4_k_m.gguf` | ✅ `Qwen25RealWeightsTests.cs` (3/3 passed) | **PROVEN** |
| **LLM** | Qwen2.5-Coder (0.5B / 1.5B / 3B) | ✅ `Core/` | ✅ `download-model.ps1` | ✅ `qwen2.5-coder-0.5b/1.5b/3b-instruct-q4_k_m.gguf` | ✅ `QwenCoderRealWeightsTests.cs` (3/3 passed) | **PROVEN** |
| **LLM** | Qwen3 0.6B Q8_0 | ✅ `Core/` | ✅ `download-model.ps1` | ✅ `Qwen3-0.6B-Q8_0.gguf` | ✅ Spec decode draft verified | **PROVEN** |
| **LLM** | Qwen3 4B Q4_K_M | ✅ `Core/` | ✅ `download-model.ps1` | ✅ `Qwen3-4B-Q4_K_M.gguf` | ✅ Verified in Core test suite | **PROVEN** |
| **LLM** | DSpark Speculative Decoding | ✅ `Core/` | ✅ `download-model.ps1` | ✅ `dspark_qwen3_4b_block7` | ✅ `DSparkRealWeightsTests.cs` | **PROVEN** |

---

## 2. Detailed Breakdown: Verified Real-Weights Coverage

### A. Speech-to-Text (ASR) & Audio
- [x] **Alibaba FunASR / Paraformer & SenseVoice** — `paraformer-zh-small.int8.onnx` (81.8 MB) + `sensevoice-small.int8.onnx` (239 MB): Verified with `FunAsrRealWeightsTests.cs` (2/2 passed).
- [x] **Alibaba Qwen3-ASR (0.6B)** — `qwen3-asr-0.6b-q4_k.gguf` (631 MB): Verified with `QwenAsrRealWeightsTests.cs`.
- [x] **Alibaba Qwen3-ForcedAligner (0.6B)** — `qwen3-forcedaligner-0.6b.safetensors` (1.8 GB): Verified with `QwenForcedAlignerRealWeightsTests.cs`.
- [x] **NVIDIA Parakeet CTC (0.6B)** — `parakeet-ctc-0.6b-q4_k.gguf` (383 MB): Verified with `ParakeetRealWeightsTests.cs`.
- [x] **OpenAI Whisper (Tiny / Base / Small / Medium)** — `ggml-tiny.bin`, `ggml-base.bin`, `ggml-small.bin`, `ggml-medium.bin`: Verified with `WhisperRealWeightsTests.cs`.
- [x] **Silero VAD** — `silero_vad.onnx` (2.2 MB): Verified with `SileroVadRealWeightsTests.cs`.

### B. Text-to-Speech (TTS)
- [x] **Piper VITS** — `en_US-lessac-medium.onnx` + `.json` (63 MB): Verified with `PiperRealWeightsTests.cs`.
- [x] **Kokoro-82M** — `kokoro-v1.0.onnx` + `af_heart.pt` (325 MB): Verified with `KokoroRealWeightsTests.cs`.
- [x] **MeloTTS** — `melotts-zh_en.onnx` (170 MB): Verified with `MeloTtsRealWeightsTests.cs`.
- [x] **Chatterbox TTS** — `chatterbox-speech_encoder.onnx` (180 MB): Verified with `ChatterboxRealWeightsTests.cs`.
- [x] **CosyVoice 300M / 2.0** — `cosyvoice_speech_tokenizer.onnx` + `campplus.onnx` + `flow.decoder.estimator.fp32.onnx` + `cosyvoice2_0.5b.safetensors`: Verified with `CosyVoiceRealWeightsTests.cs`.
- [x] **F5-TTS** — `f5tts_base.safetensors` (1.34 GB): Verified with `F5TtsRealWeightsTests.cs`.

### C. Vision & Multimodal Projectors
- [x] **MiniCPM-V 2.6 Vision Projector** — `mmproj-minicpm-v-2_6-f16.gguf` (1.04 GB): Verified with `MiniCpmVisionRealWeightsTests.cs`.
- [x] **Llama 4 Scout ViT Projector** — `mmproj-llama-4-scout-17b-16e-instruct-f16.gguf` (1.75 GB): Verified in Vision test suite.
- [x] **Gemma 4 12B Multimodal Projector** — `mmproj-gemma-4-12b-it-qat-q4_0.gguf` (2.1 GB): Verified in Vision test suite.
- [x] **Gemma 4 E4B ViT+Audio Encoder** — `gemma-4-E4B-it-mmproj.gguf` (1.2 GB): Verified in Vision test suite.
- [x] **Gemma 3 4B Vision Projector** — `mmproj-gemma-3-4b-it-f16.gguf` (780 MB): Verified in Vision test suite.

### D. Large Language Models, Speculative Decoding & Embeddings
- [x] **SmolLM2 (135M / 360M / 1.7B)** — Verified with `SmolLm2135MRealWeightsTests.cs` and `SmolLm2RealWeightsTests.cs`.
- [x] **Qwen2.5 (0.5B / 1.5B / 3B)** — Verified with `Qwen25RealWeightsTests.cs` (3/3 passed).
- [x] **Qwen2.5-Coder (0.5B / 1.5B / 3B)** — Verified with `QwenCoderRealWeightsTests.cs` (3/3 passed).
- [x] **Qwen3 (0.6B / 4B)** — Verified draft spec-decode and base forward pass.
- [x] **DSpark Speculative Decoding** — `dspark_qwen3_4b_block7` (2.8 GB): Verified with `DSparkRealWeightsTests.cs`.
- [x] **Embeddings Suite** — `all-MiniLM-L6-v2`, `bge-small`, `bge-base`, `bge-large`: Verified with `EmbeddingsRealWeightsTests.cs`.

### E. Image & Video Diffusion
- [x] **Hunyuan Video (FP8 Distilled DiT)** — `hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors` (13.1 GB): Verified with `HunyuanVideoRealWeightsTests.cs`.
- [x] **Wan Video 2.1 (1.3B DiT)** — `Wan2.1-T2V-1.3B-Q4_0.gguf` (865 MB): Verified with `WanVideoRealWeightsTests.cs`.
- [x] **LTX-Video (2B DiT)** — `ltx-video-2b-v0.9.1.safetensors` (5.7 GB): Verified with `LtxVideoRealWeightsTests.cs`.
- [x] **Real-ESRGAN x4plus** — `RealESRGAN_x4plus.safetensors` (67 MB): Verified with `RealEsrganRealWeightsTests.cs`.
- [x] **Flux VAE Autoencoder** — `ae.safetensors` (335 MB): Verified in `FluxVaeRealWeightsTests.cs`.
- [x] **Z-Image-Turbo** — `z_image_turbo-Q4_0.gguf` (3.68 GB): Verified in `ZImageRealWeightsTests.cs`.
- [x] **Stable Diffusion 1.5** — `v1-5-pruned-emaonly.safetensors` (4.2 GB): Proven in Diffusion test suite.
- [x] **SDXL Turbo 1.0** — `sd_xl_turbo_1.0_fp16.safetensors` (6.9 GB): Proven in Diffusion test suite.
