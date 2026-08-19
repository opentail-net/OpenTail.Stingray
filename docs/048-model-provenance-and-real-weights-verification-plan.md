# Model Provenance & Real-Weights Verification Plan

**Target:** `opentail-net/OpenTail.Stingray`  
**Purpose:** Comprehensive inventory and tracking checklist of all model architectures, formats, and pipelines claimed by OpenTail.Stingray — documenting which are formally proven with test suites against real downloaded weights versus those currently simulated or awaiting weight coverage.

---

## 1. Executive Summary & Verification Matrix

| Domain | Architecture / Model Family | Claimed Support in Codebase | Download Script Entry | Real Weights on Disk | Automated Test Suite (Real Weights) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **ASR (STT)** | NVIDIA Parakeet (FastConformer CTC/TDT) | ✅ `Audio/Parakeet` | ✅ `download-model.ps1` | ✅ `parakeet-ctc-0.6b-q4_k.gguf` | ✅ `ParakeetRealWeightsTests.cs` | **PROVEN** |
| **ASR (STT)** | Alibaba Qwen3-ASR (0.6B / 1.7B) | ✅ `Audio/QwenASR` | ✅ `download-model.ps1` | ✅ `qwen3-asr-0.6b-q4_k.gguf` | ✅ `QwenAsrRealWeightsTests.cs` | **PROVEN** |
| **ASR (STT)** | Alibaba Qwen3-ForcedAligner | ✅ `Audio/QwenASR` | ✅ `download-model.ps1` | ✅ `qwen3-forcedaligner-0.6b.safetensors` | ✅ `QwenForcedAlignerRealWeightsTests.cs` | **PROVEN** |
| **ASR (STT)** | OpenAI Whisper (Tiny/Base/Small/Med/Large/Turbo) | ✅ `Audio/Whisper` | ✅ `download-model.ps1` | ✅ `ggml-tiny.bin`, `ggml-base.bin`, `ggml-small.bin` | ✅ `WhisperRealWeightsTests.cs` | **PROVEN** |
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
| **Video Gen** | Hunyuan Video | ✅ `Diffusion/Hunyuan` | 🔲 | 🔄 Downloading (`hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors`) | 🔲 In Progress | In Progress |
