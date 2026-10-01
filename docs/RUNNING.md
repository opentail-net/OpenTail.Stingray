# Running models well

How to get the best result from a model on real hardware: the command to use, how much memory it
really needs, and how fast it runs. [MODELS.md](MODELS.md) says which models to pick and
[STATUS.md](STATUS.md) says what works and how that was checked. This page is the practical part.

Every number here was measured, with the date and the machine. A row without a measurement says
so. Reference machine: AMD Ryzen 7 5700G (8 cores), 64 GB DDR4, integrated GPU only, Windows 11.

## General rules

These apply to every model and were confirmed while admitting the models below.

**Working directory.**
- Run all commands from the repository root folder (`c:\Git-Public\OpenTail.Stingray`). All relative paths (such as `models/_models/...` or `models/sd15/...`) resolve from the root.
- If `stingray` is not installed as a global tool on your system `PATH`, invoke the compiled CLI directly via `.\src\OpenTail.Stingray.Cli\bin\Release\net10.0\stingray.exe` (or `dotnet run --project src/OpenTail.Stingray.Cli -c Release --`).

**Memory.**
- Weights are memory-mapped, not copied. Stingray loads the parts every token uses (embeddings,
  attention, dense and shared-expert FFNs) into RAM up front.
- A mixture-of-experts model's routed experts are loaded up front only if they fit in about 80% of
  the RAM that is free at that moment. Otherwise they are paged in on demand, and the OS keeps the
  ones in use.
- `STINGRAY_PREFAULT=1` forces everything in; `STINGRAY_PREFAULT=0` loads nothing up front.
- Stingray does not keep a second, repacked copy of the weights, so it needs noticeably less RAM
  than llama.cpp on the same file (GLM-4.5-Air row below).

**Close other heavy programs for models near your RAM size.** Two 30 GB processes at once is what
ran this 64 GB machine out of memory, not either one alone.

**BOS.** The start-of-text token is added automatically when the model's metadata asks for it. Some
models (LFM2) produce repetitive nonsense without it, so don't strip it off prompts yourself.

**Recurrent and hybrid models** (Granite 4.0-H, Nemotron-H, LFM2):
- Prompts are processed in batches since 2026-09-28: the projections of the Mamba-2 / short-conv
  layers run batched and only the conv and scan walk the tokens in order (2.8-3.4x faster prompt
  processing than before on the Q8_0 files below). `STINGRAY_RECURRENT_BATCHED_PREFILL=0` restores
  the old one-token-at-a-time path.
- Prompt caching (reusing a shared prefix between requests) is switched off automatically for them,
  because their recurrent state cannot be rewound.

**Checking quality yourself.** `stingray perplexity -m <model> -f <text> -c 2048` scores a text one
token at a time; add `--batched` to use the faster batched prompt path. Compare the `[1024,+)`
bucket with `llama-perplexity -m <model> -f <text> -c 2048 --chunks 1`.

## Part I: CPU Execution (No Dedicated GPU Required)

The models in this section run entirely on CPU cores using Stingray's AVX2/AVX-512 SIMD and OpenBLAS kernels. No dedicated graphics card or discrete VRAM is required.

Commands are CPU runs of the exact file listed. Decode is generation speed after the prompt.

### 1. Large Language Models (Text Generation)

| Model and file | Command | RAM needed | Speed on the reference machine | Notes |
|---|---|---|---|---|
| Qwen2.5 0.5B Instruct, `qwen2.5-0.5b-instruct-q4_k_m.gguf` (`stingray setup chat`) | `stingray -m models/_models/qwen2.5-0.5b-instruct-q4_k_m.gguf -p "Hello world"` | about 1 GB | Decode 62.9-69.1 tok/s over 128 tokens, prompt about 318 tok/s over 661 tokens (2026-09-28, after int8 Q5_0 kernels; was 21-25 / 77-84) | Default chat catalog model. Use `-g 0`; `-g -1` on the iGPU is about 26 tok/s. |
| Qwen2.5-Coder 0.5B Instruct, `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` | `stingray -m models/_models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf -p "Write a hello world in C#:"` | about 1 GB | Prompt 90.8 tok/s, decode 57.0 tok/s (2026-09-28) | High-throughput code generation model. |
| Qwen2.5 1.5B Instruct, `qwen2.5-1.5b-instruct-q4_k_m.gguf` | `stingray -m models/_models/qwen2.5-1.5b-instruct-q4_k_m.gguf -p "Write 3 bullet points about Mars:"` | about 1.5 GB | Prompt 113.0 tok/s, decode 24.1 tok/s (2026-09-28) | Fast general instruction model. |
| Qwen3 0.6B Base, `Qwen3-0.6B-Q8_0.gguf` | `stingray -m models/_models/Qwen3-0.6B-Q8_0.gguf -p "Hello world" --temp 0.6` | about 1.2 GB | Prompt 31.3 tok/s, decode 39.6 tok/s (2026-09-28) | Reasoning model (outputs thinking tokens). |
| Qwen3 4B Instruct, `Qwen3-4B-Q4_K_M.gguf` | `stingray -m models/_models/Qwen3-4B-Q4_K_M.gguf -p "Hello world" --temp 0.6` | about 3.2 GB | Prompt 38.0 tok/s, decode 10.7 tok/s (2026-09-28) | Reasoning model with thinking traces. |
| SmolLM2 135M Instruct, `SmolLM2-135M-Instruct-Q4_K_M.gguf` | `stingray -m models/_models/SmolLM2-135M-Instruct-Q4_K_M.gguf -p "Hello world"` | under 1 GB | Prompt 48.5 tok/s, decode 21.2 tok/s (2026-09-28) | Ultra-lightweight edge model. |
| SmolLM2 360M Instruct, `SmolLM2-360M-Instruct-Q4_K_M.gguf` | `stingray -m models/_models/SmolLM2-360M-Instruct-Q4_K_M.gguf -p "The capital of France is"` | under 1 GB | Prompt 89.2 tok/s, decode 51.4 tok/s (2026-09-28) | Very fast small model. |
| SmolLM2 1.7B Instruct, `SmolLM2-1.7B-Instruct-Q4_K_M.gguf` | `stingray -m models/_models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "The capital of France is"` | about 1.5 GB | Prompt 86.2 tok/s, decode 19.7 tok/s (2026-09-28); CPU (`-g 0`) decode 29.8 tok/s, 128 tokens (2026-10-01) | Well-balanced small instruction model. |
| SmolLM3 Q4_K_M, `SmolLM3-Q4_K_M.gguf` | `stingray -m models/_models/SmolLM3-Q4_K_M.gguf -p "Hello world" --temp 0.6` | about 2.5 GB | Prompt 119.5 tok/s, decode 11.9 tok/s (2026-09-28) | Reasoning model architecture with thinking trace. |
| Microsoft Phi-2, `phi-2.Q4_K_M.gguf` | `stingray -m models/_models/phi-2.Q4_K_M.gguf -p "Instruct: Explain gravity in one sentence.\nOutput:"` | about 2.8 GB | Prompt 5.4 tok/s, decode 11.4 tok/s (2026-09-28) | Classic 2.7B reasoning/math architecture. |
| Microsoft Phi-3 Mini 4K Instruct, `Phi-3-mini-4k-instruct-Q4_K_M.gguf` | `stingray -m models/_models/Phi-3-mini-4k-instruct-Q4_K_M.gguf -p "Instruct: Explain gravity in one sentence.\nOutput:"` | about 2.8 GB | Prompt 5.6 tok/s, decode 10.8 tok/s (2026-09-28) | 3.8B instruction model. |
| Mistral 7B Instruct v0.3, `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` | `stingray -m models/_models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf -p "Hello world"` | about 5.0 GB | Prompt 7.6 tok/s, decode 6.5 tok/s (2026-09-28); CPU (`-g 0`) decode 8.3 tok/s, 128 tokens (2026-10-01) | 32k context LLaMA-family dense transformer. |
| Ministral 8B Instruct 2410, `Ministral-8B-Instruct-2410-Q4_K_M.gguf` | `stingray -m models/_models/Ministral-8B-Instruct-2410-Q4_K_M.gguf -p "Hello world"` | about 5.5 GB | Prompt 5.7 tok/s, decode 6.4 tok/s (2026-09-28); CPU (`-g 0`) decode 7.5 tok/s, 128 tokens (2026-10-01) | Mistral-family interleaved attention architecture. |
| OLMoE 1B-7B Instruct, `OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf` | `stingray -m models/_models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf -p "Hello world"` | about 4.5 GB | Prompt 36.2 tok/s, decode 23.3 tok/s (2026-09-28) | 64-expert sparse MoE (1B active). |
| IBM Granite 4.0-H 1B, `granite-4.0-h-1b-Q8_0.gguf` | `stingray -m models/_models/granite-4.0-h-1b-Q8_0.gguf -p "Hello world"` | about 1.6 GB | Decode 21.0 tok/s, prompt 63.4 tok/s over 511 tokens (2026-09-28) | Hybrid Mamba-2 recurrent architecture. |
| Liquid LFM2 1.2B, `LFM2-1.2B-Q8_0.gguf` | `stingray -m models/_models/LFM2-1.2B-Q8_0.gguf -p "Hello world"` | about 1.3 GB | Decode 28-30 tok/s, prompt 101 tok/s over 538 tokens (2026-09-28) | Hybrid Conv/SSM. Needs automatic BOS. |
| Maincoder 1B, `Maincoder-1B-Q4_K_M.gguf` | `stingray -m models/_models/Maincoder-1B-Q4_K_M.gguf -p "Write hello world in python:"` | about 1.2 GB | Prompt 129.4 tok/s, decode 22.9 tok/s (2026-09-28) | Specialized code-generation small model. |
| StarCoder2 3B, `starcoder2-3b.Q4_K_M.gguf` | `stingray -m models/_models/starcoder2-3b.Q4_K_M.gguf -p "def fibonacci(n):"` | about 2.5 GB | Prompt 51.4 tok/s, decode 12.6 tok/s (2026-09-28) | Code LLM with 16k native window. |
| StableLM Zephyr 3B, `stablelm-zephyr-3b.Q4_K_M.gguf` | `stingray -m models/_models/stablelm-zephyr-3b.Q4_K_M.gguf -p "Hello world"` | about 2.2 GB | Prompt 34.5 tok/s, decode 14.1 tok/s (2026-09-28) | Stability AI chat model. |
| Swiss AI Apertus 8B Instruct, `swiss-ai.Apertus-8B-Instruct-2509.Q4_K_M.gguf` | `stingray -m models/_models/swiss-ai.Apertus-8B-Instruct-2509.Q4_K_M.gguf -p "Hello world"` | about 5.8 GB | Prompt 26.8 tok/s, decode 6.3 tok/s (2026-09-28) | Open multilingual foundation model. |
| Tencent Hunyuan 0.5B Instruct, `tencent_Hunyuan-0.5B-Instruct-Q8_0.gguf` | `stingray -m models/_models/tencent_Hunyuan-0.5B-Instruct-Q8_0.gguf -p "Hello world" --temp 0.6` | about 1.0 GB | Prompt 11.8 tok/s, decode 24.7 tok/s (2026-09-28) | Hunyuan reasoning dense architecture. |
| Orpheus 3B, `orpheus-3b-0.1-ft.Q4_K_M.gguf` | `stingray -m models/_models/orpheus-3b-0.1-ft.Q4_K_M.gguf -p "Hello world"` | about 3.0 GB | Prompt 59.0 tok/s, decode 11.3 tok/s (2026-09-28) | Conversational fine-tuned model. |
| Pythia 160M, `pythia-160m.Q8_0.gguf` | `stingray -m models/_models/pythia-160m.Q8_0.gguf -p "Hello world"` | under 0.5 GB | Prompt 167.3 tok/s, decode 52.3 tok/s (2026-09-28) | Compact GPT-NeoX reference architecture. |
| XVERSE 7B Chat, `xverse-7b-chat-q4_k_m.gguf` | `stingray -m models/_models/xverse-7b-chat-q4_k_m.gguf -p "Hello world"` | about 5.2 GB | Prompt 27.4 tok/s, decode 6.9 tok/s (2026-09-28) | 7B bilingual foundation model. |
| Qwen2.5-Coder 1.5B Instruct, `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` | `stingray -m models/_models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf -p "Write a python function to compute factorial:"` | about 1.5 GB | Prompt 121.5 tok/s, decode 23.6 tok/s (2026-09-28) | High-speed coding model. |
| Qwen2.5-Coder 3B Instruct, `qwen2.5-coder-3b-instruct-q4_k_m.gguf` | `stingray -m models/_models/qwen2.5-coder-3b-instruct-q4_k_m.gguf -p "Write a python function to compute factorial:"` | about 2.5 GB | Prompt 65.8 tok/s, decode 13.3 tok/s (2026-09-28) | Fast 3B code generation model. |
| Qwen2.5 3B Instruct, `qwen2.5-3b-instruct-q4_k_m.gguf` | `stingray -m models/_models/qwen2.5-3b-instruct-q4_k_m.gguf -p "Hello world"` | about 2.5 GB | Prompt 60.9 tok/s, decode 11.4 tok/s (2026-09-28) | Dense 3B instruction model. |
| Qwen3 8B Instruct, `Qwen3-8B-Q4_K_M.gguf` | `stingray -m models/_models/Qwen3-8B-Q4_K_M.gguf -p "Hello world" --temp 0.6` | about 5.0 GB | Prompt 24.1 tok/s, decode 6.2 tok/s (2026-09-28); CPU (`-g 0`) decode 7.7 tok/s, 128 tokens (2026-10-01) | Full 8B reasoning model with thinking tokens. |
| IBM Granite 4.0-H 350M, `granite-4.0-h-350m-Q8_0.gguf` | `stingray -m models/_models/granite-4.0-h-350m-Q8_0.gguf -p "Hello world"` | under 1 GB | Prompt 42.4 tok/s, decode 58.4 tok/s (2026-09-28) | Ultra-compact Mamba-2 hybrid architecture. |
| OpenAI GPT-2, `gpt2.Q8_0.gguf` | `stingray -m models/_models/gpt2.Q8_0.gguf -p "Hello world"` | under 0.5 GB | Prompt 177.4 tok/s, decode 53.2 tok/s (2026-09-28) | Classic 124M autoregressive reference model. |
| OpenAI GPT-OSS 20B (MXFP4), `gpt-oss-20b-MXFP4.gguf` | `stingray -m models/_models/gpt-oss-20b-MXFP4.gguf -p "Hello world"` | about 13 GB | Prompt 2.1 tok/s, decode 5.5 tok/s (2026-09-28) | Native MXFP4 microscopic floating-point execution on CPU. |
| NVIDIA Nemotron Nano 12B v2 VL, `nemotron-nano-12b-v2-vl-Q2_K.gguf` (text) | `stingray -m models/_models/nemotron-nano-12b-v2-vl-Q2_K.gguf -p "Hello world"` | about 4.7 GB | Prompt 3.0 tok/s, decode 6.2 tok/s (2026-09-28) | Text generation only; verified on reference machine. |
| Mistral Small 3.2 24B Instruct, `Mistral-Small-3.2-24B-Instruct-2506-Q4_K_S.gguf` | `stingray -m K:/_other_models/Mistral-Small-3.2-24B-Instruct-2506-Q4_K_S.gguf -p "Hello world"` | about 14 GB | Prompt 15.6 tok/s, decode 2.5 tok/s (2026-09-28) | 40L, 5120d, large dense LLaMA-family foundation model. |
| GLM-4.7-Flash (MoE), `GLM-4.7-Flash-Q2_K.gguf` | `stingray -m K:/_other_models/GLM-4.7-Flash-Q2_K.gguf -p "Hello world"` | about 11.5 GB | Prompt 1.9 tok/s, decode 10.6 tok/s (2026-09-28) | 47L, 2048d, headDim=256, DeepSeek2-family sparse MoE. |
| DeepSeek-V2-Lite Chat, `DeepSeek-V2-Lite-Chat.Q8_0.gguf` | `stingray -m K:/_other_models/DeepSeek-V2-Lite-Chat.Q8_0.gguf -p "Hello world"` | about 17 GB | Prompt 2.7 tok/s, decode 13.2 tok/s (2026-09-28) | 27L, 2048d, headDim=192, native Q8_0 MLA + MoE execution. |
| Microsoft Phi-3.5-MoE Instruct, `Phi-3.5-MoE-instruct-Q3_K_M.gguf` | `stingray -m K:/_other_models/Phi-3.5-MoE-instruct-Q3_K_M.gguf -p "Hello world"` | about 20 GB | Prompt 7.2 tok/s, decode 7.9 tok/s (2026-09-28) | 32L, 4096d, 16x3.8B sparse MoE architecture. |
| LG EXAONE 4.5 33B, `EXAONE-4.5-33B-Q4_K_M.gguf` | `stingray -m K:/_other_models/EXAONE-4.5-33B-Q4_K_M.gguf -p "Hello world"` | about 20 GB | Prompt 2.4 tok/s, decode 1.7 tok/s (2026-09-28) | 64L, 5120d, Korean/English bilingual foundation model. |
| Qwen3.6 35B-A3B (MoE), `Qwen3.6-35B-A3B-UD-Q6_K.gguf` | `stingray -m K:/_other_models/Qwen3.6-35B-A3B-UD-Q6_K.gguf -p "Hello world" --temp 0.6` | about 30 GB | Prompt 6.5 tok/s, decode 2.9 tok/s (2026-09-28) | Hybrid GDN + MoE architecture with Q8_K routed kernels. |

### 2. Multimodal Vision Understanding (VLM)

| Model and file | Command | RAM needed | Speed on the reference machine | Notes |
|---|---|---|---|---|
| OpenGVLab InternVL3 2B, `InternVL3-2B.Q4_K_M.gguf` | `stingray -m models/_models/InternVL3-2B.Q4_K_M.gguf --mmproj models/_models/mmproj-internvl3-2b-q8_0.gguf --image photo.png -p "Describe this image in one sentence."` | about 1.8 GB | Prompt 123.2 tok/s (256 img + 38 txt), decode 26.7 tok/s (2026-09-28) | Verified working vision captioning and object grounding. |
| Tencent Youtu-VL 4B, `youtu-vl-4b-Q8_0.gguf` | `stingray -m models/_models/youtu-vl-4b-Q8_0.gguf --mmproj models/_models/mmproj-youtu-vl-4b-BF16.gguf --image photo.png -p "Describe this picture."` | about 5.8 GB | Prompt 7.2 tok/s (64 img + 25 txt), decode 6.4 tok/s (2026-09-28) | High-accuracy visual document understanding. |
| PaddleOCR-VL 1.6, `paddleocr-vl-1.6.gguf` | `stingray -m models/_models/paddleocr-vl-1.6.gguf --mmproj models/_models/PaddleOCR-VL-1.6-GGUF-mmproj.gguf --image doc.png -p "Read the text in this image."` | about 1.5 GB | Prompt 88.7 tok/s (81 img + 17 txt) (2026-09-28) | Specialized OCR vision model. |
| Qwen3-VL 2B Instruct, `Qwen3VL-2B-Instruct-Q8_0.gguf` | `stingray -m models/_models/Qwen3VL-2B-Instruct-Q8_0.gguf --mmproj models/_models/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf --image photo.png -p "Describe this picture."` | about 2.5 GB | Prompt 23.1 tok/s (64 img + 14 txt), decode 19.7 tok/s (2026-09-28) | 2D M-RoPE image tokens with batched prompt processing. |
| IBM Granite Vision 3.2 2B, `granite-vision-3.2-2b-Q3_K_S.gguf` | `stingray -m models/_models/granite-vision-3.2-2b-Q3_K_S.gguf --mmproj models/_models/mmproj-granite-vision-3.2-2b-f16.gguf --image photo.png -p "Describe this picture."` | about 1.5 GB | Prompt 28.5 tok/s (729 img + 52 txt), decode 15.8 tok/s (2026-09-28) | High-accuracy dense vision understanding via MLP projector. |
| IBM Granite 4.0 3B Vision, `granite-4.0-3b-vision-Q4_K_M.gguf` | `stingray -m models/_models/granite-4.0-3b-vision-Q4_K_M.gguf --mmproj models/_models/mmproj-granite-4.0-3b-vision-f16.gguf --image photo.png -p "Describe this picture."` | about 2.5 GB | Prompt 104.9 tok/s (724 img + 13 txt), decode 11.8 tok/s (2026-10-01) | WindowQFormer 8-stream deepstack spatial projector with canonical Granite role framing. |
| Qwen2.5-VL 7B Instruct, `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` | `stingray -m models/_models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf --mmproj models/_models/mmproj-qwen2.5-vl-7b-f16.gguf --image photo.png -p "Describe this picture."` | about 5.0 GB | Prompt 20.3 tok/s (81 img + 25 txt), decode 7.2 tok/s (2026-09-28) | 2D M-RoPE dynamic resolution vision transformer. |
| Xiaomi MiMo-VL 7B SFT, `mimo-vl-7b-sft-Q2_K.gguf` | `stingray -m models/_models/mimo-vl-7b-sft-Q2_K.gguf --mmproj models/_models/mmproj-mimo-vl-7b-sft-Q8_0.gguf --image photo.png -p "Describe this picture."` | about 3.5 GB | Prompt 14.3 tok/s (81 img + 31 txt), decode 10.0 tok/s (2026-09-28) | 7B visual chat assistant. |
| Moonshot Kimi-VL A3B Thinking, `kimi-vl-a3b-thinking-Q2_K.gguf` | `stingray -m models/_models/kimi-vl-a3b-thinking-Q2_K.gguf --mmproj models/_models/mmproj-kimi-vl-a3b-thinking-Q8_0.gguf --image photo.png -p "Describe this picture."` | about 6.5 GB | Prompt 14.8 tok/s (81 img + 22 txt), decode 11.4 tok/s (2026-09-28) | Multimodal MoE with visual reasoning/thinking traces. |
| StepFun Step3-VL 10B, `step3-vl-10b-Q2_K.gguf` | `stingray -m models/_models/step3-vl-10b-Q2_K.gguf --mmproj models/_models/mmproj-step3-vl-10b-F16.gguf --image photo.png -p "Describe this picture."` | about 3.8 GB | Prompt 9.5 tok/s (171 img + 14 txt), decode 9.6 tok/s (2026-09-28) | 10B bilingual visual reasoning model. |
| Dots.OCR 1.5B, `dots.ocr-Q8_0.gguf` | `stingray -m models/_models/dots.ocr-Q8_0.gguf --mmproj models/_models/mmproj-dots.ocr-Q8_0.gguf --image doc.png -p "Read the text in this image."` | about 2.2 GB | Prompt 29.0 tok/s (81 img + 12 txt), decode 16.0 tok/s (2026-09-28) | Specialized OCR vision model. |
| LLaVA-1.5 7B, `llava-v1.5-7b/ggml-model-q4_k.gguf` | `stingray -m models/_models/ggml-model-q4_k.gguf --mmproj models/_models/mmproj-llava-v1.5-7b-f16.gguf --image photo.png -p "Describe this picture."` | about 4.5 GB | Prompt 53.4 tok/s (576 img + 15 txt), decode 6.9 tok/s (2026-09-28) | Direct-spliced soft image tokens in Vicuna format. |

### 3. Speech-to-Text (STT / ASR)

| Model and file | Command | RAM needed | Speed on the reference machine | Notes |
|---|---|---|---|---|
| SenseVoice Small (ONNX), `sensevoice-small.int8.onnx` | `stingray stt -m sensevoice --model-file models/_models/sensevoice-small.int8.onnx -i speech.wav` | under 1 GB | 1.58s audio transcribed in 0.17s (9.4x real-time) (2026-09-28) | Multilingual with automated timestamped segments. |
| FunASR Paraformer (ONNX), `paraformer-zh-small.int8.onnx` | `stingray stt -m paraformer --model-file models/_models/paraformer-zh-small.int8.onnx -i speech.wav` | under 1 GB | 1.58s audio transcribed in 0.07s (21.1x real-time) (2026-09-28) | Fluent Mandarin transcription pipeline. |
| OpenAI Whisper Base, `ggml-base.bin` | `stingray stt -m base --model-file models/ggml-base.bin -i speech.wav` | under 1 GB | 2.9s audio transcribed in 1.03-1.54s (2026-09-28) | Standard English/multilingual reference transcription. |
| OpenAI Whisper Small, `whisper-small.gguf` | `stingray stt -m small --model-file models/whisper-small.gguf -i speech.wav` | under 1 GB | 11.0s audio transcribed in 3.73s (3.0x real-time) (2026-09-28) | 768d / 12L multilingual model with timestamp segments. |
| OpenAI Whisper Medium, `whisper-medium.gguf` | `stingray stt -m medium --model-file models/_models/whisper-medium.gguf -i speech.wav` | about 1.8 GB | 11.0s audio transcribed in 8.59s (1.3x real-time) (2026-09-28) | 1024d / 24L high-accuracy multilingual transcription. |
| OpenAI Whisper Large v3, `whisper-large-v3.gguf` | `stingray stt -m large --model-file models/_models/whisper-large-v3.gguf -i speech.wav` | about 3.3 GB | 11.0s audio transcribed in 14.75s (0.7x real-time) (2026-09-28) | 1280d / 32L flagship multilingual transcription model. |
| NVIDIA Parakeet CTC 0.6B, `parakeet-ctc-0.6b-q4_k.gguf` | `stingray stt -m parakeet --model-file models/parakeet-ctc-0.6b-q4_k.gguf -i speech.wav` | about 1.1 GB | 11.0s audio transcribed in 1.00s (11.0x real-time) (2026-09-28) | Conformer CTC architecture with word-level timestamps. |

### 4. Text-to-Speech (TTS)

| Model and file | Command | RAM needed | Speed on the reference machine | Notes |
|---|---|---|---|---|
| Kokoro 82M (GGUF), `kokoro-82m-q8_0.gguf` | `stingray tts -e kokoro -m models/_models/kokoro-82m-q8_0.gguf -v af_heart --voices-dir models/_models -t "Hello world" -o speech.wav` | under 1 GB | 1.58s audio generated in 2.09s (1.32x RTF) (2026-09-28) | High-fidelity voice styles (`af_heart`, etc.). |
| MeloTTS (ONNX), `melotts-zh_en.onnx` | `stingray tts -e melo -m models/_models/melotts-zh_en.onnx -t "Hello world" -o speech.wav` | under 1 GB | 1.02s audio generated in 2.32s (2.27x RTF) (2026-09-28) | Fast bilingual English/Chinese synthesis. |
| Piper en_US Lessac (ONNX), `en_US-lessac-medium.onnx` | `stingray tts -e piper -m models/en_US-lessac-medium.onnx -t "Hello world" -o speech.wav` | under 1 GB | 2.83-2.93s audio in 1.12-1.35s (RTF 0.42x) (2026-09-28) | Fastest CPU TTS engine. |
| Chatterbox-Turbo (GGUF), `chatterbox-turbo-s3gen-q4_k.gguf` | `stingray tts -e chatterbox -t "Hello world, this is Chatterbox speaking." -o speech.wav` | about 1.2 GB | 2.68s audio generated in 5.40s (2.01x RTF) (2026-09-28) | Expressive conversational dual-stage speech generation. |
| Parler-TTS Mini v1, `parler-tts-mini-v1.safetensors` | `stingray tts -e parler -t "Hello world, this is Parler speaking." -o speech.wav` | about 3.5 GB | 2.44s audio generated in 17.43s (7.15x RTF) (2026-09-28) | High-fidelity description-conditioned TTS model. |
| FishSpeech S2 Pro (Q8_0), `s2-pro-q8_0.gguf` | `stingray tts -e fish -m K:/_other_models/s2-pro-q8_0.gguf -t "Hello world from Fish Speech." -o speech.wav` | about 5.5 GB | 2.00s audio generated in 22.56s (11.3x RTF) (2026-09-28) | High-fidelity dual-stage neural codec and autoregressive generator. **Prefer this Q8_0 file over `s2-pro-q4_k_m.gguf`:** for the Fast-AR stage, Q8_0 is essentially the original model (KL 0.007 nats on one deterministic position) while Q4_K_M is a materially different distribution (KL 0.98 nats, top-1 probability 0.28 -> 0.16); Q4_K_M still sounds plausible but is lower quality (measured 2026-10-01, single position; `docs/done/08-fish-speech-s2-pro-golden-reconciliation-plan.md`). |
| PersonaPlex 7B (Q8_0, duplex speech LM), `personaplex-7b-v1-q8_0.gguf` | not wired to a CLI command yet -- `dotnet test tests/OpenTail.Stingray.Tests.Audio -- --filter-class PersonaPlexGeneratorRealWeightsTests` (`STINGRAY_RUN_HEAVY_TESTS=1`) | 11.17 GiB pre-faulted (2026-09-28, was ~28 GB before the zero-copy-Q8 fix that day) | ~301ms/decode-step median (was ~660ms; 8 samples/run, 3 runs each side, isolated via `git stash` of one file) (2026-09-28) | Library/test-only; see `docs/done/audio-review-new-progress.md`'s PersonaPlex 7B entry. Speed win's direction (faster) should generalize to other CPUs (bandwidth-bound Q8_0 decode); the 2.2x magnitude is this reference machine's number, not yet confirmed elsewhere. |

### 5. Semantic Embeddings & Reranking

| Model and file | Command | RAM needed | Speed on the reference machine | Notes |
|---|---|---|---|---|
| Qwen3 Embedding 0.6B (GGUF), `qwen3-embedding-0.6b-q8_0.gguf` | `stingray embed -m models/qwen3-embedding-0.6b/qwen3-embedding-0.6b-q8_0.gguf -p "Hello world"` | about 1.0 GB | 1 text embedded in 314ms (1024-dim, L2-normed) (2026-09-28) | Dense semantic vectors with configurable pooling. |
| Qwen3 Cross/Bi-Encoder Reranker, `qwen3-embedding-0.6b-q8_0.gguf` | `stingray rerank -m models/qwen3-embedding-0.6b/qwen3-embedding-0.6b-q8_0.gguf -q "query" -d "doc1" -d "doc2"` | about 1.0 GB | 2 documents ranked in 749ms (2026-09-28) | Relevance scoring and top-N ranking. |

### 6. Image & Video Diffusion (CPU SIMD Fallback)

All diffusion models in Stingray support native CPU execution via `--device none` (or `--backend cpu`). No GPU or VRAM is required.

| Model and file | Command | RAM needed | Speed on the reference machine | Notes |
|---|---|---|---|---|
| Stable Diffusion 1.5 (CPU), `v1-5-pruned-emaonly.safetensors` | `stingray image -m models/sd15/v1-5-pruned-emaonly.safetensors -p "a photograph of an astronaut riding a horse" -W 256 -H 256 --steps 4 --device none -o output.png` | about 2.5 GB | 256×256 4-step image in 13.4s (2026-09-28) | UNet + CLIP on pure CPU SIMD (`--device none`). |
| Stable Diffusion 3.5 Medium (CPU), `sd3.5_medium-Q4_K_M.gguf` | `stingray image -m models/_models/sd3.5_medium-Q4_K_M.gguf --clip-l models/sd35-medium-aux/text_encoder/model.fp16.safetensors --clip-g models/sd35-medium-aux/text_encoder_2/model.fp16.safetensors --vae models/sd35-medium-aux/vae/diffusion_pytorch_model.safetensors --clip-tokenizer models/clip_tokenizer.json --device none --steps 4 -W 256 -H 256 -p "a red car" -o output.png` | about 5.5 GB | 256×256 4-step image in ~42s (2 steps in 21.1s) (2026-09-28) | MMDiT + dual CLIP on pure CPU SIMD (`--device none`). |
| FLUX.1-schnell (CPU), `flux1-schnell-Q4_K_S.gguf` | `stingray image -m models/flux1-schnell/flux1-schnell-Q4_K_S.gguf --vae models/flux1-schnell/ae.safetensors --clip-l models/flux1-schnell/clip_l.safetensors --clip-tokenizer models/flux1-schnell/tokenizer_clip/tokenizer.json --t5xxl models/flux1-schnell/t5xxl_fp8_e4m3fn.safetensors --t5-tokenizer models/flux1-schnell/tokenizer_t5/tokenizer.json -p "a green apple" --device none -W 256 -H 256 --steps 2 -o output.png` | about 12.0 GB | 256×256 2-step image in 50.5s (2026-09-28) | MM-DiT + CLIP-L + T5-XXL on pure CPU SIMD. |
| Z-Image-Turbo (CPU), `z_image_turbo-Q4_0.gguf` | `stingray image -m models/_models/z_image_turbo-Q4_0.gguf --vae models/z-image-turbo/vae/diffusion_pytorch_model.safetensors --qwen-encoder models/_models/Z-Image-AbliteratedV1.Q5_K_M.gguf --qwen-tokenizer models/z-image-turbo/tokenizer/tokenizer.json -p "a red apple on a wooden table" --device none -W 256 -H 256 --steps 2 -o output.png` | about 6.5 GB | 256×256 2-step image in 54.6s (2026-09-28) | S3-DiT + Qwen3-4B on pure CPU SIMD. |
| Wan 2.1 Video Diffusion 1.3B, `Wan2.1-T2V-1.3B-Q4_0.gguf` | `stingray image -m models/_models/Wan2.1-T2V-1.3B-Q4_0.gguf --vae models/wan2.1/Wan2.1_VAE.safetensors --umt5-encoder models/wan2.1/models_t5_umt5-xxl-enc-bf16.safetensors --umt5-tokenizer models/wan2.1/umt5-tokenizer.json -p "a cat" -W 256 -H 256 --steps 2 --video-frames 1 -o output.png` | about 14.5 GB | 256×256 frame in 23.2s (2026-09-28) | DiT + UMT5-XXL text encoder on CPU SIMD. |
| Wan 2.2 14B Video Diffusion, `wan2.2_t2v_high_noise_14B_Q4_K_S.gguf` | `stingray image -m K:/_other_models/wan2.2_t2v_high_noise_14B_Q4_K_S.gguf --vae models/wan2.1/Wan2.1_VAE.safetensors --umt5-encoder models/wan2.1/models_t5_umt5-xxl-enc-bf16.safetensors --umt5-tokenizer models/wan2.1/umt5-tokenizer.json -p "a cat" -W 256 -H 256 --steps 2 --video-frames 1 -o output.png` | about 18.0 GB | 256×256 frame in 383.2s (2026-09-28) | 14-billion parameter DiT + UMT5-XXL text encoder on CPU SIMD. |
| ACE-Step 1.5 Turbo (Audio Diffusion CPU), `models/acestep-v15/` | `cmd /c "set STINGRAY_BACKEND=cpu && dotnet test --filter Bench_Generate_2sAudio"` | about 1.9 GB | 2s audio generated in 31.2s (15.6x RTF) (2026-09-28) | Pure CPU SIMD execution of DiT + Oobleck VAE. |

---

## Part II: GPU Acceleration (Vulkan / Integrated or Dedicated GPU)

Pipelines and models in this section leverage GPU acceleration (measured here via Vulkan on integrated AMD Radeon Graphics; also compatible with NVIDIA CUDA and discrete GPUs).

### Running GPU-accelerated models on CPU

Every pipeline in this section can run entirely on CPU without any GPU or dedicated VRAM:
- **Image & Video Diffusion (`stingray image`)**: Add `--device none` (or `--backend cpu`). By default `stingray image` probes for Vulkan/CUDA GPU; passing `--device none` forces pure CPU SIMD kernels.
- **Large Language Models & Vision (`stingray`)**: Omit `-g -1` or specify `-g 0`. By default `stingray` uses CPU SIMD; `-g -1` (or `--ngl -1`) offloads transformer layers to GPU.
- **Audio Diffusion (`AceStepPipeline`)**: Set `STINGRAY_BACKEND=cpu` in the environment before invoking the pipeline.

The tables below provide both the GPU acceleration command and its verified CPU fallback counterpart side-by-side.

### 1. Image & Video Diffusion (Generative AI)

| Model and file | Command | RAM / VRAM needed | Speed on the reference machine | Notes |
|---|---|---|---|---|
| Stable Diffusion 1.5 (GPU), `v1-5-pruned-emaonly.safetensors` | `stingray image -m models/sd15/v1-5-pruned-emaonly.safetensors -p "a photograph of an astronaut riding a horse" -W 256 -H 256 --steps 4 -o output.png` | about 2.5 GB | 256×256 4-step image in 16.0s (2026-09-28) | UNet + CLIP on Vulkan GPU. |
| Stable Diffusion 1.5 (CPU Fallback), `v1-5-pruned-emaonly.safetensors` | `stingray image -m models/sd15/v1-5-pruned-emaonly.safetensors -p "a photograph of an astronaut riding a horse" -W 256 -H 256 --steps 4 --device none -o output.png` | about 2.5 GB | 256×256 4-step image in 13.4s (2026-09-28) | Pure CPU SIMD execution via `--device none`. |
| Stable Diffusion 3.5 Medium (GPU), `sd3.5_medium-Q4_K_M.gguf` | `stingray image -m models/_models/sd3.5_medium-Q4_K_M.gguf --clip-l models/sd35-medium-aux/text_encoder/model.fp16.safetensors --clip-g models/sd35-medium-aux/text_encoder_2/model.fp16.safetensors --vae models/sd35-medium-aux/vae/diffusion_pytorch_model.safetensors --clip-tokenizer models/clip_tokenizer.json --steps 4 -W 256 -H 256 -p "a red car" -o output.png` | about 5.5 GB | 256×256 4-step image in 25.1s (2026-09-28) | MMDiT + dual CLIP-L / OpenCLIP-bigG on Vulkan GPU. |
| Stable Diffusion 3.5 Medium (CPU Fallback), `sd3.5_medium-Q4_K_M.gguf` | `stingray image -m models/_models/sd3.5_medium-Q4_K_M.gguf --clip-l models/sd35-medium-aux/text_encoder/model.fp16.safetensors --clip-g models/sd35-medium-aux/text_encoder_2/model.fp16.safetensors --vae models/sd35-medium-aux/vae/diffusion_pytorch_model.safetensors --clip-tokenizer models/clip_tokenizer.json --device none --steps 4 -W 256 -H 256 -p "a red car" -o output.png` | about 5.5 GB | 256×256 4-step image in ~42s (2 steps in 21.1s) (2026-09-28) | Add `--device none` for pure CPU SIMD execution. |
| FLUX.1-schnell (GPU), `flux1-schnell-Q4_K_S.gguf` | `stingray image -m models/flux1-schnell/flux1-schnell-Q4_K_S.gguf --vae models/flux1-schnell/ae.safetensors --clip-l models/flux1-schnell/clip_l.safetensors --clip-tokenizer models/flux1-schnell/tokenizer_clip/tokenizer.json --t5xxl models/flux1-schnell/t5xxl_fp8_e4m3fn.safetensors --t5-tokenizer models/flux1-schnell/tokenizer_t5/tokenizer.json -p "a green apple" -W 256 -H 256 --steps 4 -o output.png` | about 12.0 GB | 256×256 4-step image in 71.2s (2026-09-28) | MM-DiT + CLIP-L + T5-XXL on Vulkan GPU. |
| FLUX.1-schnell (CPU Fallback), `flux1-schnell-Q4_K_S.gguf` | `stingray image -m models/flux1-schnell/flux1-schnell-Q4_K_S.gguf --vae models/flux1-schnell/ae.safetensors --clip-l models/flux1-schnell/clip_l.safetensors --clip-tokenizer models/flux1-schnell/tokenizer_clip/tokenizer.json --t5xxl models/flux1-schnell/t5xxl_fp8_e4m3fn.safetensors --t5-tokenizer models/flux1-schnell/tokenizer_t5/tokenizer.json -p "a green apple" --device none -W 256 -H 256 --steps 2 -o output.png` | about 12.0 GB | 256×256 2-step image in 50.5s (2026-09-28) | Add `--device none` for pure CPU SIMD execution. |
| Z-Image-Turbo (GPU), `z_image_turbo-Q4_0.gguf` | `stingray image -m models/_models/z_image_turbo-Q4_0.gguf --vae models/z-image-turbo/vae/diffusion_pytorch_model.safetensors --qwen-encoder models/_models/Z-Image-AbliteratedV1.Q5_K_M.gguf --qwen-tokenizer models/z-image-turbo/tokenizer/tokenizer.json -p "a red apple on a wooden table" -W 256 -H 256 --steps 4 -o output.png` | about 6.5 GB | 256×256 4-step image in 74.0s (2026-09-28) | S3-DiT + Qwen3-4B text encoder on Vulkan GPU SGEMM. |
| Z-Image-Turbo (CPU Fallback), `z_image_turbo-Q4_0.gguf` | `stingray image -m models/_models/z_image_turbo-Q4_0.gguf --vae models/z-image-turbo/vae/diffusion_pytorch_model.safetensors --qwen-encoder models/_models/Z-Image-AbliteratedV1.Q5_K_M.gguf --qwen-tokenizer models/z-image-turbo/tokenizer/tokenizer.json -p "a red apple on a wooden table" --device none -W 256 -H 256 --steps 2 -o output.png` | about 6.5 GB | 256×256 2-step image in 54.6s (2026-09-28) | Add `--device none` for pure CPU SIMD execution. |
| Wan 2.1 Video Diffusion 1.3B (GPU), `Wan2.1-T2V-1.3B-Q4_0.gguf` | `stingray image -m models/_models/Wan2.1-T2V-1.3B-Q4_0.gguf --vae models/wan2.1/Wan2.1_VAE.safetensors --umt5-encoder models/wan2.1/models_t5_umt5-xxl-enc-bf16.safetensors --umt5-tokenizer models/wan2.1/umt5-tokenizer.json -p "a cat" -W 256 -H 256 --steps 2 --video-frames 1 --device 0 -o output.png` | about 14.5 GB | 256×256 frame in 25.5s (DiT 2 steps in 5.6s) (2026-09-28) | DiT + streamed UMT5 on Vulkan GPU (`--device 0`). |
| Wan 2.1 Video Diffusion 1.3B (CPU Fallback), `Wan2.1-T2V-1.3B-Q4_0.gguf` | `stingray image -m models/_models/Wan2.1-T2V-1.3B-Q4_0.gguf --vae models/wan2.1/Wan2.1_VAE.safetensors --umt5-encoder models/wan2.1/models_t5_umt5-xxl-enc-bf16.safetensors --umt5-tokenizer models/wan2.1/umt5-tokenizer.json -p "a cat" -W 256 -H 256 --steps 2 --video-frames 1 --device none -o output.png` | about 14.5 GB | 256×256 frame in 23.2s (2026-09-28) | DiT + UMT5-XXL text encoder on CPU SIMD (`--device none`). |

### 2. Text-to-Music Audio Diffusion

| Model and file | Command / Invocation | RAM / VRAM needed | Speed on the reference machine | Notes |
|---|---|---|---|---|
| ACE-Step 1.5 Turbo (GPU), `models/acestep-v15/` | `dotnet test --filter Bench_Generate_2sAudio_VulkanGpu` | 1.9 GB peak | 10s audio in 17.3-17.8s (2026-09-28) | DiT runs on Vulkan GPU; VAE decode 6.5s via tiled GEMM. |
| ACE-Step 1.5 Turbo (CPU Fallback), `models/acestep-v15/` | `cmd /c "set STINGRAY_BACKEND=cpu && dotnet test --filter Bench_Generate_2sAudio"` | 1.9 GB peak | 2s audio in 31.2s (15.6x RTF) (2026-09-28) | Pure CPU SIMD execution via `STINGRAY_BACKEND=cpu`. |

### 3. Large Language & Vision Models (GPU Offload vs CPU)

Use `-g -1` (or `--ngl -1`) to offload all transformer layers to GPU VRAM, or omit `-g` (or set `-g 0`) to run on CPU cores.

| Model and file | Command | RAM / VRAM needed | Speed on the reference machine | Notes |
|---|---|---|---|---|
| Pythia 160M (GPU), `pythia-160m.Q8_0.gguf` | `stingray -m models/_models/pythia-160m.Q8_0.gguf -p "Hello world" -g -1` | under 0.5 GB VRAM | Prompt 135.5 tok/s, decode 128.9 tok/s (2026-09-28) | 2.5x faster decode on Vulkan GPU (128.9 vs 52.3 tok/s). |
| Pythia 160M (CPU Fallback), `pythia-160m.Q8_0.gguf` | `stingray -m models/_models/pythia-160m.Q8_0.gguf -p "Hello world"` | under 0.5 GB RAM | Prompt 167.3 tok/s, decode 52.3 tok/s (2026-09-28) | Default CPU SIMD execution (`-g 0`). |
| SmolLM2 135M Instruct (GPU), `SmolLM2-135M-Instruct-Q4_K_M.gguf` | `stingray -m models/_models/SmolLM2-135M-Instruct-Q4_K_M.gguf -p "Hello world" -g -1` | about 0.2 GB VRAM | Prompt 64.8 tok/s, decode 60.9 tok/s (2026-09-28) | High-speed ultra-lightweight GPU offload (all 30 layers). |
| SmolLM2 135M Instruct (CPU Fallback), `SmolLM2-135M-Instruct-Q4_K_M.gguf` | `stingray -m models/_models/SmolLM2-135M-Instruct-Q4_K_M.gguf -p "Hello world"` | under 1 GB RAM | Prompt 48.5 tok/s, decode 21.2 tok/s (2026-09-28) | Default CPU SIMD execution (`-g 0`). |
| SmolLM2 360M Instruct (GPU), `SmolLM2-360M-Instruct-Q4_K_M.gguf` | `stingray -m models/_models/SmolLM2-360M-Instruct-Q4_K_M.gguf -p "The capital of France is" -g -1` | about 0.4 GB VRAM | Prompt 29.7 tok/s, decode 28.5 tok/s (2026-09-28) | All 32 layers offloaded to Vulkan GPU. |
| SmolLM2 360M Instruct (CPU Fallback), `SmolLM2-360M-Instruct-Q4_K_M.gguf` | `stingray -m models/_models/SmolLM2-360M-Instruct-Q4_K_M.gguf -p "The capital of France is"` | under 1 GB RAM | Prompt 89.2 tok/s, decode 51.4 tok/s (2026-09-28) | Default CPU SIMD execution (`-g 0`). |
| Qwen2.5 0.5B Instruct (GPU), `qwen2.5-0.5b-instruct-q4_k_m.gguf` | `stingray -m models/_models/qwen2.5-0.5b-instruct-q4_k_m.gguf -p "Hello world" -g -1` | about 0.5 GB VRAM | Prompt 26.8 tok/s, decode 25.7 tok/s (2026-09-28) | All 24 layers offloaded to Vulkan GPU (AMD Radeon Graphics). |
| Qwen2.5 0.5B Instruct (CPU Fallback), `qwen2.5-0.5b-instruct-q4_k_m.gguf` | `stingray -m models/_models/qwen2.5-0.5b-instruct-q4_k_m.gguf -p "Hello world"` | about 1.0 GB RAM | Prompt about 318 tok/s (661 tokens), decode 62.9-69.1 tok/s (128 tokens) (2026-09-28, after int8 Q5_0 kernels; was 77-84 / 21-25) | Default CPU SIMD execution (`-g 0`). |
| Qwen2.5-Coder 0.5B Instruct (GPU), `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` | `stingray -m models/_models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf -p "Write a hello world in C#:" -g -1` | about 0.5 GB VRAM | Prompt 27.0 tok/s, decode 25.2 tok/s (2026-09-28) | All 24 layers offloaded to Vulkan GPU. |
| Qwen2.5-Coder 0.5B Instruct (CPU Fallback), `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` | `stingray -m models/_models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf -p "Write a hello world in C#:"` | about 1.0 GB RAM | Prompt 90.8 tok/s, decode 57.0 tok/s (2026-09-28) | Default CPU SIMD execution (`-g 0`). |
| Tencent Hunyuan 0.5B Instruct (GPU), `tencent_Hunyuan-0.5B-Instruct-Q8_0.gguf` | `stingray -m models/_models/tencent_Hunyuan-0.5B-Instruct-Q8_0.gguf -p "Hello world" -g -1` | about 0.7 GB VRAM | Prompt 25.1 tok/s, decode 27.3 tok/s (2026-09-28) | All 24 layers offloaded to Vulkan GPU. |
| Tencent Hunyuan 0.5B Instruct (CPU Fallback), `tencent_Hunyuan-0.5B-Instruct-Q8_0.gguf` | `stingray -m models/_models/tencent_Hunyuan-0.5B-Instruct-Q8_0.gguf -p "Hello world"` | about 1.0 GB RAM | Prompt 11.8 tok/s, decode 24.7 tok/s (2026-09-28) | Default CPU SIMD execution (`-g 0`). |
| Maincoder 1B (GPU), `Maincoder-1B-Q4_K_M.gguf` | `stingray -m models/_models/Maincoder-1B-Q4_K_M.gguf -p "Write hello world in python:" -g -1` | about 0.9 GB VRAM | Prompt 133.5 tok/s, decode 31.6 tok/s (2026-09-28) | All 32 layers offloaded to Vulkan GPU. |
| Maincoder 1B (CPU Fallback), `Maincoder-1B-Q4_K_M.gguf` | `stingray -m models/_models/Maincoder-1B-Q4_K_M.gguf -p "Write hello world in python:"` | about 1.2 GB RAM | Prompt 129.4 tok/s, decode 22.9 tok/s (2026-09-28) | Default CPU SIMD execution (`-g 0`). |
| Qwen2.5 1.5B Instruct (GPU), `qwen2.5-1.5b-instruct-q4_k_m.gguf` | `stingray -m models/_models/qwen2.5-1.5b-instruct-q4_k_m.gguf -p "Write 3 bullet points about Mars:" -g -1` | about 1.1 GB VRAM | Prompt 94.9 tok/s, decode 27.3 tok/s (2026-09-28) | All 28 layers offloaded to Vulkan GPU. |
| Qwen2.5 1.5B Instruct (CPU Fallback), `qwen2.5-1.5b-instruct-q4_k_m.gguf` | `stingray -m models/_models/qwen2.5-1.5b-instruct-q4_k_m.gguf -p "Write 3 bullet points about Mars:"` | about 1.5 GB RAM | Prompt 113.0 tok/s, decode 24.1 tok/s (2026-09-28) | Default CPU SIMD execution (`-g 0`). |
| OpenGVLab InternVL3 2B (GPU), `InternVL3-2B.Q4_K_M.gguf` | `stingray -m models/_models/InternVL3-2B.Q4_K_M.gguf --mmproj models/_models/mmproj-internvl3-2b-q8_0.gguf --image photo.png -p "Describe this image in one sentence." -g -1` | about 1.5 GB VRAM | Prompt 27.3 tok/s (256 img + 38 txt), decode 26.2 tok/s (2026-09-28) | All 28 layers offloaded to Vulkan GPU. |
| OpenGVLab InternVL3 2B (CPU Fallback), `InternVL3-2B.Q4_K_M.gguf` | `stingray -m models/_models/InternVL3-2B.Q4_K_M.gguf --mmproj models/_models/mmproj-internvl3-2b-q8_0.gguf --image photo.png -p "Describe this image in one sentence."` | about 1.8 GB RAM | Prompt 123.2 tok/s (256 img + 38 txt), decode 26.7 tok/s (2026-09-28) | Default CPU SIMD execution (batched prompt on AVX2). |
| Qwen3-VL 2B Instruct (GPU), `Qwen3VL-2B-Instruct-Q8_0.gguf` | `stingray -m models/_models/Qwen3VL-2B-Instruct-Q8_0.gguf --mmproj models/_models/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf --image photo.png -p "Describe this picture." -g -1` | about 2.2 GB VRAM | Prompt 10.5 tok/s (256 img + 14 txt), decode 10.6 tok/s (2026-09-28) | All 28 layers offloaded to Vulkan GPU. |
| Qwen3-VL 2B Instruct (CPU Fallback), `Qwen3VL-2B-Instruct-Q8_0.gguf` | `stingray -m models/_models/Qwen3VL-2B-Instruct-Q8_0.gguf --mmproj models/_models/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf --image photo.png -p "Describe this picture."` | about 2.5 GB RAM | Prompt 23.1 tok/s (64 img + 14 txt), decode 19.7 tok/s (2026-09-28) | Default CPU SIMD execution (`-g 0`). |
