# Running models well

How to get the best result from a model on real hardware: the command to use, how much memory it
really needs, and how fast it runs. [MODELS.md](MODELS.md) says which models to pick and
[STATUS.md](STATUS.md) says what works and how that was checked. This page is the practical part.

Every number here was measured, with the date and the machine. A row without a measurement says
so. Reference machine: AMD Ryzen 7 5700G (8 cores), 64 GB DDR4, integrated GPU only, Windows 11.

## General rules

These apply to every model and were confirmed while admitting the models below.

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

## Per model

Commands are CPU runs of the exact file listed (unless noted as GPU/hybrid). Decode is generation speed after the prompt.

### 1. Large Language Models (Text Generation)

| Model and file | Command | RAM needed | Speed on the reference machine | Notes |
|---|---|---|---|---|
| Qwen2.5 0.5B Instruct, `qwen2.5-0.5b-instruct-q4_k_m.gguf` (`stingray setup chat`) | `stingray -m models/_models/qwen2.5-0.5b-instruct-q4_k_m.gguf -p "Hello world"` | about 1 GB | Decode 21.3-24.6 tok/s, prompt 77-84 tok/s over 40 tokens (2026-09-28) | Default chat catalog model. |
| Qwen2.5-Coder 0.5B Instruct, `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` | `stingray -m models/_models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf -p "Write a hello world in C#:"` | about 1 GB | Prompt 90.8 tok/s, decode 57.0 tok/s (2026-09-28) | High-throughput code generation model. |
| Qwen2.5 1.5B Instruct, `qwen2.5-1.5b-instruct-q4_k_m.gguf` | `stingray -m models/_models/qwen2.5-1.5b-instruct-q4_k_m.gguf -p "Write 3 bullet points about Mars:"` | about 1.5 GB | Prompt 113.0 tok/s, decode 24.1 tok/s (2026-09-28) | Fast general instruction model. |
| Qwen3 0.6B Base, `Qwen3-0.6B-Q8_0.gguf` | `stingray -m models/_models/Qwen3-0.6B-Q8_0.gguf -p "Hello world" --temp 0.6` | about 1.2 GB | Prompt 31.3 tok/s, decode 39.6 tok/s (2026-09-28) | Reasoning model (outputs thinking tokens). |
| Qwen3 4B Instruct, `Qwen3-4B-Q4_K_M.gguf` | `stingray -m models/_models/Qwen3-4B-Q4_K_M.gguf -p "Hello world" --temp 0.6` | about 3.2 GB | Prompt 38.0 tok/s, decode 10.7 tok/s (2026-09-28) | Reasoning model with thinking traces. |
| SmolLM2 135M Instruct, `SmolLM2-135M-Instruct-Q4_K_M.gguf` | `stingray -m models/_models/SmolLM2-135M-Instruct-Q4_K_M.gguf -p "Hello world"` | under 1 GB | Prompt 48.5 tok/s, decode 21.2 tok/s (2026-09-28) | Ultra-lightweight edge model. |
| SmolLM2 360M Instruct, `SmolLM2-360M-Instruct-Q4_K_M.gguf` | `stingray -m models/_models/SmolLM2-360M-Instruct-Q4_K_M.gguf -p "The capital of France is"` | under 1 GB | Prompt 89.2 tok/s, decode 51.4 tok/s (2026-09-28) | Very fast small model. |
| SmolLM2 1.7B Instruct, `SmolLM2-1.7B-Instruct-Q4_K_M.gguf` | `stingray -m models/_models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "The capital of France is"` | about 1.5 GB | Prompt 86.2 tok/s, decode 19.7 tok/s (2026-09-28) | Well-balanced small instruction model. |
| SmolLM3 Q4_K_M, `SmolLM3-Q4_K_M.gguf` | `stingray -m models/_models/SmolLM3-Q4_K_M.gguf -p "Hello world" --temp 0.6` | about 2.5 GB | Prompt 119.5 tok/s, decode 11.9 tok/s (2026-09-28) | Reasoning model architecture with thinking trace. |
| Microsoft Phi-2, `phi-2.Q4_K_M.gguf` | `stingray -m models/_models/phi-2.Q4_K_M.gguf -p "Instruct: Explain gravity in one sentence.\nOutput:"` | about 2.8 GB | Prompt 5.4 tok/s, decode 11.4 tok/s (2026-09-28) | Classic 2.7B reasoning/math architecture. |
| Microsoft Phi-3 Mini 4K Instruct, `Phi-3-mini-4k-instruct-Q4_K_M.gguf` | `stingray -m models/_models/Phi-3-mini-4k-instruct-Q4_K_M.gguf -p "Instruct: Explain gravity in one sentence.\nOutput:"` | about 2.8 GB | Prompt 5.6 tok/s, decode 10.8 tok/s (2026-09-28) | 3.8B instruction model. |
| Mistral 7B Instruct v0.3, `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` | `stingray -m models/_models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf -p "Hello world"` | about 5.0 GB | Prompt 7.6 tok/s, decode 6.5 tok/s (2026-09-28) | 32k context LLaMA-family dense transformer. |
| Ministral 8B Instruct 2410, `Ministral-8B-Instruct-2410-Q4_K_M.gguf` | `stingray -m models/_models/Ministral-8B-Instruct-2410-Q4_K_M.gguf -p "Hello world"` | about 5.5 GB | Prompt 5.7 tok/s, decode 6.4 tok/s (2026-09-28) | Mistral-family interleaved attention architecture. |
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
| Qwen3 8B Instruct, `Qwen3-8B-Q4_K_M.gguf` | `stingray -m models/_models/Qwen3-8B-Q4_K_M.gguf -p "Hello world" --temp 0.6` | about 5.0 GB | Prompt 24.1 tok/s, decode 6.2 tok/s (2026-09-28) | Full 8B reasoning model with thinking tokens. |
| IBM Granite 4.0-H 350M, `granite-4.0-h-350m-Q8_0.gguf` | `stingray -m models/_models/granite-4.0-h-350m-Q8_0.gguf -p "Hello world"` | under 1 GB | Prompt 42.4 tok/s, decode 58.4 tok/s (2026-09-28) | Ultra-compact Mamba-2 hybrid architecture. |
| OpenAI GPT-2, `gpt2.Q8_0.gguf` | `stingray -m models/_models/gpt2.Q8_0.gguf -p "Hello world"` | under 0.5 GB | Prompt 177.4 tok/s, decode 53.2 tok/s (2026-09-28) | Classic 124M autoregressive reference model. |
| OpenAI GPT-OSS 20B (MXFP4), `gpt-oss-20b-MXFP4.gguf` | `stingray -m models/_models/gpt-oss-20b-MXFP4.gguf -p "Hello world"` | about 13 GB | Prompt 2.1 tok/s, decode 5.5 tok/s (2026-09-28) | Native MXFP4 microscopic floating-point execution on CPU. |
| NVIDIA Nemotron Nano 12B v2 VL, `nemotron-nano-12b-v2-vl-Q2_K.gguf` (text) | `stingray -m models/_models/nemotron-nano-12b-v2-vl-Q2_K.gguf -p "Hello world"` | about 4.7 GB | Decode 6.2-6.7 tok/s, prompt 11.7 tok/s over 582 tokens (2026-09-28) | Text only; image tower not wired. |
| GLM-4.5-Air REAP 82B-A12B, `cerebras_GLM-4.5-Air-REAP-82B-A12B-Q2_K.gguf` | `stingray -m models/_models/cerebras_GLM-4.5-Air-REAP-82B-A12B-Q2_K.gguf -p "..."` | 33 GB peak working set (llama.cpp on the same file: 46 GB) | Decode about 2.2 tok/s (scoring, one token at a time) | MoE, 12B active. Not admitted: perplexity 1.9% worse than llama.cpp (`docs/1-correctness/bugstofix.md`). |

### 2. Multimodal Vision Understanding (VLM)

| Model and file | Command | RAM needed | Speed on the reference machine | Notes |
|---|---|---|---|---|
| OpenGVLab InternVL3 2B, `InternVL3-2B.Q4_K_M.gguf` | `stingray -m models/_models/InternVL3-2B.Q4_K_M.gguf --mmproj models/_models/mmproj-internvl3-2b-q8_0.gguf --image photo.png -p "Describe this image in one sentence."` | about 1.8 GB | Prompt 123.2 tok/s (256 img + 38 txt), decode 26.7 tok/s (2026-09-28) | Verified working vision captioning and object grounding. |
| Tencent Youtu-VL 4B, `youtu-vl-4b-Q8_0.gguf` | `stingray -m models/_models/youtu-vl-4b-Q8_0.gguf --mmproj models/_models/mmproj-youtu-vl-4b-BF16.gguf --image photo.png -p "Describe this picture."` | about 5.8 GB | Prompt 7.2 tok/s (64 img + 25 txt), decode 6.4 tok/s (2026-09-28) | High-accuracy visual document understanding. |
| PaddleOCR-VL 1.6, `paddleocr-vl-1.6.gguf` | `stingray -m models/_models/paddleocr-vl-1.6.gguf --mmproj models/_models/PaddleOCR-VL-1.6-GGUF-mmproj.gguf --image doc.png -p "Read the text in this image."` | about 1.5 GB | Prompt 88.7 tok/s (81 img + 17 txt) (2026-09-28) | Specialized OCR vision model. |
| Qwen3-VL 2B Instruct, `Qwen3VL-2B-Instruct-Q8_0.gguf` | `stingray -m models/_models/Qwen3VL-2B-Instruct-Q8_0.gguf --mmproj models/_models/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf --image page.png -p "..."` | 2.7 GB peak working set | Decode 17.5-17.9 tok/s, prompt (784 img + 17 txt) 78-86 tok/s (2026-09-28) | 2D M-RoPE image tokens with batched prompt processing. |
| IBM Granite Vision 3.2 2B, `granite-vision-3.2-2b-Q3_K_S.gguf` | `stingray -m models/_models/granite-vision-3.2-2b-Q3_K_S.gguf --mmproj models/_models/mmproj-granite-vision-3.2-2b-f16.gguf --image photo.png -p "Describe this picture."` | about 1.5 GB | Prompt 28.5 tok/s (729 img + 52 txt), decode 15.8 tok/s (2026-09-28) | High-accuracy dense vision understanding via MLP projector. |
| Qwen2.5-VL 7B Instruct, `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` | `stingray -m models/_models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf --mmproj models/_models/mmproj-qwen2.5-vl-7b-f16.gguf --image photo.png -p "Describe this picture."` | about 5.0 GB | Prompt 20.3 tok/s (81 img + 25 txt), decode 7.2 tok/s (2026-09-28) | 2D M-RoPE dynamic resolution vision transformer. |
| Xiaomi MiMo-VL 7B SFT, `mimo-vl-7b-sft-Q2_K.gguf` | `stingray -m models/_models/mimo-vl-7b-sft-Q2_K.gguf --mmproj models/_models/mmproj-mimo-vl-7b-sft-Q8_0.gguf --image photo.png -p "Describe this picture."` | about 3.5 GB | Prompt 14.3 tok/s (81 img + 31 txt), decode 10.0 tok/s (2026-09-28) | 7B visual chat assistant. |
| Moonshot Kimi-VL A3B Thinking, `kimi-vl-a3b-thinking-Q2_K.gguf` | `stingray -m models/_models/kimi-vl-a3b-thinking-Q2_K.gguf --mmproj models/_models/mmproj-kimi-vl-a3b-thinking-Q8_0.gguf --image photo.png -p "Describe this picture."` | about 6.5 GB | Prompt 14.8 tok/s (81 img + 22 txt), decode 11.4 tok/s (2026-09-28) | Multimodal MoE with visual reasoning/thinking traces. |
| StepFun Step3-VL 10B, `step3-vl-10b-Q2_K.gguf` | `stingray -m models/_models/step3-vl-10b-Q2_K.gguf --mmproj models/_models/mmproj-step3-vl-10b-F16.gguf --image photo.png -p "Describe this picture."` | about 3.8 GB | Prompt 9.5 tok/s (171 img + 14 txt), decode 9.6 tok/s (2026-09-28) | 10B bilingual visual reasoning model. |
| Dots.OCR 1.5B, `dots.ocr-Q8_0.gguf` | `stingray -m models/_models/dots.ocr-Q8_0.gguf --mmproj models/_models/mmproj-dots.ocr-Q8_0.gguf --image doc.png -p "Read the text in this image."` | about 2.2 GB | Prompt 29.0 tok/s (81 img + 12 txt), decode 16.0 tok/s (2026-09-28) | Specialized OCR vision model. |
| LLaVA-1.5 7B, `llava-v1.5-7b/ggml-model-q4_k.gguf` | `stingray -m models/_models/ggml-model-q4_k.gguf --mmproj models/_models/mmproj-llava-v1.5-7b-f16.gguf --image test-1.png -p "..."` | 4.6 GB peak working set | Decode 7.0 tok/s, prompt (576 img + 22 txt) 7.9 tok/s (2026-09-27) | Direct-spliced soft image tokens in Vicuna format. |

### 3. Speech-to-Text (STT / ASR)

| Model and file | Command | RAM needed | Speed on the reference machine | Notes |
|---|---|---|---|---|
| SenseVoice Small (ONNX), `sensevoice-small.int8.onnx` | `stingray stt -m sensevoice --model-file models/_models/sensevoice-small.int8.onnx -i speech.wav` | under 1 GB | 1.58s audio transcribed in 0.17s (9.4x real-time) (2026-09-28) | Multilingual with automated timestamped segments. |
| FunASR Paraformer (ONNX), `paraformer-zh-small.int8.onnx` | `stingray stt -m paraformer --model-file models/_models/paraformer-zh-small.int8.onnx -i speech.wav` | under 1 GB | 1.58s audio transcribed in 0.07s (21.1x real-time) (2026-09-28) | Fluent Mandarin transcription pipeline. |
| OpenAI Whisper Base, `ggml-base.bin` | `stingray stt -m base --model-file models/ggml-base.bin -i speech.wav` | under 1 GB | 2.9s audio transcribed in 1.03-1.54s (2026-09-28) | Standard English/multilingual reference transcription. |
| NVIDIA Parakeet TDT 0.6B v2, `parakeet-tdt-0.6b-v2-q4_k.gguf` | `stingray stt -m parakeet --model-file models/parakeet-ctc-0.6b-q4_k.gguf -i speech.wav` | about 1.1 GB | 14.2s speech in 1.43s (~10x real-time) (2026-09-27) | Fast Conformer architecture with word timestamps. |

### 4. Text-to-Speech (TTS)

| Model and file | Command | RAM needed | Speed on the reference machine | Notes |
|---|---|---|---|---|
| Kokoro 82M (GGUF), `kokoro-82m-q8_0.gguf` | `stingray tts -e kokoro -m models/_models/kokoro-82m-q8_0.gguf -v af_heart --voices-dir models/_models -t "Hello world" -o speech.wav` | under 1 GB | 1.58s audio generated in 2.09s (1.32x RTF) (2026-09-28) | High-fidelity voice styles (`af_heart`, etc.). |
| MeloTTS (ONNX), `melotts-zh_en.onnx` | `stingray tts -e melo -m models/_models/melotts-zh_en.onnx -t "Hello world" -o speech.wav` | under 1 GB | 1.02s audio generated in 2.32s (2.27x RTF) (2026-09-28) | Fast bilingual English/Chinese synthesis. |
| Piper en_US Lessac (ONNX), `en_US-lessac-medium.onnx` | `stingray tts -e piper -m models/en_US-lessac-medium.onnx -t "Hello world" -o speech.wav` | under 1 GB | 2.83-2.93s audio in 1.12-1.35s (RTF 0.42x) (2026-09-28) | Fastest CPU TTS engine. |

### 5. Semantic Embeddings & Reranking

| Model and file | Command | RAM needed | Speed on the reference machine | Notes |
|---|---|---|---|---|
| Qwen3 Embedding 0.6B (GGUF), `qwen3-embedding-0.6b-q8_0.gguf` | `stingray embed -m models/qwen3-embedding-0.6b/qwen3-embedding-0.6b-q8_0.gguf -p "Hello world"` | about 1.0 GB | 1 text embedded in 314ms (1024-dim, L2-normed) (2026-09-28) | Dense semantic vectors with configurable pooling. |
| Qwen3 Cross/Bi-Encoder Reranker, `qwen3-embedding-0.6b-q8_0.gguf` | `stingray rerank -m models/qwen3-embedding-0.6b/qwen3-embedding-0.6b-q8_0.gguf -q "query" -d "doc1" -d "doc2"` | about 1.0 GB | 2 documents ranked in 749ms (2026-09-28) | Relevance scoring and top-N ranking. |

### 6. Music & Audio Diffusion

| Model and file | Command / Invocation | RAM needed | Speed on the reference machine | Notes |
|---|---|---|---|---|
| ACE-Step 1.5 Turbo (Text to Music), `models/acestep-v15/` | `AceStepPipeline.Generate` (see tests) | 1.9 GB peak | 10s audio in 17.3-17.8s (2026-09-28) | DiT runs on Vulkan GPU; VAE decode 6.5s via tiled GEMM. |
