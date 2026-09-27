# Models: where to find them

A short list of models worth starting with, one table per task, and where to download each. Every
link goes to the exact file on Hugging Face. This is a recommendation list, not a coverage list:
Stingray runs many more models (see [STATUS.md](STATUS.md)), but these are the ones we suggest
first. [RUNNING.md](RUNNING.md) has the command, the RAM each one really needs, and its measured speed.

**Columns**

- **Pick it when**: the reason to choose this one over its neighbours.
- **Size**: what you download.
- **Licence**: check it before shipping anything. *Non-commercial* means exactly that.
- **Tested**: **file** means this exact file was run by us; **family** means the model family is
  verified but this particular file has not been run yet. Evidence is in [STATUS.md](STATUS.md).

First version, 2026-09-26. Expect columns and entries to change as the list is used.

## Chat and reasoning

| Model | Pick it when | Download | Size | Licence | Tested |
|---|---|---|---|---|---|
| Qwen2.5 0.5B Instruct | Your first run: tiny, quick to download, surprisingly capable | [qwen2.5-0.5b-instruct-q4_k_m.gguf](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/blob/main/qwen2.5-0.5b-instruct-q4_k_m.gguf) | 491 MB | Apache-2.0 | file (README quick start) |
| Qwen3 4B | Good everyday assistant on a laptop (8 GB RAM) | [Qwen3-4B-Q4_K_M.gguf](https://huggingface.co/Qwen/Qwen3-4B-GGUF/blob/main/Qwen3-4B-Q4_K_M.gguf) | 2.5 GB | Apache-2.0 | family |
| Qwen3 8B | Best quality here that still fits 16 GB RAM; much faster with a GPU | [Qwen3-8B-Q4_K_M.gguf](https://huggingface.co/Qwen/Qwen3-8B-GGUF/blob/main/Qwen3-8B-Q4_K_M.gguf) | 5.0 GB | Apache-2.0 | family |
| Phi-3 mini 4k | Small, strong at reasoning, permissive licence | [Phi-3-mini-4k-instruct-q4.gguf](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/blob/main/Phi-3-mini-4k-instruct-q4.gguf) | 2.4 GB | MIT | file (compared with llama.cpp) |
| gpt-oss 20B | OpenAI's open-weight reasoning model; needs about 16 GB | [gpt-oss-20b-MXFP4.gguf](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/blob/main/gpt-oss-20b-MXFP4.gguf) | 12.1 GB | Apache-2.0 | file (compared with llama.cpp) |
| Granite 4.0-H 1B | Small IBM model with a hybrid Mamba-2 design and a permissive licence; CPU only for now | [granite-4.0-h-1b-Q8_0.gguf](https://huggingface.co/ibm-granite/granite-4.0-h-1b-GGUF/blob/main/granite-4.0-h-1b-Q8_0.gguf) | 1.6 GB | Apache-2.0 | file (compared with llama.cpp) |
| LFM2 1.2B | Very small and quick, built for on-device use; CPU only for now | [LFM2-1.2B-Q8_0.gguf](https://huggingface.co/LiquidAI/LFM2-1.2B-GGUF/blob/main/LFM2-1.2B-Q8_0.gguf) | 1.2 GB | LFM Open License v1.0: free commercial use only under $10M annual revenue | file (compared with llama.cpp) |

Language models download most easily with the CLI: `stingray pull -r Qwen/Qwen3-4B-GGUF` picks the
Q4_K_M file for you.

## Understanding images

These need two files: the model and its image projector (`mmproj`). Pass both: `stingray -m
<model> --mmproj <mmproj> --image photo.png -p "What is in this picture?"`.

| Model | Pick it when | Download | Size | Licence | Tested |
|---|---|---|---|---|---|
| Gemma 3 4B | General questions about photos and screenshots | [gemma-3-4b-it-Q4_K_M.gguf](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/blob/main/gemma-3-4b-it-Q4_K_M.gguf) + [mmproj-model-f16.gguf](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/blob/main/mmproj-model-f16.gguf) | 2.5 GB + 0.9 GB | Gemma terms | file (text); family (images) |
| dots.ocr | Reading text out of documents, receipts, forms | [dots.ocr-Q8_0.gguf](https://huggingface.co/ggml-org/dots.ocr-GGUF/blob/main/dots.ocr-Q8_0.gguf) + [mmproj-dots.ocr-Q8_0.gguf](https://huggingface.co/ggml-org/dots.ocr-GGUF/blob/main/mmproj-dots.ocr-Q8_0.gguf) | 1.9 GB + 1.3 GB | MIT | file (compared with llama.cpp) |

## Text to speech

| Model | Pick it when | Download | Size | Licence | Tested |
|---|---|---|---|---|---|
| Piper, lessac (US English) | Fast, natural English; faster than real time on a CPU | [en_US-lessac-medium.onnx](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx) + [.onnx.json](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json) | 63 MB | MIT | file (README) |
| MMS-TTS | Speech in one of 1,100+ languages (`facebook/mms-tts-<language>`) | [facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng/tree/main) (whole folder) | 145 MB | **Non-commercial** (CC-BY-NC-4.0) | family |

Other Piper voices and languages are in the same
[piper-voices](https://huggingface.co/rhasspy/piper-voices/tree/main) repository. Always download
both the `.onnx` and its `.onnx.json`.

## Speech to text

| Model | Pick it when | Download | Size | Licence | Tested |
|---|---|---|---|---|---|
| Whisper base | Quick transcription, English and major languages | [ggml-base.bin](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-base.bin) | 148 MB | MIT | file (README) |
| Whisper large-v3-turbo | Best accuracy, many languages; slower | [ggml-large-v3-turbo.bin](https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-large-v3-turbo.bin) | 1.6 GB | MIT | family |

## Image generation

| Model | Pick it when | Download | Size | Licence | Tested |
|---|---|---|---|---|---|
| FLUX.1 schnell | High-quality images in a few steps | [flux1-schnell-Q4_K_S.gguf](https://huggingface.co/city96/FLUX.1-schnell-gguf/blob/main/flux1-schnell-Q4_K_S.gguf) + [t5xxl_fp8_e4m3fn.safetensors](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp8_e4m3fn.safetensors) + [clip_l.safetensors](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/clip_l.safetensors) + [ae.safetensors](https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/ae.safetensors) | 6.8 + 4.9 + 0.25 + 0.3 GB | Apache-2.0 | family |

Image generation also needs the text encoders' tokenizer files, and the VAE (`ae.safetensors`)
requires signing in to Hugging Face and accepting the model's terms first. A guided setup that
assembles all of this is planned ([103](3-product-and-runtime/103-front-door-design.md)); until then see the FLUX row in
[STATUS.md](STATUS.md).

## Search and retrieval

| Model | Pick it when | Download | Size | Licence | Tested |
|---|---|---|---|---|---|
| bge-small-en v1.5 | Text embeddings for semantic search (English) | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/main) (whole folder) | 133 MB | MIT | family |
| ms-marco MiniLM L6 v2 | Reranking search results by relevance | [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/main) (whole folder) | 91 MB | Apache-2.0 | family |

## How entries get on this list

- The model is a current leader for its task, not an old or niche one. It is replaced when a
  clearly better open model appears.
- A permissive licence is preferred. Anything else is flagged in the Licence column.
- The file comes from the model's official repository, or a well-known converter such as
  `ggml-org`, and the link points at the exact file.
- The model family is 🟢 in [STATUS.md](STATUS.md). "Tested: file" additionally means we ran that
  exact download.
