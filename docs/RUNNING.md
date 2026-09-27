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
- Prompts are processed one token at a time, so long prompts take noticeably longer to start
  answering than on a plain transformer of the same size.
- Prompt caching (reusing a shared prefix between requests) is switched off automatically for them,
  because their recurrent state cannot be rewound.

**Checking quality yourself.** `stingray perplexity -m <model> -f <text> -c 2048` scores a text one
token at a time; add `--batched` to use the faster batched prompt path. Compare the `[1024,+)`
bucket with `llama-perplexity -m <model> -f <text> -c 2048 --chunks 1`.

## Per model

Commands are CPU runs of the exact file listed. Decode is generation speed after the prompt.

| Model and file | RAM needed | Speed on the reference machine | Command | Notes |
|---|---|---|---|---|
| IBM Granite 4.0-H 1B, `granite-4.0-h-1b-Q8_0.gguf` | about 1.6 GB | Decode 21.6 tok/s, prompt 17 tok/s (2026-09-27) | `stingray -m granite-4.0-h-1b-Q8_0.gguf -p "..."` | Hybrid Mamba-2. Decode is close to this machine's memory-bandwidth limit for the file size. |
| NVIDIA Nemotron Nano 12B v2 VL, `nemotron-nano-12b-v2-vl-Q2_K.gguf` (text) | about 4.7 GB | Decode 6.6 tok/s, prompt 5.8 tok/s (2026-09-27) | `stingray -m nemotron-nano-12b-v2-vl-Q2_K.gguf -p "..."` | Text only; the image tower is not wired. Q2_K is very lossy: prefer a larger quantisation if RAM allows (not yet measured). |
| Qwen3-VL 2B Instruct, `Qwen3VL-2B-Instruct-Q8_0.gguf` + `mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf` | 2.7 GB peak working set with one 300-token image (2026-09-27) | Decode 18.9 tok/s, prompt (image + text) 20.5 tok/s (2026-09-27) | `stingray -m Qwen3VL-2B-Instruct-Q8_0.gguf --mmproj mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf --image page.png -p "..."` | Image input is CPU only and PNG only. The image took 300 tokens for a 1.2 MP photo. |
| NVIDIA Parakeet TDT 0.6B v2 (speech to text), `parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-q4_k.gguf` | 2.8 GB peak working set (2026-09-27) | 14.2 s of speech in 3.6 s, 3.9x real time (2026-09-27) | `stingray stt -m parakeet -i speech.wav` | English, with casing and punctuation and word timestamps. The file is 378 MB but the weights are expanded to F32 in RAM, hence 2.8 GB. |
| Liquid LFM2 1.2B, `LFM2-1.2B-Q8_0.gguf` | about 1.3 GB | Decode 29.4 tok/s, prompt 13.4 tok/s (2026-09-27) | `stingray -m LFM2-1.2B-Q8_0.gguf -p "..."` | Needs BOS (added automatically). Licence: free commercial use only under $10M annual revenue. |
| GLM-4.5-Air REAP 82B-A12B, `cerebras_GLM-4.5-Air-REAP-82B-A12B-Q2_K.gguf` | 33 GB peak working set (llama.cpp on the same file: 46 GB) (2026-09-27) | Decode about 2.2 tok/s (scoring, one token at a time) | Not admitted: perplexity 1.9% worse than llama.cpp (`docs/1-correctness/bugstofix.md`) | Mixture of experts, 12B active. Leave about 13 GB free for the OS alongside it on a 64 GB machine. |
