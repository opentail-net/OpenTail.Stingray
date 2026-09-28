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

Commands are CPU runs of the exact file listed. Decode is generation speed after the prompt.

| Model and file | RAM needed | Speed on the reference machine | Command | Notes |
|---|---|---|---|---|
| Qwen2.5 0.5B Instruct, `qwen2.5-0.5b-instruct-q4_k_m.gguf` (`stingray setup chat`) | about 1 GB | Decode 21.3-24.6 tok/s, prompt 77-84 tok/s over 40 tokens, 3 runs (2026-09-28). llama.cpp on the same file (`llama-bench -t 8`): 97.8 / 379 tok/s, so ours is about 0.24x; being looked at (docs/103 item 15) | `stingray -m qwen2.5-0.5b-instruct-q4_k_m.gguf -p "..."` | The catalog's default chat model. |
| Piper en_US-lessac-medium (text to speech), `en_US-lessac-medium.onnx` + `.onnx.json` (`stingray setup speak`) | under 1 GB | 2.83-2.93 s of audio in 1.12-1.35 s, 3 runs (2026-09-28) | `stingray tts -e piper -m en_US-lessac-medium.onnx -t "..." -o out.wav` | 22.05 kHz. The voice data has its own licence (setup asks). |
| Whisper base (speech to text), `ggml-base.bin` (`stingray setup transcribe`) | under 1 GB | A 2.9 s clip in 1.03-1.54 s, 3 runs (2026-09-28) | `stingray stt -m base --model-file ggml-base.bin -i speech.wav` | Word-exact on the Piper sample above. |
| IBM Granite 4.0-H 1B, `granite-4.0-h-1b-Q8_0.gguf` | about 1.6 GB | Decode 21.0 tok/s, prompt 63.4 tok/s over 511 tokens (2026-09-28; 21.3 tok/s one token at a time) | `stingray -m granite-4.0-h-1b-Q8_0.gguf -p "..."` | Hybrid Mamba-2. Decode is close to this machine's memory-bandwidth limit for the file size. |
| NVIDIA Nemotron Nano 12B v2 VL, `nemotron-nano-12b-v2-vl-Q2_K.gguf` (text) | about 4.7 GB | Decode 6.2-6.7 tok/s, prompt 11.7 tok/s over 582 tokens (2026-09-28; 6.7 tok/s one token at a time) | `stingray -m nemotron-nano-12b-v2-vl-Q2_K.gguf -p "..."` | Text only; the image tower is not wired. Q2_K is very lossy: prefer a larger quantisation if RAM allows (not yet measured). |
| Qwen3-VL 2B Instruct, `Qwen3VL-2B-Instruct-Q8_0.gguf` + `mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf` | 2.7 GB peak working set with one 300-token image (2026-09-27) | Decode 17.5-17.9 tok/s, prompt (784 image + 17 text tokens) 78-86 tok/s, 3 runs; 32-35 s wall for a 40-token answer (2026-09-28; was 20 tok/s and 63 s before the image tokens were batched) | `stingray -m Qwen3VL-2B-Instruct-Q8_0.gguf --mmproj mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf --image page.png -p "..."` | PNG only. The image took 300 tokens for a 1.2 MP photo. Vulkan (`-g -1 --backend vulkan`, 2026-09-28): prompt (784 image + 17 text) 9.5 tok/s, decode 8.9 tok/s on this machine's integrated GPU against CPU 20.1 / 18.8 on the same run; an iGPU result, not a verdict on discrete GPUs (CLAUDE.md rule 13). Qwen-VL models run the per-token Vulkan trunk (no batched prefill). |
| LLaVA-1.5 7B, `llava-v1.5-7b/ggml-model-q4_k.gguf` + `mmproj-llava-v1.5-7b-f16.gguf` | 4.6 GB peak working set (3.80 GiB text + 0.62 GiB mmproj pre-faulted) (2026-09-27) | Decode 7.0 tok/s, prompt (576 image + 22 text) 7.9 tok/s, 3 runs (2026-09-27) | `stingray -m llava-v1.5-7b/ggml-model-q4_k.gguf --mmproj mmproj-llava-v1.5-7b-f16.gguf --image test-1.png -p "What newspaper is this and what is the main headline?"` | Direct-spliced image tokens (plain 32k LLaMA-2 vocab, no special placeholder token in vocab). Prompt formatted as Vicuna (`USER: <image>{prompt}\nASSISTANT:`). Produces 576 soft tokens (336x336 patch grid). |
| NVIDIA Parakeet TDT 0.6B v2 (speech to text), `parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-q4_k.gguf` | 1.1 GB peak working set (2026-09-27) | 14.2 s of speech in 1.43 s, about 10x real time, NativeAOT binary, 3 runs (2026-09-27; CrispASR on the same file: 1.66 s) | `stingray stt -m parakeet -i speech.wav` | English, with casing and punctuation and word timestamps. The CTC file (`--model-file parakeet-ctc-0.6b-q4_k.gguf`) runs 1.04 s on the same clip, 1.05 GB. |
| ACE-Step 1.5 Turbo (text to music), `models/acestep-v15/{turbo,vae}.safetensors` + `qwen3-embedding-0.6b-f16.gguf` | 1.9 GB peak working set; the DiT runs on Vulkan, whose shared iGPU memory may not all show in that figure (2026-09-28) | 10 s of audio in 17.3-17.8 s, 3 runs (2026-09-28, after the VAE convs moved to GEMM; was 99-102 s): DiT 8 steps 9.2-9.6 s on the iGPU, VAE decode 6.5 s on CPU (was 88-90 s). audio.cpp CPU-only on the q8_0 GGUF: 37 s | No CLI command yet; `AceStepPipeline.Generate` (see `AceStepPipelineEndToEndTests`) | Instrumental only (planner LM not ported). The DiT (about half the time now) runs on Vulkan by default. |
| Liquid LFM2 1.2B, `LFM2-1.2B-Q8_0.gguf` | about 1.3 GB | Decode 28-30 tok/s, prompt 101 tok/s over 538 tokens (2026-09-28; 30 tok/s one token at a time) | `stingray -m LFM2-1.2B-Q8_0.gguf -p "..."` | Needs BOS (added automatically). Licence: free commercial use only under $10M annual revenue. |
| GLM-4.5-Air REAP 82B-A12B, `cerebras_GLM-4.5-Air-REAP-82B-A12B-Q2_K.gguf` | 33 GB peak working set (llama.cpp on the same file: 46 GB) (2026-09-27) | Decode about 2.2 tok/s (scoring, one token at a time) | Not admitted: perplexity 1.9% worse than llama.cpp (`docs/1-correctness/bugstofix.md`) | Mixture of experts, 12B active. Leave about 13 GB free for the OS alongside it on a 64 GB machine. |
