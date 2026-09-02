# OpenTail.Stingray

**Local LLM inference, multimodal vision, speech, and image generation, written in C#.** No Python, no P/Invoke to llama.cpp, no
sidecar process — the engine itself is managed .NET 10 code that runs in your process and
NativeAOT-publishes to a single binary.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![.NET 10](https://img.shields.io/badge/.NET-10-blue)]()
[![NativeAOT](https://img.shields.io/badge/NativeAOT-ready-green)]()
[![NuGet](https://img.shields.io/nuget/v/OpenTail.Stingray.svg)](https://www.nuget.org/packages/OpenTail.Stingray)

> **Built by [opentail.net](https://opentail.net)**

## What actually works today

A brutally honest status matrix, not a feature-list — and "verified" isn't one thing, so **Status** and **Confidence** are separate columns.

| Status | Meaning |
|---|---|
| 🟢 Correct | Produces the right output today |
| 🟡 Partial | Produces real output but with a known, open correctness gap |
| 🔴 Not there | Placeholder, unwired, or not attempted |

Confidence is *how* that was checked, since an exact numerical match, a golden token sequence, a perceptual listen, and "the model loads" are very different claims:

| Confidence | Meaning |
|---|---|
| 🔬 Golden-verified | Automated test compares against a real reference (exact tensor/logit match or cosine similarity threshold) |
| 👂 Working, listened/read | Real weights, full pipeline run end-to-end, a human (or transcription) confirmed the output is correct |
| ⚪ Implemented | Code exists and runs against real weights, not independently validated this pass |
| 🔵 Unexecuted | Real, non-stubbed implementation (reads genuine named weight tensors) that has never actually been run — no checkpoint downloaded/tested, so it's unknown whether it produces correct output, garbage, or crashes |
| 🔴 Placeholder | Not functional — stubbed, unwired, or synthetic data standing in for a real component |

Sourced from [`docs/audio-review-progress.md`](docs/audio-review-progress.md) and [`docs/diffusion-samples/README.md`](docs/diffusion-samples/README.md); dates are when each was last verified.

| Capability | Status | Confidence | Backend | Notes |
|---|:---:|:---:|---|---|
| LLM inference (GGUF) | 🟢 | 🔬 | CPU / CUDA / Vulkan | Core engine — the most mature, most tested part of the project |
| MMS-TTS | 🟢 | 🔬 | CPU | Golden-verified against the real HuggingFace reference (2026-08-30) |
| XTTS-v2 | 🟢 | 🔬 | CPU | Voice cloning, 13 pieces individually golden-verified, wired end-to-end (2026-08-30) |
| Chatterbox-Turbo | 🟢 | 🔬 | CPU | Golden-tested (`ChatterboxCfmDecoderTests`), perf-optimized |
| QwenTTS | 🟢 | 👂 | CPU | Was disabled after a QK-norm bug; fixed 2026-08-29 — verified via exact, word-for-word correct ASR transcriptions on two test phrases, not a numeric golden compare |
| Parler-TTS | 🟢 | 👂 | CPU | Two correctness bugs found & fixed by ear (2026-08-28) |
| F5-TTS | 🟢 | 👂 | CPU | RoPE bug root-caused and fixed (2026-08-28) |
| Fish Speech S2 Pro | 🟢 | 👂 | CPU | User-confirmed "sonically 100% spot on" (2026-08-29) |
| Whisper ASR | 🟢 | ⚪ | CPU | 100 languages; not re-verified this pass |
| Piper TTS | 🟢 | ⚪ | CPU | Fastest engine, 0.19× RTF; not re-verified this pass |
| Kokoro | 🟢 | ⚪ | CPU | Streaming; not re-verified this pass |
| MeloTTS | 🟢 | ⚪ | CPU | Streaming; not re-verified this pass |
| CosyVoice 2 / 3 | 🟡 | 👂 | CPU | Produces real, intelligible, non-buzzing speech with real end-to-end zero-shot conditioning (mel + prompt tokens + CFG, re-verified 2026-09-01 — a prior handoff doc describing this as missing was stale); **judged by ear 2026-09-01: speaker-identity transfer quality is sub-par** — every numeric stage checked real/non-degenerate, so this is likely a genuine subtle bug, not a missing feature; not yet root-caused — see `docs/qwentts-cosyvoice3-handoff.md` |
| Stable Diffusion 1.5, SDXL-Turbo, Z-Image-Turbo | 🟢 | 👂 | CPU / Vulkan | Real timed generations confirmed this session — see [diffusion samples](docs/diffusion-samples/README.md) |
| Z-Image-Turbo | 🟢 | 👂 | CPU / Vulkan | **Fixed 2026-09-01**: `QwenTextEncoder`'s GPU path unconditionally uploaded BF16 tensors regardless of backend support; `VulkanBackend.Sgemm` had no BF16 fallback and silently reinterpreted the 2-byte BF16 buffers as 4-byte FP32, corrupting Qwen3's text embeddings to NaN before the DiT ever saw them. Now gated on `BestSgemmPrecision==Bf16`; real recognizable apple image confirmed on the Vulkan GPU path |
| Wan 2.1 / 2.2 Video | 🔴 | 👂 | CPU / Vulkan / CUDA | Five real DiT/VAE bugs fixed (RoPE, QK-norm, unpatchify, VAE scaling, VAE upsample), a further ~6x CPU perf improvement landed, and a real 256x256/12-step run (past the point where VAE-resolution artifacts confound the result) still shows a strong, regular grid artifact — this disproves the earlier "just needs more steps" theory. Every specific hypothesis checked so far (RoPE axes, AdaLN modulation, flow-schedule/CFG formula, GELU, checkpoint tensor shapes) is individually ruled out; the real remaining bug is still unlocated. See `docs/diffusion-samples/README.md` for the full handoff (2026-09-01) |
| LTX-Video | 🟢 | 🔬 | CPU | **2026-09-02**: real, checkpoint-driven port — DiT transformer, VAE decoder, the real multi-step scheduler+CFG denoising loop, and T5-v1.1-XXL encoder+tokenizer are each independently golden-verified at effectively machine precision (>0.999999 cosine similarity) against their real reference implementations; one real bug found and fixed (missing per-channel VAE latent normalization). Real end-to-end 256x256 generations initially showed heavy per-pixel corruption, DEFINITIVELY traced to the checkpoint being run far below its trained resolution regime (model card: 1216x704 flagship, "works best under 720x1280"), not this port — the actual unmodified official `LTXVideoPipeline` class produced the identical corruption at 256x256 too. Re-running both the real pipeline and this C# port at 512x512 (same prompt/seed/steps/CFG) resolved it: both independently converge on a real, recognizable image (`docs/diffusion-samples/ltx-video-python-reference-512.png`, `docs/diffusion-samples/ltx-video-csharp-port-512.png`), closing the investigation. See `docs/055-ltx-video-implementation-plan.md` for the full investigation |
| HunyuanVideo | 🔴 | ⚪ | — | DiT runs clean through every layer with real weights (2026-08-31 smoke test) — blocked on a real VAE decoder (needs its own class, same as Wan) and real dual CLIP+LLM text conditioning, neither wired yet |
| FLUX.1 | 🔴 | 👂 | CPU | **Run for the first time 2026-09-01** (`FLUX.1-schnell`, real checkpoint): six real bugs found and fixed across two rounds — a GGUF tensor-name prefix mismatch, two missing `.weight` suffixes, GEGLU-vs-plain-GELU in the MLP, a full 2D RoPE rewrite (wrong axis split + wrong rotation convention), a flow-matching Euler integration sign inversion, and a `[txt,img]`-vs-`[img,txt]` token-ordering bug — each confirmed against the real `black-forest-labs/flux` source, not guessed. Pipeline runs to completion with no crash, but a periodic tiling artifact (now with a visible seam) still dominates the image. A separate performance change (`Workspace`/`Parallel.For` rewrite) landed in the same round; re-running the identical repro twice produced byte-identical output, ruling out the initially-suspected race condition, and timing came back near the original baseline on repeat — the earlier "slower" measurement looks like shared-machine contention, not a real regression. Not yet root-caused; paused — see `docs/056-flux-tiling-artifact-handoff.md`. Same stage as Wan/LTX-Video: real port, wrong output |
| SD3/3.5 | 🔴 | 🔵 | — | `SD3/Sd3Pipeline.cs` is a real, non-stubbed port — genuine double/single-stream MMDiT blocks reading real named tensors via `IWeightLoader`, not hardcoded formulas — but no local checkpoint has ever been downloaded, so it has never actually been run once. `Sd3ConformanceTests.cs` doesn't help settle this either — it never instantiates the real pipeline, only checks array-concatenation arithmetic. Confirmed 2026-09-02 |
| Stable Audio 3 | 🔴 | 🔴 | — | Structural stub, same shape as LTX-Video before its 2026-09-02 plan: `StableAudioDiT`/`AcousticVaeDecoder`/`StableAudioPipeline` exist in `src/OpenTail.Stingray.Diffusion/StableAudio/` but take no `IWeightLoader` and read no real tensors anywhere. No local checkpoint downloaded. Confirmed 2026-09-02, not yet planned/scoped the way LTX-Video was |
| Vision (17+ architectures implemented; 11 covered by the real-weight suite) | 🟢 | 🔬 | CPU / CUDA / Vulkan | Extensively implemented (see below); `MultimodalRealWeightsTests.cs` real-weight-tests 11 architectures with a differentiation/sanity check (real, non-degenerate embeddings; two genuinely different images must NOT produce near-identical output) — not a golden numeric-correctness check on its own, only "not obviously broken." **All 6 of the "confirmed-working" architectures now have real GOLDEN numeric verification** (min per-token cosine thresholds against a hand-written numpy port of the real llama.cpp mtmd C++ reference, reading the same local mmproj GGUF), each with real bugs found and fixed along the way — the differentiation test's "not degenerate" bar had been passing despite substantially broken math in every single one: `Gemma4UV` (cosine > 0.9995, `scripts/gemma4uv_ref.py`); `Llava` (cosine > 0.999, `scripts/llava_ref.py` — 2 bugs); `Pixtral` (cosine > 0.97, `scripts/pixtral_ref.py` — 3 bugs); `GLM-4.6V` (cosine > 0.97, `scripts/glm4v_ref.py` — 6 bugs: dead fused-QKV read, non-softmaxing attention stub, missing dual-patch-embed sum, dead patch-merger, wrong/missing projector-tail tensor names, and a 4-section M-RoPE that only rotated half of `head_dim` using the wrong axis); `Exaone4`/`Qwen2.5-VL`/`MiMoVl` (all cosine > 0.97, `scripts/exaone4_ref.py`/`qwen25vl_ref.py`/`mimovl_ref.py`, 2026-09-02 — real windowed/local attention implemented for all three, `n_wa_pattern`-gated, plus the same M-RoPE bug found a third/fourth time, a missing dual-patch-embed sum in `QwenVlVisionEncoder`, dead separate-Q/K/V + wrong GQA-vs-MHA default + wrong LayerNorm-vs-RMSNorm in `MimoVlVisionEncoder`; no local Qwen2.5-VL checkpoint existed before this pass, downloaded a real one). `KimiVl` and `MiniCpmV` **fixed 2026-09-02**: `KimiVl`'s projector (`mm.1`) had its output width hardcoded to the wrong dimension (corrupting `mm.2`'s row-stride reads into garbage) and its input-norm was applied at the wrong (post-merge) dimension, reading past a 1152-element array into garbage heap memory — both exploded every value to ~1e15+ magnitude; `MiniCpmV`'s resampler cross-attention was a dead stub that never read the image-derived K/V at all, so its output was a fixed function of the learned query alone, identical for every image by construction. `YoutuVl`/`HunyuanVl` still return all-zero embeddings, not yet root-caused, explicitly lowest priority; `Step3-VL` has no local checkpoint on this machine, so it is genuinely untested. The remaining named architectures beyond these 11 (Llama 4, Gemma 3, etc. — see the feature list below) are implemented but have no real-weight test coverage in this suite at all |

Runs GGUF and SafeTensors models on CPU (AVX2/AVX-512 SIMD) and GPU (Vulkan compute shaders or CUDA cuBLAS), with:
- **OpenAI- and Anthropic-compatible API server** (/v1/chat/completions, /v1/audio/speech, /v1/audio/transcriptions), native dynamic Multi-LoRA serving,
- **Native Multimodal Vision Understanding** across 17+ architectures:
  - Alibaba Qwen2.5-VL / Qwen3-VL
  - DeepSeek-OCR & DeepSeek-OCR2
  - Mistral Pixtral 12B
  - LLaVA-1.5, LLaVA-NeXT, LLaVA-OneVision
  - OpenGVLab InternVL 2.5 / 3 / 4
  - OpenBMB MiniCPM-V 2.6
  - Zhipu AI GLM-4V, GLM-4.5V, GLM-OCR
  - NVIDIA Nemotron-V2-VL / Nemotron-4-Nano
  - Baidu / Dots PaddleOCR-VL & Dots-OCR
  - Moonshot AI Kimi K2.5 / Kimi-VL
  - LG Exaone 4.5-VL
  - IBM Granite Vision 3.2 / Granite 4.0 Vision
  - Tencent Hunyuan-VL & Tencent Youtu-VL
  - Xiaomi MiMo-VL
  - StepFun Step3-VL
  - Google Gemma 4 unified (`gemma4uv`), Gemma 4 ViT (`gemma4v`), Gemma 3 SigLIP (`gemma3`), and Meta Llama 4 (`llama4`),
- **Native Speech-to-Text (ASR) & Forced Alignment**: OpenAI Whisper Large-v3 & Turbo with 100 languages, NVIDIA NeMo Parakeet FastConformer CTC/TDT, Alibaba Qwen3-ASR 0.6B/1.7B, Qwen3-ForcedAligner, FunASR Paraformer, SenseVoice, and Silero VAD,
- **Native Text-to-Speech (TTS) Synthesis & Voice Cloning**: Alibaba Qwen3-TTS 12Hz (with ERes2NetV2 192-dim voice cloning speaker encoder), Coqui XTTS-v2 (GPT2 autoregressive codec + FiLM-conditioned HiFi-GAN, zero-shot voice cloning), Fish Speech S2 Pro, Kokoro-82M, Piper VITS, Meta MMS-TTS (Massively Multilingual VITS), F5-TTS Flow-Matching DiT, CosyVoice 2.0 / 300M, Chatterbox-Turbo, Orpheus-TTS (SNAC codec), Parler-TTS Mini, and MeloTTS Multilingual VITS with clause-level real-time streaming,
- **Native Studio-Grade DSP**: Rational windowed-sinc resamplers, ATSC A/85 broadcast downmixing, and TPDF noise-shaped dithered WAV export,
- **Native Diffusion & Video Synthesis Pipelines**: Stable Diffusion 1.5 (with ControlNet conditioning), SDXL, SD 3/3.5, FLUX.1, FLUX.2 (Klein & Kontext Multi-Reference DiT), FLUX 3 Multimodal Video+Audio DiT, Stable Audio 3 Continuous MMDiT (Variable-Length 1s-6min 44.1kHz Stereo), Z-Image-Turbo, Qwen Image & Edit, Wan 2.1/2.2 Video, HunyuanVideo, and LTX-Video with zero-dependency Animated GIF and sequence exporters.

---

## TTS engine benchmarks

Same test sentence ("Hello, I will make some lunch, darling!"), same reference speaker where voice
cloning applies, CPU-only (AVX2), full (non-streaming) generation. RTF = wall-clock seconds per
second of generated audio (lower is faster; 1.0× = real-time). TTFA = time-to-first-audio in
streaming mode, where supported.

| Rank | Engine | Family / Architecture | Sample Rate | Batch Latency (Audio Sec) | Batch RTF | Streaming TTFA | Sample |
|---|---|---|---|---|---|---|---|
| 🥇 | Piper | VITS (lessac-medium) | 22,050 Hz | 0.61s (3.24s) | 0.188× 🚀 | 89ms ⚡ | [piper-perf-turn1.wav](docs/audio-samples/piper-perf-turn1.wav) |
| 🥈 | MMS-TTS | VITS (mms-tts-eng) | 16,000 Hz | 1.42s (3.07s) | 0.463× ⚡ | 188ms ⚡ | [mms-tts-perf-turn2.wav](docs/audio-samples/mms-tts-perf-turn2.wav) |
| 🥉 | Kokoro | StyleTTS2 (82m-q8_0) | 24,000 Hz | 2.73s (2.93s) | 0.933× ⚡ | 1.27s ⚡ | [kokoro-perf-turn2.wav](docs/audio-samples/kokoro-perf-turn2.wav) |
| 4 | MeloTTS | VITS MRF (zh_en) | 44,100 Hz | 3.65s (2.73s) | 1.337× | 671ms ⚡ | [melotts-perf-turn2.wav](docs/audio-samples/melotts-perf-turn2.wav) |
| 5 | XTTS-v2 | GPT-2 + HiFi-GAN | 24,000 Hz | 11.28s (3.85s) | 2.930× | 1.33s ⚡ | [xtts-perf-turn2.wav](docs/audio-samples/xtts-perf-turn2.wav) |
| 6 | QwenTTS | Qwen-Talker 0.6B + DAC | 24,000 Hz | 6.38s (2.16s) | 2.954× | 465ms ⚡ | [qwen-tts-perf-turn5.wav](docs/audio-samples/qwen-tts-perf-turn5.wav) |
| 7 | Chatterbox | Llama3-AR + S3Gen | 24,000 Hz | 11.55s (2.44s) | 4.733× | 11.47s (sentence) | [chatterbox-perf-turn1.wav](docs/audio-samples/chatterbox-perf-turn1.wav) |
| 8 | Parler-TTS | Transformer + DAC (mini-v1) | 44,100 Hz | 14.52s (2.57s) | 5.659× | 1.73s ⚡ | [parler-perf-turn2.wav](docs/audio-samples/parler-perf-turn2.wav) |
| 9 | CosyVoice 3 | DiT Flow Matching | 24,000 Hz | 22.13s (2.96s) | 7.476× | — | [cosyvoice3-perf-turn2.wav](docs/audio-samples/cosyvoice3-perf-turn2.wav) |
| 10 | F5-TTS | DiT Flow Matching (Q8_0 + decayed CFG) | 24,000 Hz | 35.96s (2.77s) | 12.965× | 35.8s (sentence) | [f5tts-perf-turn4.wav](docs/audio-samples/f5tts-perf-turn4.wav) |
| 11 | FishSpeech | Dual-AR + Firefly | 44,100 Hz | 50.67s (3.11s) | 16.285× | 1.76s ⚡ | [fishspeech-perf-turn3.wav](docs/audio-samples/fishspeech-perf-turn3.wav) |

Raw run data: [docs/tts-benchmark-log.txt](docs/tts-benchmark-log.txt).

---

## Try it

```bash
dotnet tool install -g OpenTail.Stingray.Cli

stingray -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "Once upon a time"
stingray -m models/Qwen3-8B-Q4_K_M.gguf -p "Explain mmap" -g -1     # all layers on GPU
stingray -m models/Qwen3-8B-Q4_K_M.gguf                             # interactive chat
stingray tts -t "Hello from OpenTail Stingray!" -v af_heart -o speech.wav # native TTS
```

Or use it as a library:

```bash
dotnet add package OpenTail.Stingray --version 1.0.6
```

| Package | What it gives you |
|---|---|
| [`OpenTail.Stingray`](https://www.nuget.org/packages/OpenTail.Stingray) | The engine — all core, vision, audio, diffusion, and hardware assemblies in one package |
| [`OpenTail.Stingray.Cli`](https://www.nuget.org/packages/OpenTail.Stingray.Cli) | The `stingray` command-line tool |
| [`OpenTail.Stingray.Server`](https://www.nuget.org/packages/OpenTail.Stingray.Server) | ASP.NET Core endpoints — OpenAI, Anthropic and Responses APIs |

**Coming from llama.cpp?** Flag names follow `llama-cli` where the meaning matches, single-dash
spellings included (`-ngl`, `-ctk`, `-md`, `-fa`). Flags that cannot be honoured are refused with a
named reason rather than silently ignored.

**Requirements:** .NET 10, x86-64 with AVX2. GPU is optional — any Vulkan-capable card, or NVIDIA
with CUDA 12.x. Building from source needs the .NET 10 SDK and `dotnet build -c Release`.

---

## License

MIT License — Copyright (c) 2026 OpenTail.
