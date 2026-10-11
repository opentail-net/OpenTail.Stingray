# OpenTail.Stingray.Cli

`stingray` is the command-line front door to OpenTail.Stingray: install a recommended model for a task and run it without typing a path (`setup`, `chat`, `speak`, `transcribe`), or point it at any GGUF and tune the run yourself. It also generates images, computes embeddings, and carries the tools for checking and admitting new models (`scout`, `pull`, `inspect`, `admit-arch`). Runs on CPU (AVX2/AVX-512) or GPU (Vulkan / CUDA). No Python, no llama.cpp, no sidecar process.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![.NET 10](https://img.shields.io/badge/.NET-10-blue)]()

> **Built by [opentail.net](https://opentail.net)**

## Install

```
dotnet tool install -g OpenTail.Stingray.Cli
```

Or update:

```
dotnet tool update -g OpenTail.Stingray.Cli
```

The command is `stingray`.

## Quick Start: Path-Free Front Door

```bash
# 1. Download verified default model bundles
stingray setup chat          # Qwen2.5 0.5B Instruct (469 MB)
stingray setup speak         # Piper en_US-lessac-medium (60 MB)
stingray setup transcribe    # Whisper base (141 MB)
stingray models              # check installation status

# Optional: choose a qualified alternative or view local inventory
stingray models chat         # view chat choices (qwen2.5-0.5b, qwen2.5-1.5b, qwen2.5-7b)
stingray models use chat qwen2.5-1.5b  # set persistent favourite for chat
stingray models --local      # list installed models on disk (multi-shard GGUFs deduplicated)

# 2. Run immediately without typing paths
stingray chat                # interactive multi-turn terminal chat
stingray chat "Explain SIMD in one sentence."
stingray speak "Hello from Stingray." -o hello.wav
stingray transcribe hello.wav
```

## Is my machine ready?

```bash
stingray models              # which of chat / speak / transcribe have a model installed, and the command that fixes each gap
stingray doctor              # the local runtime and which GPU backends are usable
stingray doctor -m model.gguf   # structural check of a GGUF without running inference
```

(`stingray status` is different: it reads live metrics from a running server.) The starter
catalogue covers chat, speak and transcribe only. Everything else the engine can run, and how well
each model was checked, is in [docs/STATUS.md](../../docs/STATUS.md).

## A model that is not in the catalogue

Check it before you download it, then fetch exactly the file you checked:

```bash
stingray scout -r owner/repo --quants --budget 32G           # architecture, blockers, which quantisations fit, a pull command pinned to the inspected commit
stingray pull  -r owner/repo --revision <commit> -q Q4_K_M    # downloads that revision, so you get what scout inspected
stingray -m models/<file>.gguf -p "Hello" -g -1              # run it (drop -g -1 for CPU)
```

`scout` reads only the file index, never the weights, and only reports: without `--budget` it does
not judge whether the model fits your RAM. A local file works too (`stingray scout -m model.gguf`).
If the architecture is not admitted yet, see
[adding an architecture](../../docs/reference/adding-an-architecture.md) for the evidence required
before it is advertised as supported.

## Advanced & Path-Oriented Usage

```bash
# Text generation with explicit model path (CPU)
stingray -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "Once upon a time" --temp 0.7

# All layers on GPU (Vulkan or CUDA, auto-selected)
stingray -m models/Qwen3-8B-Q4_K_M.gguf -p "Explain mmap" --backend vulkan -g -1

# Interactive chat (omit -p to enter chat mode)
stingray -m models/Qwen3-8B-Q4_K_M.gguf

# Inspect before running
stingray list-metadata -m model.gguf
stingray capabilities                     # which model packages are supported
stingray inspect -m models/some-hf-dir    # verdict on a SafeTensors package

# Image generation (Z-Image-Turbo). A GPU (Vulkan or CUDA) is strongly recommended; add `--device none` to force CPU, which is slow.
stingray image \
  -m models/z_image_turbo-Q5_K_M.gguf \
  --vae models/z-image-turbo/vae \
  --qwen-encoder models/Z-Image-AbliteratedV1.Q5_K_M.gguf \
  --qwen-tokenizer models/z-image-turbo/tokenizer/tokenizer.json \
  -p "a serene mountain lake at sunrise" -W 512 -H 512 --steps 4 -o out.png
```

## Coming from llama.cpp

Flag names follow `llama-cli` where the meaning matches, including single-dash spellings like `-ngl`,
`-ctk`, `-md` and `-fa`, so a command copied from llama.cpp documentation generally just runs.

Where a flag *cannot* be honoured, it is **refused with a named reason rather than silently ignored** —
`-ts`, `-sm`, `-mg`, `--mlock`, `--numa`, `--presence-penalty` and `-b`/`-ub` all tell you what
Stingray does instead. Flags that are simply inert here (`-fa`, `--no-warmup`) are accepted with a
note so a pasted command line still runs. The one thing you will never get is a flag that looks
accepted and quietly does nothing.

## Common flags

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | auto-detect | Path to a GGUF file, or a SafeTensors model directory |
| `-p, --prompt` | (interactive) | Input prompt; omit to enter chat |
| `-n, --n-predict` | `512` | Maximum tokens to generate |
| `-t, --threads` | logical CPUs | CPU worker threads for the SIMD kernels |
| `--temp` | `0.7` | Sampling temperature (`0` = greedy) |
| `--top-k` | `40` | Top-k sampling |
| `--top-p` | `0.95` | Top-p nucleus sampling |
| `--min-p` | `0.05` | Min-p sampling |
| `--repeat-penalty` | `1.1` | Repetition penalty (`1.0` = off) |
| `--repeat-last-n` | `64` | Tokens the repetition penalty considers (`0` = off, `-1` = full context) |
| `--logit-bias` | — | Bias a token, `TOKEN_ID+BIAS` or `TOKEN_ID-BIAS`. Repeatable |
| `-e, --escape` | off | Process `\n`, `\t`, `\r`, `\\` in the prompt |
| `-g, --n-gpu-layers` | `0` | Layers on GPU (`0` = CPU only, `-1` = all) |
| `-c, --ctx-size` | model default | Context / max sequence length |
| `--backend` | `auto` | `auto`, `vulkan` or `cuda` |
| `-j, --json-schema` | — | Constrain the whole response to a JSON schema |
| `--chat-template` | model's own | Override with a raw Jinja2 template |
| `--tq` | off | TurboQuant KV-cache compression (~4–8× less KV memory) |
| `--tq-mode` | `auto` | `auto`, `kvarn` (4-bit K / 2-bit V) or `lloydmax` (3-bit codebooks; degrades quality on QK-norm models such as Qwen3) |

Run `stingray --help` or view the [CLI Option Inventory](../../docs/reference/cli-option-inventory.md) for every option of every command.

## Requirements

- .NET 10 runtime — the tool installs framework-dependent
- x86-64 CPU with **AVX2**
- GPU inference is optional: any Vulkan-capable GPU (AMD / Intel / NVIDIA), or an NVIDIA GPU with
  **CUDA 12.x** — the CUDA path is pinned to the 12.x runtime SONAMEs (`cudart64_12`, `cublas64_12`)

## Links

- [Library package](https://www.nuget.org/packages/OpenTail.Stingray)
- [Server package](https://www.nuget.org/packages/OpenTail.Stingray.Server)
- [Repository and documentation](https://github.com/opentail-net/OpenTail.Stingray)

---

## Acknowledgements

Forked from **[SharpInference](https://github.com/pekkah/SharpInference)** by Pekka Heikura (MIT), which remains actively developed upstream; its copyright is retained in `THIRD_PARTY_NOTICES.md`.

Interoperates with **[llama.cpp](https://github.com/ggml-org/llama.cpp)**'s GGUF format and quantization block layouts, and follows `llama-cli` flag names where the meaning matches — **no llama.cpp code is used**. **[LLamaSharp](https://github.com/SciSharp/LLamaSharp)** was studied as the reference for .NET inference API design; **no LLamaSharp code is used**, and unlike it this engine is managed C# end to end rather than P/Invoke bindings to native llama.cpp.

## License

MIT. Copyright © 2026 OpenTail.
