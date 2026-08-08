# OpenTail.Stingray.Cli

`stingray` — a command-line tool for local LLM inference and image generation, powered by [OpenTail.Stingray](https://www.nuget.org/packages/OpenTail.Stingray). Reads GGUF models and runs them on CPU (AVX2/AVX-512) or GPU (Vulkan / CUDA). No Python, no sidecar process.

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

## Usage

```
# Text generation (CPU)
stingray -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -p "Once upon a time" --temp 0.7

# All layers on GPU (Vulkan or CUDA, auto-selected)
stingray -m models/Qwen3-8B-Q4_K_M.gguf -p "Explain mmap" -g -1

# Interactive chat (omit -p to enter chat mode)
stingray -m models/Qwen3-8B-Q4_K_M.gguf

# Inspect before running
stingray list-metadata -m model.gguf
stingray capabilities                     # which model packages are supported
stingray inspect -m models/some-hf-dir    # verdict on a SafeTensors package

# Image generation (Z-Image-Turbo; CUDA or Vulkan)
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

Run `stingray --help` for the full reference.

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
