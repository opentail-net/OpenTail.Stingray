# OpenTail.Stingray Code Samples

> **Purpose:** executable C# programs that show how OpenTail.Stingray is used, and what has actually been checked about each one.
> **Prerequisites for all of them:** the [.NET 10 SDK](https://dotnet.microsoft.com/download) and a 64-bit x86 CPU with AVX2. A GPU is optional (Vulkan or CUDA). Every sample needs model files, which are never committed: each sample's own README lists them.

---

## How the samples are classified

Read the classification before you copy code from a sample.

| Classification | Meaning | What you can rely on |
|---|---|---|
| **Package consumer** | Uses only the public API of the `OpenTail.Stingray` (or `OpenTail.Stingray.Server`) NuGet package. | The sample also builds against the **packed package with no project references**, checked by `scripts/verify-package-consumers.ps1`. Copying it into your own project is safe. |
| **Source-level contributor example** | Uses engine internals (`GgufModel`, `ForwardPass`, `CpuBackend`, session runtime and similar) that are not part of the public package contract. | It builds and runs in this repository. It may break between versions and cannot be built against the package. |
| **Demonstration** | Shows a behaviour rather than a product capability, and may fall back to a simulation. | Do not cite it as an inference benchmark or as evidence of quality. |

The in-repo build of a package-consumer sample uses project references so your edits are picked up at once. To build one against the packed package, pass `-p:StingrayPackageVersion=<version>` and a local feed in `RestoreSources`; the script above does exactly that for all four.

## Sample catalogue

Verification below is what was **run on 2026-10-11** on an AMD Ryzen 7 5700G (integrated Radeon, 64 GB), with the real model files named in each sample's README. "Ran" means the program completed and its output was read; it is not a reference comparison. Details, expected output and known limitations are in each sample's README.

| Sample | Classification | Task | Model files it needs | Verification |
|---|---|---|---|---|
| **[`OpenTail.Stingray.Sample.Chat`](OpenTail.Stingray.Sample.Chat/)** | Package consumer | Multi-turn streaming chat (`Model.Load`, `ChatSession`, `ModelHome`) | A chat GGUF; `stingray setup chat` installs Qwen2.5 0.5B (469 MB) | Ran path-free with piped input (5.4 s with model load). Builds against the packed package. |
| **[`OpenTail.Stingray.Sample.Extract`](OpenTail.Stingray.Sample.Extract/)** | Package consumer | Structured extraction: schema-constrained JSON from free text, with a grounding check | A chat GGUF; `stingray setup chat` installs Qwen2.5 0.5B (469 MB) | Ran on the CPU with Qwen2.5 0.5B and 3B: correct fields from a built-in invoice email. Builds against the packed package (2026-10-11). |
| **[`OpenTail.Stingray.Sample.Speech`](OpenTail.Stingray.Sample.Speech/)** | Package consumer | Text-to-speech (Piper) and transcription (Whisper) | Piper voice `.onnx` + `.json` (63 MB), `ggml-base.bin` (141 MB) | Ran all three modes. The round trip is correct except that Whisper base **mishears the product name**. Builds against the packed package. |
| **[`OpenTail.Stingray.Sample.Diffusion`](OpenTail.Stingray.Sample.Diffusion/)** | Package consumer | Text-to-image with Stable Diffusion 1.5 | SD 1.5 safetensors (about 4.27 GB) + CLIP tokenizer | Ran 256x256, 8 steps on the GPU (31 s); the image was a recognisable apple, judged by eye. Builds against the packed package. |
| **[`ChatServer`](ChatServer/)** | Package consumer | OpenAI-compatible HTTP endpoint inside ASP.NET Core | Any chat GGUF (path as first argument) | One request returned the expected completion. Builds against the packed `OpenTail.Stingray.Server` package. |
| **[`QuickStart`](QuickStart/)** | Source-level contributor example | Chat, speak and hear on the CPU, wired from engine internals | Chat GGUF, Piper voice, `ggml-base.bin` | Ran all three verbs; `hear` transcribed the `speak` output exactly. Not a package consumer. |
| **[`OpenTail.Stingray.Sample.ToolCall`](OpenTail.Stingray.Sample.ToolCall/)** | Source-level contributor example | Tool calling loop with grammar-constrained parsing | A tool-capable chat GGUF (run with Qwen3-8B) | Ran on the CPU (38 s): three correct tool calls and a correct answer. **CUDA commands not run.** |
| **[`OpenTail.Stingray.Sample.HotRouting`](OpenTail.Stingray.Sample.HotRouting/)** | Source-level contributor example and demonstration | Several roles sharing one model with retained state | Optional `models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf`; **falls back to a simulation without it** | Ran with the real model (4.3 s, tiny generations). It demonstrates state retention, not quality or speed. |

**Not covered by any sample:** image understanding (vision), embeddings and reranking, music and sound effects, video generation, and the speech engines other than Piper and Whisper. For those use the [CLI](../src/OpenTail.Stingray.Cli/README.md), the [task guides](../docs/guides/README.md) and [docs/STATUS.md](../docs/STATUS.md); the [audio](../docs/audio-samples/README.md) and [image and video](../docs/diffusion-samples/README.md) sample collections show real output.

---

## Quick start: running the package-consumer samples

### 1. Install the models

```bash
dotnet tool install -g OpenTail.Stingray.Cli

stingray setup chat          # Qwen2.5 0.5B Instruct, 469 MB, Apache-2.0
stingray setup speak         # Piper en_US-lessac-medium, 60 MB; asks you to accept the voice's data licence
stingray setup transcribe    # Whisper base, 141 MB
```

Downloads are pinned to an exact revision and verified against a SHA-256.

### 2. Chat

```bash
dotnet run --project samples/OpenTail.Stingray.Sample.Chat -c Release
# or with any GGUF:
dotnet run --project samples/OpenTail.Stingray.Sample.Chat -c Release -- -m path/to/model.gguf
```

### 3. Speech

```bash
dotnet run --project samples/OpenTail.Stingray.Sample.Speech -c Release
```

The sample finds the voice and Whisper model in the model home; pass `--voice` and `--whisper` for explicit paths.

### 4. Diffusion

Download the checkpoint and tokenizer as its [README](OpenTail.Stingray.Sample.Diffusion/README.md#downloading-assets) describes, then:

```bash
dotnet run --project samples/OpenTail.Stingray.Sample.Diffusion -c Release -- \
  --prompt "a red apple on a wooden table" --steps 8 --width 256 --height 256 --seed 42 --output apple.png
```

### 5. HTTP server

```bash
dotnet run --project samples/ChatServer -c Release -- <path-to-model.gguf>
```

---

## Conventions and checks

1. **Package consumers stay on the public API.** They must build with `-p:StingrayPackageVersion=<version>`; `scripts/verify-package-consumers.ps1` packs the current checkout and proves it for these four samples and for every code block in the root and package READMEs that is marked `<!-- verify:consumer -->`. That proves the code compiles against the public package; it does not run inference.
2. **The classification table is checked.** `ProductSurfaceContractTests` fails if a sample listed as a package consumer has no package build mode, if a contributor example claims one, or if a sample is missing from this table or lacks a README with *Prerequisites*, *Expected output*, *Known limitations* and *Verification* sections.
3. **Never commit model weights** (`.gguf`, `.onnx`, `.bin`, `.safetensors`) or large generated outputs. Name the exact download source in the sample's README.
4. **Say how far each sample was verified**, with a date, the machine and the model files, and say so when something was not run (for example the CUDA commands).
