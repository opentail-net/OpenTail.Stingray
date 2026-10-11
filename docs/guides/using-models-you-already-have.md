# Use the models you already have

Stingray reads model files directly, so you do not need to re-download or convert what another tool already put on your disk. This guide shows how to point it at the stores of other local-AI tools, what was checked, and where it stops.

Everything marked **verified** was run on 2026-10-11 on Windows 11 (AMD Ryzen 7 5700G, integrated graphics) with real files from the tools named. Anything not marked verified was not run in that pass and is called out.

## The two things `-m` accepts

| You pass | Stingray treats it as | Notes |
|---|---|---|
| A **file** | A GGUF model | Any path works and the extension does not matter: Ollama's weight files have no extension at all. |
| A **folder** | A Hugging Face SafeTensors package | It needs `config.json`, the `.safetensors` file(s) and a `tokenizer.json`. See [SafeTensors folders](#safetensors-folders) for what that covers. |

From C#, `Model.Load(path)` takes the same file or folder.
**The file stays where it is.** Stingray opens it read-only and memory-maps it in place; it does not copy it into its own model folder and writes nothing to the file or its folder. The operating system pages the weights in from the original location as they are used (a GPU run also uploads them to GPU memory), so a file on a network drive is read over the network. Only `stingray setup` and `stingray pull` download into Stingray's own model folder.

```bash
stingray -m <file-or-folder> -p "Hello"        # add -g -1 to use the GPU
```

## A folder of GGUF files, from anywhere

GGUF is an open file format, so a `.gguf` file can live in any folder, whatever tool or download put it there. Pass the file path; Stingray does not need it to be in its own model folder:

```bash
stingray -m "D:/my-models/<model>.gguf" -p "Say hello in five words."
```

**Verified:** a Qwen3.5 2B GGUF stored in a folder outside Stingray's model home loaded and answered (prefill 61 tokens/s, decode 23 tokens/s on the CPU path). A vision model's projector is a separate `mmproj-....gguf` file; keep it beside the model and pass it with `--mmproj` (not run in this pass).

## Ollama

Ollama keeps weights as content-addressed blobs under `%USERPROFILE%\.ollama\models\blobs` (`~/.ollama/models/blobs`), named `sha256-<digest>`, and a small JSON manifest per model under `manifests\registry.ollama.ai\library\<name>\<tag>`. The weight blob **is a GGUF file**, so Stingray runs it directly and needs no Ollama installation or daemon. You only have to find which blob belongs to the model: it is the manifest layer whose media type is `application/vnd.ollama.image.model`.

PowerShell:

```powershell
$m = Get-Content "$env:USERPROFILE\.ollama\models\manifests\registry.ollama.ai\library\qwen3-vl\2b-instruct-q4_k_m" | ConvertFrom-Json
$d = ($m.layers | Where-Object mediaType -eq 'application/vnd.ollama.image.model').digest -replace ':','-'
stingray -m "$env:USERPROFILE\.ollama\models\blobs\$d" -p "Say hello in five words."
```

**Verified:** that exact snippet ran a Qwen3-VL 2B model from a real Ollama store (about 52 tokens/s prefill, 25 tokens/s decode).

**Known limitation:** pointing `-m` at the **manifest file itself** does not work in the CLI today. A real manifest has no extension and crashes with a stack trace, and a copy renamed `.json` is planned as an Ollama package and then handed to the GGUF loader, which fails with "Invalid GGUF magic". The library has an `OllamaModelPackage` reader for manifests, but it has been tested only on synthetic fixtures. Use the blob route above.

Do not edit or delete anything inside Ollama's store: Stingray only reads the blob.

## Hugging Face downloads

`huggingface-cli download` and `from_pretrained` keep files in `%USERPROFILE%\.cache\huggingface\hub\models--<owner>--<repo>\snapshots\<revision>\`. The files there are the model files themselves (symlinks into `blobs`), so the same rules apply: a `.gguf` is passed as a file and a SafeTensors model as the snapshot **folder**. **Not verified in this pass:** the Hugging Face cache on the test machine held no complete language-model snapshot to run, so this layout is described from how the cache is organised, not from a run.

You can also skip the cache and let Stingray fetch a pinned file itself: `stingray pull -r <owner/repo> --revision <commit>` (see [the README](../../README.md#trying-a-model-that-is-not-in-the-catalogue)).

## SafeTensors folders

A folder with `config.json`, `model.safetensors` and `tokenizer.json` runs as a language model if its architecture has a package profile. **Verified:** a SmolLM2 135M folder ran on the CPU (prefill 437 tokens/s, decode 56 tokens/s). The profiles are narrow, so check before you rely on one:

```bash
stingray capabilities
```

At the time of writing that lists dense Llama and Mistral, Qwen2 and Qwen3, and GPT-2, all CPU only and in F32, F16 or BF16, with the exclusions it prints for each (for example no attention bias, no sliding-window attention, no RoPE scaling, no quantised SafeTensors). Anything absent from a row is unsupported; for quantised weights or the GPU, use a GGUF of the same model. Image, video, speech and vision models distributed as SafeTensors are loaded by their own pipelines, not through this language-model route; see [docs/STATUS.md](../STATUS.md) for each.

## ONNX and whisper.cpp files

- **ONNX voices** (for example Piper): `stingray tts -e piper -m en_US-lessac-medium.onnx.json -t "Hello" -o hello.wav`. **Verified** with the Piper lessac voice. This runs through ONNX Runtime, a native component that ships as a dependency of the NuGet package (if you publish a single self-contained executable, make sure the ONNX Runtime native library is deployed with it).
- **whisper.cpp `.bin` models:** `stingray stt -m base --model-file ggml-base.bin -i hello.wav`. **Verified** with `ggml-base.bin`.

## Let Stingray list what you have

`stingray models --local` lists the GGUF models it can see, with architecture, whether Stingray supports it, the quantisation, and whether it fits this machine's memory. By default it looks in Stingray's own model folder. Add your other stores with `STINGRAY_MODEL_DIRS` (a list separated by `;` on Windows and `:` on Linux and macOS). **The scan is not recursive**: list the folder that directly contains the `.gguf` files, which may be one folder per model.

```bash
# PowerShell
$env:STINGRAY_MODEL_DIRS = "D:\my-models\gemma-4-26b"
stingray models --local
```

**Verified:** that listed the GGUF files in an outside folder next to Stingray's own, for example:

```text
File                                       Size       Architecture  Support        Quant    Fit
google_gemma-4-26B-A4B-it-Q5_K_M.gguf      17.99 GiB  gemma4        admitted       Q5_K     fits
mmproj-google_gemma-4-26B-A4B-it-f16.gguf  1.11 GiB   clip          not supported  Float16  unknown
```

The projector file shows as "not supported" because it is not a language model; it is meant for `--mmproj`. Ollama's blobs have no `.gguf` extension, so this listing does not show them: use the manifest lookup above.

## Before you run something large

A model that is listed is not necessarily one that runs well here. Check a local file without loading its weights:

```bash
stingray scout -m <file.gguf> --budget 32G      # architecture, support, whether it fits that budget
```

The CLI also refuses a run whose estimated memory does not fit the machine; `--ignore-preflight` overrides that for CPU runs, at your own risk. Which architectures are admitted, and how well each was checked, is in [docs/STATUS.md](../STATUS.md).

## Summary of what works

| Source | Route | Status |
|---|---|---|
| Any folder of GGUF files | `-m <file>.gguf` | Verified |
| Ollama store | `-m <blob>` (find it in the manifest) | Verified; manifest file itself does not work yet |
| Hugging Face cache (GGUF or SafeTensors) | `-m <snapshot file or folder>` | Not verified here |
| SafeTensors folder | `-m <folder>` | Verified for one small Llama-family model; narrow profiles, CPU only |
| ONNX voice (Piper) | `stingray tts -e piper ...` | Verified; needs ONNX Runtime |
| whisper.cpp `.bin` | `stingray stt ... --model-file` | Verified |
