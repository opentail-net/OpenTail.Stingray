# OpenTail.Stingray Code Samples

> **Purpose:** Executable C# sample applications demonstrating how to use OpenTail.Stingray for real-world tasks.  
> **Prerequisites:** [.NET 10 SDK](https://dotnet.microsoft.com/download), 64-bit x86 CPU with AVX2 (or ARM64). GPU optional (Vulkan or CUDA).

---

## Sample Catalogue

Samples are divided into two distinct categories:
- **Package Consumer Samples:** Demonstrate the public NuGet API (`OpenTail.Stingray`) as used by downstream .NET application developers.
- **Engine Contributor Samples:** Demonstrate low-level internals (raw forward passes, tensor buffers, scheduler mechanics) for contributors extending the engine.

| Directory | Audience | Task / Modality | Primary APIs Demonstrated | Verification Level |
|---|---|---|---|---|
| **[`OpenTail.Stingray.Sample.Chat`](OpenTail.Stingray.Sample.Chat/)** | **Package Consumer** | Multi-turn streaming chat | `Model.Load`, `CreateContext`, `ChatSession`, `ModelHome` | 🔬 Verified against real weights |
| **[`OpenTail.Stingray.Sample.Speech`](OpenTail.Stingray.Sample.Speech/)** | **Package Consumer** | Text-to-speech (TTS) & transcription (ASR) | `PiperPipeline`, `WhisperPipeline`, `WavReader` | 🔬 Verified against real weights |
| **[`OpenTail.Stingray.Sample.Diffusion`](OpenTail.Stingray.Sample.Diffusion/)** | **Package Consumer** | Text-to-image generation | `IDiffusionPipeline`, `StableDiffusionPipeline`, `ImageGenerationRequest` | 🔬 Checkpoint verified |
| **[`ChatServer`](ChatServer/)** | **Package Consumer** | ASP.NET Core OpenAI / Anthropic HTTP server | `AddOpenTailStingray()`, `MapOpenTailStingray()` | 🔬 Verified server endpoints |
| **[`QuickStart`](QuickStart/)** | Engine Contributor | Multi-task in-process pipeline | `GgufModel`, `CpuBackend`, `ForwardPass`, `InferenceEngine` | 🔬 Verified CLI verbs |
| **[`OpenTail.Stingray.Sample.ToolCall`](OpenTail.Stingray.Sample.ToolCall/)** | Engine Contributor | Native grammar-constrained tool calling | `GgufModel`, `ToolGrammar`, `TierPlanner` | 🔬 Verified tool loops |
| **[`OpenTail.Stingray.Sample.HotRouting`](OpenTail.Stingray.Sample.HotRouting/)** | Engine Contributor | Multi-session cold tiering & routing | `SessionRegistry`, `TierPlanner`, `ExecutionPlan` | 🔬 Structural simulation |

---

## Quick Start: Running Samples

### 1. Setup Recommended Models
Before running the samples, you can download verified default models using the `stingray` CLI tool:

```bash
dotnet tool install -g OpenTail.Stingray.Cli

# Download verified models with SHA-256 integrity:
stingray setup chat          # Qwen2.5 0.5B Instruct (469 MB)
stingray setup speak         # Piper en_US-lessac-medium (60 MB)
stingray setup transcribe    # Whisper base (141 MB)
```

### 2. Run the Consumer Chat Sample
The chat sample automatically resolves the installed model from `ModelHome`:

```bash
dotnet run --project samples/OpenTail.Stingray.Sample.Chat -c Release
```

Or pass an explicit path to any GGUF model:

```bash
dotnet run --project samples/OpenTail.Stingray.Sample.Chat -c Release -- -m path/to/model.gguf
```

### 3. Run the Speech Sample
Synthesize speech and transcribe it back in-process:

```bash
dotnet run --project samples/OpenTail.Stingray.Sample.Speech -c Release
```

### 4. Run the Diffusion Sample
Generate an image from a text prompt with real-time denoising progress:

```bash
dotnet run --project samples/OpenTail.Stingray.Sample.Diffusion -c Release -- \
  --prompt "A serene alpine mountain lake at sunrise, highly detailed digital art" \
  --output lake.png
```

---

## Development & Testing Conventions

1. **Clean Dependencies:** Package consumer samples must only consume types exposed by the `OpenTail.Stingray` package and documented framework assemblies.
2. **Model Asset Integrity:** Never commit binary model weights (`.gguf`, `.onnx`, `.bin`) to Git. Document exact download sources and hashes in each sample's README.
3. **No Simulation Confusion:** Samples demonstrating simulated or mock workflows (such as `HotRouting`) are explicitly labelled as simulations and must not be cited as measured inference benchmarks.
