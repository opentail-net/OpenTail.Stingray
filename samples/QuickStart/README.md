# QuickStart

> **Classification:** source-level contributor example. It wires the engine's internal types (`GgufModel`, `CpuBackend`, `ForwardPass`, `InferenceEngine`, the Piper and Whisper pipelines) directly. It is **not** a NuGet package consumer: if you are writing an application, start from [`OpenTail.Stingray.Sample.Chat`](../OpenTail.Stingray.Sample.Chat/) and [`OpenTail.Stingray.Sample.Speech`](../OpenTail.Stingray.Sample.Speech/), which use the public API.
> **Task:** chat, speak and hear in one small program, on the CPU.

## Prerequisites

- The .NET 10 SDK, a 64-bit x86 CPU with AVX2.
- Model files (public downloads): a chat GGUF (`stingray setup chat`, or `stingray pull -r Qwen/Qwen2.5-0.5B-Instruct-GGUF`, 469 MB); a Piper voice, both `en_US-lessac-medium.onnx` and its `.onnx.json` (about 63 MB, **research use only: the voice data forbids commercial use and redistribution**, [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium)); and `ggml-base.bin` (Whisper base, about 148 MB, [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp)). The voice's data licence applies to the Piper voice; read it before use.

## Run

```bash
dotnet run --project samples/QuickStart -c Release -- chat  <model.gguf>  "Your question"
dotnet run --project samples/QuickStart -c Release -- speak <voice.onnx>  "Text to say"  out.wav
dotnet run --project samples/QuickStart -c Release -- hear  <ggml-base.bin> in.wav
```

All paths are explicit: this sample does not use `ModelHome` or the starter catalogue.

## Expected output

`chat` prints one streamed answer; `speak` writes a WAV and prints its length; `hear` prints the transcript. The runs behind this README (Ryzen 7 5700G, CPU):

```text
chat:  Developers write unit tests to ensure that individual modules and functions work as intended, preventing failures and improving code quality and reliability.    (3.4 s, including model load)
speak: Wrote ...\qs.wav (1.5 s of audio)                                                                                                                               (2.3 s)
hear:  Hello from the Quick Start.                                                                                                                                      (3.1 s)
```

Chat answers are sampled, so the wording varies from run to run.

## Known limitations

- CPU only (it constructs a `CpuBackend`); for GPU use the CLI or the Chat sample.
- Uses engine internals that are not part of the public package contract and may change without notice.

## Verification

- **Run against real weights, 2026-10-11:** Qwen2.5 0.5B Instruct Q4_K_M, Piper `en_US-lessac-medium`, Whisper `ggml-base.bin`; the three verbs ran as shown, and `hear` transcribed the `speak` output exactly. One run each.
- **Not** verified against the packed NuGet package (it does not claim to be a package consumer).
