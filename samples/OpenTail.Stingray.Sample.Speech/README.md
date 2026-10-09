# OpenTail.Stingray.Sample.Speech

> **Audience:** Package Consumer  
> **Task:** In-process Text-to-Speech (TTS) and Speech-to-Text (ASR)  
> **API Contract:** `PiperPipeline`, `WhisperPipeline`, `WavReader`

---

## 1. What This Sample Demonstrates

This sample demonstrates how to run speech synthesis and speech recognition inside your own .NET process using pure managed C#:
- **Neural Voice Synthesis:** Generates natural 22,050 Hz speech using `PiperPipeline.FromConfigFile(...)`. Piper executes at **0.129× RTF** (much faster than real-time on ordinary CPUs).
- **Speech Recognition:** Transcribes speech audio to text using `WhisperPipeline.Load(...)`.
- **In-Memory Audio Processing:** Reads and generates standard PCM WAV buffers using `WavReader` and dithered WAV writers.
- **Path-Free Integration:** Automatically discovers default models installed via `stingray setup speak` and `stingray setup transcribe`.

---

## 2. Prerequisites & Asset Setup

You can download the verified default models with SHA-256 integrity using the CLI:

```bash
# 1. Download Piper Lessac medium voice (.onnx + .onnx.json, ~63 MB)
stingray setup speak

# 2. Download Whisper base model (ggml-base.bin, ~141 MB)
stingray setup transcribe
```

---

## 3. How to Run

### Automatic Roundtrip (Synthesizes speech then transcribes it back)
```bash
dotnet run --project samples/OpenTail.Stingray.Sample.Speech -c Release
```

### Text-to-Speech Only
```bash
dotnet run --project samples/OpenTail.Stingray.Sample.Speech -c Release -- \
    --speak "Hello from OpenTail Stingray." -o hello.wav
```

### Speech-to-Text Only
```bash
dotnet run --project samples/OpenTail.Stingray.Sample.Speech -c Release -- \
    --transcribe hello.wav
```

### Options
- `--speak <text>`: Text to synthesize.
- `-o, --output <path>`: Destination `.wav` path (default: `sample_speech.wav`).
- `--transcribe <path>`: Path to audio file to transcribe.
- `--voice <path>`: Explicit path to `.onnx.json` Piper configuration file.
- `--whisper <path>`: Explicit path to `ggml-base.bin` Whisper model file.
