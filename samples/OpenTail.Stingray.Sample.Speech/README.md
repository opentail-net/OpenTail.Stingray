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
63 MB)~63 MB)
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

---

## 4. Expected Output

The default round trip synthesizes one fixed sentence and transcribes it back. The run behind this README (Ryzen 7 5700G, CPU, 6.6 s in total):

```text
[TTS] Synthesizing speech with Piper...
      Text: "OpenTail Stingray brings fast neural voice synthesis and transcription directly to managed .NET."
      Generated 5.05s of audio in 3797ms -> sample_speech.wav
[ASR] Transcribing audio with Whisper...
      Transcription: "Benail Tina brings fast neural voice synthesis and transcription directly to managed net."
```

Note what the transcript shows: Whisper base gets the ordinary words right and **mishears the product name** ("OpenTail Stingray" became "Benail Tina"); a shorter sentence, "Hello from OpenTail Stingray.", came back as "Hello from Penial T-NAS.". The text is not the point of the sample; the pipeline working end to end is.

## 5. Known Limitations

- `stingray setup speak` asks you to accept the data licence of the Piper Lessac voice (**research use only: the Lessac Blizzard 2013 data forbids commercial use and redistribution; the Piper code is MIT**, [terms](https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html)); the sample itself does not check this, so the restriction applies to anything you do with its output.
- Whisper `base` is small and fast, not the most accurate model; proper nouns and brand names are the usual failures.
- Piper and Whisper are the only engines shown. Other speech engines have different setup needs and verification levels: see [docs/STATUS.md](../../docs/STATUS.md).
- Backend diagnostic lines may be printed to stderr.

## 6. Verification

- **Run against real weights, 2026-10-11:** Piper `en_US-lessac-medium` and Whisper `ggml-base.bin`, with the default round trip and the two single-direction modes. All three ran; the transcripts are as shown above. One run each.
- **Package contract:** the sample builds against the packed `OpenTail.Stingray` 1.0.8 NuGet package with no project references (`scripts/verify-package-consumers.ps1`, 2026-10-11).
