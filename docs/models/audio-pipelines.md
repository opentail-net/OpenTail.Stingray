# Architecture Card: Studio Audio Stack (TTS, STT & DSP)

> **Namespace:** `OpenTail.Stingray.Audio`  
> **Verification Status:** 🔬 Verified pipelines across Piper, Whisper, Kokoro, Paraformer, and DSP utilities (see [docs/STATUS.md](../STATUS.md))  
> **Supported Formats:** ONNX (`.onnx` + `.onnx.json`), GGML/GGUF (`.bin` / `.gguf`), SafeTensors  
> **Key Implementations:** `PiperPipeline.cs`, `WhisperPipeline.cs`, `WavReader.cs`, `RationalResampler.cs`

---

## 1. Architectural Highlights

Stingray's audio stack provides an integrated speech synthesis and recognition runtime, backed by studio-grade digital signal processing (DSP) written in pure managed C#.

### Text-to-Speech (TTS) & Voice Synthesis
- **Piper VITS (🥇 0.129× RTF on CPU):** Variational Inference with adversarial learning (VITS). Combines an eSpeak-ng phonemizer, duration predictor, monotonic alignment search, and a HiFi-GAN neural vocoder. Generates 2.5 seconds of clean 22,050 Hz speech in ~320 ms on an ordinary CPU.
- **Kokoro-82M:** Compact neural voice synthesis using GGUF weights with style vector conditioning.
- **Qwen3-TTS 12Hz:** Includes an ERes2NetV2 192-dimensional speaker encoder for zero-shot voice cloning from reference audio clips.
- **Coqui XTTS-v2:** Autoregressive GPT-2 text-to-codecs combined with FiLM-conditioned HiFi-GAN for multilingual voice cloning.
- **F5-TTS:** Flow-Matching Diffusion Transformer for expressive, non-autoregressive speech synthesis.

### Speech-to-Text (ASR) & Transcription
- **OpenAI Whisper (Tiny through Large-v3 & Turbo):**
  - Audio preprocessing: Log-Mel 80/128-channel spectrogram filterbanks computed using managed FFTs.
  - Encoder: 2-layer 1D convolutional downsampler followed by a standard Transformer encoder.
  - Decoder: Autoregressive Transformer with cross-attention over audio frames, producing text with word-level timestamps.
- **NVIDIA NeMo Parakeet (FastConformer CTC & TDT):** Ultra-fast end-to-end conformer acoustic model with transducer decoding.
- **FunASR Paraformer:** Non-autoregressive predictor-estimator architecture with FSMN memory blocks.
- **Voice Activity Detection (VAD):** Integrated Silero VAD and MarbleNet for automatic speech segmentation and silence suppression.

### Studio-Grade Audio DSP
- **Rational Windowed-Sinc Resampling:** Arbitrary sample-rate conversion (e.g., 16 kHz Whisper $\leftrightarrow$ 22.05 kHz Piper $\leftrightarrow$ 44.1 kHz Studio Master) with zero audible aliasing.
- **Broadcast Downmixing:** ITU-R BS.775 and ATSC A/85 compliant multi-channel to stereo downmixers.
- **Dithered WAV Exporter:** Triangular Probability Density Function (TPDF) noise-shaped dithering for pristine 16-bit and 24-bit PCM output.

---

## 2. Practical Usage & Commands

### CLI Quick Start (Path-Free Defaults)
```bash
# First-run setup: download recommended models with verified checksums
stingray setup speak         # Installs Piper en_US-lessac-medium (60 MB)
stingray setup transcribe    # Installs Whisper base (141 MB)

# Synthesize speech directly
stingray speak "OpenTail Stingray runs fully in managed C#." -o output.wav

# Transcribe an audio recording
stingray transcribe output.wav
```

### Public C# API

For a complete, runnable console application performing roundtrip speech synthesis and transcription, see the [Speech Sample (`samples/OpenTail.Stingray.Sample.Speech`)](../../samples/OpenTail.Stingray.Sample.Speech/).

#### Text to Speech (Piper)
```csharp
using System.Diagnostics;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Piper;

using var tts = PiperPipeline.FromConfigFile("en_US-lessac-medium.onnx.json");
var sw = Stopwatch.StartNew();
var result = tts.Generate(new AudioGenerationRequest
{
    Text = "Hello from local, private speech synthesis.",
    OutputPath = "greeting.wav"
});
sw.Stop();

Console.WriteLine($"Generated {result.Duration.TotalSeconds:F2}s audio in {sw.ElapsedMilliseconds}ms.");
```

#### Speech to Text (Whisper)
```csharp
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Whisper;

using var stt = WhisperPipeline.Load("ggml-base.bin");
var (samples, sampleRate, _) = WavReader.ReadWav("greeting.wav");

var result = stt.Transcribe(new SpeechToTextRequest
{
    AudioSamples = samples,
    SampleRate = sampleRate
});

Console.WriteLine($"Transcription: {result.Text}");
```

---

## 3. Verification Evidence & Benchmark Summary

Benchmarks recorded on an AMD Ryzen 7 5700G (AVX2, CPU only, single test phrase "Hello, I will make some lunch, darling!"):

| Engine | Architecture | Sample Rate | Latency | RTF (Lower is faster) | Status | Hugging Face Repository |
|---|---|---|---|---|---|---|
| **Piper** | VITS (lessac-medium) | 22,050 Hz | 0.32 s | **0.129× 🚀** | 🔬 Verified | [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) |
| **MMS-TTS** | VITS (mms-tts-eng) | 16,000 Hz | 1.30 s | **0.356× ⚡** | 🔬 Verified | [facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng) |
| **Kokoro-82M** | Flow / Style | 24,000 Hz | 2.10 s | **0.580×** | 🔬 Verified | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) |
| **Whisper Base** | Transformer Enc-Dec | 16,000 Hz | 1.20 s | **~0.48×** | 🔬 Verified | [openai/whisper-base](https://huggingface.co/openai/whisper-base) |
| **Whisper Large-v3** | Transformer Enc-Dec | 16,000 Hz | 6.80 s | **~2.72×** | 🔬 Verified | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) |

Detailed evidence logs and benchmark recordings are maintained in [docs/STATUS.md](../STATUS.md).
