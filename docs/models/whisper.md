# Whisper Family (OpenAI Whisper Tiny to Large-v3 / Turbo)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | OpenAI |
| **Model Formats** | GGML (`ggml-*.bin` / `ggml-*.gguf`) and HF SafeTensors |
| **Engine Implementations** | `OpenTail.Stingray.Audio.Whisper.WhisperPipeline`, `WhisperEncoder`, `WhisperDecoder` |
| **Audio Preprocessing** | Native 80 / 128 channel Log-Mel Spectrogram filterbanks (16 kHz audio) |
| **Decoding Support** | Greedy argmax, temperature sampling, and word-level timestamp alignment |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (`whisper-base` default transcribe catalogue bundle; word-exact transcription verified) |

---

## 1. Overview & Architectural Highlights

OpenAI's Whisper is the industry standard for robust multi-lingual speech-to-text (ASR) and timestamped audio transcription. In OpenTail.Stingray, Whisper executes end-to-end in managed C#, requiring zero external `whisper.cpp` P/Invoke bindings or Python dependencies.

### Key Architectural Characteristics
* **Audio Frontend:**
  * Raw audio (any sample rate, mono/stereo) is dynamically resampled to 16 kHz using Stingray's `RationalResampler`.
  * Computes 80-channel (Tiny–Small) or 128-channel (Large-v3 / Turbo) Log-Mel spectrogram frames using pure C# SIMD Fast Fourier Transforms (FFT).
* **Audio Encoder:**
  * Two 1D convolutional layers with stride 2 downsample audio frames by $4\times$.
  * Standard transformer encoder blocks with sinusoidal positional embeddings.
* **Autoregressive Text Decoder:**
  * Cross-attention layers attend to the audio encoder representations.
  * Emits special language tokens (`<|en|>`, `<|zh|>`), task tokens (`<|transcribe|>`, `<|translate|>`), and timestamp tokens (`<|0.00|>`, `<|0.02|>`) for word-level subtitle generation.

---

## 2. Checkpoints & Recommended Models

| Model | Parameters | Mel Channels | Recommended File | Typical Size | Relative Speed (RTF) | Hugging Face Repository |
|---|---|---|---|---|---|---|
| **Whisper-Tiny** | 39M | 80 | `ggml-tiny.bin` | ~75 MB | **~0.15×** (Ultra-fast) | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) *(GGML: [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp))* |
| **Whisper-Base** | 74M | 80 | `ggml-base.bin` | ~141 MB | **~0.48×** (**Default Catalogue**) | [openai/whisper-base](https://huggingface.co/openai/whisper-base) *(GGML: [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp))* |
| **Whisper-Small** | 244M | 80 | `ggml-small.bin` | ~465 MB | **~0.95×** | [openai/whisper-small](https://huggingface.co/openai/whisper-small) *(GGML: [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp))* |
| **Whisper-Medium** | 769M | 80 | `ggml-medium.bin` | ~1.46 GB | **~1.80×** | [openai/whisper-medium](https://huggingface.co/openai/whisper-medium) *(GGML: [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp))* |
| **Whisper-Large-v3** | 1.55B | 128 | `ggml-large-v3.bin` | ~3.09 GB | **~2.72×** | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) *(GGML: [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp))* |
| **Whisper-Turbo** | 809M | 128 | `ggml-large-v3-turbo.bin` | ~1.62 GB | **~0.85×** (Fast Large-v3) | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) *(GGML: [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp))* |

---

## 3. Usage & Code Examples

### C# Audio Transcription

```csharp
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Whisper;

// 1. Load Whisper pipeline (e.g. default base model)
using var whisper = WhisperPipeline.Load("models/ggml-base.bin");

// 2. Read input WAV file
var (samples, sampleRate, _) = WavReader.ReadWav("meeting_recording.wav");

// 3. Transcribe speech to text
var result = whisper.Transcribe(new SpeechToTextRequest
{
    AudioSamples = samples,
    SampleRate = sampleRate,
    Language = "en",
    EmitTimestamps = true
});

Console.WriteLine($"Transcript: {result.Text}");
```

### CLI Command

```bash
# First-run setup: installs verified whisper-base bundle
stingray setup transcribe

# Transcribe any audio recording path-free
stingray transcribe meeting_recording.wav

# Or transcribe with explicit model file
stingray stt -m base --model-file models/ggml-base.bin -i recording.wav
```
