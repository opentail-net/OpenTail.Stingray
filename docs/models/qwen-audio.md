# Qwen Audio Family (Qwen3-TTS & Qwen3-ASR)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | Alibaba Cloud (Qwen Team) |
| **Model Formats** | SafeTensors & GGUF |
| **Engine Implementations** | `OpenTail.Stingray.Audio.QwenTTS.QwenTtsPipeline`, `OpenTail.Stingray.Audio.QwenASR` |
| **Speech Generation (TTS)** | Qwen3-TTS 12Hz with ERes2NetV2 192-dim voice cloning speaker encoder |
| **Speech Recognition (ASR)** | Qwen3-ASR 0.6B / 1.7B and Qwen3-ForcedAligner (word timestamps) |
| **Output Sample Rate** | 24,000 Hz studio audio (TTS) / 16,000 Hz input (ASR) |
| **Status & Confidence** | 🟢 **Admitted & Verified** (Exact word-for-word correct ASR transcriptions verified on test phrases) |

---

## 1. Overview & Architectural Highlights

The Qwen Audio family delivers studio-quality speech generation, neural voice cloning, and high-accuracy speech recognition built upon Alibaba's unified multimodal architecture.

### Key Architectural Characteristics
* **Qwen3-TTS 12Hz:**
  * Uses a high-compression discrete neural audio codec operating at only 12 frames per second (12Hz), enabling fast autoregressive token generation.
  * Incorporates an **ERes2NetV2 192-dimensional speaker encoder**, allowing high-fidelity zero-shot voice cloning from a reference recording.
* **Qwen3-ASR (Speech-to-Text):**
  * Transformer encoder-decoder architecture handling diverse accents, background noise, and multi-speaker overlapping speech.
  * Parameter sizes from 0.6B up to 1.7B.
* **Qwen3-ForcedAligner:**
  * Computes frame-accurate phonetic and word boundary timestamps, essential for subtitle synchronization and audio karaoke timing.

---

## 2. Checkpoints & Recommended Models

| Model | Task | Parameters | Typical Size |
|---|---|---|---|
| **Qwen3-TTS-12Hz** | Voice Synthesis & Cloning | ~800M | ~1.6 GB |
| **Qwen3-ASR-0.6B** | Speech Recognition | 0.6B | ~1.2 GB |
| **Qwen3-ASR-1.7B** | Speech Recognition | 1.7B | ~3.4 GB |
| **Qwen3-ForcedAligner** | Word-Level Timestamps | 0.6B | ~1.2 GB |

---

## 3. Usage & Code Examples

### C# Voice Cloning with Qwen3-TTS

```csharp
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.QwenTTS;

// 1. Initialize Qwen3-TTS pipeline
using var tts = new QwenTtsPipeline();

// 2. Generate speech with reference voice cloning
var result = tts.Generate(new AudioGenerationRequest
{
    Text = "OpenTail Stingray is the unified, native .NET 10 multimodal AI engine.",
    ReferenceAudioPath = "samples/speaker_reference.wav",
    OutputPath = "qwen_output.wav"
});

Console.WriteLine($"Synthesized {result.Duration.TotalSeconds:F2}s of 24kHz audio.");
```

### CLI Speech Recognition

```bash
# Transcribe with Qwen3-ASR
stingray stt -e qwen3 -m models/qwen3-asr-0.6b -i meeting.wav
```
