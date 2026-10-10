# XTTS-v2 Family (Coqui XTTS-v2 Zero-Shot Voice Cloning)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | Coqui AI |
| **Model Formats** | SafeTensors / PyTorch Checkpoint |
| **Engine Implementations** | `OpenTail.Stingray.Audio.Xtts.XttsPipeline`, `XttsGpt` |
| **Speaker Conditioning** | Zero-shot voice cloning from 3–6 second reference audio clip |
| **Architecture** | Autoregressive GPT-2 text-to-codec + FiLM-conditioned HiFi-GAN vocoder |
| **Output Sample Rate** | 24,000 Hz |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (13 stages individually golden-verified, wired end-to-end) |

---

## 1. Overview & Architectural Highlights

Coqui XTTS-v2 is a state-of-the-art multilingual voice cloning pipeline capable of replicating any speaker's voice timbre, pitch inflection, and accent from a short (3–10 second) reference audio recording.

### Key Architectural Characteristics
* **13 Golden-Verified Stages in Pure C#:**
  In OpenTail.Stingray, every stage of the XTTS-v2 graph is verified against the reference PyTorch implementation:
  1. Mel Spectrogram computation and normalization
  2. ResNet speaker encoder (extracts 512-dim speaker embedding $s$)
  3. Perceiver cross-attention resampler
  4. Byte-level BPE text tokenizer
  5. GPT-2 autoregressive audio token transformer
  6. DVAE latent projection
  7. HiFi-GAN FiLM-conditioned neural vocoder
* **Multilingual Zero-Shot Transfer:**
  Clone an English speaker's voice and have them speak fluent German, French, Spanish, Japanese, or Mandarin while retaining their vocal signature.
* **Pure Managed Implementation:**
  Runs entirely in C# without Python, PyTorch, or CUDA toolchains installed.

---

## 2. Checkpoints & Recommended Models

| Model | Format | Size | Function | Hugging Face Repository |
|---|---|---|---|---|
| **XTTS-v2 Main Model** | `model.safetensors` | ~1.8 GB | GPT-2 codec predictor & Vocoder | [coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2) |
| **XTTS-v2 Speaker Encoder** | `speaker_encoder.safetensors` | ~45 MB | Voice embedding extractor | [coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2) |
| **XTTS-v2 Vocoder Config** | `config.json` | ~4 KB | Hyperparameters & sample rate | [coqui/XTTS-v2](https://huggingface.co/coqui/XTTS-v2) |

---

## 3. Usage & Code Examples

### C# Zero-Shot Voice Cloning

```csharp
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Xtts;

// 1. Initialize XTTS-v2 voice cloning pipeline
using var xtts = XttsPipeline.Load("models/xtts-v2");

// 2. Clone voice from a reference audio clip
var result = xtts.Generate(new XttsRequest
{
    Text = "This voice was synthesized by matching a six-second reference sample.",
    SpeakerAudioPath = "samples/my_voice_reference.wav",
    Language = "en",
    OutputPath = "cloned_speech.wav"
});

Console.WriteLine($"Cloned voice audio generated: {result.Duration.TotalSeconds:F2}s.");
```

### CLI Command

```bash
# Clone voice from reference clip
stingray tts -e xtts2 \
             -m models/xtts-v2 \
             --speaker-audio samples/my_voice_reference.wav \
             -t "Hello world in my cloned voice." \
             -o cloned.wav
```
