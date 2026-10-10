# Piper VITS Family (Fast Local Neural TTS)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | Rhasspy / Michael Hansen |
| **Model Formats** | ONNX (`.onnx` + `.onnx.json` config pair) |
| **Engine Implementations** | `OpenTail.Stingray.Audio.Piper.PiperPipeline`, `VitsGenerator` |
| **Synthesis Speed** | **🥇 0.129× Real-Time Factor (RTF)** on standard CPU (~320 ms per 2.5s sentence) |
| **Phonemizer** | Native managed phoneme map / eSpeak-ng tokenization |
| **Output Sample Rate** | 22,050 Hz (Medium models) / 16,000 Hz (Low) |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (`piper-lessac` default speech catalogue bundle) |

---

## 1. Overview & Architectural Highlights

Piper is an ultra-fast, local neural text-to-speech engine based on the VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) architecture. In OpenTail.Stingray, Piper serves as the default voice synthesis engine (`stingray setup speak`), delivering crisp, human-like voice synthesis with sub-second latency on modest CPUs.

### Key Architectural Characteristics
* **End-to-End VITS Architecture:**
  * **Text Encoder:** Transforms phoneme tokens into prior latent distributions.
  * **Duration Predictor:** Predicts phoneme durations via a stochastic flow model.
  * **Monotonic Alignment Search (MAS):** Aligns phoneme representations with audio frames.
  * **HiFi-GAN Neural Vocoder:** Generates 22,050 Hz PCM waveform samples directly from latent states.
* **Blazing CPU Inference Speed:**
  Generates audio at over 7× faster than real time (0.129× RTF) on an ordinary desktop CPU without dedicated GPU acceleration.
* **Licensing & Safety:**
  Voice checkpoints (e.g. `en_US-lessac-medium`) have explicit data licenses; Stingray’s front door prompts the user for license consent upon first setup.

---

## 2. Checkpoints & Recommended Voices

| Voice | Language | Quality | ONNX Model File | Config File | Size | Hugging Face Repository |
|---|---|---|---|---|---|---|
| **en_US-lessac-medium** | English (US) | Medium (22 kHz) | `en_US-lessac-medium.onnx` | `en_US-lessac-medium.onnx.json` | ~60 MB (**Default Catalogue**) | [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) |
| **en_US-libritts-high** | English (US) | High (22 kHz) | `en_US-libritts-high.onnx` | `en_US-libritts-high.onnx.json` | ~110 MB | [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) |
| **en_GB-alba-medium** | English (UK) | Medium (22 kHz) | `en_GB-alba-medium.onnx` | `en_GB-alba-medium.onnx.json` | ~60 MB | [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) |
| **de_DE-eva_k-x_low** | German | Low (16 kHz) | `de_DE-eva_k-x_low.onnx` | `de_DE-eva_k-x_low.onnx.json` | ~25 MB | [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) |

---

## 3. Usage & Code Examples

### C# High-Speed Speech Synthesis

```csharp
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Piper;

// 1. Initialize Piper from its companion JSON configuration
using var tts = PiperPipeline.FromConfigFile("voices/en_US-lessac-medium.onnx.json");

// 2. Synthesize audio directly to a pristine WAV file
var result = tts.Generate(new AudioGenerationRequest
{
    Text = "OpenTail Stingray runs high-speed voice synthesis locally in pure managed C#.",
    OutputPath = "speech.wav"
});

Console.WriteLine($"Synthesized {result.Duration.TotalSeconds:F2}s audio in {result.ComputeTimeMs}ms.");
```

### CLI Command

```bash
# First-run setup: installs verified piper-lessac voice
stingray setup speak

# Generate speech path-free
stingray speak "Hello from private, on-device AI." -o output.wav

# Play or inspect output
stingray transcribe output.wav
```
