# Kokoro Family (Kokoro-82M Compact Neural TTS)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | Hexgrad / StyleTTS 2 Community |
| **Model Formats** | GGUF (`kokoro-v0_19.gguf`) and ONNX |
| **Engine Implementations** | `OpenTail.Stingray.Audio.Kokoro.KokoroPipeline`, `StyleTtsGenerator` |
| **Parameters** | 82 Million parameters |
| **Output Sample Rate** | 24,000 Hz studio audio |
| **Style Conditioning** | Multi-speaker voice embeddings (e.g., `af_bella`, `am_adam`, `bf_emma`) |
| **Status & Confidence** | 🟢 **Admitted & Golden-Verified** (Level 2 Proven, verified voice generation) |

---

## 1. Overview & Architectural Highlights

Kokoro-82M is an open-weights, ultra-compact text-to-speech model derived from the StyleTTS 2 architecture. Despite containing only 82M parameters, it achieves natural speech cadence, pitch inflections, and emotional expression matching models 10× its size.

### Key Architectural Characteristics
* **StyleTTS 2 Framework:**
  * Uses a diffusion-based style predictor to sample natural prosody, duration, and pitch variations.
  * Employs differentiable duration modeling rather than hard alignment matrices.
* **Style Vector Voice Switching:**
  * Voice identities are controlled by compact 128-dimensional or 256-dimensional style vectors. Switching speaker voices does not require reloading model weights; the application simply swaps the style tensor slice.
* **Studio 24 kHz Quality:**
  * Produces 24,000 Hz audio output, providing richer high-frequency acoustics than legacy 16 kHz or 22 kHz TTS engines.
* **GGUF Quantization:**
  * Fully quantized in GGUF (`Q8_0` or `FP16`), requiring only ~85 MB of storage.

---

## 2. Checkpoints & Recommended GGUFs

| Model | Format | Recommended File | Typical Size | Hugging Face Repository |
|---|---|---|---|---|
| **Kokoro-82M v0.19** | GGUF | `kokoro-v0_19-Q8_0.gguf` | ~85 MB | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) |
| **Kokoro-82M Voices** | Tensor pack | `voices.bin` / `voices.json` | ~5 MB | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) |

---

## 3. Usage & Code Examples

### C# Voice Generation with Speaker Style Selection

```csharp
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Kokoro;

// 1. Initialize Kokoro-82M pipeline
using var model = KokoroModel.Load("models/kokoro-v0_19-Q8_0.gguf");
using var kokoro = new KokoroPipeline(model);

// 2. Synthesize with chosen speaker style
var result = kokoro.Generate(new AudioGenerationRequest
{
    Text = "Kokoro delivers 24kHz expressive voice synthesis at just 82 million parameters.",
    Voice = "af_bella", // e.g. Bella (American Female), Adam, Emma
    OutputPath = "kokoro_sample.wav"
});

Console.WriteLine($"Synthesized {result.Duration.TotalSeconds:F2}s audio.");
```

### CLI Command

```bash
# Generate speech with Kokoro
stingray tts -e kokoro -m models/kokoro-v0_19-Q8_0.gguf --voice af_bella -t "Welcome to OpenTail Stingray." -o welcome.wav
```
