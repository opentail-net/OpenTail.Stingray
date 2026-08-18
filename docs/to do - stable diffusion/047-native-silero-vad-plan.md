# Plan â Native Silero Voice Activity Detection (VAD) for OpenTail.Stingray

**Reference:** `examples/whisper.cpp` (`src/whisper.cpp` lines 4369â4750, `tests/test-vad.cpp`)  
**Target:** `opentail-net/OpenTail.Stingray` (`src/OpenTail.Stingray.Audio/Vad/`)  
**Execution:** **100% native managed C# (.NET 10) â zero external binaries, Python, P/Invoke, or sidecar process**

---

# Status

**COMPLETED (100% Native C# Port)**

OpenTail.Stingray supports Text Generation (LLMs), Text Embeddings, Cross-Encoder Reranking, Multimodal Vision, Native Diffusion (Images & Video), a 5-Engine Text-to-Speech (TTS) Suite, native OpenAI Whisper Speech-to-Text (ASR), and native Silero Voice Activity Detection (VAD).

---

# 1. Architectural Analysis of Silero VAD

Silero VAD is a compact, ultra-fast recurrent neural network (~500k parameters) operating on 512-sample audio windows (31.25ms at 16kHz).

### 1.1 Complete Silero VAD Processing Graph
```text
Audio Frame (512 samples @ 16kHz PCM)
        â
        â¼
Reflective 1D Padding (64, 64) ââ> 640 samples
        â
        â¼
STFT Convolution (Forward Basis [256, 1, 258], hop=128)
        â
        â¼ Complex STFT [2, 129, 4]
Magnitude Layer: sqrt(Real^2 + Imag^2) ââ> [129, 4]
        â
        â¼
4-Stage Conv1D Encoder (ReLU):
  - Layer 0: [3, 129, 128], stride=1
  - Layer 1: [3, 128, 64], stride=2
  - Layer 2: [3, 64, 64], stride=2
  - Layer 3: [3, 64, 128], stride=1
        â
        â¼ Feature Vector [128]
Recurrent LSTM Layer (Hidden Dim = 128, stateful h_state, c_state)
  - Input Gate:   i_t = Ï(W_ii x + W_hi h + b_i)
  - Forget Gate:  f_t = Ï(W_if x + W_hf h + b_f)
  - Cell Gate:    g_t = tanh(W_ig x + W_hg h + b_g)
  - Output Gate:  o_t = Ï(W_io x + W_ho h + b_o)
  - State Update: c_t = f_t * c_{t-1} + i_t * g_t
  - Hidden State: h_t = o_t * tanh(c_t)
        â
        â¼
Final Projection Conv1D + Sigmoid
        â
        â¼
Speech Probability: P(speech) â [0.0, 1.0]
        â
        â¼
VadSegmenter (threshold, min_speech_ms, min_silence_ms, speech_pad_ms)
        â
        â¼
Speech Timestamp Segments List: [(t0, t1), (t2, t3), ...]
```

---

# 2. Design & Implementation Structure

Target layout within `src/OpenTail.Stingray.Audio/Vad`:

```text
src/OpenTail.Stingray.Audio
âââ Vad
â   âââ IVoiceActivityDetector.cs // VAD interface, parameters, and segment records
â   âââ SileroVad.cs              // Pure C# Silero VAD neural network (STFT + CNN + LSTM)
â   âââ VadSegmenter.cs           // Segment thresholding & boundary aggregation
âââ Whisper
    âââ WhisperPipeline.cs        // Silence filtering integration using VAD
```

---

# 3. Phased Implementation Plan

### Phase 1: VAD Interfaces & Data Contracts [COMPLETED]
* Implemented `IVoiceActivityDetector.cs` in `OpenTail.Stingray.Audio.Vad`.

### Phase 2: Silero Neural Network Engine [COMPLETED]
* Implemented `SileroVad.cs` with STFT magnitude extraction, 4-stage Conv1D feature encoder, stateful LSTM cell updates, and sigmoid probability generation.

### Phase 3: Segment Aggregation & Boundary Extraction [COMPLETED]
* Implemented `VadSegmenter.cs` mirroring `whisper_vad_segments_from_probs`.

### Phase 4: Whisper STT & CLI Integration [COMPLETED]
* Updated `WhisperPipeline.cs` and `ISpeechToTextPipeline.cs` to support VAD silence pruning.
* Updated `SttCommand.cs` with `--vad` option and refreshed `docs/cli-option-inventory.md` (190 options).

### Phase 5: Automated Testing & Verification [COMPLETED]
* Created `SileroVadTests.cs` in `OpenTail.Stingray.Tests.Audio`.
* Verified full solution build across `OpenTail.Stingray.slnx` (39/39 Audio tests passed, 367/367 CLI tests passed, 12/12 Server tests passed, 528/528 Core tests passed, 0 errors, 0 warnings).
