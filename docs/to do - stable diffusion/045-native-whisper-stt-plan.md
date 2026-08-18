# Plan — Native Whisper Automatic Speech Recognition (ASR) & Translation Support for OpenTail.Stingray

**Reference:** `examples/whisper.cpp` (`include/whisper.h`, `src/whisper.cpp`)  
**Target:** `opentail-net/OpenTail.Stingray` (`src/OpenTail.Stingray.Audio/Whisper/`)  
**Execution:** **100% native managed C# (.NET 10) — zero external binaries, Python, P/Invoke, or sidecar process**

---

# Status

**READY FOR IMPLEMENTATION**

OpenTail.Stingray currently supports Text (LLMs), Multimodal Vision, Native Diffusion (Images & Video), and a 5-Engine Text-to-Speech (TTS) Suite.

This plan specifies the native C# port of **OpenAI Whisper (whisper.cpp)**, unlocking **Speech-to-Text (ASR)**, **Speech Translation**, and **Timestamp-Accurate Audio Transcription** across 99+ languages in `OpenTail.Stingray`.

---

# 1. Architectural Analysis of OpenAI Whisper

Whisper is an encoder-decoder Transformer trained on 680,000 hours of multilingual speech data.

### 1.1 Complete Whisper End-to-End Transcription & Translation Graph
```text
Audio Input (16kHz WAV / PCM)
        │
        ▼
WhisperMelExtractor (80 or 128 Mel Bins, 16kHz, n_fft=400, hop=160)
        │
        ▼ Log-Mel Spectrogram [80, T_frames]
2× Conv1D Downsamplers (stride 2, GELU) ──> Sequence Length T / 2
        │
        ▼
Sinusoidal Positional Embeddings
        │
        ▼
Audio Transformer Encoder (4 to 32 Layers, Pre-LN Self-Attention)
        │
        ▼ Audio Representation [T_enc, d_model]
        │
        └───────────────────────────────────────────────┐
                                                        ▼ Cross-Attention
Prompt Tokens (e.g. <|startoftranscript|><|en|><|transcribe|><|notimestamps|>)
        │
        ▼
Text Transformer Decoder (4 to 32 Layers, Causal Self-Attention + Audio Cross-Attention)
        │
        ▼
Autoregressive Greedy / Beam Search Token Generation
        │
        ▼
WhisperTokenizer (BPE Decoding + Timestamp Alignment)
        │
        ▼
Transcription Result (Text + Segments + Timestamps + Detected Language)
```

### 1.2 Key Specifications
* **Sample Rate:** 16000 Hz mono PCM.
* **Mel-Spectrogram:** 80 mel channels (standard Whisper) or 128 mel channels (Whisper large-v3), 25ms window (400 samples), 10ms hop (160 samples).
* **Audio Encoder:** 2× Conv1D downsampling (stride 2 $\rightarrow 4\times$ reduction, 1 audio frame per 20ms) + Transformer layers.
* **Text Decoder:** Causal self-attention + cross-attention over encoder output.
* **Special Tokens:** `<|startoftranscript|>`, `<|en|>`, `<|zh|>`, `<|transcribe|>`, `<|translate|>`, `<|notimestamps|>`, `<|endoftranscript|>`.

---

# 2. Design & Implementation Structure

Target layout within `src/OpenTail.Stingray.Audio`:

```text
src/OpenTail.Stingray.Audio
├── ISpeechToTextPipeline.cs       // Standard ASR interface & request/result records
└── Whisper
    ├── WhisperMelExtractor.cs     // 80/128-channel 16kHz log-mel spectrogram extractor
    ├── WhisperTokenizer.cs        // Multilingual BPE tokenizer with timestamp decoding
    ├── WhisperEncoder.cs          // Conv1D downsamplers + Audio Transformer Encoder
    ├── WhisperDecoder.cs          // Causal + Cross-Attention Autoregressive Decoder
    └── WhisperPipeline.cs         // ISpeechToTextPipeline implementation
```

---

# 3. Phased Implementation Plan

### Phase 1: Third-Party Notices & Licensing
* Add Whisper / whisper.cpp (MIT - Copyright (c) 2023-2026 The ggml authors / OpenAI) notice to `THIRD_PARTY_NOTICES.md`.

### Phase 2: ASR Interfaces & Mel Extractor
* Implement `ISpeechToTextPipeline.cs` in `OpenTail.Stingray.Audio`.
* Implement `WhisperMelExtractor.cs` (16kHz audio $\rightarrow$ 80/128 log-mel features).

### Phase 3: Tokenizer, Audio Encoder & Autoregressive Decoder
* Implement `WhisperTokenizer.cs` with language tags and timestamp decoding.
* Implement `WhisperEncoder.cs` with Conv1D downsamplers and sinusoidal positional encoding.
* Implement `WhisperDecoder.cs` with causal self-attention, cross-attention, and greedy search.

### Phase 4: Pipeline, CLI & Server Integration
* Implement `WhisperPipeline.cs`.
* Add `SttCommand.cs` in `OpenTail.Stingray.Cli` (`stingray stt -i audio.wav --language en --task transcribe|translate`).
* Implement OpenAI-compatible `POST /v1/audio/transcriptions` and `POST /v1/audio/translations` in `OpenTail.Stingray.Server`.

### Phase 5: Automated Testing & Verification
* Create `WhisperTests.cs` in `OpenTail.Stingray.Tests.Audio`.
* Build and verify full solution `OpenTail.Stingray.slnx`.
