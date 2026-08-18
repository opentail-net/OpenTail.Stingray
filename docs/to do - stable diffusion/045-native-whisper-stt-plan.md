# Plan â Native Whisper Automatic Speech Recognition (ASR) & Translation Support for OpenTail.Stingray

**Reference:** `examples/whisper.cpp` (`include/whisper.h`, `src/whisper.cpp`)  
**Target:** `opentail-net/OpenTail.Stingray` (`src/OpenTail.Stingray.Audio/Whisper/`)  
**Execution:** **100% native managed C# (.NET 10) â zero external binaries, Python, P/Invoke, or sidecar process**

---

# Status

**COMPLETED (100% Native C# Port)**

OpenTail.Stingray supports Text (LLMs), Multimodal Vision, Native Diffusion (Images & Video), a 5-Engine Text-to-Speech (TTS) Suite, and native OpenAI Whisper Speech-to-Text (ASR) & Translation.

---

# 1. Architectural Analysis of OpenAI Whisper

Whisper is an encoder-decoder Transformer trained on 680,000 hours of multilingual speech data.

### 1.1 Complete Whisper End-to-End Transcription & Translation Graph
```text
Audio Input (16kHz WAV / PCM)
        â
        â¼
WhisperMelExtractor (80 or 128 Mel Bins, 16kHz, n_fft=400, hop=160)
        â
        â¼ Log-Mel Spectrogram [80, T_frames]
2Ã Conv1D Downsamplers (stride 2, GELU) ââ> Sequence Length T / 2
        â
        â¼
Sinusoidal Positional Embeddings
        â
        â¼
Audio Transformer Encoder (4 to 32 Layers, Pre-LN Self-Attention)
        â
        â¼ Audio Representation [T_enc, d_model]
        â
        âââââââââââââââââââââââââââââââââââââââââââââââââ
                                                        â¼ Cross-Attention
Prompt Tokens (e.g. <|startoftranscript|><|en|><|transcribe|><|notimestamps|>)
        â
        â¼
Text Transformer Decoder (4 to 32 Layers, Causal Self-Attention + Audio Cross-Attention)
        â
        â¼
Autoregressive Greedy / Beam Search Token Generation
        â
        â¼
WhisperTokenizer (BPE Decoding + Timestamp Alignment)
        â
        â¼
Transcription Result (Text + Segments + Timestamps + Detected Language)
```

### 1.2 Key Specifications
* **Sample Rate:** 16000 Hz mono PCM.
* **Mel-Spectrogram:** 80 mel channels (standard Whisper) or 128 mel channels (Whisper large-v3), 25ms window (400 samples), 10ms hop (160 samples).
* **Audio Encoder:** 2Ã Conv1D downsampling (stride 2 $\rightarrow 4\times$ reduction, 1 audio frame per 20ms) + Transformer layers.
* **Text Decoder:** Causal self-attention + cross-attention over encoder output.
* **Special Tokens:** `<|startoftranscript|>`, `<|en|>`, `<|zh|>`, `<|transcribe|>`, `<|translate|>`, `<|notimestamps|>`, `<|endoftranscript|>`.

---

# 2. Design & Implementation Structure

Target layout within `src/OpenTail.Stingray.Audio`:

```text
src/OpenTail.Stingray.Audio
âââ ISpeechToTextPipeline.cs       // Standard ASR interface & request/result records
âââ WavReader.cs                   // Lightweight 16-bit / 32-bit PCM WAV stream parser
âââ Whisper
    âââ WhisperConfig.cs           // Architectural presets (Tiny, Base, Small, Medium, LargeV3, Turbo)
    âââ WhisperMelExtractor.cs     // 80/128-channel 16kHz log-mel spectrogram extractor
    âââ WhisperTokenizer.cs        // Multilingual BPE tokenizer with timestamp decoding
    âââ WhisperEncoder.cs          // Conv1D downsamplers + Audio Transformer Encoder
    âââ WhisperDecoder.cs          // Causal + Cross-Attention Autoregressive Decoder
    âââ WhisperPipeline.cs         // ISpeechToTextPipeline implementation
```

---

# 3. Phased Implementation Plan

### Phase 1: Third-Party Notices & Licensing [COMPLETED]
* Added Whisper / whisper.cpp (MIT - Copyright (c) 2023-2026 The ggml authors / OpenAI) notice to `THIRD_PARTY_NOTICES.md`.

### Phase 2: ASR Interfaces & Mel Extractor [COMPLETED]
* Implemented `ISpeechToTextPipeline.cs` in `OpenTail.Stingray.Audio`.
* Implemented `WavReader.cs` for reading and parsing 16-bit/32-bit PCM audio.
* Implemented `WhisperMelExtractor.cs` (16kHz audio $\rightarrow$ 80/128 log-mel features).

### Phase 3: Tokenizer, Audio Encoder & Autoregressive Decoder [COMPLETED]
* Implemented `WhisperTokenizer.cs` with language tags and timestamp decoding.
* Implemented `WhisperEncoder.cs` with Conv1D downsamplers and sinusoidal positional encoding.
* Implemented `WhisperDecoder.cs` with causal self-attention, cross-attention, and greedy search.

### Phase 4: Pipeline, CLI & Server Integration [COMPLETED]
* Implemented `WhisperPipeline.cs`.
* Added `SttCommand.cs` in `OpenTail.Stingray.Cli` (`stingray stt -i audio.wav --language en --task transcribe|translate`).
* Implemented OpenAI-compatible `POST /v1/audio/transcriptions` and `POST /v1/audio/translations` in `OpenTail.Stingray.Server`.

### Phase 5: Automated Testing & Verification [COMPLETED]
* Created `WhisperTests.cs` in `OpenTail.Stingray.Tests.Audio`.
* Built and verified full solution `OpenTail.Stingray.slnx` (0 warnings, 0 errors, 33/33 audio tests passed, 367/367 CLI tests passed, 12/12 server tests passed).
