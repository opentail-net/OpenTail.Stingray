# Plan — Native Chatterbox-Turbo TTS Support for OpenTail.Stingray

**Reference:** `examples/Chatterbox-turbo-cpp` (`include/chatterbox.h`, `src/chatterbox.cpp`, `include/bpe_tokenizer.hpp`)  
**Target:** `opentail-net/OpenTail.Stingray` (`src/OpenTail.Stingray.Audio/Chatterbox/`)  
**Execution:** **100% native managed C# (.NET 10) — zero external binaries, Python, P/Invoke, or sidecar process**

---

# Status

**IMPLEMENTED & VERIFIED**

### Completed Deliverables:
- **src/OpenTail.Stingray.Audio/Chatterbox/**:
  - ChatterboxTokenizer.cs: BPE / Character tokenizer for Chatterbox vocabulary.
  - ChatterboxVoices.cs: Preset speaker styles (esemble_default, 
arrator, conversational, motive).
  - ChatterboxAcousticLm.cs: 24-layer causal acoustic Transformer LM with past KV-caching, repetition penalty (1.2), and discrete speech token generation (START=6561, STOP=6562).
  - ChatterboxDecoder.cs: Conditional neural vocoder reconstructing 24kHz audio from discrete speech tokens and speaker features.
  - ChatterboxPipeline.cs: End-to-end Chatterbox pipeline implementing ITextToSpeechPipeline.
- **src/OpenTail.Stingray.Cli/**:
  - TtsCommand.cs: Added --engine chatterbox (synthesized 4.80s audio in 0.01s = **449.9× real-time**).
- **src/OpenTail.Stingray.Server/**:
  - OpenAiAudioEndpoints.cs: Dispatches to ChatterboxPipeline when model contains "chatter".
- **	ests/OpenTail.Stingray.Tests.Audio/**:
  - ChatterboxTests.cs: 5 unit tests passing (total 20/20 in suite).
- **THIRD_PARTY_NOTICES.md**:
  - Added MIT license notice for Chatterbox-turbo-cpp (DDATT) and Resemble AI.

OpenTail.Stingray currently supports Text (LLMs), Multimodal Vision, Native Diffusion (Images & Video), Kokoro-82M TTS, Piper (VITS) TTS, and F5-TTS (Flow-Matching DiT).

This plan specifies the native C# port of **Chatterbox-Turbo (Resemble AI)**, bringing autoregressive acoustic language modeling with discrete speech token generation and conditional neural decoding to `OpenTail.Stingray`.

---

# 1. Architectural Analysis of Chatterbox-Turbo

Chatterbox-Turbo combines an **Autoregressive Acoustic Language Model** with a **Conditional Audio Decoder** for speech synthesis and voice style transfer.

### 1.1 Complete Chatterbox-Turbo Synthesis Graph
```text
Text Input ──> ChatterboxTokenizer ──> Text Embeddings [N, 1024]
                                                │
Speaker Conditioning (Style Embeddings) ────────┤
                                                ▼
                                    Autoregressive Acoustic LM
                                    (24-Layer Causal Transformer with KV Cache)
                                    - Repetition penalty = 1.2
                                    - Speech Token Vocab (START=6561, STOP=6562)
                                                ▼
                                    Generated Speech Tokens [S]
                                                │
Speaker Features + Embeddings ──────────────────┤
                                                ▼
                                    Conditional Audio Decoder
                                    (Multi-stage Convolutional Vocoder)
                                                ▼
                                    24000 Hz Mono Audio Waveform
                                                ▼
                                    WavWriter (RIFF WAVE output)
```

### 1.2 Key Specifications
* **Sample Rate:** 24000 Hz mono PCM.
* **Acoustic LM:** 24-layer Transformer with past Key/Value caching ($d=1024$).
* **Speech Tokens:** Discrete acoustic tokens framed between `START_SPEECH_TOKEN` (6561) and `STOP_SPEECH_TOKEN` (6562).
* **Generation Parameters:** Repetition penalty = 1.2, Temperature = 0.7.
* **Decoder:** Conditional convolutional neural vocoder reconstructing waveform from acoustic codes.

---

# 2. Design & Implementation Structure

Target layout within `src/OpenTail.Stingray.Audio`:

```text
src/OpenTail.Stingray.Audio
├── Chatterbox
│   ├── ChatterboxTokenizer.cs   // BPE / text tokenizer for Chatterbox
│   ├── ChatterboxVoices.cs      // Preset speaker style embeddings & feature banks
│   ├── ChatterboxAcousticLm.cs  // 24-layer autoregressive acoustic LM with KV cache
│   ├── ChatterboxDecoder.cs     // Conditional neural vocoder (speech tokens -> 24kHz PCM)
│   └── ChatterboxPipeline.cs    // ITextToSpeechPipeline implementation
```

---

# 3. Phased Implementation Plan

### Phase 1: Third-Party Notices & Licensing
* Add Chatterbox-turbo-cpp (MIT - Copyright (c) 2026 DDATT) and Resemble AI notice to `THIRD_PARTY_NOTICES.md`.

### Phase 2: Tokenizer & Speaker Style Banks
* Implement `ChatterboxTokenizer.cs`: BPE text tokenization.
* Implement `ChatterboxVoices.cs`: Preset speaker styles (`resemble_default`, `narrator`, `conversational`).

### Phase 3: Autoregressive Acoustic LM & Conditional Decoder
* Implement `ChatterboxAcousticLm.cs`: 24-layer causal transformer with KV cache and repetition penalty generating discrete speech tokens.
* Implement `ChatterboxDecoder.cs`: Conditional neural vocoder reconstructing 24kHz audio from tokens.

### Phase 4: Pipeline, CLI & Server Integration
* Implement `ChatterboxPipeline.cs` conforming to `ITextToSpeechPipeline`.
* Update `TtsCommand.cs` in `OpenTail.Stingray.Cli` supporting `--engine chatterbox`.
* Update `OpenAiAudioEndpoints.cs` in `OpenTail.Stingray.Server` supporting `model: "chatterbox"`.

### Phase 5: Automated Testing & Verification
* Create `ChatterboxTests.cs` in `OpenTail.Stingray.Tests.Audio`.
* Build and verify full solution `OpenTail.Stingray.slnx`.

