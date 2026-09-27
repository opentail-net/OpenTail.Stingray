# Plan — Native Kokoro-82M Text-to-Speech (TTS) Support for OpenTail.Stingray

**Reference:** `examples/KOKORO-GPT2` (`src/models/kokoro/main.cpp`, `src/models/kokoro/phonemizer.cpp`), `examples/kokoro.cpp` (`Kokoro.cpp`, `EnG2P.h`), and `examples/kokoro-server`  
**Target:** `opentail-net/OpenTail.Stingray` (`src/OpenTail.Stingray.Audio`)  
**Execution:** **100% local/native C# (.NET 10) — zero external binaries, Python, P/Invoke, or sidecar process**

---

# Status

**IMPLEMENTED & VERIFIED**

### Completed Deliverables:
- **src/OpenTail.Stingray.Audio/**:
  - WavWriter.cs: 16-bit PCM RIFF WAVE exporter at 24kHz.
  - ITextToSpeechPipeline.cs: Universal audio request/result contracts.
  - KokoroPhonemizer.cs: G2P vocabulary and token indexing (178 token IDs).
  - KokoroVoices.cs: Preset 256-dim speaker style vectors (f_heart, f_bella, m_adam, f_alice, m_george, etc.).
  - KokoroModel.cs: 3-layer PLBERT transformer encoder + duration predictor + 4-stage AdaIN-ResBlock decoder + 24kHz iSTFT waveform vocoder.
  - KokoroPipeline.cs: End-to-end TTS inference pipeline.
- **src/OpenTail.Stingray.Cli/**:
  - TtsCommand.cs: Added stingray tts CLI command with voice selection and real-time factor reporting.
- **src/OpenTail.Stingray.Server/**:
  - OpenAiAudioEndpoints.cs: Exposed official OpenAI /v1/audio/speech endpoint.
- **	ests/OpenTail.Stingray.Tests.Audio/**:
  - KokoroTests.cs: 6/6 unit tests passing.
- **THIRD_PARTY_NOTICES.md**:
  - Added third-party notices for KOKORO-GPT2, kokoro.cpp, and kokoro-infer.

OpenTail.Stingray currently supports Text (LLM), Multimodal Vision (Gemma 4 / Qwen-VL), and Native Image & Video Diffusion (SD 1.5, SDXL, SD 3/3.5, FLUX.1, Z-Image, Qwen Image, Wan 2.1/2.2, HunyuanVideo, LTX-Video).

This plan specifies the native C# implementation of **Kokoro-82M Text-to-Speech**, adding high-quality real-time voice synthesis and the OpenAI `/v1/audio/speech` endpoint to Stingray.

---

# 1. Objective

Add native C# support for the **Kokoro-82M** Text-to-Speech architecture.

The primary targets are:
1. **`OpenTail.Stingray.Audio`**: Core Audio & TTS assembly with Kokoro-82M synthesis engine.
2. **Native Phonemizer & G2P Frontend**: Converting plain text to phoneme sequences and token IDs.
3. **PLBERT & AdaIN Decoder Vocoder**: 3-layer phoneme transformer + duration predictor + AdaIN-ResBlock decoder + 24kHz iSTFT waveform generation.
4. **OpenAI Audio API (`/v1/audio/speech`)**: REST endpoint in `OpenTail.Stingray.Server` returning audio streams (`audio/wav`, `audio/pcm`).
5. **CLI `tts` Command**: Command in `OpenTail.Stingray.Cli` for quick synthesis (`stingray tts -p "Hello world" -o out.wav`).

Target repository layout:

```text
src/OpenTail.Stingray.Audio
├── ITextToSpeechPipeline.cs
├── WavWriter.cs
└── Kokoro
    ├── KokoroPhonemizer.cs
    ├── KokoroModel.cs
    ├── KokoroVoices.cs
    └── KokoroPipeline.cs
```

---

# 2. Architectural Analysis & Specification

### 2.1 Full Pipeline Flow (`KOKORO-GPT2/src/models/kokoro/main.cpp`)
```text
Text Input ──> KokoroPhonemizer ──> Token IDs [N]
                                         │
Style Vector [256] (Voice) ───────┐      ▼
                                  ├──> PLBERT Encoder [N, 512]
                                  │      ▼
                                  ├──> Duration Predictor [N] ──> Frame Count T
                                  │      ▼
                                  ├──> Length Regulator / Frame Upsampling [T, 512]
                                  │      ▼
                                  └──> AdaIN-ResBlocks Decoder [T, 512]
                                         ▼
                                  iSTFT / HiFi-GAN Vocoder
                                         ▼
                                  Audio Samples [24000 Hz float PCM]
                                         ▼
                                  WavWriter (RIFF WAVE output)
```

### 2.2 Model Components
1. **PLBERT Transformer Encoder:**
   * 3-layer ALBERT / Transformer with embedding dimension $d=512$, $8$ attention heads, and intermediate dimension $2048$.
2. **Style Vectors (`KokoroVoices.cs`):**
   * 256-dimensional speaker embeddings defining pitch, cadence, and vocal timbre (e.g. `af_heart`, `af_bella`, `am_adam`, `bf_alice`, `bm_george`).
3. **Duration Predictor:**
   * Predicts frame durations per phoneme token ($T_i \in [1, 20]$ frames).
4. **AdaIN-ResBlocks Decoder:**
   * 4-stage residual network with Adaptive Instance Normalization modulating feature channels with the 256-dim voice style vector.
5. **iSTFT / Waveform Generator:**
   * Inverse Short-Time Fourier Transform with Hann window (n_fft=1024, hop_length=256) synthesizing clean 24kHz float audio samples.

---

# 3. Phased Implementation Strategy

### Phase 1: Create `OpenTail.Stingray.Audio` & Core Abstractions
* Create `src/OpenTail.Stingray.Audio/OpenTail.Stingray.Audio.csproj`.
* Implement `WavWriter.cs` for 16-bit PCM 24kHz `.wav` header and sample streaming.
* Implement `ITextToSpeechPipeline.cs` with `AudioGenerationRequest` and `AudioGenerationResult`.

### Phase 2: Kokoro Phonemizer & Voice Embeddings
* Implement `KokoroPhonemizer.cs`:
  * Vocabulary dictionary mapping words and IPA phoneme symbols to token indices $[0, 177]$.
* Implement `KokoroVoices.cs`:
  * Default style vector presets (`af_heart`, `am_adam`, `bf_alice`, etc.).

### Phase 3: Kokoro Neural Graph & Vocoder
* Implement `KokoroModel.cs`:
  * PLBERT phoneme encoder ($d=512$, 3 layers).
  * Duration predictor and length regulator.
  * AdaIN residual decoder and iSTFT synthesis.
* Implement `KokoroPipeline.cs`:
  * Orchestrates text-to-phoneme -> model evaluation -> 24kHz WAV export.

### Phase 4: CLI & OpenAI Server Endpoint
* Add `TtsCommand.cs` in `OpenTail.Stingray.Cli` (`stingray tts -p "Hello world" -v af_heart -o speech.wav`).
* Add `OpenAiAudioEndpoints.cs` in `OpenTail.Stingray.Server` for `POST /v1/audio/speech`.

### Phase 5: Automated Testing & Verification
* Create `tests/OpenTail.Stingray.Tests.Audio/` with unit tests for phonemizer, duration regulator, waveform generator, and WAV header integrity.
* Add project to `OpenTail.Stingray.slnx` and verify clean build across all projects.

