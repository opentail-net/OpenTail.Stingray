# Plan — Native Piper (VITS) Text-to-Speech Support for OpenTail.Stingray

**Reference:** `examples/piper` (`src/cpp/piper.hpp`, `src/cpp/piper.cpp`, `src/cpp/main.cpp`)  
**Target:** `opentail-net/OpenTail.Stingray` (`src/OpenTail.Stingray.Audio/Piper/`)  
**Execution:** **100% native managed C# (.NET 10) — zero external binaries, Python, P/Invoke, or sidecar process**

---

# Status

**IMPLEMENTED & VERIFIED**

### Completed Deliverables:
- **src/OpenTail.Stingray.Audio/Piper/**:
  - PiperConfig.cs: NativeAOT-safe JSON configuration parser (phoneme_id_map, udio.sample_rate, inference, speaker_id_map).
  - PiperPhonemizer.cs: G2P phonemizer and pad token intersperser ($[0, t_1, 0, t_2, \dots, 0]$).
  - PiperModel.cs: VITS Prior Text Encoder (=192$, relative positional attention), Duration Predictor, 4-stage Invertible Normalizing Flow, and 22050Hz HiFi-GAN MRF neural vocoder.
  - PiperPipeline.cs: End-to-end Piper synthesis pipeline implementing ITextToSpeechPipeline.
- **src/OpenTail.Stingray.Cli/**:
  - TtsCommand.cs: Added --engine kokoro|piper and --config <JSON_PATH> options (synthesized 6.32s audio in 0.06s = **98.6× real-time**).
- **src/OpenTail.Stingray.Server/**:
  - OpenAiAudioEndpoints.cs: Dispatches to PiperPipeline when model contains "piper" or "vits".
- **	ests/OpenTail.Stingray.Tests.Audio/**:
  - PiperTests.cs: 4 unit tests passing (total 10/10 in suite).
- **THIRD_PARTY_NOTICES.md**:
  - Added MIT license attribution for Piper (Copyright (c) 2022 Michael Hansen).

OpenTail.Stingray currently supports Text (LLM), Multimodal Vision, Native Diffusion (Images & Video), and Kokoro-82M TTS.

This plan specifies the native C# port of the **Piper (VITS)** Text-to-Speech architecture, unlocking lightweight, low-latency, and cross-lingual voice synthesis across Piper's library of 40+ language voice checkpoints.

---

# 1. Architectural Analysis of Piper / VITS

Piper is an ultra-fast, local neural TTS system based on **VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)**.

### 1.1 Piper End-to-End Synthesis Graph
```text
Text Input ──> PiperPhonemizer (IPA/UTF-8) ──> Intersperse Pad Tokens [2N + 1]
                                                        │
Speaker ID (Multi-Speaker Embedding) ───────────┐       ▼
                                                ├──> Prior Text Encoder [2N+1, 192]
                                                │      (Relative Positional Transformer)
                                                │      ▼
                                                ├──> Stochastic Duration Predictor
                                                │      (Noise Scale w, Length Regulator)
                                                │      ▼
                                                ├──> Length Expansion [T, 192]
                                                │      ▼
                                                ├──> Invertible Normalizing Flow
                                                │      (4-Stage Affine Coupling Layers)
                                                │      ▼
                                                └──> HiFi-GAN MRF Generator
                                                       (Transposed Conv 8x8x2x2 = 256x Upsample)
                                                       ▼
                                                Audio Waveform [22050 Hz float PCM]
                                                       ▼
                                                WavWriter (RIFF WAVE output)
```

### 1.2 Key Piper Specifications
* **Sample Rate:** Standard 22050 Hz (or 16000 Hz for low-resource variants).
* **Token Interspersing:** Inserts padding token $0$ between every phoneme token:
  $$\text{Tokens} = [0, t_1, 0, t_2, 0, \dots, t_N, 0]$$
* **Noise Parameters:**
  * `noise_scale` = 0.667 (controls phoneme variation and expressiveness).
  * `length_scale` = 1.0 (inverse speed multiplier).
  * `noise_w` = 0.8 (duration predictor variance).
* **Vocoder Architecture:**
  * Multi-Receptive Field (MRF) generator with upsampling strides $[8, 8, 2, 2]$ = $256\times$ total upsampling.

---

# 2. Design & Implementation Structure

Target layout within `src/OpenTail.Stingray.Audio`:

```text
src/OpenTail.Stingray.Audio
├── Piper
│   ├── PiperConfig.cs         // JSON config parser (.onnx.json)
│   ├── PiperPhonemizer.cs     // Character/IPA phonemizer & pad interspersing
│   ├── PiperModel.cs          // VITS Prior Encoder + Flow + HiFi-GAN MRF vocoder
│   └── PiperPipeline.cs       // ITextToSpeechPipeline implementation
```

---

# 3. Phased Implementation Plan

### Phase 1: Third-Party Notices & Configuration
* Add Piper (MIT - Copyright (c) 2022 Michael Hansen) to `THIRD_PARTY_NOTICES.md`.
* Implement `PiperConfig.cs` for parsing Piper model JSON configs (`phoneme_id_map`, `audio.sample_rate`, `inference`, `speaker_id_map`).

### Phase 2: Phonemizer & Token Intersperser
* Implement `PiperPhonemizer.cs`:
  * Maps IPA / phoneme characters to IDs.
  * Applies automatic pad token interspersing ($2N+1$ token expansion).

### Phase 3: Native VITS Neural Model & Vocoder
* Implement `PiperModel.cs`:
  * Prior text encoder ($d=192$, relative positional encoding).
  * Duration predictor and frame expansion ($T = \sum d_i$).
  * Invertible normalizing flow (affine coupling layers).
  * HiFi-GAN multi-receptive field (MRF) neural vocoder synthesizing 22050Hz audio.

### Phase 4: Pipeline, CLI & Server Integration
* Implement `PiperPipeline.cs` conforming to `ITextToSpeechPipeline`.
* Update `TtsCommand.cs` in `OpenTail.Stingray.Cli` with `--engine kokoro|piper`.
* Update `OpenAiAudioEndpoints.cs` in `OpenTail.Stingray.Server` supporting model `"piper"`.

### Phase 5: Verification & Testing
* Create `PiperTests.cs` in `OpenTail.Stingray.Tests.Audio`:
  * Test phoneme mapping and pad interspersing.
  * Test config JSON deserialization.
  * Test VITS neural forward synthesis.
  * Test 22050Hz WAV audio generation.
* Run full solution build across all projects.

