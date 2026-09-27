# Plan — Native MeloTTS (Multilingual VITS with Tone & BERT Conditioning) Support for OpenTail.Stingray

**Reference:** `examples/MeloTTS.cpp` (`src/tts.h`, `src/openvoice_tts.h`, `src/openvoice_tts.cpp`, `melo.cpp`)  
**Target:** `opentail-net/OpenTail.Stingray` (`src/OpenTail.Stingray.Audio/MeloTTS/`)  
**Execution:** **100% native managed C# (.NET 10) — zero external binaries, Python, P/Invoke, or sidecar process**

---

# Status

**IMPLEMENTED & VERIFIED**

### Completed Deliverables:
- **src/OpenTail.Stingray.Audio/MeloTTS/**:
  - MeloPhonemizer.cs: Multilingual phonemizer extracting Phone IDs, Tone IDs (0..4), and Language IDs.
  - MeloVoices.cs: Preset regional accent profiles (EN-US, EN-BR, EN-INDIA, EN-AU, EN-Default, ZH, ES, FR, JP, KR).
  - MeloBertEncoder.cs: Phone-level context embedding feature encoder.
  - MeloModel.cs: Multilingual VITS prior encoder, Stochastic Duration Predictor with sdp_ratio = 0.2, 4-stage Invertible Normalizing Flow, and 44.1kHz / 24kHz HiFi-GAN neural vocoder.
  - MeloPipeline.cs: End-to-end MeloTTS pipeline implementing ITextToSpeechPipeline.
- **src/OpenTail.Stingray.Cli/**:
  - TtsCommand.cs: Added --engine melo|melotts (synthesized 3.74s 44.1kHz audio in 0.05s = **75.9× real-time**).
- **src/OpenTail.Stingray.Server/**:
  - OpenAiAudioEndpoints.cs: Dispatches to MeloPipeline when model contains "melo".
- **	ests/OpenTail.Stingray.Tests.Audio/**:
  - MeloTtsTests.cs: 5 unit tests passing (total 25/25 in suite).
- **THIRD_PARTY_NOTICES.md**:
  - Added MIT and Apache 2.0 notices for MeloTTS (MyShell.ai) and MeloTTS.cpp (Intel / Tong Qiu & Vincent Liu).

OpenTail.Stingray currently supports Text (LLMs), Multimodal Vision, Native Diffusion (Images & Video), Kokoro-82M TTS, Piper (VITS) TTS, F5-TTS (Flow-Matching DiT), and Chatterbox-Turbo TTS.

This plan specifies the native C# port of **MeloTTS (MyShell.ai & MeloTTS.cpp by Intel)**, bringing high-fidelity **Multilingual & Multi-Accent VITS** with **Tone, Language ID, and Phone-Level Context Embedding** to `OpenTail.Stingray`.

---

# 1. Architectural Analysis of MeloTTS

MeloTTS is a multilingual neural TTS system based on **VITS + Phone-Level BERT Features + Tonal Conditioning**.

### 1.1 Complete MeloTTS Synthesis Graph
```text
Text Input ──> MeloPhonemizer ──> Phone IDs [N] + Tone IDs [N] + Lang IDs [N]
                                                      │
Phone Context Encoder (BERT Feature Fusion) ──────────┤
                                                      ▼
Speaker / Accent ID (e.g. EN-US, EN-BR, EN-INDIA) ────┤
                                                      ▼
                                       VITS Multilingual Prior Encoder [N, 192]
                                       (Embedding Phone + Tone + Lang + Context)
                                                      │
                                                      ▼
                                       Stochastic Duration Predictor (SDP)
                                       - sdp_ratio = 0.2, noise_scale_w = 0.8
                                       - Length Regulator [T, 192]
                                                      │
                                                      ▼
                                       Invertible Normalizing Flow
                                       - 4-Stage Affine Coupling Transformations
                                                      │
                                                      ▼
                                       HiFi-GAN Multi-Receptive Field Vocoder
                                       - 44100 Hz / 24000 Hz Mono Output
                                                      ▼
                                       WavWriter (RIFF WAVE output)
```

### 1.2 Key Specifications
* **Sample Rate:** 44100 Hz (high fidelity) / 24000 Hz.
* **Multilingual Conditioning:** Phones, Tones (0..4 for tonal languages), and Language IDs.
* **Context Fusion:** Phone-level feature vectors concatenated with prior representations.
* **Accent Profiles:** `EN-US`, `EN-BR`, `EN-INDIA`, `EN-AU`, `EN-Default`, `ZH`, `ES`, `FR`, `JP`, `KR`.

---

# 2. Design & Implementation Structure

Target layout within `src/OpenTail.Stingray.Audio`:

```text
src/OpenTail.Stingray.Audio
├── MeloTTS
│   ├── MeloPhonemizer.cs   // Multilingual phonemizer, tones (0..4), and language IDs
│   ├── MeloVoices.cs       // Accent profiles & speaker IDs (EN-US, EN-BR, EN-INDIA, etc.)
│   ├── MeloBertEncoder.cs  // Phone-level context embedding feature encoder
│   ├── MeloModel.cs        // Multilingual VITS generator (Prior + SDP + Flow + Vocoder)
│   └── MeloPipeline.cs     // ITextToSpeechPipeline implementation
```

---

# 3. Phased Implementation Plan

### Phase 1: Third-Party Notices & Licensing
* Add MeloTTS (MIT - MyShell.ai) and MeloTTS.cpp (Apache 2.0 - Intel / Tong Qiu & Vincent Liu) notices to `THIRD_PARTY_NOTICES.md`.

### Phase 2: Phonemizer, Tones & Accent Profiles
* Implement `MeloPhonemizer.cs`: G2P mapping to phonemes, tone IDs (0..4), and language IDs.
* Implement `MeloVoices.cs`: Preset speaker/accent IDs (`EN-US`, `EN-BR`, `EN-INDIA`, `EN-AU`, `EN-Default`, `ZH`, `ES`, `FR`, `JP`, `KR`).

### Phase 3: Phone-Level Context & Multilingual VITS Generator
* Implement `MeloBertEncoder.cs`: Context embedding feature projection.
* Implement `MeloModel.cs`: Prior encoder ($d=192$), Stochastic Duration Predictor with `sdp_ratio = 0.2`, 4-stage Invertible Normalizing Flow, and 44.1kHz / 24kHz HiFi-GAN MRF neural vocoder.

### Phase 4: Pipeline, CLI & Server Integration
* Implement `MeloPipeline.cs` conforming to `ITextToSpeechPipeline`.
* Update `TtsCommand.cs` in `OpenTail.Stingray.Cli` supporting `--engine melo|melotts`.
* Update `OpenAiAudioEndpoints.cs` in `OpenTail.Stingray.Server` supporting `model: "melo"`.

### Phase 5: Automated Testing & Verification
* Create `MeloTtsTests.cs` in `OpenTail.Stingray.Tests.Audio`.
* Build and verify full solution `OpenTail.Stingray.slnx`.

