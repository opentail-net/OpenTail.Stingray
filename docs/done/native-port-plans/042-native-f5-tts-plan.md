# Plan — Native F5-TTS (Flow-Matching DiT with Voice Cloning) Support for OpenTail.Stingray

**Reference:** `examples/CrispASR` (`src/f5_tts.h`, `src/f5_tts.cpp`, `models/convert-f5-tts-to-gguf.py`)  
**Target:** `opentail-net/OpenTail.Stingray` (`src/OpenTail.Stingray.Audio/F5TTS/`)  
**Execution:** **100% native managed C# (.NET 10) — zero external binaries, Python, P/Invoke, or sidecar process**

---

# Status

**IMPLEMENTED & VERIFIED**

### Completed Deliverables:
- **src/OpenTail.Stingray.Audio/F5TTS/**:
  - F5MelExtractor.cs: 100-channel 24kHz Mel-Spectrogram extractor with Hann windowing, STFT, and log-compression.
  - F5TextEncoder.cs: Character tokenizer (2546 vocab) + 4-stage ConvNeXtV2 text feature encoder.
  - F5DiTModel.cs: 22-layer AdaLN-Zero DiT backbone with 1D RoPE, input concatenation (+100+512=712$), and Euler flow-matching solver (32 ODE steps, sway sampling, CFG=2.0).
  - F5VocosVocoder.cs: 8-stage ConvNeXt neural vocoder with magnitude/phase linear head and 24kHz iSTFT synthesis.
  - F5TtsPipeline.cs: End-to-end Flow-Matching DiT pipeline implementing ITextToSpeechPipeline with Zero-Shot Voice Cloning.
- **src/OpenTail.Stingray.Cli/**:
  - TtsCommand.cs: Added --engine f5tts, --ref-audio <PATH>, and --ref-text <TEXT> voice cloning options.
- **src/OpenTail.Stingray.Server/**:
  - OpenAiAudioEndpoints.cs: Dispatches to F5TtsPipeline when model contains "f5".
- **	ests/OpenTail.Stingray.Tests.Audio/**:
  - F5TtsTests.cs: 5 unit tests passing (total 15/15 in suite).
- **THIRD_PARTY_NOTICES.md**:
  - Added MIT license notice for SWivid/F5-TTS and CrispASR.

OpenTail.Stingray currently supports Text (LLMs), Multimodal Vision, Native Diffusion (Images & Video), Kokoro-82M TTS, and Piper (VITS) TTS.

This plan specifies the native C# port of **F5-TTS (SWivid/F5-TTS)**, bringing state-of-the-art **Flow-Matching Diffusion Transformer (DiT)** speech synthesis and **Zero-Shot Voice Cloning** with reference audio conditioning to `OpenTail.Stingray`.

---

# 1. Architectural Analysis of F5-TTS

F5-TTS is a non-autoregressive speech synthesis system based on **Flow-Matching Diffusion Transformers (DiT)** and the **Vocos** neural vocoder.

### 1.1 Complete F5-TTS End-to-End Synthesis Graph
```text
Text Input + Ref Text ──> ConvNeXtV2 Text Encoder ──> Text Features [N, 512]
                                                              │
Ref Audio (24kHz WAV) ──> Mel Spectrogram Extractor ──> Cond Mel [T_ref, 100]
                                                              │
Noise Prior z_0 ~ N(0, I) [T, 100] ───────────────────────────┤
                                                              ▼
                                             Concat Inputs [T, 100+100+512 = 712]
                                                              ▼
                                                   Input Embedding [T, 1024]
                                                              +
                                                   ConvPosEmbedding (2× Conv1d k=31)
                                                              │
Timestep t ∈ [0, 1] ──> Sinusoidal Timestep Embed [1024] ─────┤
                                                              ▼
                                                   22-Layer DiT Backbone (d=1024)
                                                   - AdaLN-Zero (6-way scale/shift/gate)
                                                   - Bidirectional Attention with 1D RoPE
                                                   - Modulated LayerNorm + GeLU FFN (2048)
                                                              ▼
                                                   AdaLN-Final + Linear(1024, 100)
                                                              ▼
                                                   Predicted Velocity v_t [T, 100]
                                                              │
Euler Flow-Matching ODE Solver (32 steps, CFG=2.0) ───────────┘
                                                              ▼
                                                   Denoised Mel Latents [T, 100]
                                                              ▼
                                                   Vocos Neural Vocoder
                                                   - 8 ConvNeXt blocks (512-dim)
                                                   - Mag + Phase Linear Head (1026)
                                                   - 24kHz iSTFT Synthesis
                                                              ▼
                                                   24000 Hz Mono Audio Waveform
                                                              ▼
                                                   WavWriter (RIFF WAVE output)
```

### 1.2 Key Architectural Specifications
* **Sample Rate:** 24000 Hz mono PCM.
* **Mel-Spectrogram Channels:** 100 mel channels (n_fft=1024, hop_length=256, win_length=1024).
* **Text Encoder:** Character embedding ($2546 \times 512$) + 4 ConvNeXtV2 blocks.
* **DiT Backbone:** 22 Transformer layers, hidden dimension $d=1024$, $16$ attention heads ($d_{\text{head}}=64$), with AdaLN-Zero modulation.
* **ODE Sampler:** Euler flow-matching solver with Sway sampling ($\text{sway\_coef} = -1.0$) and Classifier-Free Guidance ($\text{CFG} = 2.0$).
* **Vocoder:** Vocos neural vocoder converting 100-channel mel latents directly to 24kHz audio via inverse STFT.

---

# 2. Design & Implementation Structure

Target layout within `src/OpenTail.Stingray.Audio`:

```text
src/OpenTail.Stingray.Audio
├── F5TTS
│   ├── F5MelExtractor.cs      // 100-channel 24kHz Mel-Spectrogram feature extractor
│   ├── F5TextEncoder.cs       // Character tokenizer & 4-stage ConvNeXtV2 text encoder
│   ├── F5DiTModel.cs          // 22-layer AdaLN-Zero DiT with 1D RoPE
│   ├── F5VocosVocoder.cs      // 8-stage ConvNeXt + Mag/Phase iSTFT vocoder
│   └── F5TtsPipeline.cs       // End-to-end flow-matching pipeline with voice cloning
```

---

# 3. Phased Implementation Plan

### Phase 1: Third-Party Notices & Licensing
* Add SWivid/F5-TTS (MIT License) notice to `THIRD_PARTY_NOTICES.md`.

### Phase 2: Mel-Spectrogram Extractor & Text Encoder
* Implement `F5MelExtractor.cs`:
  * 100-channel Mel filterbank with Hann windowing, STFT, and log-compression.
* Implement `F5TextEncoder.cs`:
  * Character vocabulary tokenization (2546 symbols).
  * 4-layer ConvNeXtV2 feature projection.

### Phase 3: DiT Backbone & Flow-Matching ODE Solver
* Implement `F5DiTModel.cs`:
  * Timestep embedding with sinusoidal frequencies.
  * AdaLN-Zero modulation projections.
  * 22 DiT blocks with 1D RoPE self-attention and gated MLP.
  * Euler Flow-Matching trajectory with sway sampling and CFG.

### Phase 4: Vocos Neural Vocoder
* Implement `F5VocosVocoder.cs`:
  * 8 ConvNeXt residual blocks with LayerNorm.
  * Magnitude and phase projection ($512 \rightarrow 1026$).
  * iSTFT synthesis generating 24kHz audio samples.

### Phase 5: Pipeline Orchestration, CLI & Server Integration
* Implement `F5TtsPipeline.cs` conforming to `ITextToSpeechPipeline`.
  * Supports Zero-Shot Voice Cloning via `--ref-audio <PATH>` and `--ref-text <TEXT>`.
* Update `TtsCommand.cs` in `OpenTail.Stingray.Cli` with `--engine f5tts`, `--ref-audio`, `--ref-text`.
* Update `OpenAiAudioEndpoints.cs` in `OpenTail.Stingray.Server` supporting `model: "f5-tts"`.

### Phase 6: Automated Testing & Verification
* Create `F5TtsTests.cs` in `OpenTail.Stingray.Tests.Audio`:
  * Test Mel-spectrogram extraction.
  * Test ConvNeXt text encoder.
  * Test DiT velocity prediction and Euler ODE flow matching.
  * Test Vocos vocoder synthesis.
  * Test voice cloning pipeline end-to-end.
* Build and verify full solution `OpenTail.Stingray.slnx`.

