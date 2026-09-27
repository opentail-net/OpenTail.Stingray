# Implementation Plan — Native Whisper Speech-to-Text (ASR) & Translation in OpenTail.Stingray

**Execution:** **100% native managed C# (.NET 10) — zero external binaries or Python dependencies**

---

## User Review Required

> [!NOTE]
> OpenAI Whisper uses an **Encoder-Decoder Transformer Architecture** with 80/128-channel 16kHz log-Mel spectrogram feature extraction, $4\times$ convolutional sequence downsampling, and autoregressive decoding for **Speech-to-Text (ASR)**, **Speech Translation**, and **Timestamp-Accurate Transcription** across 99+ languages. All modules will be written in 100% native managed C# without GGML C++ bindings or sidecar binaries.

---

## Proposed Changes

### 1. Attributions & Licensing
#### [MODIFY] [`THIRD_PARTY_NOTICES.md`](file:///C:/Git-Public/OpenTail.Stingray/THIRD_PARTY_NOTICES.md)
* Add attribution and MIT license notice for Whisper / whisper.cpp (Copyright (c) 2023-2026 The ggml authors / OpenAI).

---

### 2. Audio Library (`OpenTail.Stingray.Audio/Whisper`)
#### [NEW] [`src/OpenTail.Stingray.Audio/ISpeechToTextPipeline.cs`](file:///C:/Git-Public/OpenTail.Stingray/src/OpenTail.Stingray.Audio/ISpeechToTextPipeline.cs)
* Standardized `ISpeechToTextPipeline` interface, `TranscriptionRequest`, and `TranscriptionResult` (with segment-level timestamps).

#### [NEW] [`src/OpenTail.Stingray.Audio/Whisper/WhisperMelExtractor.cs`](file:///C:/Git-Public/OpenTail.Stingray/src/OpenTail.Stingray.Audio/Whisper/WhisperMelExtractor.cs)
* 80/128-channel 16kHz Log-Mel spectrogram extractor ($n_{\text{fft}}=400$, hop length 160).

#### [NEW] [`src/OpenTail.Stingray.Audio/Whisper/WhisperTokenizer.cs`](file:///C:/Git-Public/OpenTail.Stingray/src/OpenTail.Stingray.Audio/Whisper/WhisperTokenizer.cs)
* Multilingual BPE tokenizer with language tags (`<|en|>`, `<|zh|>`), task tags (`<|transcribe|>`, `<|translate|>`), and timestamp tokens.

#### [NEW] [`src/OpenTail.Stingray.Audio/Whisper/WhisperEncoder.cs`](file:///C:/Git-Public/OpenTail.Stingray/src/OpenTail.Stingray.Audio/Whisper/WhisperEncoder.cs)
* 2× Conv1D downsamplers (stride 2) + Sinusoidal Positional Embeddings + Audio Transformer Encoder.

#### [NEW] [`src/OpenTail.Stingray.Audio/Whisper/WhisperDecoder.cs`](file:///C:/Git-Public/OpenTail.Stingray/src/OpenTail.Stingray.Audio/Whisper/WhisperDecoder.cs)
* Autoregressive Transformer Decoder with causal self-attention + cross-attention over audio features.

#### [NEW] [`src/OpenTail.Stingray.Audio/Whisper/WhisperPipeline.cs`](file:///C:/Git-Public/OpenTail.Stingray/src/OpenTail.Stingray.Audio/Whisper/WhisperPipeline.cs)
* Implements `ISpeechToTextPipeline` for end-to-end transcription and translation.

---

### 3. CLI & Server Integration
#### [NEW] [`src/OpenTail.Stingray.Cli/SttCommand.cs`](file:///C:/Git-Public/OpenTail.Stingray/src/OpenTail.Stingray.Cli/SttCommand.cs)
* Add `stingray stt -i audio.wav --language en --task transcribe` command.

#### [MODIFY] [`src/OpenTail.Stingray.Server/Endpoints/OpenAiAudioEndpoints.cs`](file:///C:/Git-Public/OpenTail.Stingray/src/OpenTail.Stingray.Server/Endpoints/OpenAiAudioEndpoints.cs)
* Implement `POST /v1/audio/transcriptions` and `POST /v1/audio/translations`.

---

### 4. Verification & Testing
#### [NEW] [`tests/OpenTail.Stingray.Tests.Audio/WhisperTests.cs`](file:///C:/Git-Public/OpenTail.Stingray/tests/OpenTail.Stingray.Tests.Audio/WhisperTests.cs)
* Unit tests for Whisper Mel extraction, BPE tokenization, Audio Encoder, Decoder cross-attention, and end-to-end transcription.

---

## Verification Plan

### Automated Tests
```powershell
dotnet test "C:\Git-Public\OpenTail.Stingray\tests\OpenTail.Stingray.Tests.Audio\OpenTail.Stingray.Tests.Audio.csproj"
dotnet build "C:\Git-Public\OpenTail.Stingray\OpenTail.Stingray.slnx"
```

### CLI Verification
```powershell
stingray stt -i sample.wav --language en --task transcribe
```
