# Chatterbox-Turbo TTS in OpenTail.Stingray: Architecture, Optimizations & Hybrid Execution

## 1. Overview

**Chatterbox-Turbo** is a state-of-the-art fast text-to-speech architecture designed for low-latency, high-fidelity conversational voice synthesis.

In OpenTail.Stingray, Chatterbox-Turbo is implemented with a dual-execution strategy:
1. **Pure C# AVX2 Engine (Default / GGUF)**: A 100% dependency-free, memory-efficient SIMD engine that loads model weights from GGUF containers and executes entirely in managed .NET 10 code.
2. **Native C++ Accelerator (Optional / ONNX)**: An auto-detecting acceleration path that executes the fused S3Gen neural decoder via native ONNX Runtime when `conditional_decoder.onnx` is present.

---

## 2. End-to-End Pipeline Architecture

```
                       [ Input Text ]
                              │
                              ▼
                    ┌───────────────────┐
                    │ BPE Tokenizer     │
                    └─────────┬─────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│ Stage 1: T3 Acoustic LM (Autoregressive Token Generation)    │
│ • Architecture: GPT-2 style 24-layer Transformer (Dim: 1024)  │
│ • Container: chatterbox-turbo-t3-q4_k.gguf (or F16)           │
│ • Outputs: ~55–60 discrete acoustic speech tokens             │
│ • Optimizations: 16-head parallel AVX2 context reduction,     │
│   zero-alloc stack attention scores, flat KV cache streaming. │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼ [ Speech Tokens ]
┌───────────────────────────────────────────────────────────────┐
│ Stage 2: S3Gen Conditional Neural Decoder                     │
│                                                               │
│  Option A: Pure C# AVX2 Engine (GGUF Mode)                    │
│  ├─ S3Gen Flow Encoder (Tokens -> Mel Conditioning)           │
│  ├─ S3Gen CFM Estimator (112-block UNet, 2-step Euler solve)  │
│  └─ S3Gen HiFT Vocoder (Mel Spectrogram -> 24kHz PCM Waveform)│
│                                                               │
│  Option B: Native Fused Graph Accelerator (ONNX Mode)         │
│  └─ Single-shot native C++ execution (conditional_decoder.onnx│
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
                    [ 24 kHz Mono Audio PCM ]
```

---

## 3. Benchmark History & Performance Progression

All benchmarks measured on identical hardware generating an English utterance (prompt: *"Hello, I will make some lunch darling!."*, ~2.4s audio duration):

| Engine Stage | Baseline (Pure C#) | Optimized Pure C# | Standalone C++ Reference | **Stingray (Accelerated)** |
| :--- | :--- | :--- | :--- | :--- |
| **T3 Acoustic LM** | 6,329 ms | 5,694 ms | 3,895 ms | **4,944 ms** |
| **S3Gen Flow Encoder** | 2,028 ms | 567 ms | ~400 ms | *(fused in native call)* |
| **S3Gen CFM Estimator** | 16,259 ms | 4,665 ms | ~1,800 ms | *(fused in native call)* |
| **S3Gen HiFT Vocoder** | 1,001 ms | 898 ms | ~675 ms | *(fused in native call)* |
| **Total S3Gen Decoder** | **19,288 ms (19.3s)** | **6,136 ms (6.1s)** | **2,875 ms (2.9s)** | **1,636 ms (1.64s)** |
| **Total End-to-End Time** | **25.65 s (10.9× RTF)**| **11.86 s (4.7× RTF)** | **6.77 s (2.8× RTF)** | **6.62 s (2.96× RTF)** |

---

## 4. Key Engineering Optimizations in Pure C#

1. **Multi-Head Parallel Attention with AVX2 Context Vectorization**:
   - In `ChatterboxAcousticLm.cs`, all 16 attention heads are dispatched across worker threads.
   - Vectorized context accumulation utilizes 8 unrolled AVX2 FMA registers (`Fma.MultiplyAdd`), eliminating 9.3 million scalar loop iterations per utterance.
2. **Zero-Allocation ResNet & Scratch Buffers**:
   - In `CfmUNetKernels.cs`, pre-rented scratch buffers in `UnetScratchBuffers` eliminate over 114,000 transient heap allocations during CFM Euler ODE integration.
3. **Dual-Pass Concurrent CFM Integration**:
   - `Parallel.Invoke` computes the conditional ($dxdt_{\text{cond}}$) and unconditional ($dxdt_{\text{uncond}}$) velocity fields simultaneously on separate core clusters, maximizing per-core L1/L2 cache locality.
4. **Pristine Float32 Fidelity**:
   - Maintains full 32-bit floating point precision across all activations and layers, completely avoiding FP16 truncation artifacts and pitch modulation wobble.

---

## 5. CLI Usage & Model Formats

### Running in Pure C# GGUF Mode:
```bash
dotnet run -c Release --project src/OpenTail.Stingray.Cli/OpenTail.Stingray.Cli.csproj -- \
  tts -e chatterbox \
  -m models/chatterbox-turbo-t3-q4_k.gguf \
  -t "Hello, I will make some lunch darling!." \
  -o output.wav
```

### Running with Native ONNX Decoder Acceleration:
When `conditional_decoder.onnx` is located alongside the model or in the working directory, Stingray automatically detects it and activates the single-shot native accelerator (**6.62s total latency**). If the file is omitted, it seamlessly falls back to pure C# (**11.86s total latency**).
