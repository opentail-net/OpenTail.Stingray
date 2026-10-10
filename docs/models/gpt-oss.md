# gpt-oss (OpenAI open-weight reasoning MoE)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | OpenAI |
| **GGUF Architecture Key** | `gpt-oss` |
| **Engine Implementation** | `GptFamilyArchitectures.GptOss` → `GptOssForwardPass` (CPU) / `GptOssGpuForwardPass` (Vulkan) |
| **Forward Pass Family** | `ForwardPassFamily.GptOss` |
| **Modalities** | Text (reasoning model) |
| **Weights format** | MXFP4 (native microscaling 4-bit floating point) |
| **Continuous batching** | Not supported |
| **Status & Confidence** | 🟢 **Admitted 2026-09-26** on gpt-oss-20b MXFP4 (see the [STATUS row](../STATUS.md)) |

Related: [Qwen series](qwen.md) (other large MoE on the same CPU path), [OLMoE](olmoe.md).

---

## 1. Overview & Architectural Highlights

gpt-oss is OpenAI's open-weight reasoning family. Stingray runs it through its own forward pass rather than the generic dense one, because it differs from the Llama-style block in several places.

### Key Architectural Characteristics (as implemented in `GptOssForwardPass`)
* **Attention sinks:** a learned per-head sink term takes part in the softmax.
* **Alternating sliding-window attention (1:1 SWA):** sliding-window and full-attention layers alternate.
* **Biased MoE:** the routed experts and router carry bias terms.
* **OpenAI-style SwiGLU:** the gated activation uses the model's own clamped variant, not the plain SiLU gate.
* **YaRN RoPE scaling** for long context.
* **MXFP4 weights:** executed natively on CPU, no up-conversion of the whole file.

### Verification
* Teacher-forced against `llama-server`: **22/24 exact argmax**, worst gap **0.063 logits**, which is inside llama.cpp's own `-fa on/off` noise (`GptOssRealWeightSmokeTests`).
* Routed by both the CLI and the server.

---

## 2. Checkpoints & Recommended GGUFs

| Model | Hugging Face Repository | File | Size | Notes |
|---|---|---|---|---|
| **gpt-oss-20b** | [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) | `gpt-oss-20b-MXFP4.gguf` | 12.1 GB | The file Stingray was verified on. Apache-2.0. |

The same repo also carries `eagle3-gpt-oss-20b-*` speculative-decoding drafts; Stingray does not use them.

Only the 20B MXFP4 file has been checked. No gpt-oss-120b receipt exists, so it is not claimed here.

---

## 3. Hardware Requirements & Performance Profile

Measured in [docs/RUNNING.md](../RUNNING.md) (reference AMD Ryzen 7 5700G, CPU):

| Model | Peak RAM | Prompt | Decode | Date |
|---|---|---|---|---|
| gpt-oss-20b MXFP4 | about 13 GB | 2.1 tok/s | 5.5 tok/s | 2026-09-28 |

The STATUS row quotes about 11 tok/s CPU decode, which does not match RUNNING.md's 5.5 tok/s. The two rows are not reconciled in the docs; RUNNING.md has the reproducible command.

**GPU:** the architecture descriptor lists Vulkan among its supported backends and a `GptOssGpuForwardPass` exists, but the STATUS row still says "No GPU path yet" and no GPU receipt is recorded. Treat Vulkan as unverified for this family until the matrix says otherwise.

---

## 4. Usage

```bash
stingray -m models/_models/gpt-oss-20b-MXFP4.gguf -p "Hello world"
```
