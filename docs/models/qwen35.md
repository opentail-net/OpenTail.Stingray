# Qwen3.5 / 3.6 / 3.8 Hybrid Gated DeltaNet (+ MoE) and Ornith

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | Alibaba Cloud (Qwen Team); Ornith 1.0 is a third-party fine-tune on the same architecture |
| **GGUF Architecture Keys** | `qwen35` (dense hybrid), `qwen35moe` (hybrid + MoE) |
| **Engine Implementation** | `QwenArchitectures.Qwen35` / `Qwen35moe` |
| **Forward Pass Family** | `ForwardPassFamily.HybridGdn` via `CommonForwardPassFactory.CreateHybridGdn` (CPU; CUDA offload through `CudaHybridGdnForwardPass`) |
| **Thinking Mode** | Reasoning models: output starts with a thinking trace |
| **Speculation** | MTP self-speculation (`--spec-type mtp`) |
| **Status & Confidence** | 🟢 **Admitted** (2026-09-28): Ornith-1.0 9B and Qwen3.8-27B match llama.cpp exactly under greedy decoding (see the [STATUS row](../STATUS.md)) |

Related: [Qwen series](qwen.md) (dense Qwen2.5/3 and Qwen3-Coder), [Qwen-VL](qwen-vl.md), [Granite](granite.md) and [LFM2](lfm2.md) (other hybrid recurrent families).

---

## 1. Overview & Architectural Highlights

These models replace most full-attention layers with **Gated DeltaNet**, a linear-time recurrent memory, and keep a full-attention layer at a fixed interval. The `qwen35moe` variant adds a routed mixture-of-experts feed-forward.

* **Gated DeltaNet layers:** recurrent state instead of a growing KV cache on those layers. The chunked recurrence is pinned by `GdnKernelsTests`.
* **Periodic full attention:** `full_attention_interval = 4`, so every fourth block is ordinary attention.
* **MoE variants:** `Qwen3.6-35B-A3B` runs with Q8_K routed-expert kernels.
* **MTP layer:** the checkpoint's multi-token-prediction layer is loaded and used for `--spec-type mtp` self-speculation (draft depth 1).

Prefill speed against llama.cpp is recorded in STATUS as about 0.96x for Qwen3.6-35B-A3B after the GDN/MoE parallelisation, with a Phase 8 note of 0.63x on the Q6_K file. Source plan: [02-qwen35moe-plan](../done/02-qwen35moe-plan.md).

---

## 2. Checkpoints

| Model | Hugging Face Repository | File used in docs | Notes |
|---|---|---|---|
| **Qwen3.8-27B** | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | `Qwen3.8-27B-UD-Q3_K_XL.gguf` | Exact greedy match vs llama.cpp. The repo currently lists Q4_0, Q8_0 and UD-Q4_K_M; the UD-Q3_K_XL file used for the receipt was not in the list I checked, so confirm the file name before downloading. |
| **Qwen3.6-35B-A3B** | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | `Qwen3.6-35B-A3B-UD-Q6_K.gguf` | MoE. Same caveat: the repo listing I checked showed Q8_0 and UD-Q4_K_M, so confirm the Q6_K file name. |
| **Ornith 1.0 9B** | [ornith-ai/Ornith-1.0-9B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.0-9B-GGUF) | `deepreinforce-ai_Ornith-1.0-9B-Q4_K_M.gguf` | Exact greedy match vs llama.cpp. The repo has `ornith-1.0-9b-Q4_K_M.gguf`; the file name in RUNNING.md carries a different prefix, so it likely came from another uploader. |

---

## 3. Hardware Requirements & Performance Profile

Measured in [docs/RUNNING.md](../RUNNING.md) (reference AMD Ryzen 7 5700G, CPU):

| Model | Peak RAM | Decode | Notes |
|---|---|---|---|
| Ornith 1.0 9B Q4_K_M | about 5.9 GB | 4.8-6.3 tok/s over 128 tokens (2026-10-01) | First tokens are a thinking trace: raise `-n` to see the answer. |
| Qwen3.8-27B UD-Q3_K_XL | about 13 GB | 1.6-2.1 tok/s (2026-10-01); 2.3-2.4 re-measured 2026-10-08 | Prefill 8.8 tok/s on a 620-token prompt. Load takes about 25 s. With `--spec-type mtp --no-thinking` (greedy): 2.7 tok/s (+12%). |
| Qwen3.6-35B-A3B UD-Q6_K | about 30 GB | 2.9 tok/s (prompt 6.5 tok/s, 2026-09-28) | Hybrid GDN + MoE with Q8_K routed kernels. |

---

## 4. Usage

```bash
# Plain greedy run (reasoning model: allow room for the thinking trace)
stingray -m models/_models/Qwen3.8-27B-UD-Q3_K_XL.gguf \
         -p "Write a short paragraph about the ocean." -n 128 --temp 0

# MTP self-speculation, greedy
stingray -m models/_models/Qwen3.8-27B-UD-Q3_K_XL.gguf \
         -p "Write a short paragraph about the ocean." -n 128 --temp 0 \
         --spec-type mtp --no-thinking
```
