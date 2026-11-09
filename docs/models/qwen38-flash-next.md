# Qwen 3.8 Flash Next (`qwen4exp`)

[← Back to Architecture Cards](README.md)

| Property | Value |
|---|---|
| **Provider** | Alibaba Cloud (Qwen Team) |
| **GGUF Architecture Key** | `qwen4exp` |
| **Engine Implementation** | `Qwen4ExpArchitecture` (descriptor), `Qwen4ExpForwardPass` (CPU) |
| **Forward Pass Family** | `ForwardPassFamily.HybridGdn` |
| **Status** | Admitted 2026-11-09 for text generation and tool calls, CPU only. See [STATUS](../STATUS.md). |
| **Checkpoint used** | `unsloth/Qwen3.8-Flash-Next-GGUF`, UD-IQ1_S, three shards, 72.5 GB, memory-mapped |
| **Licence** | Qwen Community License 1.0 |

## Architecture as implemented

- Hybrid Gated DeltaNet layers and QSA sparse attention layers. The attention indexer uses a K-pool of 4 and selects the top 2048 keys.
- A four-stream hyper-connection residual: the hidden state is the hyper-connection count times the embedding width.
- A per-layer token embedding table (`per_layer_token_embd.weight`, about 28.8 GB, in shard 3). Rows are gathered from the memory map as needed.
- IMRoPE rotary embeddings.
- Mixture of experts with 512 experts.
- A shared multi-token-prediction (MTP) head in its own GGUF (`mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`, 1.9 GB). The MTP decoder path is measured. It is not admitted as a product path.

## Scope

Admitted: text generation, verified greedy against llama.cpp, and tool calls through the server round trip.

Not admitted: image input, video input, MTP speculation as a product path, and the per-step MTP oracle. The reasons and the open phases are in the plan's D4 entry.

## Memory and running

- The preflight estimate is about 66.0 GiB for a CPU run at the default context. The reference machine has 63.3 GiB. The CLI and the server refuse the checkpoint on that host unless the override is set (`--ignore-preflight` for the CLI, `STINGRAY_IGNORE_PREFLIGHT=1` for the server).
- Whether this override should be documented for users is an open owner decision (plan, Phase 8). Until it is decided, this checkpoint has no RUNNING.md row and no WHAT-YOU-CAN-DO line.

## Speed

Single samples on the reference machine (63.3 GiB RAM, paged checkpoint):

- 2026-11-08, `MeasureTextThroughput_OnTheRealCheckpoint`: prefill 1.30 tok/s over 63 prompt tokens, decode 1.69 tok/s over 12 steps.
- 2026-11-09, the same test in the Core full gate: prefill 2.22 tok/s, decode 2.51 tok/s.

These are two samples. The performance pass has not been done.

## Evidence

Every number and result on this page is recorded in [the Qwen 3.8 plan](../2-coverage/2026-11-08-qwen38-remaining-work-plan.md). The sourced status row is in [STATUS](../STATUS.md).
