# Plan: close the CPU-prefill handoff, one MoE family at a time (open items only)

**Status (2026-10-14):** steps 1-6 and most of the extension wave are **closed** and archived verbatim in
[../done/2026-10-04-moe-handoff-closure-plan-history.md](../done/2026-10-04-moe-handoff-closure-plan-history.md)
(plan, per-family results log, parked review feedback, extension-wave results). Context:
[coverage matrix](2026-10-04-moe-handoff-coverage.md), [batched MoE prefill plan](2026-10-03-batched-moe-prefill-plan.md).

Receipts held (family, path): olmoe / qwen3moe / qwen3 / qwen2moe / llama+experts / phimoe / llama4 on `VulkanHybrid`;
olmoe / qwen2moe / phimoe on `VulkanFullGpu`; glm4moe and afmoe on `VulkanLayerSplit`; **none on `CudaHybrid`** (no NVIDIA GPU here).
Speed league for the MoE lot: ours is 0.21-0.84x of llama.cpp on CPU (PerformanceLeague.md "MoE lot").

## What "closed" means (unchanged)

A family/path is closed when `PrefillHandoffFamilies.Receipts` has a line for it, backed by a passing real-weight run, and its STATUS and
coverage rows are updated. A receipt needs **Stage A** (the CPU path is admitted: right tokens on a real checkpoint against an independent
reference such as `llama-server`, per `docs/reference/numerics-investigation-method.md`) and **Stage B** (the handoff is exact: K/V
byte-identical to the CPU pass, equal logits, 40 decode tokens within the near-tie rule, boundary lengths, `startPos > 0`, at least one mixed
CPU/GPU split). No family-level admission authorizes an untested path. Full wording: archived history, "What closed means".

## Open items

| # | Item | Family | Detail (archived) |
|---|---|---|---|
| 1 | **Stage B not run.** Rerun `LayerSplitCpuPrefillHandoffTests.HunyuanA13B_Handoff_GpuKvEqualsCpuAndDecodeAgrees` alone on an idle machine (`STINGRAY_HYBRID_CPU_PREFILL=all`, `-g 4`, 35 GB checkpoint; the first attempt was stopped for low memory). No `(hunyuan-moe, VulkanLayerSplit)` receipt exists until it passes. | `hunyuan-moe` | Extension wave results, Stage B entry |
| 2 | **CUDA evidence.** No family has a `CudaHybrid` receipt; needs an NVIDIA machine. Also CUDA hybrid has no LongRoPE (phimoe). | all | Review feedback table; steps 2, 3, 4 |
| 3 | **Stage A2 routing check** (selected expert ids, routing weights, top-k, renormalisation, shared-expert contribution vs the reference, first 8 layers / 32 tokens). Parked; PPL stands in. | llama+experts, phimoe, glm4moe, hunyuan-moe, afmoe | Review feedback table |
| 4 | **`-g` above 4 for GLM:** the layer split uploads every expert of each resident layer (128 per layer); needs the expert slot cache. | `glm4moe` | GLM handoff entry |
| 5 | **Sliding window beyond 2048 tokens** not covered; full-GPU parity check, hybrid and CUDA gate for the attention output gate not done. | `afmoe` | afmoe entries |
| 6 | **Vulkan full-GPU receipts missing** (Mixtral-style; llama4 has no pure full-GPU pass at 52 GB). Phi-MoE full-GPU only at the short-factor regime (ctx 1024); long factors (ctx > 4096) not run. | llama+experts, phimoe | steps 2, 3 |
| 7 | **`STINGRAY_MOE_SLOTS` boundary:** only 16 and 4 slots were run; the 2- and 4-slot warm-up cost question (is the warm-up pure overhead with 8 large experts, gate it by expert count?) is unanswered. | llama+experts | step 2 |
| 8 | **License check** for checkpoints that become permanent fixtures (Qwen1.5-MoE-A2.7B is Tongyi Qianwen, not Apache/MIT); confirm against the project's checkpoint-license policy. | qwen2moe | Review feedback table |
| 9 | **Backlog, classify only:** `cohere2moe` (no small checkpoint; needs LayerNorm, parallel residual, sliding-window/NoPE mix in GPU passes; revisit if a small variant ships), `step35`, `exaone-moe` (too large for this machine), `grovemoe` (inspect first: expert layout may not fit the generic slot-cache model). | various | Extension wave backlog |
| 10 | **Legacy per-expert Mixtral layout** (`blk.N.ffn_gate.E.weight`) is not loaded; would be a separate loader task. | `llama` + experts | step 2 results |

**Stay outside this class** (own handoff mechanism needed, none is a gate fix): `deepseek2` (MLA latent cache), recurrent/conv-state families
(`lfm2moe`, `qwen35moe`, `qwen3next`, `granitehybrid`, Nemotron-H), `gpt-oss` (own forward pass).

**Standing rules that apply to any open item:** one multi-GB checkpoint at a time, never alongside another heavy run (contention numbers are not
speed numbers); no GPU conclusion from the iGPU alone (CLAUDE.md rule 13); a green real-weight test must show real-weight timing (rule 12).
