# Completed items split from 2026-10-04-moe-handoff-closure-plan.md (2026-10-30)

Finished items moved verbatim out of [../2-coverage/2026-10-04-moe-handoff-closure-plan.md](../2-coverage/2026-10-04-moe-handoff-closure-plan.md). Open items stay there. Each block notes the section it came from.

> from: ## Tracking
- [x] Step 0 shared test plumbing (receipts keyed by (family, path); fingerprints; Theory over model files)

> from: ## Tracking
- [x] 1. `qwen2moe` (2026-10-04): Stage A (found and fixed 3 bugs, below), Stage B on the Vulkan hybrid at 1 and 4 GPU layers, receipt `(qwen2moe, VulkanHybrid)`. Vulkan full-GPU (2026-10-04, later): `GpuForwardPass` now applies the shared-expert sigmoid gate and uses own shared-expert scratch; `GpuCpuPrefillHandoffTests` on Qwen1.5-MoE passes (prefill cosine 0.9995, decode 0.998-1.0), receipt `(qwen2moe, VulkanFullGpu)`. Not done for this family: CUDA, router-level check (Stage A2; PPL 4.7838 against 4.8144 ± 0.78 stands in for it). Weights kept at `H:\_models` (F: is full; `models/_models` points at F:) until Mixtral needs the room.

> from: ## Tracking
- [x] 2. `llama+experts` (Mixtral, 2026-10-04): Stage A, Stage B on the Vulkan hybrid at 1 and 4 GPU layers, receipt `(llama+experts, VulkanHybrid)`. Not done: Vulkan full-GPU, CUDA, Stage A2 routing check (PPL stands in), the 2- and 4-slot `STINGRAY_MOE_SLOTS` boundary and warm-up cost question (only 16 and 4 slots were run). Weights kept in `H:\_models`.

> from: ## Tracking
- [x] 3. `phimoe` (2026-10-04): LongRoPE, norm biases and LM-head bias in the Vulkan hybrid, Stage B at both RoPE regimes, receipt `(phimoe, VulkanHybrid)`. Vulkan full-GPU (2026-10-04, later): `GpuForwardPass` now adds the RMSNorm biases and `output.bias`; `GpuCpuPrefillHandoffTests` on Phi-3.5-MoE (ctx 1024, short factors only) passes (prefill cosine 0.99995, decode >= 0.9987), receipt `(phimoe, VulkanFullGpu)`. Not done: full-GPU at the long-factor regime (ctx > 4096), CUDA hybrid (no LongRoPE), Stage A2.

> from: ## Tracking
- [x] 4. `glm4moe` (2026-10-04, partial by design): Stage A done and the CPU path admitted; the first "no GPU path to hand to" gap was closed later the same day: Vulkan layer split `-g 4` plus a layer-split CPU-prefill handoff, receipt `(glm4moe, VulkanLayerSplit)` (see the last GLM results entry). Disk: GLM-4.5-Air-Q2_K (45.3 GB) is on `H:\_models`, nothing was deleted (H: had 235 GB free).

> from: ## Tracking
- [x] 5. `llama4` (2026-10-04, CPU Stage A only): reading done, Scout Q3_K_M in `H:/_models/Q3_K_M/`, Stage A passes (below). No handoff receipt: the Vulkan paths were not exercised, so the handoff stays Unverified for this family.

> from: ## Tracking
- [x] 6. **Speed league for the MoE lot** (done 2026-10-04: rows in PerformanceLeague.md "MoE lot"; ours is 0.21-0.84x of llama.cpp on CPU on all five, llama.cpp has no GPU backend here; found and fixed the GLM `-g -1` crash by capping auto offload to a 4-layer split) (added 2026-10-04 at the user's request): measure, then record dated rows in [PerformanceLeague.md](../../PerformanceLeague.md) (its existing table format and methodology: `tools/llama.cpp/llama-bench.exe -m <gguf> -p 512 -n 128 -t 6`, versus our CLI on `docs/reference/benchmark-prompt.txt -n 24 --temp 0 --single-turn --no-display-prompt`). **Never run while another heavy test or download is running** (the numbers would be contention, not speed); one model at a time, several repetitions, median.
  - Models: Qwen1.5-MoE-A2.7B Q4_K_M, Mixtral-8x7B i1-Q4_K_S, Phi-3.5-MoE Q3_K_M, GLM-4.5-Air Q2_K, Llama-4-Scout Q3_K_M (all in `H:\_models`); OLMoE and Qwen3-Coder-30B already have rows and are re-run only if the code under them changed.
  - **Part A, llama.cpp alone, three modes:** (1) CPU (`-ngl 0`); (2) mixed CPU+GPU (`-ngl N`) and (3) GPU-only (`-ngl 99`), each *only if the vendored `tools/llama.cpp` has a GPU backend that works on this machine*. It has the RPC backend and no Vulkan backend at the time of writing (the backfill plan says building one is a separate project), so modes 2 and 3 are recorded as "not available here" unless that is built; do not substitute a different build without saying so. The machine has only an integrated GPU sharing system RAM (CLAUDE.md rule 13), so a GPU row here is not evidence about discrete GPUs.
  - **Part B, our engine, after all pending code changes in this plan are committed and verified:** the same three modes: CPU (`-g 0`), mixed (`-g N`: Vulkan hybrid, or the layer split for GLM, with N stated) and GPU-only (`-g -1`, which falls back to the hybrid when the model does not fit; say which path actually ran). Same prompt lengths, same thread count, reported as prefill and decode tokens per second and as a ratio to llama.cpp in the same mode.
  - Rules: report what ran (path, thread count, quantisation, context), keep failures and "not available" as rows rather than omitting them, and do not conclude "the GPU path is slow" from the iGPU alone.

