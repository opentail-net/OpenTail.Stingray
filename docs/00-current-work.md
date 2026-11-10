# Current work: what to do next

As of 2026-11-08. This file is the entry point. The open work is in one plan, [2026-11-08-open-work-merged.md](2026-11-08-open-work-merged.md). The Qwen 3.8 port has its own plan. Finished items are in [done/](done). The previous version of this file is in git history.

## How to use this file

1. Work the **Queue** top to bottom. Do not reorder because another item looks more interesting.
2. Each item has a **Next step** and a **Done when** in its plan. If an item stalls, write the blocker in the plan and move to the next one (CLAUDE.md).
3. An item is done only when its **Done when** line is true, with real weights and an independent reference where one applies. Then update the STATUS.md row, the RUNNING.md row if a command changed, and commit. Move the finished text to [done/](done).
4. Heavy runs one at a time (64 GB host). Perf claims need an idle machine, interleaved A/B, three or more runs and equal output (CLAUDE.md rule 7).
5. A new item needs a named failure, a user-visible limit, a measurable gain or a recurring cost.
6. **One parent per item.** An item's text and its checkbox live in one parent document: the open plan that holds its Why, Next step and Done when. This file keeps only the items that have no open parent, and points to every other item's parent. It does not copy their state.

## Queue

### 1. Open work (merged plan, in order)

The open items and their state are in the merged plan (Part A, Part B, Part C), with a checkbox on each. They are not copied here.

### 2. Qwen 3.8 Flash Next (`qwen4exp`), in progress

Source: [2026-11-08-qwen38-remaining-work-plan](2-coverage/2026-11-08-qwen38-remaining-work-plan.md). Qwen 3.8 stays not admitted and not advertised until Phase 8 (CLAUDE.md rule 14). Its owner decisions (parity criterion, unsloth fork reference, validator behaviour, admission) are in that plan.

The parent of its open items is that plan's unchecked boxes. They are not copied here.

### 3. Items with no open parent

These items have no open owner document, so this file is their parent. Decide whether each one goes into the merged plan or is dropped.

- [ ] **Memory returned to the OS after unload.** Two-model `singleslot` host, four switch cycles, working set after each unload; trace the private-memory growth of about 130 MB per cycle. Source: former hub item E4 (its 2026-11-05 measurements are in git history of this file).
- [ ] **Admit one real model with the playbook.** Candidate `ornith-ai/Ornith-1.5-35B-A3B-GGUF`. Start with `stingray scout -r <repo> -f <file> --budget <host RAM> -c <ctx>` before any download. Source: [scout plan](3-product-and-runtime/2026-10-09-checkpoint-scout-and-ai-admission-plan.md), [playbook](reference/architecture-admission-agent-playbook.md).
- [ ] **Configuration ownership.** List every config switch with its owner and mark the obsolete-looking ones for an owner decision. Source: [QoL plan](3-product-and-runtime/2026-11-08-product-plan.md).

Moved to their parents, no longer tracked here:
- Gemma 4 26B-A4B word-3 divergence: parent is [2026-11-07 TensorSharp reference comparison](1-correctness/2026-11-07-gemma4-tensorsharp-reference-comparison.md), which records the root cause as fixed.
- Protocol corpus status per case: parent is the product plan, item A4.
- Multi-model durable session restore: parent is the [032 multi-model plan](3-product-and-runtime/032-multi-model-inference-runtime-plan.md), phase 7.

### 4. Coverage ports (behind Qwen; rule 14 applies)

Ported families stay not admitted and not advertised until a real checkpoint and an independent reference verify them. Status ladder: [ported-families-todo](2-coverage/ported-families-todo.md).

Each port's open boxes live in its parent. They are not copied here:

1. **GLM-5.x**: parent is the [programme plan](2-coverage/2026-11-07-coverage-completion-programme-plan.md), Phase 2B.
2. **DeepSeek V4**: parent is the programme plan, Phase 3.
3. **DiffusionGemma**: parent is its [port plan](2-coverage/2026-10-03-diffusiongemma-port-plan.md), the Level 4 box.
4. **MiniMax-H3** and **Muse-Glimmer**: parents are their port plans, [MiniMax-H3](2-coverage/2026-10-03-minimax-h3-port-plan.md) and [Muse-Glimmer](2-coverage/2026-10-03-muse-glimmer-port-plan.md).
5. **Candidate reassessment**: parent is the programme plan, Phase 4.

Lower priority, not planned: AI21 Jamba, Kimi Linear, Arcee AFM, ServiceNow Apriel, Ant Ling (`bailingmoe2`), MiniMax-M2 (too large), DeepSeek-OCR v1 (no checkpoint), LTX-2.3 / 2.5 (a later campaign), ONNX shared-pipeline extraction (SenseVoice, Paraformer).

### 5. Performance (paused 2026-10-30, owner decision)

Resume from [4-performance/2026-11-08-performance-plan.md](4-performance/2026-11-08-performance-plan.md). Rules: [performance-measurement-rules](reference/performance-measurement-rules.md). Do not reopen the closed Q4_K repacked-GEMM investigation. When the owner resumes, this is the order:

- [ ] 1. SmolLM2 prefill at about 0.89x of llama.cpp: the Q4_K Path-2 GEMM is 65% of trunk time; then RoPE and attention.
- [ ] 2. int8 alignment with ggml Q8_K (ADR-0003 gap): the Q8_KS variant of the repacked Q4_Kx8 GEMM.
- [ ] 3. Batched prefill for partially offloaded dense models and the remaining per-token paths.
- [ ] 4. MTP on Qwen3.6-27B (UD-Q3_K_XL) is slower than plain (0.77-0.81x). Check the tensor-dtype coverage of the shared-decode kernels first. Source: [ADR-0002](reference/adr-0002-hybrid-gdn-mtp-chunked-prefill.md).
- [ ] 5. Raw-quant matvec bandwidth on Vulkan (gpt-oss MXFP4 decode, about 18 GB/s).
- [ ] 6. FLUX.1 DiT: 132 s against the 99.8 s sd.cpp target ([diffusion-performance Part 2](4-performance/diffusion/diffusion-performance.md#part-2-flux1-vulkan-gemm-handoff)).
- [ ] 7. Image-token prefill in VLMs, the remaining per-token segment path.
- 8. Cross-model DRY and perf-doc consistency: parent is [diffusion-performance Phase 9](4-performance/diffusion/diffusion-performance.md), an open box. Not copied here.
- [ ] 9. MusicGen / AudioGen performance and DRY passes.
- [ ] 10. CosyVoice3 ODE step count: the fewer-steps A/B by ear and with a Whisper round trip (never run; the default is still 10).
- [ ] 11. Stable Audio 3 `IStableAudio3Engine` consolidation.

The rest of the performance boxes are in [perf-sweep-plan](4-performance/2026-11-08-performance-plan.md), [diffusion-performance](4-performance/diffusion/diffusion-performance.md), [vulkan-gemm](4-performance/gpu/vulkan-gemm/vulkan-gemm.md) and [080](4-performance/audio/080-tts-asr-gpu-residency-checklist.md).

## Owner decisions and actions outside the plans

- [ ] **Release 1.0.10.** Not ready: no tag, no version bump, no pack. Everything since 1.0.9 stays unpublished until the owner says so.
- [ ] **Stable Audio 3 Small quality.** Waiting on the owner's listening verdict on `docs/audio-samples/sa3_small_*_POSTTRAINED_*`. If the post-trained clips are better, add GGUF loading.

The other owner decisions are in Part B of the merged plan, each with a recommended default.

## Parked by owner decision (do not start)

Parent: Part C of the merged plan, which lists these items. Not copied here.

## Rules

1. **Order.** Take items in this order.
2. **A stalled item** gets its blocker written down, in its plan and in [1-correctness/bugstofix.md](1-correctness/bugstofix.md) when it is a bug. Then the next item starts.
3. **Evidence.** "Done" needs the check in the item's Done-when line, with real weights and an independent reference where one applies.
4. **Estimates are guesses** unless an item says it was investigated. Record the real time taken when an item closes.
5. **Disk.** Do not download a checkpoint unless the item says so. Check `models/`, `models/_models/` and the other `_models` folders first.
