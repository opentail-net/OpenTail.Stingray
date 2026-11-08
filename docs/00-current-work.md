# Current work: what to do next

As of 2026-11-08. This file is the entry point. The open work is in one plan, [2026-11-08-open-work-merged.md](2026-11-08-open-work-merged.md). The Qwen 3.8 port has its own plan. Finished items are in [done/](done). The previous version of this file is in git history.

## How to use this file

1. Work the **Queue** top to bottom. Do not reorder because another item looks more interesting.
2. Each item has a **Next step** and a **Done when** in its plan. If an item stalls, write the blocker in the plan and move to the next one (CLAUDE.md).
3. An item is done only when its **Done when** line is true, with real weights and an independent reference where one applies. Then update the STATUS.md row, the RUNNING.md row if a command changed, and commit. Move the finished text to [done/](done).
4. Heavy runs one at a time (64 GB host). Perf claims need an idle machine, interleaved A/B, three or more runs and equal output (CLAUDE.md rule 7).
5. A new item needs a named failure, a user-visible limit, a measurable gain or a recurring cost.

## Queue

### 1. Open work (merged plan, in order)

Part A of the plan is for starting now, in this order. Part B waits for an owner decision, an approval, a download, or a quiet machine. Part C is blocked or parked. Section numbers below are the plan's.

- **A1–A4, bugs and test reliability:** cancellation flake under low CPU; fixed timing ceilings in two tests; the finder sweep for heavy test projects; the hybrid MoE NaN.
- **A5–A8, memory and handoff:** device memory figure; warm-up after a dense-llama handoff; receipts for more dense families; handoff cases not yet exercised.
- **A9–A13, numerics, sweeps, housekeeping:** MoE activation numerics; stale open boxes; housekeeping; OpenTelemetry acceptance; heavy runs on a quiet machine.
- **Part B, waiting:** resident-bytes accounting choice (B1); fair turn gate (B2, after B1); which Phi-3.5-MoE file is real (B3); Gemma 2 9B golden (B4, disk first); Bonsai2 reference approval (B5); Piper licence (B6); MoE expert GEMM numerics choice (B7); Q5 API follow-ups (B8); performance and scout follow-ups (B9).
- **Part C, blocked or parked:** hardware, checkpoints that do not fit, unverified families, parked items.

### 2. Qwen 3.8 Flash Next (`qwen4exp`), in progress

Source: [2026-11-08-qwen38-remaining-work-plan](2-coverage/2026-11-08-qwen38-remaining-work-plan.md). Qwen 3.8 stays not admitted and not advertised until Phase 8 (CLAUDE.md rule 14). Its owner decisions (parity criterion, unsloth fork reference, validator behaviour, admission) are in that plan.

3. **Broader real-weight verification (Phase 1).** Runnable now on the 62 GB checkpoint on E:. First step: three fixed prompts, one crossing a prefill/decode chunk boundary, compared with the pinned llama.cpp `bed0a8566` through `Qwen4ExpRefDumpTests`.
4. **Memory accounting completion (Job 3).** Source: [programme plan](2-coverage/2026-11-07-coverage-completion-programme-plan.md), Job 3. First step: full Core run on the part (a) change, then `HeldSnapshotBytes` and its residency and reservation tests.
5. **Real-checkpoint cancellation memory check (Phase 2).** Assert sequence-cache disposal in `Qwen4ExpBatchingTests`.
6. **Logit-level image comparison (Phase 5).** Compare the first generated logits with `llama-mtmd-cli`, then assert exact 40-token equality.

### 3. Items not yet in the merged plan

These six were in the old queue and did not make it into the merged plan. They are kept here so nothing is lost. Decide whether each one goes into the merged plan or is dropped.

- **Gemma 4 26B-A4B, one bounded step.** For the word-3 divergence: capture llama.cpp's top-k logprobs with `llama-server` `/completion` and `n_probs`. If the reference is confident, bisect the layer stack. If it is near a tie, park it. Do not re-measure on new prompts. Source: [2026-11-07 TensorSharp reference comparison](1-correctness/2026-11-07-gemma4-tensorsharp-reference-comparison.md).
- **Memory returned to the OS after unload.** Two-model `singleslot` host, four switch cycles, working set after each unload; trace the private-memory growth of about 130 MB per cycle. Source: former hub item E4 (its 2026-11-05 measurements are in git history of this file).
- **Protocol corpus status per case.** Record a status and error type for every case in `protocol-smoke.json`; add streaming tool-call and error cases for Anthropic and Chat Completions. Source: former hub item E5 (git history of this file).
- **Admit one real model with the playbook.** Candidate `ornith-ai/Ornith-1.5-35B-A3B-GGUF`. Start with `stingray scout -r <repo> -f <file> --budget <host RAM> -c <ctx>` before any download. Source: [scout plan](3-product-and-runtime/2026-10-09-checkpoint-scout-and-ai-admission-plan.md), [playbook](reference/architecture-admission-agent-playbook.md).
- **Multi-model durable session restore.** Source: [032 multi-model plan](3-product-and-runtime/032-multi-model-inference-runtime-plan.md). First step: write the restore test that fails today.
- **Configuration ownership.** List every config switch with its owner and mark the obsolete-looking ones for an owner decision. Source: [QoL plan](3-product-and-runtime/2026-11-08-product-plan.md).

### 4. Coverage ports (behind Qwen; rule 14 applies)

Ported families stay not admitted and not advertised until a real checkpoint and an independent reference verify them. Status ladder: [ported-families-todo](2-coverage/ported-families-todo.md).

1. **GLM-5.x** (`glm-dsa`, `glm5next`): paged real-weight run and K-pool selection gating attention reads. Blocked on disk for the full checkpoint ([programme plan](2-coverage/2026-11-07-coverage-completion-programme-plan.md), Phase 2B).
2. **DeepSeek V4**: YaRN rope scaling (factor 16) verified against the pinned reference, then a non-zero `compress_ratios` test. Paged run blocked on size (86 GB file).
3. **DiffusionGemma**: Q4_K_M Level 4 run. Checkpoint needed.
4. **MiniMax-H3**, **Muse-Glimmer** (admitted 2026-10-03, text, CPU only): next steps in their plans.
5. **Candidate reassessment** (report only, cheap): survey Muse Spark, MiniMax M3.1 and others.

Lower priority, not planned: AI21 Jamba, Kimi Linear, Arcee AFM, ServiceNow Apriel, Ant Ling (`bailingmoe2`), MiniMax-M2 (too large), DeepSeek-OCR v1 (no checkpoint), LTX-2.3 / 2.5 (a later campaign), ONNX shared-pipeline extraction (SenseVoice, Paraformer).

### 5. Performance (paused 2026-10-30, owner decision)

Resume from [4-performance/2026-11-08-performance-plan.md](4-performance/2026-11-08-performance-plan.md). Rules: [performance-measurement-rules](reference/performance-measurement-rules.md). Do not reopen the closed Q4_K repacked-GEMM investigation. When the owner resumes, this is the order:

1. SmolLM2 prefill at about 0.89x of llama.cpp: the Q4_K Path-2 GEMM is 65% of trunk time; then RoPE and attention.
2. int8 alignment with ggml Q8_K (ADR-0003 gap): the Q8_KS variant of the repacked Q4_Kx8 GEMM.
3. Batched prefill for partially offloaded dense models and the remaining per-token paths.
4. MTP on Qwen3.6-27B (UD-Q3_K_XL) is slower than plain (0.77-0.81x). Check the tensor-dtype coverage of the shared-decode kernels first. Source: [ADR-0002](reference/adr-0002-hybrid-gdn-mtp-chunked-prefill.md).
5. Raw-quant matvec bandwidth on Vulkan (gpt-oss MXFP4 decode, about 18 GB/s).
6. FLUX.1 DiT: 132 s against the 99.8 s sd.cpp target ([diffusion-performance Part 2](4-performance/diffusion/diffusion-performance.md#part-2-flux1-vulkan-gemm-handoff)).
7. Image-token prefill in VLMs, the remaining per-token segment path.
8. Cross-model DRY and perf-doc consistency (diffusion umbrella Phase 9).
9. MusicGen / AudioGen performance and DRY passes.
10. CosyVoice3 ODE step count: the fewer-steps A/B by ear and with a Whisper round trip (never run; the default is still 10).
11. Stable Audio 3 `IStableAudio3Engine` consolidation.

The rest of the performance boxes are in [perf-sweep-plan](4-performance/2026-11-08-performance-plan.md), [diffusion-performance](4-performance/diffusion/diffusion-performance.md), [vulkan-gemm](4-performance/gpu/vulkan-gemm/vulkan-gemm.md) and [080](4-performance/audio/080-tts-asr-gpu-residency-checklist.md).

## Owner decisions and actions outside the plans

- **Release 1.0.10.** Not ready: no tag, no version bump, no pack. Everything since 1.0.9 stays unpublished until the owner says so.
- **Stable Audio 3 Small quality.** Waiting on the owner's listening verdict on `docs/audio-samples/sa3_small_*_POSTTRAINED_*`. If the post-trained clips are better, add GGUF loading.

The other owner decisions are in Part B of the merged plan, each with a recommended default.

## Parked by owner decision (do not start)

- The HTTP image endpoints stay 501 (`/v1/images/generations` and `/edits` are not connected to the real pipelines).
- Gemma 2 GPU soft-capping (Gemma 2 is CPU only).
- The Qwen3-Coder NaN hunt. Keep the canary line; reopen only if the canary fires.
- The browser UI for the local server (Qwen phase 7F).

## Rules

1. **Order.** Take items in this order.
2. **A stalled item** gets its blocker written down, in its plan and in [1-correctness/bugstofix.md](1-correctness/bugstofix.md) when it is a bug. Then the next item starts.
3. **Evidence.** "Done" needs the check in the item's Done-when line, with real weights and an independent reference where one applies.
4. **Estimates are guesses** unless an item says it was investigated. Record the real time taken when an item closes.
5. **Disk.** Do not download a checkpoint unless the item says so. Check `models/`, `models/_models/` and the other `_models` folders first.
