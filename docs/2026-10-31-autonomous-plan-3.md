# Plan 3 (2026-10-31): what remains after plan 2

Follows [plan 2](2026-10-31-autonomous-plan-2.md) (open items) and its [done half](done/2026-10-31-autonomous-plan-2-completed-items-2026-10.md). Scope is OpenTail.Stingray only. A new item must name an identifiable failure, a user-visible limitation, a measurable gain, or a recurring cost it removes.

Contracts this must satisfy: `PrefillHandoffFamilies.Receipts` (admission by evidence); `GpuForwardPass.VulkanHybridUnsupportedReason` (the hybrid refuses models whose scale fields it does not apply); `ArchitectureDescriptor.SupportedBackends` (the planner honours it); perf claims need an idle machine, interleaved A/B, 3+ runs and equal output.

## Checklist

- [ ] **P1c** The NaN in hybrid MoE decode (Qwen3-Coder `-g 4`): 1 sighting in 18 control runs, then 0 in 12 and 0 in the first 27 of a 40-run hunt with the finiteness assertion. Open until reproduced or until a decision to stop hunting is recorded
- [ ] **Q1** Full-GPU against CPU perplexity sweep over the dense checkpoints on disk (gemma-3 full GPU is unchecked); record each delta
- [ ] **Q2** Apply embedding, residual, logit and attention-scale fields in the Vulkan hybrid so Granite and Gemma 3 can partial-offload (then lift the refusal for them, with a parity row each). Needs an owner decision, see below
- [ ] **Q3** GPU soft-capping for gemma2 (attention and final logits), so the dense GPU path matches the CPU (perplexity 15.38 against 12.58 today). Needs an owner decision
- [ ] **Q4** Handoff gaps: Llama-3.1 checkpoint, LongRoPE, sliding-window llama, contexts above 4096, Qwen2.5-7B / Coder / VL, `startPos > 0` (a 300-token follow-up costs about 20 s against 2 to 3 s; not built)
- [ ] **Q5** API follow-ups: request-level tests for the `tool_choice: none` wiring (only a unit test covers it), the parameter audit of the other endpoints, the error envelope on other endpoints, the stale `stop_sequences` snippet in the design doc, a fake engine that emits tool calls so the telemetry canary covers tool-call argument leakage, and tests that each accepted sampling field reaches the engine
- [ ] **Q6** Release verification: build the release artifacts with `release.yml` and install/run them on a clean machine (CI proves builds and tests, not an installable release)
- [ ] **P2d** CUDA passes have no scale references (blocked: no NVIDIA hardware)
- [ ] Memory note updated to the final state

## Owner decisions and blocked items

Decisions taken on the owner's behalf, all reversible: `CreateContext` kept unused; `FairModelScheduling` stays off (measured a net loss on the CPU: worst decode gap 1,183 ms against 81 ms, wall time 12% worse); dense-llama admission kept; Qwen3-Coder row uses a 0.97 per-step floor plus median above 0.999; the hybrid refuses scaled models; the planner notes CPU-only architectures; API choices (400 for `n != 1` and out-of-range sampling values, `tool_choice` any/tool refused, chunk size 0 means auto, the error envelope gained a nested `error` object additively).

Open for the owner:
- Close the superseded PR #81 (the other session's partial P8; P8 was finished in #92)
- Q2: implement the scales in the hybrid, or keep refusing Granite and Gemma 3
- Q3: implement GPU soft-capping, or keep gemma2 CPU-only
- Q4 `startPos > 0`: worth building only if long follow-up prompts matter
- Keep the 0.99 bar for dense rows and the relaxed criterion for the MoE row, or tighten

Blocked: CUDA and discrete-GPU hardware (P2d, CUDA receipts, shared-GPU gate evaluation); a Llama-3.1 checkpoint; an idle machine for clean perf numbers; the NaN if it never reproduces.
