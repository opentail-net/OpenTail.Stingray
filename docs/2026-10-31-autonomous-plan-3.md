# Plan 3 (2026-10-31): what remains after plan 2

Follows [plan 2](2026-10-31-autonomous-plan-2.md) (open items) and its [done half](done/2026-10-31-autonomous-plan-2-completed-items-2026-10.md). Scope is OpenTail.Stingray only. A new item must name an identifiable failure, a user-visible limitation, a measurable gain, or a recurring cost it removes.

Contracts this must satisfy: `PrefillHandoffFamilies.Receipts` (admission by evidence); `GpuForwardPass.VulkanHybridUnsupportedReason` (the hybrid refuses models whose scale fields it does not apply); `ArchitectureDescriptor.SupportedBackends` (the planner honours it); perf claims need an idle machine, interleaved A/B, 3+ runs and equal output.

## Checklist

- [ ] **P1c** The NaN in hybrid MoE decode (Qwen3-Coder `-g 4`): 1 sighting in 18 control runs, then 0 in 12, then 0 in a 40-run hunt with the finiteness assertion (0 non-finite, 0 failed, min step cosine 0.9838 to 0.9979): 0 in 52 runs since the sighting. Not reproduced and not explained; open until reproduced or until the owner decides to stop hunting
- [x] **Q1** Full-GPU against CPU perplexity sweep over the dense checkpoints on disk. Done 2026-10-31 (`stingray perplexity -c 512` on the first 512 tokens of `wiki.test.raw`, `-g 0` against `-g -1`; the CPU run uses an fp32 KV cache and the GPU run bf16, so a fraction of a percent is not kernel error). No model shows a gap like gemma-2 did (+22 percent). Qwen3-0.6B Q8_0 19.07 / 19.12 (+0.3%); Falcon3-3B 10.16 / 10.09 (-0.6%); SmolLM3 8.893 / 8.901 (+0.1%); Qwen3-4B 14.20 / 14.12 (-0.5%); gemma-3-4b 16.43 / 16.66 (+1.4%, the largest; consistent with its step-level cosine of 0.992 / 0.9993); Granite 3.3-2B 7.460 / 7.404 (-0.8%); ERNIE-4.5-0.3B Q8_0 16.30 / 16.37 (+0.4%); Maincoder-1B 15.80 / 15.85 (+0.4%). One 512-token window per model, not a full-corpus figure
- [x] **Q2** Apply embedding, residual, logit and attention-scale fields in the Vulkan hybrid so Granite and Gemma 3 can partial-offload. Done 2026-10-31 (the owner asked for it, with "refuse if not useful"). Gemma 3 also needed the sliding-window RoPE base (10K on local layers; the hybrid used 1M everywhere), applied to both the GPU and CPU layers. Against the CPU pass on real weights, `-g 8` and `-g 20`: Granite 3.3-2B minimum per-step cosine 0.996, median 0.9995 (was -0.36); gemma-3-4b minimum 0.990, median 0.9992 (was 0.47), the same as the full Vulkan pass (0.992 / 0.9993). The refusal is now a soft-cap guard (Gemma 2 only). Ten mutations of the production change each fail a test (the logit-scale one only after a norm-ratio check was added, because cosine ignores a uniform scale). **Not faster on this machine**: decode `-g 8` 18.1 tok/s against 21.3 for `-g 0` (Granite) and 14.1 against 14.5 (gemma-3-4b), short-prompt prefill 6 against 45 tok/s, three interleaved runs each, machine not idle. This is the integrated GPU (rule 13), so it does not say a discrete GPU would lose; what it enables is a correct split, not a proven speed-up. No handoff receipt was added for either family
- [ ] **Q3** GPU soft-capping for gemma2 (attention and final logits), so the dense GPU path matches the CPU (perplexity 15.38 against 12.58 today). Needs an owner decision
- [ ] **Q4** Handoff gaps: Llama-3.1 checkpoint, LongRoPE, sliding-window llama, contexts above 4096, Qwen2.5-7B / Coder / VL, `startPos > 0` (a 300-token follow-up costs about 20 s against 2 to 3 s; not built)
- [ ] **Q5** API follow-ups (the `tool_choice: none` endpoint test landed in PR 96): the parameter audit of the other endpoints, the error envelope on other endpoints, the stale `stop_sequences` snippet in the design doc, a fake engine that emits tool calls so the telemetry canary covers tool-call argument leakage, and tests that each accepted sampling field reaches the engine
- [ ] **Q6** Release verification (the release workflow already runs `scripts/verify-nuget-package.ps1`, so the claim that nothing proves installation was too strong; what is missing is a machine that is not this one)
  - [x] 2026-10-31 on `main` at `f7f981eb`: the script packed all three packages at 1.0.9 and passed: a library consumer restored from the local feed and ran, a server consumer built, and the CLI installed as a dotnet tool and printed `1.0.9+f7f981eb`. Then the installed tool (a temp `--tool-path`, outside the repo) answered a prompt on Qwen3-0.6B Q8_0: CPU `-g 0` 41.6 tok/s decode, Vulkan hybrid `-g 4` 26.7 tok/s decode, same text
  - [ ] Run the installed package on a machine with no repo, no SDK from this checkout and no GPU driver tweaks, and install the package that nuget.org serves rather than the locally packed one (blocked: there is no second machine; the NativeAOT self-test in CI covers the AOT binary only)
- [ ] **P2d** CUDA passes have no scale references (blocked: no NVIDIA hardware)
- [ ] Memory note updated to the final state

## Owner decisions and blocked items

Decisions taken on the owner's behalf, all reversible: `CreateContext` kept unused; `FairModelScheduling` stays off (measured a net loss on the CPU: worst decode gap 1,183 ms against 81 ms, wall time 12% worse); dense-llama admission kept; Qwen3-Coder row uses a 0.97 per-step floor plus median above 0.999; the hybrid refuses scaled models; the planner notes CPU-only architectures; API choices (400 for `n != 1` and out-of-range sampling values, `tool_choice` any/tool refused, chunk size 0 means auto, the error envelope gained a nested `error` object additively).

Closed: the superseded PR #81 (its file was taken over by #92; closed 2026-10-31 after a diff against `main`).

Open for the owner:
- Q2: implement the scales in the hybrid, or keep refusing Granite and Gemma 3
- Q3: implement GPU soft-capping, or keep gemma2 CPU-only
- Q4 `startPos > 0`: worth building only if long follow-up prompts matter
- Keep the 0.99 bar for dense rows and the relaxed criterion for the MoE row, or tighten

Blocked: CUDA and discrete-GPU hardware (P2d, CUDA receipts, shared-GPU gate evaluation); a Llama-3.1 checkpoint; an idle machine for clean perf numbers; the NaN if it never reproduces.
