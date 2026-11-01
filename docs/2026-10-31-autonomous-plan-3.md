# Plan 3 (2026-10-31): what remains after plan 2

> Completed items moved to [done/2026-10-31-autonomous-plan-3-completed-items-2026-10.md](done/2026-10-31-autonomous-plan-3-completed-items-2026-10.md) on 2026-11-01. This file keeps only the open items; a finished item that still has an open child stays here with it.

Follows [plan 2](2026-10-31-autonomous-plan-2.md) (open items) and its [done half](done/2026-10-31-autonomous-plan-2-completed-items-2026-10.md). Scope is OpenTail.Stingray only. A new item must name an identifiable failure, a user-visible limitation, a measurable gain, or a recurring cost it removes.

Contracts this must satisfy: `PrefillHandoffFamilies.Receipts` (admission by evidence); `GpuForwardPass.VulkanHybridUnsupportedReason` (the hybrid refuses models whose scale fields it does not apply); `ArchitectureDescriptor.SupportedBackends` (the planner honours it); perf claims need an idle machine, interleaved A/B, 3+ runs and equal output.

## Checklist

- [ ] **Q4** Handoff gaps
  - [x] `startPos > 0` on the Vulkan hybrid (the owner said long follow-ups matter). Built 2026-11-01: the hybrid's earlier K/V rows go into the temporary CPU cache, the CPU pass prefills only the new tokens, all rows are handed back. A 400-token follow-up after a 48-token first turn, three interleaved runs each, machine not idle: Falcon3-3B `-g 4` 2.6 s against 23.0 s sequential (about 9x), Qwen3-0.6B `-g 8` 1.3 s against 14.4 s (about 11x). Logits and every K/V row are byte-identical to the CPU pass for dense rows; MoE rows (OLMoE) are held to the decode comparison, because one full-class run moved the OLMoE logits by about 3 percent (a router near-tie). Decode afterwards agrees with the sequential path, minimum cosine 0.9988 or better. Four mutations (no seeding, wrong start position, handing back only the new rows, wrong stored length) each fail tests
  - [x] `startPos > 0` on the full-offload Vulkan pass (`GpuPrefillHandoff`), 2026-11-01. Non-Gemma-4 MoE: the GPU's earlier K/V rows are read back as F32 (fp16 widens exactly), the CPU pass prefills the new tokens, all rows go back. OLMoE `-g -1`, 400-token follow-up after a 48-token first turn, three interleaved runs: 3.1 s against 21.8 s sequential (about 7x). Decode agrees with the all-sequential run, minimum cosine 0.9991, for the default fp16 KV and for F32, with a first turn above and below the gate. Four mutations each fail tests. Gemma 4 still refuses a later turn (per-layer head dimensions, aliased KV layers)
  - [x] `startPos > 0` on the Vulkan layer split, 2026-11-01. The GPU layers' earlier rows are read back into the CPU part's cache, the CPU pass prefills the new tokens against the full history, the GPU layers get all rows. Trinity-Mini `-g 4`: second-turn prefill logits cosine 0.99985 against the all-CPU pass, decode minimum 0.9987; three mutations each fail the test. **Speed depends on the CPU trunk**: Trinity-Mini (afmoe, per-token CPU trunk because of its attention output gate) is not faster, 7.6 to 7.9 s against 7.8 s sequential for a 93-token turn; Hunyuan-A13B (batched trunk) 11.9 s against 19.3 s for a 102-token turn (about 1.6x), timed with the receipt gate lifted (`STINGRAY_HYBRID_CPU_PREFILL=all`) because `hunyuan-moe` has no layer-split parity receipt, so it is still not admitted
  - [ ] Llama-3.1 checkpoint (none on disk), LongRoPE, sliding-window llama, contexts above 4096, Qwen2.5-7B / Coder / VL
- [ ] **Q5** API follow-ups
  - [x] `tool_choice: none` endpoint test (PR 96)
  - [x] Parameter audit of the other endpoints, 2026-11-01: [the audit](reference/request-parameter-audit-other-endpoints-2026-11-01.md) covers `/v1/responses`, `/v1/audio/speech`, `/v1/images/*`, `/v1/embeddings`, `/v1/rerank`. Fixed: `/v1/responses` refuses non-empty `tools`, `previous_response_id` and out-of-range sampling values, and all its 400s now carry the error body; `/v1/audio/speech` refuses a `response_format` other than `wav`; **`/v1/images/generations` and `/edits` returned a prompt-seeded gradient with a 200 and never ran diffusion** (the README claimed SD/SDXL/FLUX), now 501 with the docs corrected (owner chose this). 18 tests, five mutations fail them
  - [x] `/v1/responses` reported 0 input tokens in both modes (it read the text-only stream, which drops the engine's usage chunk). Fixed 2026-11-01: it now reads the chunk stream, so `input_tokens`, `output_tokens` and `total_tokens` are real; two tests, mutation checked (found by the external review)
  - [x] `/v1/audio/transcriptions` and `/translations` audited 2026-11-01: a non-WAV body was a 500 and bad `temperature`, `response_format` and boolean flags were silently replaced; all are 400s that name the field now (10 tests, four mutations fail them)
  - [x] `/tokenize`, `/detokenize`, `/props`, `/status` and `/api/embed` read 2026-11-01: no defect found (recorded in the audit)
  - [x] `/v1/sessions/*` audited 2026-11-01: the turn digest ignored `allowed_choices` and the sampling fields were unchecked; fixed (plan 4, C1)
  - [ ] The `/v1/rerank` `top_n` range check (F5), the audio `model` and `language` fields and the speech `speed` fallback: moved to [plan 4](2026-11-01-plan-4.md), item C2
  - [x] The stale `stop_sequences` snippet in the design doc: the design sketch now carries a note naming the shipped record and what `stop_sequences` does (2026-11-01)
  - [x] A fake engine that emits tool calls so the telemetry canary covers tool-call argument leakage (2026-11-01: `ToolCallArgumentCanary_NeverAppearsInAnyServerTelemetry`, with a positive control that the call is parsed and returned in both modes; a mutation that puts the arguments on a span fails it and does not fail the older test)
  - [x] Tests that each accepted sampling field reaches the engine (2026-11-01: chat temperature, top_p, top_k, max_tokens, logit_bias; Anthropic temperature, top_p, top_k, max_tokens; Responses temperature, top_p, max_output_tokens, with a default-versus-override control; seed, the penalties and the thinking budget were already covered; three endpoint mutations fail them)

- [ ] **Q7** Connect the HTTP image API to the real diffusion pipelines (the ones `stingray image` runs), so `/v1/images/generations` and `/edits` can stop answering 501. Needs model selection, weights, timing and memory handling; own plan
- [ ] **Q6** Release verification (the release workflow already runs `scripts/verify-nuget-package.ps1`, so the claim that nothing proves installation was too strong; what is missing is a machine that is not this one)
  - [x] 2026-10-31 on `main` at `f7f981eb`: the script packed all three packages at 1.0.9 and passed: a library consumer restored from the local feed and ran, a server consumer built, and the CLI installed as a dotnet tool and printed `1.0.9+f7f981eb`. Then the installed tool (a temp `--tool-path`, outside the repo) answered a prompt on Qwen3-0.6B Q8_0: CPU `-g 0` 41.6 tok/s decode, Vulkan hybrid `-g 4` 26.7 tok/s decode, same text
  - [x] 2026-11-01: the package nuget.org serves (1.0.9) was installed into clean projects outside the repo on this machine: the README sample built (0 errors, 0 warnings) and the CLI tool ran a real model (43 tok/s decode); speak and 	ranscribe ran from the installed tool
  - [ ] The same on a machine that is not this one (no repo, no checkout SDK, no GPU driver tweaks; blocked: there is no second machine; the NativeAOT self-test in CI covers the AOT binary only)
- [ ] Memory note updated to the final state

## Owner decisions and blocked items

Decisions taken on the owner's behalf, all reversible: `CreateContext` kept unused; `FairModelScheduling` stays off (measured a net loss on the CPU: worst decode gap 1,183 ms against 81 ms, wall time 12% worse); dense-llama admission kept; Qwen3-Coder row uses a 0.97 per-step floor plus median above 0.999; the hybrid refuses scaled models; the planner notes CPU-only architectures; API choices (400 for `n != 1` and out-of-range sampling values, `tool_choice` any/tool refused, chunk size 0 means auto, the error envelope gained a nested `error` object additively).

Closed: the superseded PR #81 (its file was taken over by #92; closed 2026-10-31 after a diff against `main`).

Decided by the owner (2026-11-01): keep the hybrid scale work (Q2, merged in PR 98); build the `startPos > 0` handoff (Q4, merged for the three Vulkan paths); keep the 0.99 bar for dense rows and the relaxed criterion for the MoE row for now ("may change my mind later").

Open for the owner:
- Q3: implement GPU soft-capping, or keep gemma2 CPU-only

Blocked: CUDA and discrete-GPU hardware (P2d, CUDA receipts, shared-GPU gate evaluation); a Llama-3.1 checkpoint; an idle machine for clean perf numbers; the NaN if it never reproduces.
