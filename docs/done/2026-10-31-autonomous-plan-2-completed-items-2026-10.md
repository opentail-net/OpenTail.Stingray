# Completed items split from 2026-10-31-autonomous-plan-2.md (2026-10-31)

Finished items moved verbatim out of [../2026-10-31-autonomous-plan-2.md](../2026-10-31-autonomous-plan-2.md). Open items stay there. Each block is preceded by the heading it sat under. Relative links inside the blocks were rebased for this folder.

> from: ## Checklist
- [x] Owner decisions recorded as taken provisionally (PR 72)

> from: ## Checklist
- [x] Plan 2 written and linked from `docs/00-current-work.md` (PR 72)

> from: ## Checklist
- [x] **P1** First-decode warm-up after a handoff: profiled, fixed, measured (PR 73)

> from: ## Checklist
- [x] **P1b** Housekeeping and the Qwen3-Coder row
  - [x] `LtxVaeNoiseIsolationTest` writes to a temp directory (PR 75)
  - [x] `SimdKernelsLegacyQ8_1Tests` timing test allows three attempts, mutation-checked (PR 75)
  - [x] Qwen3-Coder `-g 4` row diagnosed: failure rate measured on old and new builds (PR 76); the run-to-run variation traced to timing-dependent expert placement, and the row's acceptance criterion made explicit (PR 83)

> from: ## Checklist
- [x] **P6** `Audio.Fast` and `Vulkan.Fast` in CI
  - [x] `ci.yml` (floors 70 and 38) (PR 74)
  - [x] `release.yml` (PR 74)
  - [x] Whisper+VAD test skips visibly without `silero_vad.onnx` (PR 74)

> from: ## Checklist
- [x] **P3** Close the llama receipt gaps
  - [x] Context above 1024: Falcon3-3B 2,110 tokens and orpheus 1,774 tokens at context 4096 (new env-selected test `Handoff_EnvironmentChosenModel_LongPrompt`)
  - [x] A prompt at the 32-token minimum: permanent test `Handoff_PromptOfExactlyTheMinimumLength_TakesTheHandoff_AndOneTokenLessDoesNot`
  - [x] Llama-3.x `rope_freqs`: orpheus (Llama-3.2) carries the tensor and was already a passing row; the receipt wrongly listed it as not exercised
  - [x] Measure the `startPos > 0` cost before building anything: recorded from today's measured rates (see the P3 progress note); not built
  - [x] Receipt text updated

> from: ## Checklist
- [x] **P7** Documented request parameters audited (unknown JSON properties stay accepted); findings F1 to F4 of the audit are resolved (PRs 82, 84, 85 and 86); F5 (request-level tests for supported fields such as top_p and the Anthropic sampling fields reaching the engine) is only partly covered, since the range tests send them but do not assert they arrive
  - [x] Audit of every field the chat and messages endpoints read, with its tests and classification (PR 82)
  - [x] P7b: `max_completion_tokens`, `seed`, `top_k` implemented; `n` other than 1 and out-of-range sampling values refused with a 400 naming the field; `parallel_tool_calls`, `user` and `metadata` documented as intentionally ignored (30 tests, mutation-checked)
  - [x] P7c: `stop` and `stop_sequences` (one filter around the engine stream in all four handlers; generation is cancelled, not just the text hidden). 35 tests: 19 on the filter including 400 random inputs against a character-level reference, 16 through the endpoints; five production mutations each fail tests (the endpoint stream not filtered 4, no `stop_sequence` reason 2, no cancellation 7, no holdback 3, no validation 6)
  - [x] P7d: `stream_options.include_usage` (final usage chunk) and Anthropic `tool_choice` (auto and none honoured; any and tool refused with a 400). Limit: the `none` wiring has no failing endpoint test
  - [x] P7e: error bodies now also carry the nested `error` object on both APIs, additively (the flat fields stay; Anthropic's top-level `type` is not changed to `error`)

> from: ## Checklist
- [x] **P8** HTTP-disconnect cancellation, end to end: single-model half by the delegate session (PR 81, taken over unchanged), multi-model half with the runtime handle added here
  - [x] Check which existing tests already cover it (done in PR 81)
  - [x] Test: generation started, client disconnects, cancellation reaches the engine, slot released, next request completes (single-model host, delegate's test)
  - [x] Test: the same in a multi-model host with a fake loader (`NamedModelOptions.EngineFactory`): positive control that the runtime holds one handle while generating, then the handle is back to zero after the disconnect and the model serves the next request
  - [x] Mutation-checked: never disposing the runtime handle fails only the multi-model test; ignoring the client disconnect token fails both; the delegate's three mutations are in PR 81's description

> from: ## Checklist
- [x] **P4** Fair gate measured (two resident models, interleaved, 3 runs). Machine 10 to 16 percent busy, not idle; stated with the numbers. Result: off by default stays right, and the single-request engine is not worth mediating (plan 032, Slice 5)
  - [x] P4b `STINGRAY_PREFILL_CHUNK` and the server option `PrefillChunkTokens` never reached the batching engine; wired, 4 real-model tests, three production mutations each fail tests

> from: ## Checklist
- [x] **P5** OpenTelemetry acceptance (PR 80; limit: the fake engine emits no tool calls, so tool-call arguments are not covered)
  - [x] Allow-list test over every instrument and span
  - [x] Server-side prompt canary through the HTTP endpoints

> from: ## Checklist
- [x] **P9** Reproducibility helper used by the benchmark scripts (PR 78; self-test run again here: all checks pass)

> from: ## Checklist
- [x] **P10** `stingray config --show` (narrow), with a redaction test (PR 79; source is environment only and the output says so)

## Plan table rows of the finished items (from the original plan)

| # | Item | Done when |
|---|---|---|
| P1 | Profile the ~1.9 s first-decode warm-up after a handoff; fix only if the profile confirms the cause | 24-token decode rate within 5% of the 160-token rate, greedy output unchanged, or a written reason it cannot be hidden |
| P1b | Housekeeping first (ChatGPT review round 4, 2026-10-31, asked for it): `LtxVaeNoiseIsolationTest` writes to a temp directory; the flaky SimdKernels timing test is made deterministic or quarantined from correctness CI; the Qwen3-Coder-30B `-g 4` slots-16 handoff row (step 18 cosine 0.980) is diagnosed (old build vs new, same row) | `git status` clean after a full sweep; the row passes, or is a recorded MoE-specific finding with its measured cause |
| P3 | Close the llama receipt gaps: ctx above 1024, a 32-token prompt, the `startPos > 0` cost | each measured; receipt text updated |
| P4 | Measure the fair gate (two resident models, interleaved, 3 runs) | numbers in plan 032 slice 5 |
| P5 | OpenTelemetry: allow-list test over every instrument and span, server canary (spec: [2026-10-31-server-lifecycle-and-canary-specs.md](2026-10-31-server-lifecycle-and-canary-specs.md), Spec B) | both tests mutation-checked |
| P6 | CI: add `Audio.Fast` and `Vulkan.Fast` to the CPU-safe job and the release job (floors 70 and 38) | both suites run in CI; merged only if the workflow change is green |
| P7 | Audit the parameters Stingray documents as supported on the OpenAI and Anthropic endpoints (read first: the request DTOs and where each field is used). Do **not** reject unknown JSON properties: clients send vendor fields and that must keep working | a list of documented parameters; each is implemented, or rejected / reported as ignored, or has a recorded reason |
| P8 | Cancellation, end to end through real HTTP: start a generation, wait until it has begun, disconnect the client, check cancellation reaches the engine's active operation, the concurrency slot is released, the next request completes, and no session or runtime handle leaks. A GPU kernel already running need not stop at once (read first: `ConcurrencyLimitTests`, `ModelRuntimeManagerTests`, `ServerTests`, which mention cancellation; spec: [2026-10-31-server-lifecycle-and-canary-specs.md](2026-10-31-server-lifecycle-and-canary-specs.md), Spec A) | the test, mutation-checked, or the existing one cited |
| P9 | Reproducibility helper: a small shared PowerShell function used by the benchmark scripts, printing commit and dirty state, model file name and identity, binary/build configuration, backend and device, context and generation settings, input and run count, date. Identify the model by size and modified time plus a hash cached per experiment; never rehash a 12 GB file per run | the Performance League scripts print it; one worked example |
| P10 | `stingray config --show`, narrow: recognised `STINGRAY_*` settings that are set, with value (redacted where sensitive), source (default, environment, CLI) and class from the inventory; says so where the effective value cannot yet be resolved. Not a configuration framework | prints for a real command line; a test over redaction |

> from: the original "Order" line
Order: P1, P1b (done except P1c), P6 (done), P1c, P2, P3, P7, P8, P4, P5, P9, P10. P9 and P10 may move up only if they prove genuinely small; they do not displace correctness work.

## Progress notes and review of the finished items (from the original plan)

> from: ## Progress
### P1 - first-decode warm-up after a handoff: done (2026-10-31)

Profiled with a temporary stopwatch around the phases of `HybridForwardPass.Forward` (removed again), SmolLM2-1.7B `-g 12`, 69-token
prompt: the first step after the handoff took 1976 ms, of which 1784 ms was the GPU half (CPU half 27 ms, KV copy not involved); every
later step took 46 to 49 ms (GPU half 19 to 20 ms). So the cause is lazy GPU decode setup that a sequential prefill pays during the
prompt and a handoff skips. The hypothesis in the session report (pipelines created on first use) is consistent with this but the
profile does not say which object it is; the fix does not need to know.

Fix: `WarmDecodePathForHandoff()` runs one throwaway `Forward(0, 0)` at the end of the constructor, then `ResetCache()`, only for
dense non-TurboQuant placements that can use the handoff. The handoff overwrites every GPU row it keeps and truncates the CPU cache.

| Decode with handoff | Before | After (3 runs) | Without handoff (3 runs) |
|---|---|---|---|
| 24 tokens | 7.9 tok/s | 20.3, 20.5, 20.8 | 20.9, 21.4, 22.0 |
| 64 tokens | 12.5 tok/s | 21.2, 21.4, 21.7 | 21.6, 22.0, 22.0 |

Runs were interleaved (handoff on, then off, per length). Greedy text was identical in all 12 runs of the after-change set (3 with and
3 without the handoff, at each of two lengths). Prefill is unchanged at 93 to 108 tok/s; the 1.8 s moved from the first decode to model
load. Handoff exactness class (`HybridCpuPrefillHandoffTests`, heavy): every dense row passes with the warm-up (Mistral-7B, SmolLM2
-g 4 and 12, Qwen3-0.6B, Falcon3, and the MoE rows OLMoE and Phi-3.5-MoE). One row failed: Qwen3-Coder-30B-A3B `-g 4` slots 16, decode
step 18 cosine 0.97997 (all other steps above 0.9966). That is a MoE model; the warm-up returns early when an expert slot manager
exists, so it does not run there. I did not run the old build on that row, so "pre-existing" is an inference from the code path, not a
measurement; it is carried as an open finding (P2 housekeeping).

> from: ## Progress
### P6 - fast suites in CI: done (PR 74, 2026-10-31)

`Audio.Fast` (floor 70, 94 discovered) and `Vulkan.Fast` (floor 38, 48 discovered) now run in `ci.yml` and `release.yml`. The
first CI run failed: `WhisperPipeline_TranscribeWithVad_SuccessfullyPrunesSilence` needs `silero_vad.onnx` and threw on the
model-free runner. It passed here only because the model is on disk. It now skips visibly, and both suites were also run from copies
outside the repo so no `models/` folder was reachable (Audio.Fast 94 total, 5 skipped, 0 failed; Vulkan.Fast 48 passed). The second CI
run was green.

> from: ## Progress
### P1b - housekeeping: two of three done (2026-10-31)

- `LtxVaeNoiseIsolationTest` writes to the temp directory. Ran for real (11 s, 1 passed, not skipped) and `git status` stayed clean.
  The tracked png stays as the evidence for `docs/done/077`. Other Diffusion tests still write curated samples into
  `docs/diffusion-samples` on purpose (rule 9).
- `SimdKernelsLegacyQ8_1Tests.FastKernels_AreFasterThanDequantFallback`: idle speedup is 4.9x to 5.5x (3 runs), so the 1.0x bar only
  fails when contention hits every trial of an attempt. It now allows three independent attempts and fails only if all three miss.
  Mutation-checked: with the bar at 100x it ran three attempts and failed; at 1.0x it passes. The Q8_0 sibling already only prints.
- Heavy `HybridCpuPrefillHandoffTests` with the warm-up: 24 tests, 1 failed, 1 skipped, 80 minutes. The one failure is the
  Qwen3-Coder-30B `-g 4` slots-16 row.

> from: ## Progress
### P1b - the Qwen3-Coder row: diagnosed as far as measured (2026-10-31)

The row is flaky on the build before the warm-up too, so the warm-up did not cause it (and cannot: it returns early whenever an expert
slot manager exists). Counts of runs with at least one decode step at or below the 0.99 bar, standalone, 51 s each:

| Build / setting | Runs failing |
|---|---|
| Before the warm-up (commit of PR 72), interleaved with the next row | 4 of 8 |
| With the warm-up (current main) | 1 of 8 here, and 3 of 8 in an earlier batch |
| Current main, `STINGRAY_MOE_PREDICT_PREFETCH=0` | 7 of 10 |

Minimum step cosine in failing runs: 0.978 to 0.990; in passing runs 0.991 to 0.997. Every run passes the K/V byte-equality and
prefill-logit equality checks, so the handoff itself is exact; the variation is in hybrid MoE decode. A control added to the test class
(`Control_SequentialVsSequential_EnvironmentChosenModel`, skips without `STINGRAY_HANDOFF_TEST_MODEL`) compares two independent
sequential hybrids on the same forced tokens: 1 of 6 runs had a step at 0.9807 and 1 of 6 returned NaN for every step (one side's
logits were NaN or all zero; the test does not say which). No NaN appeared in the 34 handoff-vs-sequential runs above.

Refuted: predictive prefetch (the failure rate did not fall without it). Not established: the cause. Candidates not yet tested are a
GPU-versus-CPU expert choice that depends on the slot cache's contents (a legitimate numerical difference when a top-k choice is near a
tie, which the dense-calibrated 0.99 bar does not allow for), and a race or uninitialised read in the slot cache's lazy load, which
would also explain the NaN. The NaN is the part that needs finding; the cosine dips alone may be the bar being too tight for MoE.

Open item P1c: capture which side returns NaN (log the first non-finite logit and the expert cache state), then test the two
candidates. Until then the row stays as it is (heavy-only, not in CI), and `docs/STATUS.md` row 142 carries the caveat.

> from: ## Progress
### P3 - the gaps the llama receipt named (2026-10-31)

| Gap | Result |
|---|---|
| `rope_freqs` (Llama-3.x scaling) | **not a gap.** The orpheus-3b checkpoint (Llama-3.2) carries `rope_freqs.weight`; `HybridForwardPass` applies it on the GPU layers and in the CPU table, and orpheus was already a passing row (the receipt listed the feature as unexercised by mistake). No Llama-3.1 checkpoint is on the drives, so 3.1 itself is untested. |
| Context above 1024 | passes. Falcon3-3B `-g 4`: 2,110-token prompt, context 4096, prefill cosine 0.999835, 40 decode steps >= 0.99979 (199 s, mostly the token-by-token reference). orpheus `-g 8`: 1,774-token prompt, context 4096, prefill cosine 0.997314, decode >= 0.99921 (220 s). |
| 32-token minimum | passes exactly at the gate: a 31-token prompt is refused with the "shorter than" reason, a 32-token prompt takes the handoff, and its prefill logits and 8 teacher-forced decode steps agree with the sequential path (Falcon3-3B `-g 4`, 20 s). Permanent test. |
| `startPos > 0` (later chat turns) | measured, not built. The hybrid refuses the handoff there, so the new tokens prefill one at a time at the sequential rate: SmolLM2-1.7B `-g 12` 14.0 to 14.6 tok/s at 69 tokens (18.1 to 18.3 at 503), Mistral-7B `-g 4` 7.7, against 93 to 108 (178 to 182 at 503) and 44 to 49 with the handoff. A 300-token follow-up is about 20 s against 2 to 3 s. Short follow-ups (a few dozen tokens) cost 1 to 4 s. These rates come from the earlier measurements this day; SmolLM2 and Mistral are no longer on disk, so they were not re-measured. Building it means the CPU pass must hold the earlier turns' K/V; revisit if long pasted follow-ups matter. |

> from: ## Progress
### P4 - the fair gate, measured (2026-10-31)

Full write-up in `docs/3-product-and-runtime/032-multi-model-inference-runtime-plan.md`, Slice 5. Means over three interleaved pairs, both models on the CPU backend, a decode-heavy
Qwen2.5-0.5B (200 tokens) beside a prefill-heavy Qwen2.5-1.5B (three 1,245-token prompts):

| Gate | Decode model: median / p95 / worst gap | Decode model total | Prefill model total | Both done |
|---|---|---|---|---|
| off | 27 / 37 / 81 ms | 5.6 s | 7.9 s | 7.9 s |
| on, chunk 256 (default) | 12 / 44 / 1,183 ms | 8.8 s | 6.7 s | 8.8 s |
| on, chunk 128 | p95 507, worst 634 ms | 9.1 s | 7.0 s | 9.1 s |
| on, chunk 64 | p95 304, worst 356 ms | 9.4 s | 7.4 s | 9.4 s |
| on, chunk 32 | p95 181, worst 220 ms | 9.7 s | 8.0 s | 9.7 s |
| on, chunk 16 | p95 107, worst 143 ms | 10.1 s | 8.9 s | 10.1 s |

Without the gate the OS shares the cores and the decode model never waits more than 82 ms at any chunk size; with it, the decode model waits one prefill iteration per turn. Smaller
chunks shrink the wait and cost throughput. So on the CPU the gate is a net loss and stays off; mediating the single-request engine is not worth building. A shared GPU is the case it
could still help, and this machine (an iGPU, with both models on the CPU in this test) cannot show that. While running it I found that the chunk size setting had no effect at all
(P4b): the first sweep gave identical results for chunks of 256, 64, 32 and 0, which is how the dead wiring showed up.

> from: ## Progress
## Review of the strategic gap analysis (2026-10-31, ChatGPT follow-up)

The follow-up asked for 18 gaps and then, on being asked what is realistic, cut them down. I checked what could be checked against the
repository and took the same cut. The rule used: adopt what is small, has a demonstrated gap and reuses what exists; defer the rest
without discarding it.

**Adopted.** (a) Add the CPU-safe fast suites to CI. Verified: `ci.yml` and `release.yml` run Core, Pipeline, TurboQuant, Cli,
Vision, Server.Fast, Sessions.Fast and ForwardPass.Fast; `Audio.Fast` (94 tests, 6 s) and `Vulkan.Fast` (48 tests, 0.5 s) were not
in either, and `Vulkan.Fast` had three failing tests from 2026-10-11 until PR 70 that CI could not see (P6). `Embeddings` was left
out on purpose: 205 s and 8 skips without models. (b) The handoff boundaries and the Qwen3-Coder row (already P1b and P3). (c) Two
small checks the review raised that I have not yet looked at in the code: silently accepted request parameters (P7) and end-to-end
cancellation (P8). Both start by reading what exists; either may turn out to be already covered.

**Deferred, with the reason.** A unified qualification framework: receipts, goldens and the status matrix already do most of it, and
a second system would duplicate them. A generation-conformance suite, a scheduler, a resource-failure programme, a fuzzing programme,
a `doctor` command, an SBOM and a backend qualification matrix: each is real but large, and none has a failure behind it today.
Take one up when a specific defect or a release requirement appears, and fix a concrete security or lifecycle defect at once if
one is found; the deferral covers programmes, not defects. Not testable on this machine at all: CUDA, discrete GPU, ARM64.

**One point of agreement worth keeping.** The review's closing rule is to close more items than are opened. This plan therefore adds
only P6 to P8, each small, and removes nothing from the earlier list.

---

## More completed items moved from 2026-10-31-autonomous-plan-2.md on 2026-11-01

Finished items moved verbatim out of [../2026-10-31-autonomous-plan-2.md](../2026-10-31-autonomous-plan-2.md). Open items stay there. Each block is preceded by the heading it sat under. Relative links inside the blocks were rebased for this folder.

> from: ## Checklist
- [x] Plan 3 written (merged as [plan 3](../2026-10-31-autonomous-plan-3.md), PR 95) and a final report listing only owner decisions and blocked items (posted to the owner 2026-11-01; plan 3 repeats the decisions and blocked items)

> from: ## Checklist
- [x] Memory note `project_session_2026_10_31_loop_state.md` updated to the final state (kept current through PR 108, 2026-11-01)
