# Plan 2 (2026-10-31): carry-on work, done without owner input

Follows [the session report and next plan](2026-10-31-session-report-and-next-plan.md), section 7. The owner's three decisions were
taken provisionally (same file, "Decisions taken on the owner's behalf"). Scope is OpenTail.Stingray only; no CUDA, no rule-14 families,
no voice-licence recipes.

Contracts this must satisfy: `PrefillHandoffFamilies.Receipts` is admission by evidence (a receipt needs a passing parity row and the
evidence text); the handoff is bit-compared to the token-by-token path by `HybridCpuPrefillHandoffTests`; perf claims need an idle
machine, interleaved A/B, 3+ runs and equal output (rule 7); real-weight tests must be timed (rule 12).

| # | Item | Done when |
|---|---|---|
| P1 | Profile the ~1.9 s first-decode warm-up after a handoff; fix only if the profile confirms the cause | 24-token decode rate within 5% of the 160-token rate, greedy output unchanged, or a written reason it cannot be hidden |
| P1b | Housekeeping first (ChatGPT review round 4, 2026-10-31, asked for it): `LtxVaeNoiseIsolationTest` writes to a temp directory; the flaky SimdKernels timing test is made deterministic or quarantined from correctness CI; the Qwen3-Coder-30B `-g 4` slots-16 handoff row (step 18 cosine 0.980) is diagnosed (old build vs new, same row) | `git status` clean after a full sweep; the row passes, or is a recorded MoE-specific finding with its measured cause |
| P1c | Find the NaN in hybrid MoE decode (Qwen3-Coder `-g 4`): find which side returns non-finite logits in the control, log the first one with the slot cache state, test the two candidates in the P1b write-up | the cause measured, or the NaN reproduced under a fixed setting; then a fix or a recorded decision about the 0.99 bar for MoE rows |
| P2 | Receipts for Qwen2.5, Phi-3, gemma-2, Granite on the Vulkan hybrid, one passing row each | permanent rows pass with no override; a family that fails stays Unverified with the measured cosine recorded |
| P3 | Close the llama receipt gaps: ctx above 1024, a 32-token prompt, the `startPos > 0` cost | each measured; receipt text updated |
| P4 | Measure the fair gate (two resident models, interleaved, 3 runs) | numbers in plan 032 slice 5 |
| P5 | OpenTelemetry: allow-list test over every instrument and span, server canary (spec: [2026-10-31-server-lifecycle-and-canary-specs.md](2026-10-31-server-lifecycle-and-canary-specs.md), Spec B) | both tests mutation-checked |
| P6 | CI: add `Audio.Fast` and `Vulkan.Fast` to the CPU-safe job and the release job (floors 70 and 38) | both suites run in CI; merged only if the workflow change is green |
| P7 | Audit the parameters Stingray documents as supported on the OpenAI and Anthropic endpoints (read first: the request DTOs and where each field is used). Do **not** reject unknown JSON properties: clients send vendor fields and that must keep working | a list of documented parameters; each is implemented, or rejected / reported as ignored, or has a recorded reason |
| P8 | Cancellation, end to end through real HTTP: start a generation, wait until it has begun, disconnect the client, check cancellation reaches the engine's active operation, the concurrency slot is released, the next request completes, and no session or runtime handle leaks. A GPU kernel already running need not stop at once (read first: `ConcurrencyLimitTests`, `ModelRuntimeManagerTests`, `ServerTests`, which mention cancellation; spec: [2026-10-31-server-lifecycle-and-canary-specs.md](2026-10-31-server-lifecycle-and-canary-specs.md), Spec A) | the test, mutation-checked, or the existing one cited |
| P9 | Reproducibility helper: a small shared PowerShell function used by the benchmark scripts, printing commit and dirty state, model file name and identity, binary/build configuration, backend and device, context and generation settings, input and run count, date. Identify the model by size and modified time plus a hash cached per experiment; never rehash a 12 GB file per run | the Performance League scripts print it; one worked example |
| P10 | `stingray config --show`, narrow: recognised `STINGRAY_*` settings that are set, with value (redacted where sensitive), source (default, environment, CLI) and class from the inventory; says so where the effective value cannot yet be resolved. Not a configuration framework | prints for a real command line; a test over redaction |

Order: P1, P1b (done except P1c), P6 (done), P1c, P2, P3, P7, P8, P4, P5, P9, P10. P9 and P10 may move up only if they prove genuinely small; they do not displace correctness work.

Rule for adding to this plan: a new item must name an identifiable failure, a user-visible limitation, a measurable gain, or a recurring cost it removes. Without one of those it is not built yet.

Each item ends in its own merged PR, `Tests.Core` run before every merge. A stalled item is documented and the next one started.

## Checklist

Ticked means done **and merged to `main`** (or, for a measurement, recorded in a merged document). Anything in an open PR, in progress,
or only partly answered stays unticked, with a note. Last checked 2026-10-31 20:10 against `main` at `8ffb7c97`: PRs 72 to 86 are merged except 81, the delegate session's partial P8. Items the delegate session finished (PRs 78 to 82) were ticked after reading each PR description and re-running the P9 self-test; the CPU-safe CI suites run all of their tests on every PR.

- [x] Owner decisions recorded as taken provisionally (PR 72)
- [x] Plan 2 written and linked from `docs/00-current-work.md` (PR 72)
- [x] **P1** First-decode warm-up after a handoff: profiled, fixed, measured (PR 73)
- [x] **P1b** Housekeeping and the Qwen3-Coder row
  - [x] `LtxVaeNoiseIsolationTest` writes to a temp directory (PR 75)
  - [x] `SimdKernelsLegacyQ8_1Tests` timing test allows three attempts, mutation-checked (PR 75)
  - [x] Qwen3-Coder `-g 4` row diagnosed: failure rate measured on old and new builds (PR 76); the run-to-run variation traced to timing-dependent expert placement, and the row's acceptance criterion made explicit (PR 83)
- [ ] **P1c** Find the NaN in hybrid MoE decode (the bar decision is made; the NaN is open)
  - [ ] Capture the NaN: 1 in 18 control runs, not reproduced in 12 later runs; the control test now asserts finite logits and names the run and step
  - [x] Test the slot-cache candidates: placement timing confirmed (8 of 8 runs bit-identical with prefetcher and warm-up off); expert warm-up alone and predictive prefetch alone are not the cause
  - [x] Record a decision about the bar: 0.97 per-step floor plus a median above 0.999 for this row only (PR 83)
- [x] **P6** `Audio.Fast` and `Vulkan.Fast` in CI
  - [x] `ci.yml` (floors 70 and 38) (PR 74)
  - [x] `release.yml` (PR 74)
  - [x] Whisper+VAD test skips visibly without `silero_vad.onnx` (PR 74)
- [ ] **P2** Handoff receipts on the Vulkan hybrid (Qwen2.5 done; Granite open; P2b opened) (one passing permanent row each; a failing family stays Unverified with its cosine recorded)
  - [x] Qwen2.5 (receipt on the Vulkan hybrid; three permanent rows)
  - [x] Phi-3: tried, cannot run on the hybrid (fused `attn_qkv`, full offload only), stays Unverified; the full-GPU handoff path for it is untested
  - [x] gemma-2: tried, fails (min cosine 0.90 handoff vs sequential); the cause is the Vulkan hybrid and full GPU passes, not the handoff (see P2b); stays Unverified
  - [ ] Granite: the only dense checkpoint on the drives is the Granite Vision 3.2 LM, which fails like gemma-2 and has a CPU perplexity of 1,135 on plain text (itself suspicious); needs a dense Granite text checkpoint (`stingray pull`) before this can be answered
  - [ ] **P2b** The Vulkan GPU passes accept gemma2 and granite but do not match the CPU: gemma-2 full-GPU perplexity 15.38 against 12.58; hybrid decode cosine about 0.80 against the CPU. The dense GPU path has no attention or final-logit soft-cap (`FinalLogitSoftcap` and `AttnLogitSoftcap` are read into the hyperparameters and applied only in the Gemma 4 trunk). Either refuse such models on the GPU with a clear reason (and make the planner fall back to the CPU) or implement soft-capping; first find who consults `GpuForwardPass.UnsupportedReason` today (nobody in `src/` calls it outside the partial-offload check)
  - Note: SmolLM2 and Mistral-7B were no longer in `models/` at 17:27 and are on no `_models` drive; re-pull before A/B work that needs them
- [ ] **P3** Close the llama receipt gaps
  - [ ] Context above 1024 (`STINGRAY_HANDOFF_TEST_CTX`)
  - [ ] A prompt at the 32-token minimum
  - [ ] Measure the `startPos > 0` cost before building anything
  - [ ] Receipt text updated
- [x] **P7** Documented request parameters audited (unknown JSON properties stay accepted); findings F1 to F4 of the audit are resolved (PRs 82, 84, 85 and 86); F5 (request-level tests for supported fields such as top_p and the Anthropic sampling fields reaching the engine) is only partly covered, since the range tests send them but do not assert they arrive
  - [x] Audit of every field the chat and messages endpoints read, with its tests and classification (PR 82)
  - [x] P7b: `max_completion_tokens`, `seed`, `top_k` implemented; `n` other than 1 and out-of-range sampling values refused with a 400 naming the field; `parallel_tool_calls`, `user` and `metadata` documented as intentionally ignored (30 tests, mutation-checked)
  - [x] P7c: `stop` and `stop_sequences` (one filter around the engine stream in all four handlers; generation is cancelled, not just the text hidden). 35 tests: 19 on the filter including 400 random inputs against a character-level reference, 16 through the endpoints; five production mutations each fail tests (the endpoint stream not filtered 4, no `stop_sequence` reason 2, no cancellation 7, no holdback 3, no validation 6)
  - [x] P7d: `stream_options.include_usage` (final usage chunk) and Anthropic `tool_choice` (auto and none honoured; any and tool refused with a 400). Limit: the `none` wiring has no failing endpoint test
  - [x] P7e: error bodies now also carry the nested `error` object on both APIs, additively (the flat fields stay; Anthropic's top-level `type` is not changed to `error`)
- [ ] **P8** HTTP-disconnect cancellation, end to end (PR 81 is open and partial: disconnect, cancellation reaching the engine, slot freed and three mutation checks are covered; releasing the runtime handle is not)
  - [x] Check which existing tests already cover it (done in PR 81)
  - [ ] Test: generation started, client disconnects, cancellation reaches the engine, slot released, next request completes, no leaked handle
  - [ ] Mutation-checked
- [ ] **P4** Fair gate measured (two resident models, interleaved, 3 runs; idle machine, so not while another agent is running)
- [x] **P5** OpenTelemetry acceptance (PR 80; limit: the fake engine emits no tool calls, so tool-call arguments are not covered)
  - [x] Allow-list test over every instrument and span
  - [x] Server-side prompt canary through the HTTP endpoints
- [x] **P9** Reproducibility helper used by the benchmark scripts (PR 78; self-test run again here: all checks pass)
- [x] **P10** `stingray config --show` (narrow), with a redaction test (PR 79; source is environment only and the output says so)
- [ ] Plan 3 written and a final report listing only owner decisions and blocked items
- [ ] Memory note `project_session_2026_10_31_loop_state.md` updated to the final state

## Progress

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

### P6 - fast suites in CI: done (PR 74, 2026-10-31)

`Audio.Fast` (floor 70, 94 discovered) and `Vulkan.Fast` (floor 38, 48 discovered) now run in `ci.yml` and `release.yml`. The
first CI run failed: `WhisperPipeline_TranscribeWithVad_SuccessfullyPrunesSilence` needs `silero_vad.onnx` and threw on the
model-free runner. It passed here only because the model is on disk. It now skips visibly, and both suites were also run from copies
outside the repo so no `models/` folder was reachable (Audio.Fast 94 total, 5 skipped, 0 failed; Vulkan.Fast 48 passed). The second CI
run was green.

### P1b - housekeeping: two of three done (2026-10-31)

- `LtxVaeNoiseIsolationTest` writes to the temp directory. Ran for real (11 s, 1 passed, not skipped) and `git status` stayed clean.
  The tracked png stays as the evidence for `docs/done/077`. Other Diffusion tests still write curated samples into
  `docs/diffusion-samples` on purpose (rule 9).
- `SimdKernelsLegacyQ8_1Tests.FastKernels_AreFasterThanDequantFallback`: idle speedup is 4.9x to 5.5x (3 runs), so the 1.0x bar only
  fails when contention hits every trial of an attempt. It now allows three independent attempts and fails only if all three miss.
  Mutation-checked: with the bar at 100x it ran three attempts and failed; at 1.0x it passes. The Q8_0 sibling already only prints.
- Heavy `HybridCpuPrefillHandoffTests` with the warm-up: 24 tests, 1 failed, 1 skipped, 80 minutes. The one failure is the
  Qwen3-Coder-30B `-g 4` slots-16 row.

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

### P2 - handoff receipts for other dense families on the Vulkan hybrid (2026-10-31)

Env-selected test, `STINGRAY_HYBRID_CPU_PREFILL=all` to lift Unverified, 150-token prompt, 40 teacher-forced decode steps, checkpoints from `F:\_models`:

| Checkpoint, split | Result | Min decode cosine |
|---|---|---|
| Qwen2.5-1.5B Q4_K_M, `-g 8` | pass | 0.99896 |
| Qwen2.5-1.5B Q4_K_M, `-g 4` | pass | 0.99921 |
| Qwen2.5-3B Q4_K_M, `-g 12` | pass | 0.99851 |
| Phi-3-mini-4k Q4_K_M, `-g 8` and `-g 16` | the hybrid refuses it: no path for fused `attn_qkv` | n/a |
| gemma-2-2b-it Q4_K_M, `-g 8` | fail | 0.90340 |
| granite-vision-3.2-2b Q3_K_S, `-g 8` | fail | 0.76407 |

To see whether the two failures were the handoff, the three-way diagnostic compares the handoff hybrid, the sequential hybrid and an all-CPU pass on the same tokens:

| `-g 8` | handoff vs CPU | sequential vs CPU | handoff vs sequential |
|---|---|---|---|
| Qwen2.5-1.5B (control) | min 0.99846, 0 steps below 0.99 | min 0.99800, 0 | min 0.99981, 0 |
| gemma-2-2b-it | min 0.78286, 40 of 40 below | min 0.79863, 40 of 40 | min 0.89898, 40 of 40 |
| Granite vision 3.2 LM | min 0.69092, 40 of 40 | min 0.81136, 40 of 40 | min 0.69543, 40 of 40 |

So for gemma-2 and Granite the hybrid itself disagrees with the CPU pass, before any handoff is involved. Perplexity of the full GPU pass against the CPU on 384 tokens
of `scripts/kvarn-gate/wiki.test.raw` separates "different numbers" from "worse model": Qwen2.5-1.5B 11.07 against 11.19 (the control, bf16 KV), gemma-2 15.38
against 12.58 (about 22 percent worse), Granite vision LM 1,172.7 against 1,135.7. With a 14-token prompt the greedy text is identical on the CPU, the hybrid and the
full GPU, so the degradation is in the numbers, not visible in a short answer. Gemma 2 defines both an attention and a final-logit soft-cap; the dense GPU path applies neither.

Verified without the override: Qwen2.5 takes the handoff by default (16 s and 24 s, real runs), and gemma-2 is refused with the receipt message. Mutation check: moving the
receipt to another path fails `ReceiptedFamilies_AreAdmitted` and `DenseQwen2_IsAdmittedOnTheVulkanHybridOnly`.

### P1c - Qwen3-Coder row: cause of the flakiness found (2026-10-31, evening)

All runs: Qwen3-Coder-30B-A3B Q4_K_M, `-g 4`, 16 slots, 150-token prompt, 40 teacher-forced decode steps, standalone (about 50 s each).

| Experiment | Result |
|---|---|
| Handoff row, old build (before the warm-up), 8 runs | 4 failed |
| Handoff row, current build, 22 runs in three batches (3 of 8, 1 of 8, 3 of 6) | 7 failed |
| `STINGRAY_HYBRID_CPU_PREFILL_WARM=0` (no expert warm-up), 10 runs | 3 failed |
| `STINGRAY_MOE_PREDICT_PREFETCH=0` (no next-layer prefetch only), 10 runs | 7 failed |
| Control: two sequential hybrids, 18 runs | 1 step at 0.981, 1 run with NaN logits |
| Three-way, CPU's greedy tokens, 6 runs | 0 dips in all three pairs (min cosine 0.996) |
| Three-way, the hybrid's greedy tokens, 8 runs | handoff vs CPU dipped in 4, handoff vs sequential in 3, sequential vs CPU in 0 |
| Three-way, hybrid tokens, prefetcher and expert warm-up both off, 8 runs | all 8 runs **bit-identical** (handoff vs CPU min 0.98952, sequential vs CPU min 0.99392) |

What this shows. The hybrid's own greedy tokens are the same in every run (same hash over 8 runs), and K/V rows and prefill logits are
byte-exact every time, so the handoff is exact. The run-to-run variation disappears when the always-on same-layer prefetcher and the
expert warm-up are both off, so it comes from which experts happen to be resident on the GPU (GPU kernel) versus computed in the CPU
fallback at each step, which depends on when background uploads land. With that source removed one step still sits at 0.9895 against the
CPU reference (sequential's worst against the CPU is 0.9939): a small residue I have not explained. Over 2,000 decode steps from 50
runs the median cosine is 0.99955, 5 percent of steps are below 0.9968, 1 percent below 0.990 and the minimum is 0.978.

Decision (taken here, reversible): the Qwen3-Coder row uses a 0.97 per-step floor plus a median above 0.999; K/V, prefill logits and the
argmax near-tie rule are unchanged, and every other row keeps 0.99. Verified: 8 of 8 runs pass; with the floor at 0.9999 the row fails;
with the median bound at 0.99999 it fails on the median message. The env-selected test accepts
`STINGRAY_HANDOFF_TEST_MIN_STEP_COSINE` (a test-only variable). Two env-selected diagnostics were added and assert little by design:
`Control_SequentialVsSequential_EnvironmentChosenModel` (asserts finite logits) and `Diagnostic_ThreeWay_EnvironmentChosenModel`.

Not explained: the single NaN. It appeared once, in the first six control runs, and not in 12 further control runs, so it is rare
(about 1 in 18) and I could not capture which side produced it. The unproven candidate is a read of an expert slot while its upload is
in flight, which timing-dependent placement makes plausible; it stays open under P1c.
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
