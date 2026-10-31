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
| P5 | OpenTelemetry: allow-list test over every instrument and span, server canary | both tests mutation-checked |
| P6 | CI: add `Audio.Fast` and `Vulkan.Fast` to the CPU-safe job and the release job (floors 70 and 38) | both suites run in CI; merged only if the workflow change is green |
| P7 | Audit the parameters Stingray documents as supported on the OpenAI and Anthropic endpoints (read first: the request DTOs and where each field is used). Do **not** reject unknown JSON properties: clients send vendor fields and that must keep working | a list of documented parameters; each is implemented, or rejected / reported as ignored, or has a recorded reason |
| P8 | Cancellation, end to end through real HTTP: start a generation, wait until it has begun, disconnect the client, check cancellation reaches the engine's active operation, the concurrency slot is released, the next request completes, and no session or runtime handle leaks. A GPU kernel already running need not stop at once (read first: `ConcurrencyLimitTests`, `ModelRuntimeManagerTests`, `ServerTests`, which mention cancellation) | the test, mutation-checked, or the existing one cited |
| P9 | Reproducibility helper: a small shared PowerShell function used by the benchmark scripts, printing commit and dirty state, model file name and identity, binary/build configuration, backend and device, context and generation settings, input and run count, date. Identify the model by size and modified time plus a hash cached per experiment; never rehash a 12 GB file per run | the Performance League scripts print it; one worked example |
| P10 | `stingray config --show`, narrow: recognised `STINGRAY_*` settings that are set, with value (redacted where sensitive), source (default, environment, CLI) and class from the inventory; says so where the effective value cannot yet be resolved. Not a configuration framework | prints for a real command line; a test over redaction |

Order: P1, P1b (done except P1c), P6 (done), P1c, P2, P3, P7, P8, P4, P5, P9, P10. P9 and P10 may move up only if they prove genuinely small; they do not displace correctness work.

Rule for adding to this plan: a new item must name an identifiable failure, a user-visible limitation, a measurable gain, or a recurring cost it removes. Without one of those it is not built yet.

Each item ends in its own merged PR, `Tests.Core` run before every merge. A stalled item is documented and the next one started.

## Checklist

Ticked means done **and merged to `main`** (or, for a measurement, recorded in a merged document). Anything in an open PR, in progress,
or only partly answered stays unticked, with a note. Last checked 2026-10-31 17:30 against `main` at `b9b00e1b` plus PR 76 (open).

- [x] Owner decisions recorded as taken provisionally (PR 72)
- [x] Plan 2 written and linked from `docs/00-current-work.md` (PR 72)
- [x] **P1** First-decode warm-up after a handoff: profiled, fixed, measured (PR 73)
- [ ] **P1b** Housekeeping and the Qwen3-Coder row (two of three sub-items done; the third continues as P1c)
  - [x] `LtxVaeNoiseIsolationTest` writes to a temp directory (PR 75)
  - [x] `SimdKernelsLegacyQ8_1Tests` timing test allows three attempts, mutation-checked (PR 75)
  - [ ] Qwen3-Coder `-g 4` row diagnosed: failure rate measured on old and new builds and prefetch refuted (in PR 76, not merged); the cause is **not** established, so this stays open and continues as P1c
- [ ] **P1c** Find the NaN in hybrid MoE decode and decide about the 0.99 bar for MoE rows
  - [ ] Capture which side returns non-finite logits in the control test
  - [ ] Test the slot-cache candidates (GPU-or-CPU expert choice; lazy-load race)
  - [ ] Fix, or record a decision about the bar
- [x] **P6** `Audio.Fast` and `Vulkan.Fast` in CI
  - [x] `ci.yml` (floors 70 and 38) (PR 74)
  - [x] `release.yml` (PR 74)
  - [x] Whisper+VAD test skips visibly without `silero_vad.onnx` (PR 74)
- [ ] **P2** Handoff receipts on the Vulkan hybrid (one passing permanent row each; a failing family stays Unverified with its cosine recorded)
  - [ ] Qwen2.5
  - [ ] Phi-3
  - [ ] gemma-2
  - [ ] Granite
  - Note: SmolLM2 and Mistral-7B were no longer in `models/` at 17:27 and are on no `_models` drive; re-pull before A/B work that needs them
- [ ] **P3** Close the llama receipt gaps
  - [ ] Context above 1024 (`STINGRAY_HANDOFF_TEST_CTX`)
  - [ ] A prompt at the 32-token minimum
  - [ ] Measure the `startPos > 0` cost before building anything
  - [ ] Receipt text updated
- [ ] **P7** Documented request parameters audited (unknown JSON properties stay accepted)
- [ ] **P8** HTTP-disconnect cancellation, end to end
  - [ ] Check which existing tests already cover it
  - [ ] Test: generation started, client disconnects, cancellation reaches the engine, slot released, next request completes, no leaked handle
  - [ ] Mutation-checked
- [ ] **P4** Fair gate measured (two resident models, interleaved, 3 runs; idle machine, so not while another agent is running)
- [ ] **P5** OpenTelemetry acceptance
  - [ ] Allow-list test over every instrument and span
  - [ ] Server-side prompt canary through the HTTP endpoints
- [ ] **P9** Reproducibility helper used by the benchmark scripts
- [ ] **P10** `stingray config --show` (narrow), with a redaction test
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
candidates. Until then the row stays as it is (heavy-only, not in CI), and `docs/STATUS.md` row 142 carries the caveat.## Review of the strategic gap analysis (2026-10-31, ChatGPT follow-up)

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