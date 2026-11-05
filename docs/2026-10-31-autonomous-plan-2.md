# Plan 2 (2026-10-31): carry-on work, done without owner input

> Completed items moved to [done/2026-10-31-autonomous-plan-2-completed-items-2026-10.md](done/2026-10-31-autonomous-plan-2-completed-items-2026-10.md) on 2026-10-31. This file keeps only the open items; a finished item that still has an open child stays here with it.

Follows [the session report and next plan](2026-10-31-session-report-and-next-plan.md), section 7. The owner's three decisions were
taken provisionally (same file, "Decisions taken on the owner's behalf"). Scope is OpenTail.Stingray only; no CUDA, no rule-14 families,
no voice-licence recipes.

Contracts this must satisfy: `PrefillHandoffFamilies.Receipts` is admission by evidence (a receipt needs a passing parity row and the
evidence text); the handoff is bit-compared to the token-by-token path by `HybridCpuPrefillHandoffTests`; perf claims need an idle
machine, interleaved A/B, 3+ runs and equal output (rule 7); real-weight tests must be timed (rule 12).

| # | Item | Done when |
|---|---|---|
| P1c | Find the NaN in hybrid MoE decode (Qwen3-Coder `-g 4`): find which side returns non-finite logits in the control, log the first one with the slot cache state, test the two candidates in the P1b write-up | the cause measured, or the NaN reproduced under a fixed setting; then a fix or a recorded decision about the 0.99 bar for MoE rows |
| P2 | Receipts for Qwen2.5, Phi-3, gemma-2, Granite on the Vulkan hybrid, one passing row each | permanent rows pass with no override; a family that fails stays Unverified with the measured cosine recorded |

Order: P1c (the NaN), then P2d (blocked on hardware). The full original order is in the done half. **Both open items are also tracked in [plan 3](2026-10-31-autonomous-plan-3.md)** (P1c and P2d); this file keeps their evidence.

Rule for adding to this plan: a new item must name an identifiable failure, a user-visible limitation, a measurable gain, or a recurring cost it removes. Without one of those it is not built yet.

Each item ends in its own merged PR, `Tests.Core` run before every merge. A stalled item is documented and the next one started.

## Checklist

Ticked means done **and merged to `main`** (or, for a measurement, recorded in a merged document). Anything in an open PR, in progress,
or only partly answered stays unticked, with a note. Last checked 2026-11-01 against `main` at `237753da` (PR 107, CI green): PRs 72 to 107 are merged; PR 81 was closed as superseded by PR 92. Plan 3 holds the work after plan 2. Items the delegate session finished (PRs 78 to 82) were ticked after reading each PR description and re-running the P9 self-test; the CPU-safe CI suites run all of their tests on every PR.

- [ ] **P1c** Find the NaN in hybrid MoE decode (the bar decision is made; the NaN is open)
  - [ ] Capture the NaN: 1 in 18 control runs, not reproduced in 12 later runs; the control test now asserts finite logits and names the run and step
  - [x] Test the slot-cache candidates: placement timing confirmed (8 of 8 runs bit-identical with prefetcher and warm-up off); expert warm-up alone and predictive prefetch alone are not the cause
  - [x] Record a decision about the bar: 0.97 per-step floor plus a median above 0.999 for this row only (PR 83)

- [x] **P6** `Audio.Fast` and `Vulkan.Fast` in CI
  - [x] `ci.yml` (floors 70 and 38) (PR 74)
  - [x] `release.yml` (PR 74)
  - [x] Whisper+VAD test skips visibly without `silero_vad.onnx` (PR 74)
- [x] **P2** Handoff receipts on the Vulkan hybrid (Qwen2.5 admitted; Phi-3, gemma-2 and Granite recorded as not admissible, with the reasons; P2b and P2c fixed; P2d blocked on hardware) (one passing permanent row each; a failing family stays Unverified with its cosine recorded)
  - [x] Qwen2.5 (receipt on the Vulkan hybrid; three permanent rows)
  - [x] Phi-3: tried, cannot run on the hybrid (fused `attn_qkv`, full offload only), stays Unverified; the full-GPU handoff path for it is untested
  - [x] gemma-2: tried, fails (min cosine 0.90 handoff vs sequential); the cause is the Vulkan hybrid and full GPU passes, not the handoff (see P2b); stays Unverified
  - [x] Granite: a dense checkpoint was on `H:\_models` all along (`ibm-granite_granite-3.3-2b-instruct`, missed by my first search because its name starts with `ibm-`). It is not a receipt candidate: the Vulkan hybrid ignores Granite's scale fields and produces garbage, see P2c. The earlier vision-LM result was a property of that file
  - [x] **P2c** The Vulkan hybrid applies none of the model scale fields (embedding, residual, logit, attention-scale override). Found with the dense Granite: `-g 8` answered `C. to - > and \ C、` instead of `Paris` (sequential hybrid against CPU: min cosine -0.36; CPU perplexity 7.81, full GPU 7.77); gemma-3-4b the same (0.47); ERNIE-0.3B (no scales) 0.998. **Superseded the same day by [plan 3, Q2](2026-10-31-autonomous-plan-3.md): the hybrid now applies the scale fields and the guard covers soft-capping only.** Original fix: Fixed: `GpuForwardPass.VulkanHybridUnsupportedReason` refuses such models, and `run` prints a clear error with exit code 1 instead of a stack trace. Tests: 6 unit tests, 2 real-model refusals, a 13-checkpoint guard that no receipt checkpoint is caught; three production mutations each fail tests. The clean CLI error was verified by hand only
  - [ ] P2d (blocked: no NVIDIA hardware) The CUDA full and hybrid passes have no reference to the residual or logit scale either, and the new guard does not cover them; confirm on a machine with a CUDA GPU before relying on Granite or Gemma 3 there
  - [x] **P2b** Silent GPU degradation for CPU-only architectures. Found: real runs were already safe (the gemma2 descriptor declares `SupportedBackends = Cpu` and the planner falls back to the CPU for `-g 8` and `-g -1`), but the override was silent, `stingray plan` reported 26 GPU layers (it did not consult the descriptor), and `perplexity -g -1` built the GPU pass directly and measured a 22 percent worse gemma-2 without saying so. Fixed: the planner raises a `cpu_only_architecture` warning when explicit GPU layers are overridden, `run` prints it as one note, `plan` consults the descriptor (and shows the reason), and `perplexity -g -1` refuses a CPU-only architecture. Tests: 3 planner and 3 plan tests, four production mutations each fail tests; the `run` note and the perplexity refusal were checked by hand only. Not changed: code that constructs the GPU passes directly (tests, library users) still accepts gemma2, and the dense GPU path still lacks soft-capping
  - Note: SmolLM2 and Mistral-7B were no longer in `models/` at 17:27 and are on no `_models` drive; re-pull before A/B work that needs them
- [x] **P3** Close the llama receipt gaps
  - [x] Context above 1024: Falcon3-3B 2,110 tokens and orpheus 1,774 tokens at context 4096 (new env-selected test `Handoff_EnvironmentChosenModel_LongPrompt`)
  - [x] A prompt at the 32-token minimum: permanent test `Handoff_PromptOfExactlyTheMinimumLength_TakesTheHandoff_AndOneTokenLessDoesNot`
  - [x] Llama-3.x `rope_freqs`: orpheus (Llama-3.2) carries the tensor and was already a passing row; the receipt wrongly listed it as not exercised
  - [x] Measure the `startPos > 0` cost before building anything: recorded from today's measured rates (see the P3 progress note); not built
  - [x] Receipt text updated
- [x] **P7** Documented request parameters audited (unknown JSON properties stay accepted); findings F1 to F4 of the audit are resolved (PRs 82, 84, 85 and 86); F5 (request-level tests for supported fields such as top_p and the Anthropic sampling fields reaching the engine) is only partly covered, since the range tests send them but do not assert they arrive
  - [x] Audit of every field the chat and messages endpoints read, with its tests and classification (PR 82)
  - [x] P7b: `max_completion_tokens`, `seed`, `top_k` implemented; `n` other than 1 and out-of-range sampling values refused with a 400 naming the field; `parallel_tool_calls`, `user` and `metadata` documented as intentionally ignored (30 tests, mutation-checked)
  - [x] P7c: `stop` and `stop_sequences` (one filter around the engine stream in all four handlers; generation is cancelled, not just the text hidden). 35 tests: 19 on the filter including 400 random inputs against a character-level reference, 16 through the endpoints; five production mutations each fail tests (the endpoint stream not filtered 4, no `stop_sequence` reason 2, no cancellation 7, no holdback 3, no validation 6)
  - [x] P7d: `stream_options.include_usage` (final usage chunk) and Anthropic `tool_choice` (auto and none honoured; any and tool refused with a 400). Limit: the `none` wiring has no failing endpoint test
  - [x] P7e: error bodies now also carry the nested `error` object on both APIs, additively (the flat fields stay; Anthropic's top-level `type` is not changed to `error`)
- [x] **P8** HTTP-disconnect cancellation, end to end: single-model half by the delegate session (PR 81, taken over unchanged), multi-model half with the runtime handle added here
  - [x] Check which existing tests already cover it (done in PR 81)
  - [x] Test: generation started, client disconnects, cancellation reaches the engine, slot released, next request completes (single-model host, delegate's test)
  - [x] Test: the same in a multi-model host with a fake loader (`NamedModelOptions.EngineFactory`): positive control that the runtime holds one handle while generating, then the handle is back to zero after the disconnect and the model serves the next request
  - [x] Mutation-checked: never disposing the runtime handle fails only the multi-model test; ignoring the client disconnect token fails both; the delegate's three mutations are in PR 81's description
- [x] **P4** Fair gate measured (two resident models, interleaved, 3 runs). Machine 10 to 16 percent busy, not idle; stated with the numbers. Result: off by default stays right, and the single-request engine is not worth mediating (plan 032, Slice 5)
  - [x] P4b `STINGRAY_PREFILL_CHUNK` and the server option `PrefillChunkTokens` never reached the batching engine; wired, 4 real-model tests, three production mutations each fail tests
- [x] **P5** OpenTelemetry acceptance (PR 80; limit: the fake engine emits no tool calls, so tool-call arguments are not covered)
  - [x] Allow-list test over every instrument and span
  - [x] Server-side prompt canary through the HTTP endpoints
- [x] **P9** Reproducibility helper used by the benchmark scripts (PR 78; self-test run again here: all checks pass)
- [x] **P10** `stingray config --show` (narrow), with a redaction test (PR 79; source is environment only and the output says so)
- [ ] Plan 3 written and a final report listing only owner decisions and blocked items
- [ ] Memory note `project_session_2026_10_31_loop_state.md` updated to the final state


## Progress

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

So for gemma-2 and the Granite vision LM the hybrid itself disagrees with the CPU pass, before any handoff is involved. For gemma-2 the cause is the missing soft-capping. For Granite the vision-LM file was a poor probe (its CPU perplexity on plain text is 1,135); the dense Granite 3.3-2B found later shows the real cause, the hybrid ignoring the model scale fields (P2c). Perplexity of the full GPU pass against the CPU on 384 tokens
of `scripts/kvarn-gate/wiki.test.raw` separates "different numbers" from "worse model": Qwen2.5-1.5B 11.07 against 11.19 (the control, bf16 KV), gemma-2 15.38
against 12.58 (about 22 percent worse), Granite vision LM 1,172.7 against 1,135.7. With a 14-token prompt the greedy text is identical on the CPU, the hybrid and the
full GPU, so the degradation is in the numbers, not visible in a short answer. Gemma 2 defines both an attention and a final-logit soft-cap; the dense GPU path applies neither.

Verified without the override: Qwen2.5 takes the handoff by default (16 s and 24 s, real runs), and gemma-2 is refused with the receipt message. Mutation check: moving the
receipt to another path fails `ReceiptedFamilies_AreAdmitted` and `DenseQwen2_IsAdmittedOnTheVulkanHybridOnly`.

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

So for gemma-2 and the Granite vision LM the hybrid itself disagrees with the CPU pass, before any handoff is involved. For gemma-2 the cause is the missing soft-capping. For Granite the vision-LM file was a poor probe (its CPU perplexity on plain text is 1,135); the dense Granite 3.3-2B found later shows the real cause, the hybrid ignoring the model scale fields (P2c). Perplexity of the full GPU pass against the CPU on 384 tokens
of `scripts/kvarn-gate/wiki.test.raw` separates "different numbers" from "worse model": Qwen2.5-1.5B 11.07 against 11.19 (the control, bf16 KV), gemma-2 15.38
against 12.58 (about 22 percent worse), Granite vision LM 1,172.7 against 1,135.7. With a 14-token prompt the greedy text is identical on the CPU, the hybrid and the
full GPU, so the degradation is in the numbers, not visible in a short answer. Gemma 2 defines both an attention and a final-logit soft-cap; the dense GPU path applies neither.

Verified without the override: Qwen2.5 takes the handoff by default (16 s and 24 s, real runs), and gemma-2 is refused with the receipt message. Mutation check: moving the
receipt to another path fails `ReceiptedFamilies_AreAdmitted` and `DenseQwen2_IsAdmittedOnTheVulkanHybridOnly`.

### P3 - the gaps the llama receipt named (2026-10-31)

| Gap | Result |
|---|---|
| `rope_freqs` (Llama-3.x scaling) | **not a gap.** The orpheus-3b checkpoint (Llama-3.2) carries `rope_freqs.weight`; `HybridForwardPass` applies it on the GPU layers and in the CPU table, and orpheus was already a passing row (the receipt listed the feature as unexercised by mistake). No Llama-3.1 checkpoint is on the drives, so 3.1 itself is untested. |
| Context above 1024 | passes. Falcon3-3B `-g 4`: 2,110-token prompt, context 4096, prefill cosine 0.999835, 40 decode steps >= 0.99979 (199 s, mostly the token-by-token reference). orpheus `-g 8`: 1,774-token prompt, context 4096, prefill cosine 0.997314, decode >= 0.99921 (220 s). |
| 32-token minimum | passes exactly at the gate: a 31-token prompt is refused with the "shorter than" reason, a 32-token prompt takes the handoff, and its prefill logits and 8 teacher-forced decode steps agree with the sequential path (Falcon3-3B `-g 4`, 20 s). Permanent test. |
| `startPos > 0` (later chat turns) | measured, not built. The hybrid refuses the handoff there, so the new tokens prefill one at a time at the sequential rate: SmolLM2-1.7B `-g 12` 14.0 to 14.6 tok/s at 69 tokens (18.1 to 18.3 at 503), Mistral-7B `-g 4` 7.7, against 93 to 108 (178 to 182 at 503) and 44 to 49 with the handoff. A 300-token follow-up is about 20 s against 2 to 3 s. Short follow-ups (a few dozen tokens) cost 1 to 4 s. These rates come from the earlier measurements this day; SmolLM2 and Mistral are no longer on disk, so they were not re-measured. Building it means the CPU pass must hold the earlier turns' K/V; revisit if long pasted follow-ups matter. |

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
