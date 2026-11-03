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
- [x] **P2** Handoff receipts on the Vulkan hybrid (Qwen2.5 admitted; Phi-3, gemma-2 and Granite recorded as not admissible, with the reasons; P2b and P2c fixed; P2d blocked on hardware) (one passing permanent row each; a failing family stays Unverified with its cosine recorded)
  - [x] Qwen2.5 (receipt on the Vulkan hybrid; three permanent rows)
  - [x] Phi-3: tried, cannot run on the hybrid (fused `attn_qkv`, full offload only), stays Unverified; the full-GPU handoff path for it is untested
  - [x] gemma-2: tried, fails (min cosine 0.90 handoff vs sequential); the cause is the Vulkan hybrid and full GPU passes, not the handoff (see P2b); stays Unverified
  - [x] Granite: a dense checkpoint was on `H:\_models` all along (`ibm-granite_granite-3.3-2b-instruct`, missed by my first search because its name starts with `ibm-`). It is not a receipt candidate: the Vulkan hybrid ignores Granite's scale fields and produces garbage, see P2c. The earlier vision-LM result was a property of that file
  - [x] **P2c** The Vulkan hybrid applies none of the model scale fields (embedding, residual, logit, attention-scale override). Found with the dense Granite: `-g 8` answered `C. to - > and \ C、` instead of `Paris` (sequential hybrid against CPU: min cosine -0.36; CPU perplexity 7.81, full GPU 7.77); gemma-3-4b the same (0.47); ERNIE-0.3B (no scales) 0.998. **Superseded the same day by [plan 3, Q2](2026-10-31-autonomous-plan-3.md): the hybrid now applies the scale fields and the guard covers soft-capping only.** Original fix: Fixed: `GpuForwardPass.VulkanHybridUnsupportedReason` refuses such models, and `run` prints a clear error with exit code 1 instead of a stack trace. Tests: 6 unit tests, 2 real-model refusals, a 13-checkpoint guard that no receipt checkpoint is caught; three production mutations each fail tests. The clean CLI error was verified by hand only
  - [x] **P2b** Silent GPU degradation for CPU-only architectures. Found: real runs were already safe (the gemma2 descriptor declares `SupportedBackends = Cpu` and the planner falls back to the CPU for `-g 8` and `-g -1`), but the override was silent, `stingray plan` reported 26 GPU layers (it did not consult the descriptor), and `perplexity -g -1` built the GPU pass directly and measured a 22 percent worse gemma-2 without saying so. Fixed: the planner raises a `cpu_only_architecture` warning when explicit GPU layers are overridden, `run` prints it as one note, `plan` consults the descriptor (and shows the reason), and `perplexity -g -1` refuses a CPU-only architecture. Tests: 3 planner and 3 plan tests, four production mutations each fail tests; the `run` note and the perplexity refusal were checked by hand only. Not changed: code that constructs the GPU passes directly (tests, library users) still accepts gemma2, and the dense GPU path still lacks soft-capping
  - Note: SmolLM2 and Mistral-7B were no longer in `models/` at 17:27 and are on no `_models` drive; re-pull before A/B work that needs them

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
