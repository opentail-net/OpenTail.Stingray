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
| P2 | Receipts for Qwen2.5, Phi-3, gemma-2, Granite on the Vulkan hybrid, one passing row each | permanent rows pass with no override; a family that fails stays Unverified with the measured cosine recorded |
| P3 | Close the llama receipt gaps: ctx above 1024, a 32-token prompt, the `startPos > 0` cost | each measured; receipt text updated |
| P4 | Measure the fair gate (two resident models, interleaved, 3 runs) | numbers in plan 032 slice 5 |
| P5 | OpenTelemetry: allow-list test over every instrument and span, server canary | both tests mutation-checked |

Order: P1, P1b, P2, P3, P4, P5. Each ends in its own merged PR, `Tests.Core` run before every merge. A stalled item is documented and the
next one started.

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
