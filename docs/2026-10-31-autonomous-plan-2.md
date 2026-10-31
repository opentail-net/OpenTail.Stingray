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
| P2 | Receipts for Qwen2.5, Phi-3, gemma-2, Granite on the Vulkan hybrid, one passing row each | permanent rows pass with no override; a family that fails stays Unverified with the measured cosine recorded |
| P3 | Close the llama receipt gaps: ctx above 1024, a 32-token prompt, the `startPos > 0` cost | each measured; receipt text updated |
| P4 | Measure the fair gate (two resident models, interleaved, 3 runs) | numbers in plan 032 slice 5 |
| P5 | OpenTelemetry: allow-list test over every instrument and span, server canary | both tests mutation-checked |
| P6 | Housekeeping: `LtxVaeNoiseIsolationTest` temp output, flaky SimdKernels timing test | `git status` clean after a full sweep |

Order: P1, P2, P3, P4, P5, P6. Each ends in its own merged PR, `Tests.Core` run before every merge. A stalled item is documented and the
next one started.

## Progress

(filled in as items land)
