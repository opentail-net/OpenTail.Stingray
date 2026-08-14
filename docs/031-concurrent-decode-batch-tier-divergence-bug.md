# Bug (FIXED): concurrent prefill output depended on how many OTHER sessions were packed alongside it

**Status: fixed.** Root cause was **not** decode's kernel tiering (the original hypothesis below,
now disproven) — it was `ForwardPass.PrefillPackedMulti` letting OpenBLAS SGEMM engage or not
based on the COMBINED packed batch size across MULTIPLE UNRELATED sessions, which is not a
property of any single request. Found while stress-testing `HotSession`/`ContinuousBatchingEngine`
under concurrent load, unrelated to any of docs/028's three phases or docs/029's Q8-vs-F32
mechanism.

## One-paragraph summary (corrected)

Run N identical sessions (same prompt, greedy/temperature-0 sampling — deterministic by
construction) concurrently against one `ContinuousBatchingEngine`. For roughly 5 ≤ N ≤ 15, at
least one session would silently diverge from the rest — the same wrong token, from the same
session index, every run: not a race, not thread-count noise. The actual mechanism: when
`ContinuousBatchingEngine.RunPrefillStep` admits multiple prompts in the same tick, it calls
`ForwardPass.PrefillPackedMulti`, which packs every admitted session's prompt tokens into ONE
combined batch of size `N = sum of each session's chunk length` and runs the whole layer stack's
matmuls (`MatMulBatchedCached`) over that combined `N`. `SimdKernels.MatMulBatched` engages
OpenBLAS SGEMM whenever `N >= MinBatchForBlas` (default 16) and falls back to a dot-product/
tiered kernel otherwise — and **that gate does not know or care that the combined `N` is made up
of several independent prompts**, only one of which is the session you're looking at. A short
prompt (5 tokens) prefilled alone never gets near 16 and always takes the non-BLAS path; the same
prompt packed with 7 other short prompts (`N = 40`) crosses the threshold and takes BLAS instead.
BLAS SGEMM and the dot-product kernels are both individually correct — they are just not
bit-identical to each other (different summation order) — so a session's own numerics silently
depended on how many *other*, unrelated sessions happened to be packed alongside it in the same
admission tick. The drift is tiny per layer but compounds over several decode steps and
eventually flips a close-margin greedy argmax decision.

## Why the original hypothesis was wrong

The first write-up of this bug (see git history of this file) attributed it to
`SimdKernels.MatMulBatched`'s small-batch tiered dispatch (`MatVec4In`/`MatVec2In`/`MatVec`),
reached from **decode**, based on a bisection that varied session count and watched decode output
diverge in the same N range. That bisection was real, but its attribution was wrong — it never
isolated prefill from decode, and this codebase's continuous-batching path always prefills before
it decodes, so "N concurrent sessions" conflates "N sessions packed in one `PrefillPackedMulti`
call" with "N sessions later decoding together." Two controlled diagnostics (raw `ForwardPass`
API calls, bypassing `ContinuousBatchingEngine`/`HotSession` entirely) separated the two:

1. **`BatchForwardMulti` (decode) at N=8, in isolation**: zero divergence between "slot 1 in an
   8-wide batch" and "slot 1 computed alone." Decode's own tiered dispatch is safe, exactly as
   its own bit-identity test (`SimdKernelsQ8KSTests.MatVec4In_BitwiseMatchesSingleMatVec`) claims.
2. **The full engine repro, with `prefillChunkTokens: 0`** (forces `RunPrefillStep`'s
   one-prompt-at-a-time branch, which calls `PrefillWithCache` per session and never calls
   `PrefillPackedMulti`, even though the SAME 8 sessions still decode together afterward): all 8
   sessions produced identical, correct output. This is the decisive result — it proves the
   defect requires `PrefillPackedMulti` specifically, not shared decode.

From there, a raw `PrefillPackedMulti(N=8 identical 5-token prompts)` call (no engine at all)
reproduced the divergence directly against a solo `PrefillWithCache` reference. Toggling
`SimdKernels.Q8PrefillEnabled` off did **not** make it go away (ruling out the Q8 activation-
quantization path, and confirming `MinBatchForQ8Prefill`'s default of 1 was never actually the
active variable here, since it applies uniformly to solo and packed alike). Forcing
`SimdKernels.MinBatchForBlas` to 1000 — so the packed call's `N=40` stayed on the same non-BLAS
route solo's `N=5` was already using — made the divergence disappear completely, with both Q8 on
and Q8 off. That isolated the BLAS/non-BLAS crossover as the sole remaining variable and the
confirmed mechanism.

## The fix

`SimdKernels.MatMulBatched` gained an `allowBlas` parameter (default `true`, preserving existing
behavior everywhere). `ForwardPass.MatMulBatchedCached`/`MatMulBatchedDualCached` gained the same
parameter, threaded through to both their own dequant-cache BLAS branch and to
`SimdKernels.MatMulBatched`. `PrefillPackedMulti`'s six matmul call sites now pass
`allowBlas: false` — packed multi-session prefill never lets its combined `N` decide BLAS
eligibility, so a session's own prefill numerics no longer depend on how many other sessions
happened to be packed alongside it. Solo prefill (`Prefill`/`PrefillWithCache`) is unaffected —
every call site there keeps the default `true` and behaves exactly as before.

This trades away BLAS's throughput benefit specifically for packed admission of long prompts
(prompts individually long enough to justify BLAS on their own now stay on the dot-product/Q8
path when prefilled via packed admission) in exchange for determinism: packed prefill output no
longer depends on unrelated concurrent traffic. Q8Prefill's own dot8/dot4/dot1 tiering already
provides substantial weight-read amortization across packed tokens independent of BLAS, so the
practical throughput cost is real but narrower than it sounds — most real concurrent traffic
(interactive chat messages, individually well under 16 tokens) was never BLAS-eligible on its own
account anyway, and is exactly the case this bug hit.

## Verification

`HotSessionConcurrencyStressTests` (5, 10, 40 concurrent sessions, plus 4/8 boundary tests) is
fully green — `Boundary_8Concurrent_StillDiverges` (kept its original name; the divergence it
demonstrated is gone) now passes alongside everything else. A direct raw-API repro
(`PrefillPackedMulti(N=8 identical 5-token prompts)` compared against solo `PrefillWithCache`,
replaying 3 greedy decode steps) matches solo exactly at every step post-fix, both with Q8Prefill
at its default (on) and explicitly disabled.

Full regression: `Tests.ForwardPass.Fast` (602/602), `Tests.Sessions.Fast` (394/394),
`Tests.Server.Fast` (260/260), and the real-model `ContinuousBatchingTests` class all pass except
three PRE-EXISTING, unrelated failures confirmed (via `git stash`, re-running against the pre-fix
baseline) to fail identically with or without this change: `PrefillWithCache_Chunked_MatchesFull`
(the codebase's existing, already-documented deliberate-red test —
see `docs/00-current-work.md`), `PrefillWithCache_SingleToken_MatchesForward` (flagged as a
still-open issue in `docs/029-prefill-batch-composition-numerics-bug.md` / `docs/bugstofix.md`),
and `PrefillWithCache_DequantCacheOnOff_BitIdentical`. None of the three touch
`PrefillPackedMulti`; all three are solo-`PrefillWithCache`-only, and this fix does not change
solo `PrefillWithCache`'s behavior at all (its call sites keep the default `allowBlas: true`).

## Reproduction

```
STINGRAY_RUN_HEAVY_TESTS=1 dotnet test tests/OpenTail.Stingray.Tests.Sessions \
  --filter-method "*HotSessionConcurrencyStressTests*"
```

All five tests in `tests/OpenTail.Stingray.Tests.Sessions/HotSessionConcurrencyStressTests.cs`
now pass. Needs `models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf` on disk.
