# Bug: concurrent decode output depends on how many OTHER sessions share the batch (5–15 wide)

**Status: root cause narrowed to a precise, reproducible boundary condition. Not fixed — the exact
non-associative line inside `SimdKernels`'s small-batch tiered dispatch has not been isolated. Found
while stress-testing `HotSession`/`ContinuousBatchingEngine` under concurrent load, unrelated to any
of docs/028's three phases or docs/029's Q8-vs-F32 mechanism.**

## One-paragraph summary

Run N identical sessions (same prompt, greedy/temperature-0 sampling — deterministic by
construction) concurrently against one `ContinuousBatchingEngine`. For **N ≤ 4** and **N ≥ 16**,
every session produces byte-identical output, as it must. For **5 ≤ N ≤ 15**, at least one session
silently diverges from the rest — in the reproduced case, one session continues generating a token
past the point every other (otherwise-identical) session stopped. This is not a race, not
nondeterminism, not thread-count-sensitive noise: the same wrong token, from the same session
index, every single run. The boundary lines up exactly with
`SimdKernels.MatMulBatched`'s own internal dispatch: `N < MinBatchForBlas` (default 16) takes a
small-batch tiered path (`MatVec4In` for groups of 4, `MatVec2In` for groups of 2, plain `MatVec`
for the odd one out); `N ≥ MinBatchForBlas` takes an entirely different OpenBLAS GEMM path. The
defect lives somewhere in the tiered path, specifically whenever a batch needs **more than one**
call into that dispatch to cover all N rows — never when a single call handles the whole batch.

## Why this matters

This is the **core continuous-batching decode path** — every `HotSession` turn, every concurrent
request, real production traffic. It is not gated behind any opt-in flag, not related to Q8/prefill
precision (confirmed: reproduces identically with `STINGRAY_CPU_PREFILL_Q8=0`), and it affects a
batch-size range (5–15 concurrent sequences) squarely inside normal operating conditions — most
real deployments will see this range constantly, not as an edge case.

## How this was found

Requested as a stress test for the `HotSession` continuous-batching path: "pretend there are 5,
then 10, then 40 requests at the same time." All three levels ran the identical prompt
("The capital of France is") with greedy sampling on `SmolLM2-1.7B-Instruct-Q4_K_M.gguf`, real CPU
dense model, one shared `ContinuousBatchingEngine`/`HotSessionRuntime`, `maxBatchSize` set to match
each level's concurrency. Because every session's input is identical and sampling is deterministic,
any single session disagreeing with the rest is decisive evidence of a defect — there is no
"expected" source of variation to explain it away.

N = 5 and N = 10 both failed on the first run:

```
session 1 of 5 produced [7042,30,198] but session 0 produced [7042,30] for the IDENTICAL prompt
under greedy sampling -- 5-way concurrent load corrupted or crossed session state.
```

N = 40 passed cleanly.

## Bisection

| N | Result | `SimdKernels.MatMulBatched` dispatch for this N |
|---|---|---|
| 2 | pass | one `MatVec2In` call (rows 0–1) |
| 3 | pass | one `MatVec2In` call (0–1) + one `MatVec` call (2) |
| **4** | **pass** | **one `MatVec4In` call (0–3), nothing else** |
| **5** | **fail** | `MatVec4In` (0–3) + `MatVec` (4) |
| 6 | fail | `MatVec4In` (0–3) + `MatVec2In` (4–5) |
| 7 | fail | `MatVec4In` (0–3) + `MatVec2In` (4–5) + `MatVec` (6) |
| **8** | **fail** | **`MatVec4In` (0–3) + `MatVec4In` (4–7) — two calls of the SAME tier, no other tier involved** |
| 10 | fail | `MatVec4In` ×2 (0–3, 4–7) + `MatVec2In` (8–9) |
| **40** | **pass** | **`N ≥ MinBatchForBlas` (16) → OpenBLAS GEMM path entirely, tiered dispatch never runs** |

N = 3 (two different kernel calls: `MatVec2In` then `MatVec`) passes. N = 8 (two calls of the
*same* kernel, `MatVec4In` twice, no other tier) fails. This rules out "mixing different tiers" as
the necessary ingredient — the common factor across every failing N is specifically that
**`MatVec4In` is invoked, and something else (another `MatVec4In` call, a `MatVec2In` call, or a
`MatVec` call) also runs within the same `MatMulBatched`/`BatchForwardMulti` step.** N = 4 (the
only case where `MatVec4In` runs exactly once and nothing else follows it) is clean. This points at
`MatVec4In` itself — most likely some state it reads or writes is not fully isolated between
separate invocations or from whatever runs immediately after it — but the exact line has not been
located; `MatVec4In`'s per-call parameters (`output0..3`, `input0..3`, a freshly-sliced `weights`
pointer) looked properly scoped on a first read of `SimdKernels.cs` (Q4_K branch: a `Parallel.For`
over weight rows writing into the four caller-supplied output pointers, no obviously-shared static
or thread-local buffer) — the defect is real and precisely bounded, but not yet pinned to a
specific line.

## What's confirmed and what isn't

**Confirmed:**
- Deterministic and 100% reproducible at every N tested in [5, 10] (5, 6, 7, 8, 10 all fail
  identically on repeat runs).
- Unrelated to `docs/029`'s Q8-vs-F32 mechanism (reproduces with `STINGRAY_CPU_PREFILL_Q8=0`).
- Confined to `SimdKernels.MatMulBatched`'s `batchSize < MinBatchForBlas` branch — N ≥ 16 (BLAS
  path) is clean at N = 40.
- Reproduces via `BatchForwardMulti` (decode), which always calls `SimdKernels.MatMulBatched` with
  `allowQ8: false` — i.e. this is NOT a quantization-precision issue at all, it reproduces in the
  supposedly-exact F32 tiered path.
- Not a `HotSession`/`ContinuousBatchingEngine` scheduling bug in the sense of "wrong session's data
  read" — the affected session's own prior tokens (positions 1–2) matched every other session
  exactly; only the decision at one specific later step diverged, consistent with a numeric result
  differing by kernel-tier composition rather than a session getting the wrong cache/position
  entirely.

**Not yet confirmed (plausible, not verified):**
- Whether this **also** affects prefill, not just decode. `PrefillCore`'s `MatMulBatchedCached`
  eventually calls the same `SimdKernels.MatMulBatched` in its own fallback branch, gated the same
  way — if several different sessions' prompts get admitted via `PrefillPackedMulti` and land in
  the same 5–15 combined-row range, the same defect class plausibly applies there too. Not
  independently tested; the reproduction here isolates decode specifically (positions 1–2 matched
  across all sessions, meaning prefill and the first two decode steps were NOT where this
  particular repro's divergence originated — but that doesn't rule out prefill being vulnerable
  under different conditions).
- The precise non-associative operation inside `MatVec4In`/`MatVec2In`/`MatVec`'s interaction.
  `SimdKernelsQ8KSTests.MatVec4In_BitwiseMatchesSingleMatVec` (cited in `SimdKernels.cs`'s own
  `BatchedMatVecTierEnabled` comment) apparently asserts bit-identity between `MatVec4In` and a
  solo `MatVec` call — worth re-running that specific test and checking whether it covers the
  scenario found here (a `MatVec4In` call immediately followed by ANOTHER batched call in the same
  step, rather than in isolation).
- Whether GPU backends (`CudaForwardPass`) have an equivalent tiered dispatch with the same
  vulnerability, or whether this is CPU-`SimdKernels`-specific.

## Reproduction

```
STINGRAY_RUN_HEAVY_TESTS=1 dotnet test tests/OpenTail.Stingray.Tests.Sessions \
  --filter-method "*HotSessionConcurrencyStressTests*"
```

`tests/OpenTail.Stingray.Tests.Sessions/HotSessionConcurrencyStressTests.cs` — `Stress_5ConcurrentSessions`
and `Stress_10ConcurrentSessions` are **left red on purpose**, the same standing pattern this
codebase already uses for `ContinuousBatchingTests.PrefillWithCache_Chunked_MatchesFull` (see
`docs/00-current-work.md`, "Known defect — one deliberate red test"). `Boundary_4Concurrent_AllMatch`
and `Boundary_8Concurrent_StillDiverges` are the bisection evidence, kept as permanent regression
markers for the exact boundary. `Stress_40ConcurrentSessions` is expected to pass (BLAS path).
Needs `models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf` on disk.

## Suggested next steps for whoever picks this up

1. Confirm/refute the prefill exposure question above — construct a repro where several different
   sessions' PROMPTS (not just decode steps) land in the same packed 5–15 range and check for the
   same class of divergence.
2. Read `MatVec2In`/`MatVec4In`'s full implementation for every `DType` branch (only Q4_K's branch
   was read here) — the model used is Q4_K_M, but other quant formats' branches weren't checked and
   could have a different or additional issue.
3. Check whether `Parallel.For`'s `s_parallelOpts` (shared across calls) has any per-call state
   that could leak between two back-to-back `MatVec4In` invocations in the same `BatchForwardMulti`
   step — the N=8 case (two `MatVec4In` calls, nothing else) is the narrowest reproduction and the
   best starting point for a kernel-level bisection, since it removes tier-mixing as a variable
   entirely.
4. Once the exact line is found, the fix likely follows the same shape as
   `SimdKernelsQ8KSTests.MatVec4In_BitwiseMatchesSingleMatVec`'s existing contract: assert bit
   identity between a value computed via any tier combination and the same value computed as a lone
   `MatVec` call, and fix until that holds for every N, not just the specific N used to invent the
   original test.
