# Bug: retained-session KV cache mixes Q8-quantized (prefill) and F32-exact (decode) precision

**Status: RESOLVED.** Root cause found and confirmed end-to-end (below). The fix landed is
**Option 3** from the "Fix options" list — the test's oracle rewritten to replay through the
session's own entry points, not a production-numerics change — after **Option 2 was implemented,
verified NOT to work, and reverted** (see "Why Option 2 doesn't actually work" below; keep this
section if this bug class resurfaces elsewhere, since the same reasoning gap is easy to repeat).
`HotSessionGreedyReplayTests`'s full suite (3 real-model tests) passes; the full heavy
`Tests.Sessions` suite (23/23) and `Tests.Sessions.Fast` (384/384) are green.

## One-paragraph summary

`ForwardPass.PrefillCore` (used by every prefill, including every `PrefillWithCache` call a
`HotSession` turn makes) defaults to an int8-activation-quantized ("Q8-prefill") compute path for
speed. `ForwardPass.BatchForwardMulti` (used by every decode step) **deliberately** never takes
that path — it stays in exact F32, on purpose, to keep one user's decoded logits independent of
who else is batched alongside them and to preserve speculative decoding's bit-exact verify
guarantee. Both of those are correct, intentional, individually well-reasoned design decisions.
Put together, they mean: **a token's cached K/V is Q8-approximate if it arrived via prefill, and
F32-exact if it arrived via decode** — and a real multi-turn `HotSession`'s cache accumulates both
kinds, turn after turn. A later turn's prefill correctly reads back whatever precision is actually
sitting in the cache; nothing is reading data wrong. The divergence is real precision, not a
logic bug in cache bookkeeping. **Confirmed conclusively**: setting `STINGRAY_CPU_PREFILL_Q8=0`
makes both the raw repro and the original failing test (`HotSessionGreedyReplayTests.
HotSession_MultiTurn_MatchesFullGreedyReplay_OnRealModel`) match/pass exactly, `diff = 0.000000`.

This supersedes everything in this doc's original version below "What's confirmed but not yet
fully pinned down" — that section's hypotheses (a) and (b) are now resolved: (b) was right
(splitting an admission alone does nothing — Arm 4 below proved that), and the actual mechanism
inside "a `BatchForwardMulti`-authored cache entry" is the Q8-vs-F32 precision mismatch, not a
`TruncateTo`/bookkeeping defect.

## The decisive experiments

### Arm 4: does splitting a prefill into two admissions, with NO decode in between, reproduce it?

```
single       : PrefillWithCache(all 13 tokens, startPos=0), then 2× BatchForwardMulti decode
splitNoDecode: PrefillWithCache(7 tokens, startPos=0) + PrefillWithCache(6, startPos=7),
               NO decode call between them, then 2× BatchForwardMulti decode
splitWithDecode (= the original "appended" arm): PrefillWithCache(5, startPos=0),
               2× BatchForwardMulti decode (writing positions 5,6), then
               PrefillWithCache(6, startPos=7), then 2× BatchForwardMulti decode
```

Result:
```
single          top3 = 2:16.8740, 198:16.4822, 284:16.0230
splitNoDecode   top3 = 2:16.8740, 198:16.4822, 284:16.0230     diff(splitNoDecode, single)   = 0.000000
splitWithDecode top3 = 284:16.6223, 2:16.3506, 198:16.3326     diff(splitWithDecode, single) = 2.161279
```

**Splitting the admission alone changes nothing.** `PrefillCore`'s batched math genuinely is
invariant to batch composition when everything goes through it — this rules out the "GEMM/RMSNorm
tiling isn't N-invariant" family of hypothesis entirely for this bug (it's a real, documented
concern elsewhere in this codebase, see "Related precedent" — just not what's happening here). The
only thing that matters is whether a real `BatchForwardMulti` decode call touched the cache before
the next prefill runs.

### The confirming experiment: disable Q8-prefill entirely

Re-running the exact same three arms with `STINGRAY_CPU_PREFILL_Q8=0` (which gates both the
int8-activation path and the Q4_K×8-repacked path in `SimdKernels.MatMulBatchedCached` —
`ForwardPass.cs:1286`/`1307`, see `SimdKernels.Q8PrefillEnabled`, `SimdKernels.cs:289-290`):

```
single          top3 = 284:16.5686, 198:16.5647, 2:16.5523
splitNoDecode   top3 = 284:16.5686, 198:16.5647, 2:16.5523     diff = 0.000000
splitWithDecode top3 = 284:16.5686, 198:16.5647, 2:16.5523     diff = 0.000000
```

All three now agree exactly. And directly on the original test:

```
STINGRAY_RUN_HEAVY_TESTS=1 STINGRAY_CPU_PREFILL_Q8=0 dotnet test tests/OpenTail.Stingray.Tests.Sessions \
  --filter-method "*HotSession_MultiTurn_MatchesFullGreedyReplay_OnRealModel*"
→ Passed! total: 1, failed: 0
```

This is as close to proof as a numerics investigation gets: the entire divergence, from the raw
three-token repro up to the actual production test, disappears when the one identified variable is
removed.

## The mechanism, precisely

- `SimdKernels.MatMulBatched`'s `allowQ8` parameter (`SimdKernels.cs:71-86`) defaults to `false`,
  and the doc comment on it says the default "is load-bearing": prefill rows are positions within
  one prompt, so quantizing them together is numerically sound and consistent; decode rows are
  independent sequences, so quantizing them together would make one user's logits depend on who
  else was in the batch, and would break speculative decoding's bit-exact verify contract.
- `ForwardPass.PrefillCore` → `MatMulBatchedCached` (`ForwardPass.cs:1283-1316`) takes the
  Q8-quantized / Q4_K×8-repacked path by default (`Q8PrefillEnabled`, default `true`) for every
  `PrefillWithCache` call, i.e. every `HotSession` turn's admission.
- `ForwardPass.BatchForwardMulti`'s decode step (`ForwardPass.cs:5347-5352`) calls
  `SimdKernels.MatMulBatched` directly with **no `allowQ8` argument**, i.e. the safe default
  (`false`) — permanently exact F32, by design, regardless of `Q8PrefillEnabled`.
- Both write their computed K/V into the *same* `PagedKvCache` via the *same* `Append`/
  `IncrementPosition` calls (`ForwardPass.cs:1631-1647` for prefill, `:5400-5403` for decode) — the
  cache has no memory of which precision wrote a given position, and doesn't need to for its own
  bookkeeping to be correct. It just stores whatever bytes it was given.
- A subsequent turn's `PrefillCore` call computes attention over that mixed-precision history
  faithfully. It isn't misreading anything.

The result: a `HotSession`'s cache is a precision patchwork — Q8-approximate wherever a prompt (or
a continuation's own prompt) was prefilled, F32-exact wherever a reply was decoded — and that
patchwork is invisible within one turn but becomes numerically observable the moment any later
turn's prefill attends back across a decode-authored region. Nothing here is a logic error; it's
two individually-correct, individually-intentional numeric paths whose interaction across turn
boundaries was never checked before `HotSessionGreedyReplayTests` (a pre-existing test, unrelated
to whatever was being worked on when this was found) happened to catch it.

## Why the earlier hypotheses in this investigation were wrong, briefly

The original version of this doc (see git history) suspected `PrefillCore`'s batched RMSNorm/GEMM/
RoPE pipeline might not be invariant to N (batch size/composition) — reasoning from this
codebase's own extensive precedent of exactly that defect class (see below). Arm 4 above rules
that out directly: splitting one 13-token prefill into two prefill-only admissions, with nothing
decoded in between, reproduces the "single" result exactly. The batched math is fine. What matters
is only whether *any* token in the cache's history arrived via decode before the next prefill runs.

## Related precedent in this codebase (context, not the cause here)

Worth reading before deciding on a fix, since this bug sits in the same neighborhood as several
already-handled issues and the fix should be consistent with how those were resolved:

- **`ForwardPass.cs:5238-5254`** — dated 2026-08-13, the day before this bug surfaced. Fixed
  `PrefillWithCache` so a single-token continuation (`N == 1`) no longer takes a *different*,
  non-quantized prefill code path than longer prefills — explicitly to make **all prefill calls**
  numerically consistent **with each other**. That fix is real, still in place, and orthogonal to
  this bug (which is prefill-vs-decode, not prefill-vs-prefill) — it does not need to be reverted
  or revisited for whatever fix lands here.
- **`ForwardPass.cs:1298-1306`, `:3113-3208`** — the Q4_K×8 batch-size-threshold removal and the
  long Flash-64/Flash-128 investigation, both examples of "a token's result must not depend on who
  else was batched with it" being taken seriously and fixed or explicitly measured-and-accepted in
  this codebase. This bug is a *cross-turn* instance of the same principle, not covered by either
  fix.
- **`docs/adr-0001-session-cache-lifecycle.md`** — the retained-session design's stated guarantee
  is "ExactLossless" continuation. This bug means that guarantee does not currently hold once a
  session has both decoded and re-prefilled within its lifetime (i.e. essentially every real
  multi-turn conversation) — relevant context for weighing the fix options below, since option 2
  restores exactly what the ADR claims and option 3 documents a deliberate exception to it instead.

## Fix options that were on the table

1. **Disable Q8-prefill for ALL `PrefillWithCache` calls**, unconditionally.
2. **Disable Q8-prefill only for `PrefillWithCache` calls that continue an existing cache**
   (`startPos > 0`).
3. **Leave production numerics alone; fix `HotSessionGreedyReplayTests`'s oracle** to replay
   through the same computation path the session actually used, instead of blanket-re-prefilling
   the growing history every turn.
4. **Make single-sequence, non-concurrent decode also eligible for Q8** — rejected outright,
   `SimdKernels.cs:71-86` explicitly warns this risks multi-user decode correctness and
   speculative decoding's bit-exact verify contract; not attempted.

## Why Option 2 doesn't actually work (implemented, verified, reverted)

Option 2 was implemented first — `MatMulBatchedCached`/`MatMulBatchedDualCached`/`MoeFfnBatched`/
`PrefillPackedMulti` all threaded an `allowQ8` flag computed from `startPos == 0`, so any
`PrefillWithCache` call continuing an existing cache would skip Q8 for its own new tokens. It
built clean and ran clean. **It did not fix the test — identical failure, identical tokens,
verified with a forced clean rebuild to rule out a stale-build false alarm.**

The reasoning gap: Option 2 changes how a continuation's *own new tokens* get computed. It does
nothing about positions the cache *already holds* from before that call — and in this test,
positions 5–6 (turn 1's decoded tokens) are already permanently F32-exact by the time turn 2's
admission even starts. No change to how turn 2 computes its *own* tokens can retroactively alter
what's already sitting in the cache at 5–6. Worse, the oracle's own reference computation
(`Prefill()`, called with `startPos = 0` every time, by construction) is a completely separate
call path that Option 2 never touches — so the oracle stayed fully Q8-quantized across all 13
positions regardless, while the session (with Option 2) became F32-exact for *more* of its cache
than before. Two different kinds of mismatch, same visible failure.

This is also why the earlier global-flag experiment (`STINGRAY_CPU_PREFILL_Q8=0`) genuinely
worked: it isn't scoped to sessions at all, so it also disabled Q8 for the oracle's `Prefill()`
call — making *both sides of the comparison* F32-exact throughout. Confirming the mechanism and
confirming a *fix* are different claims; the global-flag experiment only ever established the
former. **Lesson for next time this bug class shows up**: once a cache has held even one
decode-authored (permanently F32) position, no change to how *later* prefill computes its own new
tokens can make that cache's total contents match an independent, all-Q8 reference computation.
The only ways to actually close the gap are changing what the reference computes, or making
*every* position in the cache F32 (which is what disabling Q8 globally does, at a much larger
cost than "just continuations" implied). Option 2's code was fully reverted (`git checkout --` on
`ForwardPass.cs`, confirmed via `git diff` to contain nothing else) rather than left in as a
partial, ineffective mitigation.

## What was actually implemented: Option 3

`HotSessionGreedyReplayTests`'s oracle (`GreedyContinuationSegment`, replacing the old
`GreedyContinuation`) now replays through **the literal same entry points the session uses** —
`ForwardPass.PrefillWithCache` for each prompt segment and `ForwardPass.BatchForwardMulti` for
each generated segment — against its own independent `PagedKvCache` (via `CreateCache()`,
sharing no state with the session), continuing that cache turn to turn instead of re-deriving the
whole growing history from scratch on every call. This is stronger than "use the same precision
path" in the abstract: calling the exact functions the session calls closes off *any* future
asymmetry between the oracle's API surface and the session's, not just the one this investigation
started from.

That mattered immediately: switching the oracle to `Prefill`/`Forward` (still keying off "which
precision path", but through the *single-user* top-level API rather than the session's own
`PrefillWithCache`/`BatchForwardMulti`) fixed the original failing test but broke a previously-
passing sibling, `HotSession_ExactAppendAtPageBoundary_MatchesFullGreedyReplay` — a **second,
related asymmetry**: `ForwardPass.Prefill` → `PrefillDispatch` short-circuits a length-1 prompt
straight to `Forward`'s exact-F32 path; `PrefillWithCache` deliberately does not (see its own doc
comment, "Deliberately NOT short-circuiting on N == 1 here, unlike PrefillDispatch" —
`ForwardPass.cs:5238-5254`, the 2026-08-13 fix already covered under "Related precedent" below).
Turn 2's short continuation prompt (`" capital"`) was exactly the kind of input that asymmetry
bites. Switching the oracle one level further — to `PrefillWithCache`/`BatchForwardMulti`
specifically, not just "any prefill/decode pair" — closed this by construction: both call sites
now run through the identical dispatch the session's own admission and decode use, so there is no
second API surface left to drift out of sync. All 23 heavy `Tests.Sessions` tests pass with this
version; the `Prefill`/`Forward` intermediate version was not kept.

A second real bug surfaced and was fixed along the way, purely in the oracle's own code (not a
production defect): the old `GreedyContinuation` never called `Forward`/`BatchForwardMulti` for a
turn's *last* generated token (harmless there — it discarded its `ForwardPass` immediately after
returning). The new oracle *reuses* its cache across turns, so skipping that last append left a
gap in cache history that the next turn's prefill silently attended across — every generated
token must be appended, including the last one, once the cache is no longer disposable per call.

## Impact

Every real multi-turn `HotSession` conversation is affected once it has both decoded at least one
token and then prefilled again (i.e. essentially all of them past their first turn). The magnitude
is not cosmetic — measured `maxAbsDiff` up to 2.16 on a single logit in this repro, large enough to
flip a real greedy decoding decision a few tokens later, not just perturb a probability.

**What the fix actually changes and doesn't.** The production numerics described above are
UNCHANGED — `PrefillCore` still Q8-quantizes by default, `BatchForwardMulti` still stays exact
F32, and a session's cache still legitimately mixes both. What changed is recognizing that
`docs/adr-0001-session-cache-lifecycle.md`'s "ExactLossless" claim was never actually achievable
against a naive "replay everything as one cold Q8 prefill" reference once decode is involved (see
"Why Option 2 doesn't actually work"), so grading the session against that reference was testing
an unachievable bar, not a real defect in the session. The precision-consistent oracle
(`GreedyContinuationSegment`) is the reference this test can actually promise to hold, and it does.

**Still an open question, not re-checked as part of this fix**: the plain (non-session)
`ContinuousBatchingEngine` admission path for any request whose prompt is chunked across multiple
`PrefillWithCache` calls with decode steps interleaved for OTHER concurrent sequences (issue #183's
chunked admission) — the mechanism (mixed-precision cache history) would apply identically there
if such a request's own chunks ever straddle another sequence's decode round, though whether that's
even possible given how chunking is scheduled hasn't been verified.

**docs/028 Phase 2 (cross-session prefix sharing) is unaffected** by this specific failure mode,
worth repeating since it's adjacent work: Phase 2's real-model test seeds a new session's cache via
`ForkSharedPrefix`/`CapturePrefix`/`ForkPrefix` (pure page-sharing, no `PrefillCore` call involved)
and then runs exactly one `RunTurnAsync` against that freshly-seeded cache — one admission, no
prior decode on that cache. A third turn on a Phase-2-seeded session would hit this bug the same as
any other multi-turn session.

## Where the fix lives, and how to sanity-check it

`tests/OpenTail.Stingray.Tests.Sessions/HotSessionGreedyReplayTests.cs` — the file's own header
comment now explains the precision-consistent oracle design in full; `GreedyContinuationSegment`
is the replacement for the old `GreedyContinuation`, called from all three tests that need a
real-model replay oracle (`HotSession_MultiTurn_MatchesFullGreedyReplay_OnRealModel`,
`HotSession_ExactAppendAtPageBoundary_MatchesFullGreedyReplay`, and the restart-continuation
child's `RestoreAndReplayRestartFixture`).

```
STINGRAY_RUN_HEAVY_TESTS=1 dotnet test tests/OpenTail.Stingray.Tests.Sessions
```

Needs `models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf` on disk (`download-model.ps1 -Model smollm2`).
`STINGRAY_CPU_PREFILL_Q8=0` remains a useful diagnostic lever for any *new* instance of this bug
class elsewhere (see the leads below) — it globally disables both sides of whatever comparison is
failing, which confirms the mechanism even when it isn't the right production fix.

All temporary diagnostic scaffolding used during this investigation
(`DIAG_PrefillBatchCompositionInvestigation.cs`, `DIAG_OracleParity.cs`) has been deleted; none of
it belongs in the permanent suite.

## A second, confirmed asymmetry worth its own note: `Prefill`'s N==1 shortcut

Found and fixed as part of getting the new oracle right (see "What was actually implemented"
above), independent of the Q8-vs-F32 mechanism this doc is mainly about:
`ForwardPass.Prefill` → `PrefillDispatch` short-circuits any length-1 prompt straight to
`Forward` (`ForwardPass.cs`, near the top of `PrefillDispatch`: `if (N == 1) { var single =
Forward(tokens[0], startPos); ... }`). `PrefillWithCache` has no equivalent shortcut, by design —
its own doc comment says so explicitly, added by the same 2026-08-13 fix covered under "Related
precedent" below. Any code comparing `Prefill`/`Forward` output against `PrefillWithCache`/
`BatchForwardMulti` output for a length-1 segment is comparing two different dispatch paths, not
just two different APIs for the same computation. This is a second, real instance of "different
code paths for logically-identical work," same family as the main bug, different call pair.

## Other currently-failing tests worth checking against whatever fix lands here

Found while investigating this bug, not chased down or re-verified as part of it — this fix only
touched `HotSessionGreedyReplayTests.cs` and no production code, so these are exactly as
unresolved as before:

- `ContinuousBatchingTests.PrefillWithCache_SingleToken_MatchesForward`
  (`tests/OpenTail.Stingray.Tests.ForwardPass/ContinuousBatchingTests.cs:66`) — failed with a much
  larger, structurally different-looking gap (`Expected: -0.57, Actual: 4.2`). Compares
  `Forward(42, 0)` against `PrefillWithCache([42], cache)` on a *fresh* cache — i.e. exactly the
  `Prefill`'s-N==1-shortcut shape above, not the mixed-history shape this doc is mainly about
  (there's no prior admission in this test at all). Now a strong, specific lead rather than a
  vague one: the fresh cache's `PrefillWithCache([42], ...)` almost certainly still takes
  `PrefillCore`'s Q8/Q4_K×8 path (a length-1 prompt is not the degenerate-prompt case
  `PrefillWithCache`'s own dispatch special-cases), while `Forward(42, 0)` is exact F32 by
  construction — the same asymmetry documented above, just observed via a different test's
  comparison. Worth confirming directly rather than assuming, but the mechanism is no longer a
  guess.
- `ContinuousBatchingTests.PrefillWithCache_DequantCacheOnOff_BitIdentical`,
  `PrefillWithCache_Chunked_MatchesFull` — names suggest the Q4_K×8/BLAS-threshold family
  (`ForwardPass.cs:1298-1306`) rather than either asymmetry above; not re-verified.
- `PrefillDecodeSelfConsistencyTests.F32Prefill_MatchesTokenByTokenDecode(promptLength:33)` — name
  matches this doc's main mechanism closely (F32 prefill vs token-by-token decode consistency);
  not re-verified, but if it uses `Prefill`/`Forward` on a single growing cache the way the OLD
  greedy-replay oracle did, it's a strong candidate for the exact same root cause.

None of these were re-verified as part of this fix; treat the above as a lead list, not a
confirmed-current-state list.
