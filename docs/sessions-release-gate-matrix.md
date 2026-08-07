# Sessions release gate — conformance matrix

**Assessed:** 2026-08-07 against `tests/OpenTail.Stingray.Tests.Sessions` (79 test methods, 12
files, all passing). This maps the seven dimensions the release gate calls for onto what is
actually asserted, and records one dimension that is **not covered** despite appearing so. A second was initially
recorded as uncovered and is corrected at the foot of this document — read that before citing this
table.

## Matrix

| Dimension | Status | Evidence |
|---|---|---|
| Hot reuse | **Covered** | `HotSessionRuntime_CreateAndOpenByAddress_RoutesToCorrectSession`, `CowSessionSnapshot_BranchAndMerge_PreservesParentBlocks`, prefix/resume cases (31 name matches) |
| Persistence | **Covered** | `FileSessionManifest_SaveAndLoad_PreservesMetadata`, `SegmentPackStore_SaveAndLoadBlock_VerifiesChecksum` (14) |
| Restart | **Covered** | `FileSessionJournal_AppendsAndRecoversRecords`, `FileSessionJournal_TornWriteWithAbsurdLength_RecoversPriorRecordsInsteadOfThrowing` |
| Corrupt packs | **Covered** | `ColdSession_Open_RejectsACorruptPersistedKvPack`, `..._RejectsACorruptPersistedOperationLedger` (10) |
| Quotas / eviction | **Covered** | `ColdSession_WithPagedKvCache_EvictsToDiskAndRestoresExactKv`, `EvictToDisk_ReEviction_ReclaimsPacksTheNewManifestNoLongerReferences` (14) |
| Rollback | **Covered** (2026-08-07) | `HotSessionRollbackTests`, via a new test seam — see the resolution at the foot |
| Multi-model routing | **Covered** | `SessionModelBudgetTests` (4 tests, all passing) — see the 2026-08-07 correction below |

## Why two dimensions read as covered but are not

A keyword scan over test *names* reports all seven green. That is the trap: name matching produces
false coverage, and both gaps below were originally scored "COVERED" by it.

- **Rollback** matched `FileSessionJournal_TruncatesCorruptedTrailingPayload` — on the substring
  "truncat". That test is journal corruption recovery; it has nothing to do with rolling back a
  session turn.
- **Multi-model routing** matched `HotSessionRuntime_CreateAndOpenByAddress_RoutesToCorrectSession`,
  which routes to the correct **session**, not to a different **model**.

Verifying against production symbols rather than test titles is what separated them.

### Gap 1 — turn rollback

`RetainedSequenceState.RollbackLastTurn` (RetainedSequenceState.cs:175) has exactly one caller:
`HotSession.CompensateUncommittedTurn` (HotSession.cs:226). No test in the repository references
either symbol.

Two properties make this the worst dimension to leave untested:

1. It is an **error path**. It runs only when a turn failed, was cancelled, or was left uncommitted
   — i.e. when the system is already in a degraded state and correctness matters most.
2. The call is wrapped in `try { ... } catch { /* ... */ }` which **swallows every exception**, with
   the comment "A failed rollback discards the cache and allows a fresh state on the next turn."
   That may well be the right recovery policy, but it means a rollback that silently stops working
   produces no signal anywhere — no test, no log, no metric.

Suggested coverage: drive a turn that fails mid-generation, assert the cursor returns to the prior
`SessionCursor`, that `MaterializedPosition` and resident-bytes accounting return to their
pre-turn values, and that the next turn on the same session produces the same output as if the
failed turn had never run.

### Gap 2 — multi-model routing

`modelKey` is threaded through the public surface — `ColdSessionRuntime.Create(sessionId, modelKey)`
→ `HotSessionRuntime.Create` → `HotSession._modelKey`, defaulting to `engine.ModelId` — and has
**zero** test references anywhere in the repository.

Suggested coverage: create two sessions with distinct `modelKey` values in one runtime, assert
lookups do not cross, that a session created under one key is not returned to a caller asking under
another, and that persistence round-trips the key so restart cannot silently merge two models'
state.

## Caveat on this assessment

Coverage here is judged by symbol reference and test reading, not by line coverage instrumentation.
A dimension marked "Covered" means at least one test genuinely exercises it, not that its edge cases
are complete. The two gaps are the confident findings; the five "Covered" rows are a floor, not a
ceiling.

## Follow-up 2026-08-07 — rollback is untested *and* not reachable from a test

An attempt to write the rollback test found the gap is deeper than missing coverage.

`CompensateUncommittedTurn` only calls `RollbackLastTurn` when `generationCompleted` is true. Tracing
what can fault **after** that flag is set and **before** `operationCommitted`:

| Failure point | Injectable from a test? |
|---|---|
| `BuildNextCursor` invariant checks (`TurnStartPosition`/`MaterializedPosition` disagree) | Only by driving the engine into a state it is designed never to reach |
| `_store.Transition(...)` | **No** — `_store` is a concrete `InMemorySessionStore`, not an interface |
| `reservation.Complete(...)` | **No** — internal budget object |
| `_store.Complete(...)` | **No** — same concrete store |

Cancellation does not reach it either. The `OperationCanceledException` catch runs the same
compensation, but a token cancelled during generation throws out of the `await foreach` while
`generationCompleted` is still **false**, so the rollback branch is skipped; and after generation
completes there is no further token observation to throw on.

So `RetainedSequenceState.RollbackLastTurn` is reachable only via a genuine internal fault — an OOM,
or a bug in the store's state machine. It has, as far as this repository can demonstrate, **never
executed**, and its one call site swallows every exception it might raise.

### Recommendation

Do not write a test that contorts the engine into an invalid state to reach this — such a test
pins the contortion, not the behaviour. Add a narrow seam instead, then cover the path honestly.
The smallest change that would work: give `HotSession` its store through an interface (or an
`internal` delegate hook used only by tests) so a test can fail `Transition`/`Complete` at a chosen
point and assert that the cursor, `MaterializedPosition`, and resident-byte accounting all return
to their pre-turn values, and that the next turn produces the same result as if the failed turn had
never run.

Until that seam exists, this dimension of the release gate cannot be evidenced, and the matrix
should say so rather than carry an aspirational tick. The honest status is **"not covered, and not
coverable without a production change"** — which is a stronger reason to make the change than any
amount of missing-test bookkeeping.


## CORRECTION 2026-08-07 — multi-model routing IS covered

The "Gap 2" entry above was **wrong** and is retracted. `SessionModelBudgetTests` covers the
dimension with four passing tests:

- `SessionModelBudget_EnforcesPerModelPartitionLimits` — `model-a` capped at 1,000 bytes against
  `model-b` at 10,000, asserting `GetModelResidentBytes` attributes usage to the right model.
- `SessionModelBudget_RejectsOverBudgetTurnForSpecificModel` — a cap on one model does not constrain
  the other.
- `SessionModelBudget_ConcurrentInFlightReservations_EnforcesModelCap`
- `SessionModelBudget_RenewalCannotExceedModelCapByItsOwnReservation`

Routing is expressed through `SessionAddress(tenant, role, thread, model)`, and the model dimension
carries a real per-model resource partition rather than being a bare lookup tag.

### How the false negative happened, since it is instructive

The first pass scored dimensions by matching **test names**, which produced a false POSITIVE for
this dimension (a test that routes to the correct *session* matched "route"). Correcting for that,
the second pass grepped for the production **symbol** `modelKey` — and found zero test references,
producing a false NEGATIVE. The tests never write that token: they use `SessionAddress(..., "model-a")`
and `GetModelResidentBytes("model-a")`. Eight occurrences of `model-a`, zero of `modelKey`.

So both passes were wrong, in opposite directions, for the same underlying reason: **a proxy was
substituted for reading the tests.** Test names are a proxy for behaviour; an implementation
parameter name is a proxy for a concept. Neither survives contact with code that is spelled
differently from the searcher's expectation. The only method that worked was opening the file.

The rollback gap recorded above was found by reading call sites and remains correct — it was
verified by tracing every fault point, not by a keyword search.


## RESOLVED 2026-08-07 — rollback is now covered

The seam recommended above was added and the dimension is covered. `HotSession` gained one
test-only hook, `FaultBeforeCommitForTests`, invoked immediately before the commit — the only point
at which `generationCompleted`, `cursorPublished` and `resourcesFinalized` are all set, and
therefore the only state in which `CompensateUncommittedTurn` runs its full body. It is null on
every production path.

`HotSessionRollbackTests` covers it with two cases, and **only one of them actually tests rollback**:

| Test | Covers rollback? |
|---|---|
| `FailedTurn_RestoresCursorAndRevision_LeavingNoTraceOfTheFailedTurn` | **No** — cursor restore is a separate compensation branch |
| `TurnAfterAFailedTurn_MatchesTheTurnThatWouldHaveFollowedSuccess` | **Yes** |

That split was established by **mutation testing**, not by inspection: commenting `RollbackLastTurn`
out of the production path makes the second test fail while the first still passes. Without that
check the pair would have looked like belt-and-braces coverage while half of it was inert — the
same vacuous-green pattern this document already records twice.

Two details are load-bearing and should survive any tidy-up:

- The fake's `Assert.Equal(startPos, retained.LogicalPosition)` in `PrefillWithCache`. A first draft
  omitted it; without it a rollback that silently did nothing still passes both tests, because
  nothing else notices a KV cache left advanced.
- The comparison against a **control run** (same sequence without the injected fault) rather than
  hard-coded expected counts. It asserts the failed turn left no trace, which is the actual
  contract, instead of pinning numbers that would drift.

**Result: the rollback path executed for the first time and behaves correctly** — the cache rewinds,
the cursor restores, and the following turn is indistinguishable from one that never saw a failure.
The production catch block still swallows exceptions from rollback, which remains worth revisiting,
but the path is now exercised rather than assumed.
