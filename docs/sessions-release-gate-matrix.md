# Sessions release gate — conformance matrix

**Assessed:** 2026-08-07 against `tests/OpenTail.Stingray.Tests.Sessions` (89 test methods, 12
files, all passing). This maps the seven dimensions the release gate calls for onto what is
actually asserted, and records two dimensions that are **not covered** despite appearing so.

## Matrix

| Dimension | Status | Evidence |
|---|---|---|
| Hot reuse | **Covered** | `HotSessionRuntime_CreateAndOpenByAddress_RoutesToCorrectSession`, `CowSessionSnapshot_BranchAndMerge_PreservesParentBlocks`, prefix/resume cases (31 name matches) |
| Persistence | **Covered** | `FileSessionManifest_SaveAndLoad_PreservesMetadata`, `SegmentPackStore_SaveAndLoadBlock_VerifiesChecksum` (14) |
| Restart | **Covered** | `FileSessionJournal_AppendsAndRecoversRecords`, `FileSessionJournal_TornWriteWithAbsurdLength_RecoversPriorRecordsInsteadOfThrowing` |
| Corrupt packs | **Covered** | `ColdSession_Open_RejectsACorruptPersistedKvPack`, `..._RejectsACorruptPersistedOperationLedger` (10) |
| Quotas / eviction | **Covered** | `ColdSession_WithPagedKvCache_EvictsToDiskAndRestoresExactKv`, `EvictToDisk_ReEviction_ReclaimsPacksTheNewManifestNoLongerReferences` (14) |
| **Rollback** | **NOT COVERED** | see below |
| **Multi-model routing** | **NOT COVERED** | see below |

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
