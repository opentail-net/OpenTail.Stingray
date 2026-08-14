# TODO: delete `InferenceSession`/`InferenceRuntime` and whatever Phase 0 marked redundant

**Status: not started. This is a TODO, not a plan — the plan is already written, in
`docs/028-inference-session-to-hotsession-migration-plan.md`'s "End state" section. This doc exists
so the pending deletion has its own trackable home instead of being buried at the bottom of a
migration plan whose own header now says "done, verified" for all three phases.**

## Why this is safe to do now

`docs/028`'s stated precondition — "Once Phases 1-3 are done and each has a passing, real test
behind it" — is met:

- Phase 1 (KV memory governance): done, verified.
- Phase 2 (cross-session prefix sharing): done, verified.
- Phase 3 (fork/branching): done, verified.

Every genuinely novel capability `InferenceSession`/`InferenceRuntime` had has been ported onto
`HotSession`'s real, production-wired foundation, each with its own real-model test:
`SessionResourceBudgetEvictionTests.cs` + `HotSessionTests.cs`/`HotSessionRollingReservationTests.cs`
(Phase 1), `CrossSessionPrefixForkingTests.cs` + `CrossSessionPrefixSharingRealModelTests.cs`
(Phase 2), `HotSessionForkTests.cs` + `HotSessionForkRealModelTests.cs` (Phase 3).

## What to delete

From Phase 0's audit (`docs/028`, "Phase 0 — Redundancy audit" section) — final tally was 2
confirmed novel (now ported), 3 confirmed redundant or alternate-design, 1 partial:

- **`InferenceSession.cs`, `InferenceRuntime.cs`** and their supporting types — the entire
  `InferenceSession`/`IInferenceSession` architecture. Confirmed zero references from
  `src/OpenTail.Stingray.Server` at the time of the original audit — re-confirm with a fresh grep
  before deleting, in case something changed since.
- **`ISessionStore.cs`, `FileSessionStore.cs`** — confirmed redundant with the pre-existing
  `FileSessionJournal`/`SessionStateCodec` that already do this job for `HotSession`.
- **`InferenceSessionGrammarExtensions.cs`** — confirmed redundant with
  `ChatTemplate.BuildToolArgumentConstraint`/`BuildForcedToolCallConstraint`, already
  architecture-independent at the Server layer.
- **`ResponseContinuationToken.cs`, `StaleContinuationException.cs`, `SessionDelta.cs`,
  `SessionDeltaWireCompressor.cs`** — classified as an alternate API design for a problem
  `HotSession` already solves differently (`expected_revision` + `SessionContinuationDiagnostic`),
  not a gap. Delete unless a deliberate decision is made to adopt the token-based shape instead —
  that would be a real design choice, not a byproduct of this cleanup, and hasn't been made.
- **`CrossSessionPrefixSynthesizer.cs`, `IActiveSessionRegistry.cs`,
  `InMemorySessionManager.cs`** (or whatever the actual registry file is named) — superseded by
  Phase 2's `HotSession`-native scan (`HotSessionRuntime.CreateWithSharedPrefixHint`, using
  `HotSessionRuntime`'s own `_sessions` dictionary). Note:
  `docs/bugstofix.md` flagged a real bug in `CrossSessionPrefixSynthesizer.cs` (hardcoded
  `("default-model","default-kv")` namespace) — that finding becomes moot once the file is deleted,
  not something to fix first.
- **`SessionBranchingExtensions.cs`** and any other `InferenceSession`-only branching code —
  superseded by Phase 3's `HotSessionRuntime.Fork`.
- **~20+ test files** exercising the above (`KvMemoryGovernorTests.cs`, `SessionBranchingTests.cs`,
  `ForkAndVoteTests.cs`, and others per Phase 0's original enumeration) — these were real, valuable
  oracles *during* migration (see `docs/028`'s "silver lining" section); once each phase's
  `HotSession`-native test exists and passes, the `InferenceSession`-side test has no further job.

## What NOT to delete without a separate decision

- **`GenerationResult.cs`, `GenerationStream.cs`, `FinishReason.cs`** — Phase 0 marked these
  "partial": `HotSessionTurnResult` exists but doesn't bundle `FinishReason`/`ToolCalls`/
  `ContinuationToken`/`Metrics` into one record the way `GenerationResult` does. Check whether
  anything ported during Phases 1-3 ended up depending on this shape before deleting it — if
  nothing does, it's likely safe, but this wasn't explicitly re-verified when Phase 0 was closed.
- **`ISessionMetadata.cs`, `SessionMetadata.cs`, `ISessionMetrics.cs`, `SessionMetrics.cs`,
  `SessionMetricsSnapshot.cs`** — Phase 0 confirmed these **novel** (no `HotSession` equivalent
  exists). Deleting `InferenceSession` without first deciding whether `HotSession` needs
  session-scoped metadata/metrics would silently drop a capability, not just remove dead code. Port
  first, or explicitly decide the capability isn't needed, before deleting.

## How to execute this when picked up

1. Fresh grep for any reference to the types above from `src/OpenTail.Stingray.Server` or anywhere
   else outside `OpenTail.Stingray.Sessions` and its own tests — confirm the zero-production-
   references premise still holds (things may have changed since the original Phase 0 audit).
2. Decide explicitly on the two "not without a separate decision" items above — port, deliberately
   drop, or defer with a reason — rather than deleting by default.
3. Delete the confirmed-redundant files and their tests together, in one change, so the diff reads
   as "remove a superseded system" rather than a series of partial, confusing removals.
4. Run the full test suite (not just Sessions-scoped) to catch any reference this doc's audit
   missed.
5. Update `docs/028`'s "End state" section to record that this happened, and update
   `docs/00-current-work.md` to drop the pointer to this doc once done.
