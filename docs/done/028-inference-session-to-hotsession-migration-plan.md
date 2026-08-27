# Reconciling `InferenceSession` into `HotSession` — Migration Plan

## Why this exists

The repository currently contains two independent, non-communicating session
architectures:

- **`HotSession` / `HotSessionRuntime` / `ContinuousBatchingEngine`** — present before the
  recent growth wave, wired into every production entry point that matters:
  `src/OpenTail.Stingray.Server/Endpoints/SessionEndpoints.cs` (the real `/v1/sessions` HTTP
  handler), `SessionRuntimeRelay.cs`, `InferenceEngineLoader.cs`. This is what an actual API
  request touches. It does continuous batching (real, shipping, multi-user throughput).
- **`InferenceSession` / `InferenceRuntime`** — added since, entirely new (confirmed against
  `priorworkingstate`: none of it existed before). ~1,490 lines across the two main files, 20+
  dedicated test files, genuinely sophisticated (pressure-based eviction, CoW forking, session
  trees, cross-session prefix sharing). **Zero references anywhere in `src/OpenTail.Stingray.Server`.**
  It has never served a real request and does not support continuous batching at all.

This split was not the plan — and it's better-documented than it first looked. Two design
lineages exist in `docs/`, and they disagree with each other:

- **`docs/002`/`003`** ("Native Sessions Implementation Plan" / "Inference State Architecture &
  Transition Plan") propose an ambitious from-scratch redesign: a `Runtime` owning independent
  per-session state (`Session A/B/C`, each with its own KV state), explicitly listing
  "Multi-session batching" as something this architecture *would eventually enable* — i.e., not
  yet solved, a future goal of the redesign.
- **`docs/adr-0001-session-cache-lifecycle.md`** (explicitly dated 2026-08-02, the newer of the
  two) directly considered that shape of solution — a separate, session-aware batching engine —
  and **rejected it by name**: *"Separate `SessionBatchingEngine`. Rejected because it would copy
  complex scheduling behavior which is already exercised in `ContinuousBatchingEngine`."* The
  accepted decision was to extend `ContinuousBatchingEngine` itself with a narrow, explicit
  cache-retention contract (`RetainedSequenceState`, `GenerateRetainedChunksAsync`) — which is
  `HotSession`'s actual lineage.
- **`docs/006`** (KV Memory Governor) and **`docs/008`** (Zero-Copy Session Branching) are
  consistent with the ADR, not with 002/003 — both explicitly instruct building *against the
  existing session lifecycle* ("use the existing session lifecycle state rather than inventing
  another independent state machine"; "integrate with existing session suspension/resumption").

So `HotSession` isn't just the more battle-tested option — it's the implementation of an
explicit, accepted architecture decision that directly evaluated and rejected the
`InferenceSession` approach, for a stated, still-valid reason (duplicating
`ContinuousBatchingEngine`'s complex scheduling behavior). `InferenceSession`/`InferenceRuntime`
reads as an implementation of the earlier, superseded 002/003 vision, built without apparent
awareness that the project had already closed that door. The individual features it implements
are real and valuable (its governor is exactly what today's investigation independently found
`HotSession` is missing) — the shape it was built in is what's wrong, not the ideas.

## The silver lining

This is not a rewrite from a design doc. `InferenceSession`'s code compiles, runs, and has 20+
passing test files behind it today — a working reference implementation and a working oracle,
not just prior art on paper. That materially lowers the cost of every phase below:

- Algorithm logic (pressure-threshold checks, eviction ordering, branch-vote aggregation) can
  likely be adapted close to directly even where the underlying storage substrate differs
  (`CpuKvCache`'s shared pool vs. `PagedKvCache`'s per-session allocation with its own
  `_pool`/`.AddRef()` primitive) — the hard thinking already happened once.
- The existing test files (`KvMemoryGovernorTests.cs`, `SessionBranchingTests.cs`,
  `ForkAndVoteTests.cs`, ...) are a real oracle for "what should happen" when writing the
  `HotSession`-targeted equivalents, not something to invent from scratch.
- Where Phase 0 finds a file is genuinely redundant with something `HotSession` already has, that
  redundant `InferenceSession` code still served a purpose: independent confirmation, from a
  different implementation, that the existing `HotSession` behavior is the right behavior.

## Decision

Do not migrate the production path onto `InferenceSession` — see the earlier discussion for why
(unproven under real load, no continuous batching, and "swap the foundation to gain features" is
the same shape of risk that produced today's KV-cache bug, just with a much bigger blast radius).

Instead: **port each genuinely novel capability from `InferenceSession` onto `HotSession`'s real
foundation**, one bounded phase at a time, each verified against something real (not just unit
tests against fakes) before moving to the next. Retire `InferenceSession` only once nothing in
it is uniquely valuable anymore.

## Phase 0 — Redundancy audit (do first, cheap, prevents wasted work)

Not every `InferenceSession`-only file is a missing `HotSession` capability. Several groups look
like they may duplicate something `HotSession`/the Server layer already does under a different
name (`HotSession` already streams tokens back through the real API, already persists via
`SessionStateCodec`/`FileSessionJournal`, and tool/grammar support already exists at the Server
layer independent of sessions). Before committing engineering time to porting these, classify
each as **novel** or **redundant**:

| File(s) | Claimed purpose | Status |
|---|---|---|
| `GenerationResult.cs`, `GenerationStream.cs`, `FinishReason.cs` | Structured streaming result | **Partial.** `HotSessionTurnResult` already exists (`Operation`, `Cursor`, `Chunks`, `IsIdempotentReplay`) but doesn't bundle `FinishReason`/`ToolCalls`/`ContinuationToken`/`Metrics` into one record the way `GenerationResult` does. Whether `GenerationStream`'s `IAsyncEnumerable` wrapping is itself novel needs a closer look when this phase starts. |
| `ISessionMetadata.cs`, `SessionMetadata.cs` | App-level metadata bag on a session | Still unverified — not checked this pass. |
| `ISessionMetrics.cs`, `SessionMetrics.cs`, `SessionMetricsSnapshot.cs` | Session-level metrics | **Confirmed novel.** Grep-verified: zero reference to session-scoped metrics anywhere in `HotSession.cs` or `SessionEndpoints.cs`. |
| `ISessionStore.cs`, `FileSessionStore.cs` | Persistent session storage | **Confirmed redundant.** `FileSessionJournal`/`SessionStateCodec` predate the growth wave and already do this for `HotSession`. `FileSessionStore` is a second, parallel implementation of the same job. |
| `InferenceSessionGrammarExtensions.cs` | Schema-constrained generation on a session | **Confirmed redundant.** `ChatTemplate.BuildToolArgumentConstraint`/`BuildForcedToolCallConstraint` already provide this at the Server layer, architecture-independent — usable by `HotSession` today without this file. |
| `ResponseContinuationToken.cs`, `StaleContinuationException.cs` | Versioned continuation handle, no-silent-rewind | **Alternate design, not a gap.** `HotSession` already solves "safe resumption without silent rewind" — via caller-supplied `expected_revision` + `SessionContinuationDiagnostic`/`DiagnoseContinuation`, not an opaque encoded token. Two different API shapes for the same solved problem; not something to port, a design choice to make deliberately if the token-based shape is preferred over the revision-based one. |
| `SessionDelta.cs`, `SessionDeltaWireCompressor.cs` | Incremental delta + wire compression | **Same bucket as the continuation token above, not standalone.** `SessionDelta` is defined directly in terms of `ResponseContinuationToken` (`BaseToken`/`ResultToken` fields) — it's the wire-delta format for the token-based continuation design, not an independent capability. Tied to the same "different API shape for an already-solved problem" classification. |
| `ISessionMetadata.cs`, `SessionMetadata.cs` | App-level metadata bag on a session | **Confirmed novel.** Grep-verified: zero reference to session-scoped metadata anywhere in `HotSession.cs` or `SessionEndpoints.cs`. `HotSession` has no way for a host application to attach arbitrary key-value context (user identity, workflow phase) to a session today. |

Phase 0 audit is complete. Final tally: 2 confirmed novel (`SessionMetrics`, `SessionMetadata`),
3 confirmed redundant or alternate-design (persistence, grammar extensions, continuation-token
family), 1 partial (`GenerationResult`/`GenerationStream`/`FinishReason` — `HotSessionTurnResult`
exists but doesn't bundle the same fields; needs a closer look if/when ported). Nothing gets
ported until Phase 2/3 actually need it; the "confirmed redundant" ones are staying exactly where
they are (see "The silver lining" above for why that's not wasted effort).

## Phase 1 — KV memory governance (highest confidence, already scoped)

**Correction to the original seam investigation**, found while starting this phase: `HotSession`
does *not* have zero governance. `HotSessionRuntime`/`SessionResourceBudget` (pre-existing, not
part of the growth wave) already tracks total resident bytes — globally and per-model — inclusive
of idle-retained sessions, and correctly rejects new admission
(`SessionResourceBudgetExceededException`) once a configured budget is exhausted. That part is
real and correctly designed. What's actually missing is narrower: `SessionResourceBudget`'s own
doc comment states it outright — *"It does not evict: a session with running work keeps its
reservation until the operation commits, rolls back, or fails."* There is no automatic reclaim
from idle sessions under pressure. The system fails closed (hard-rejects new work) instead of
freeing space the way `KvMemoryGovernor`'s pressure-based reclaim does for the other
architecture.

- The work is an eviction/reclaim policy layered on top of `SessionResourceBudget`, not a
  ground-up budget-tracking system (that part already exists and is well-designed — checked/
  reserved/committed bytes, per-model sub-budgets, rollback-safe reservation objects). Port
  `KvMemoryGovernor`'s reclaim *policy* (what counts as reclaimable, eviction ordering under
  pressure) onto this existing tracker.
- `ContinuousBatchingEngine`'s own `_committedTokens`/`_kvTokenBudget` gate is unrelated and
  correct as-is (in-flight tokens only, by design) — this phase doesn't touch it.
- **Verification**: characterization test — multiple `HotSession`s sharing one
  `HotSessionRuntime`, driven to fill `SessionResourceBudget`'s configured limit via idle-retained
  reservations, then assert what happens to a new admission attempt. Today: hard rejection via
  `SessionResourceBudgetExceededException`, always, regardless of whether any idle session could
  safely be reclaimed. Write that assertion first (proving the precise, corrected gap), then
  implement reclaim and watch the same test's expected outcome change.

## Phase 2 — Cross-session prefix sharing (done, verified)

**Correction to the original scoping**, found while starting this phase: `CrossSessionPrefixSynthesizer`
cannot simply be pointed at `HotSessionRuntime` — it's built entirely against `IInferenceSession`
(`session.TokenHistory`, `.KvSequence`, `.ModelId`, `.State`), which `HotSession` does not
implement and should not be made to implement (that would merge the two architectures the rest of
this plan deliberately keeps apart). The obvious "just expose the cache" fix is also blocked by
design: `docs/adr-0001-session-cache-lifecycle.md`'s guardrails state outright — *"The public
session API cannot expose a concrete `PagedKvCache` type."*

**Second correction, found immediately after**: while looking for where a new capability
interface should live, `ContinuousBatchingEngine` turned out to already have a full, tested,
wired cross-request prefix cache (`IPrefixCacheableBatchedForwardPass.CapturePrefix`/`ForkPrefix`,
an LRU list, hit/miss/eviction counters — issue #183-era code) that the CPU `ForwardPass` already
implements. It's just explicitly disabled for retained-session admissions (`req.RetainedState is
null` gates it off), because relaxing that gate to serve retained turns directly would mean
computing `RetainedSequenceState`'s `TurnStartPosition` as an *output* of admission instead of an
*input* the session's cursor supplies beforehand — which is precisely the invariant a nearby
diagnostic guard exists to protect, added after a real hot-session replay divergence bug. So the
actual gap was never "no forking primitive exists" — it's "the existing primitive is reachable
only from the session layer's own seeding step, before a turn starts, not from inside admission."

The shipped design routes around that constraint entirely rather than touching the fragile
admission-time path, and needed no new capability interface at all — just two internal methods
reusing the engine's own `IPrefixCacheableBatchedForwardPass`:

- `RetainedSequenceState.TryForkSharedPrefix(IPrefixCacheableBatchedForwardPass, int)` (Engine,
  internal) — runs entirely under the handle's own lock (atomic with `Reserve()`, so it can't race
  a concurrent turn start), aligns down to `PrefixCacheBlockSize`, and calls `CapturePrefix` then
  `ForkPrefix` — the exact two calls `ContinuousBatchingEngine.RetainPrefix` already makes for its
  own cache, just used once instead of kept for repeated reuse.
- `RetainedSequenceState.SeedWithForkedCache(...)` (Engine, internal) — seeds a never-used handle
  as if a turn had already materialized the forked length. This is the genuinely new piece; nothing
  before this filled a `RetainedSequenceState` from anything but a real completed turn or
  deserialized bytes (`ImportState`/`RestoreKvBytes`).
- `ContinuousBatchingEngine.TryForkSharedPrefix(RetainedSequenceState, int)` — thin pass-through
  exposing the above through the engine reference `HotSession` already holds.
- `HotSession.TryForkSharedPrefixCache`/`SeedFromSharedPrefix` (Sessions, internal) — source- and
  destination-side accessors. Token-history matching needed no new surface at all:
  `HotSession.Cursor.ExecutionLog` was already public and gives exact, prefix-comparable token
  identity — the `IActiveSessionRegistry` dependency flagged below was never actually load-bearing
  for this.
- `HotSessionRuntime.CreateWithSharedPrefixHint(ImmutableArray<int>, ...)` (public) — scans the
  runtime's existing `_sessions` dictionary (already there for Phase 1's `ReclaimIdleBytes`) for
  the idle sibling with the longest common token prefix, forks, seeds, and returns how many leading
  tokens of the caller's intended prompt were seeded — best-effort throughout; any failure reason
  (no match, backend doesn't support forking, budget rejects the seeded size) falls back to an
  ordinary cold session, never to session-creation failure.
- `IActiveSessionRegistry`/`InMemorySessionManager` (the `InferenceSession`-side registry) were not
  needed and were not ported — `HotSessionRuntime`'s own `_sessions` already does that job.

**Verification, both landed**: an orchestration suite against a fake `IPrefixCacheableBatchedForwardPass`
(`tests/OpenTail.Stingray.Tests.Sessions.Fast/CrossSessionPrefixForkingTests.cs` — matching,
page-alignment flooring, and every fallback-to-cold path) plus a real-model proof
(`tests/OpenTail.Stingray.Tests.Sessions/CrossSessionPrefixSharingRealModelTests.cs`) that checks
two independent things a correctness-only check can't distinguish: the engine's new
`CrossSessionPrefixTokensShared` counter proves the fork path actually ran rather than silently
falling back, and a full greedy-replay oracle (mirroring `HotSessionGreedyReplayTests`'s own
token-not-text replay reasoning) proves the shared pages compute bit-identical output to a
from-scratch prefill. Both pass on a real CPU dense model. Full Sessions.Fast/Server.Fast/ForwardPass.Fast
and the full heavy Sessions suite show no regressions; one pre-existing heavy-suite failure
(`HotSessionGreedyReplayTests.HotSession_MultiTurn_MatchesFullGreedyReplay_OnRealModel`) was
confirmed unrelated by reverting every Phase 2 file and reproducing the identical failure without
them — a real, standing issue in the original single-session retained-replay path, out of scope
for this phase. Since resolved: see docs/029-prefill-batch-composition-numerics-bug.md (root cause
was the test's own replay oracle grading the session against an unachievable reference, not a
session defect; fixed by replaying through the session's own entry points).

## Phase 3 — Fork / branching / consensus voting (done, verified)

**Confirmed novel, but materially lower-risk than originally scoped once `docs/008`/`010` were
re-read against what Phases 1 and 2 already established.** `HotSession` has zero fork support
today (grep-verified) — but the hard, novel primitive it would need already exists and is already
proven: `PagedKvCache.ForkSharedPrefix` (ref-counted `AddRef`/`Release`, same mechanism Phase 2
uses) plus `EnsureWritableBlock`'s automatic copy-on-write on first write to a shared block is
exactly the "share physical pages, CoW only on divergence" contract `docs/008` asks for, verified
against a real model in Phase 2's own test. What was missing is session-level plumbing, not
physical-layer machinery — the same pattern as Phases 1 and 2.

- **`docs/010`'s central concern (duplicating a *stateful* forward-pass instance per branch, with
  per-backend costs and GDN state flagged as the hardest tier) does not apply to `HotSession` at
  all.** `docs/010` is about `InferenceSession`'s architecture, where each session's own
  `ForwardPass` instance embeds its own internal `PagedKvCache`. `ContinuousBatchingEngine`/
  `HotSession` never had that problem: the forward pass is already one shared, stateless-per-
  sequence instance (immutable weights + lazily-populated caches only), and each session's mutable
  state already lives in an *externally*-owned `ISequenceKvCache`
  (`CreateCache()` → `PrefillWithCache`/`BatchForwardMulti(cache)`). Forking a `HotSession` means
  forking that external cache, not duplicating a forward pass.
- **Checked and closed**: whether GDN-hybrid models (`HybridGdnForwardPass`/
  `CudaHybridGdnForwardPass`, qwen35moe-style) could break this, since `docs/010` flags GDN state
  as "destructively updated per token and NOT arbitrarily rewindable" and it lives inside the
  forward-pass instance rather than externalized. Confirmed moot: neither implements
  `IBatchedForwardPass`, so neither can be wired into `ContinuousBatchingEngine`/`HotSession` at
  all today, fork or no fork — a pre-existing scope boundary, not a new risk this phase
  introduces. Fork support only needs to cover backends that actually implement
  `IBatchedForwardPass` (CPU dense today; CUDA dense once/if it implements
  `IPrefixCacheableBatchedForwardPass` the same way — not yet confirmed).
- **Checked and closed**: `docs/008`'s RNG-independence requirement (Test 10 — branches must not
  contend over one shared mutable RNG). Already true by construction:
  `ContinuousBatchingEngine.ActivateSeq` creates a fresh `Random()` per admitted request, not a
  shared instance field — no cross-sequence RNG state exists to leak between a parent and its
  forks in the first place. Per-request grammar/constraint state (`req.Sp.Constraint?.Reset()`) is
  the same story. Fork doesn't need to do anything special here.
- **The one real design decision**: `ForkSharedPrefix(prefixLength)` requires page-aligned length
  (a multiple of `PagedKvCache.PageSize`), and a session forking mid-conversation is essentially
  never sitting exactly on that boundary. Decided: fork zero-copy-shares everything up to the last
  full page; the caller's next `RunTurnAsync` call on each branch supplies whatever text covers
  the remaining (at most `PageSize - 1`-token) tail — the same "caller must encode the shared
  portion in isolation" contract Phase 2's `CreateWithSharedPrefixHint` already documents, not a
  new kind of caveat. Rejected: giving each branch an exact, immediate snapshot of the parent's
  precise current position, which would need a new token-level (not text) admission path bypassing
  `RunTurnAsync` — real, bounded, but more surface area than this phase's payoff justifies right
  now.
- **Fork semantics, per `docs/008`**: atomic and all-or-nothing (§30) — if any of `count` branches
  fails partway (source not idle, backend doesn't support forking, budget rejects a branch),
  every already-created branch in that call is torn down and its reservation/pages released before
  the failure propagates; a caller never has to distinguish "3 of 4 branches exist" from "the call
  failed." `count >= 1` validated (§16); `count == 1` still valid, same semantics as a single fork.
  Lives on `HotSessionRuntime` (`Fork(HotSession parent, int count)`), not on `HotSession` itself —
  following this codebase's own established convention (`Create`/`CreateWithSharedPrefixHint`/
  `Open`/`ImportState` are all runtime-level factory methods), which is exactly what `docs/008`
  itself instructs when it says existing API conventions should win over its own suggested shape.
  Fork and Generate stay separate (§4): `Fork` returns ready `HotSession`s; driving `RunTurnAsync`
  on them (concurrently or not) is entirely the caller's business, reusing the existing
  continuous-batching engine — no new scheduler, matching §14/§34's explicit "do not overbuild."
- **Implementation**: `HotSessionRuntime.Fork` (`src/OpenTail.Stingray.Sessions/HotSession.cs`)
  reuses `TryForkSharedPrefixCache`/`SeedFromSharedPrefix` from Phase 2 directly — no new
  engine-level code was needed, only the session-level orchestration and rollback. One design
  correction made while implementing: the first draft computed the page-aligned shared length
  itself using `PagedKvCache.PageSize` directly, a concrete-type constant that has no business in
  the Sessions layer (the same layering discipline Phase 2 already established — the backend's own
  reported `PrefixCacheBlockSize` is what must be authoritative). Fixed to pass the token-only
  prefix length as a *cap* and let `TryForkSharedPrefixCache`'s own return value report the actual
  aligned length used, matching `CreateWithSharedPrefixHint`'s existing pattern exactly.
- **Verification**: `docs/008`'s own test list (§27) was the acceptance bar. A fake-based
  orchestration suite (`tests/OpenTail.Stingray.Tests.Sessions.Fast/HotSessionForkTests.cs`, 10
  tests) covers independent branches sharing the parent's prefix, the zero-copy claim (via the
  `CrossSessionPrefixTokensShared` counter Phase 2 added), alignment flooring, `count` validation,
  cold-branch forking from a session with no materialized content, atomic rollback with no residual
  reservation when the parent isn't idle, parent/sibling isolation, and nested forking. A real-model
  test (`tests/OpenTail.Stingray.Tests.Sessions/HotSessionForkRealModelTests.cs`) forks two branches
  from a real 16-token parent state, drives each to a genuinely different continuation (the actual
  copy-on-write trigger), and checks both that the engine's shared-prefix counter fired for each
  branch and that each branch's output exactly matches a precision-consistent replay oracle.

  That real-model test caught a real bug in itself before it caught anything in the production
  code: its first version built the oracle by cold-`PrefillWithCache`-ing the parent's whole
  16-token history in one call — which silently re-Q8-quantizes whatever position the parent
  actually *decoded* (permanently F32-exact), reproducing the exact
  docs/029-prefill-batch-composition-numerics-bug.md mismatch on the shared prefix itself rather
  than testing anything about forking. Fixed by replaying the parent's own execution log segment by
  segment (prefill for prompt segments, decode-with-the-parent's-own-recorded-token for generated
  segments), the same fix shape as `HotSessionGreedyReplayTests`'s corrected oracle. Once fixed, it
  passed immediately — meaning `Fork`/CoW were correct from the start; only the test's own oracle
  had the bug. **Phase 2's real-model test (`CrossSessionPrefixSharingRealModelTests.cs`) had the
  identical latent flaw** — it happened to pass because the affected position's numeric influence
  wasn't large enough to flip that test's specific greedy choice, the same kind of coincidence that
  let `HotSession_ExactAppendAtPageBoundary_MatchesFullGreedyReplay` pass before docs/029 was found.
  Fixed with the same per-segment-replay pattern while this was fresh, rather than leaving a known
  landmine for whoever next touches it. Full heavy `Tests.Sessions` suite: 24/24. `Tests.Sessions.Fast`:
  394/394 (384 + the 10 new fork tests). `Tests.Server.Fast`/`Tests.ForwardPass.Fast`: unaffected,
  no production code outside `HotSession.cs` was touched this phase.

## End state

Once Phases 1-3 are done and each has a passing, real test behind it: `InferenceSession`,
`InferenceRuntime`, and whatever Phase 0 marks as redundant get deleted, not archived. Keeping
a parallel, unintegrated system around after its useful parts have been ported is exactly the
"two systems that don't talk to each other" problem this plan exists to close — leaving the
carcass around invites a third AI session to find it, assume it's real, and write another
`perspective-no-future.txt`.

**2026-08-27: done.** `docs/030`'s deletion executed — `InferenceSession`/`InferenceRuntime` and
the ~20 files Phase 0 marked redundant (plus `KvMemoryGovernor` and friends, its Phase-1
predecessor, and `SessionTree`/`BranchVoteResult`/`InferenceSessionConsensusExtensions`, missed by
the original file list but part of the same island) are gone. One real gap surfaced during the
deletion re-check: `SamplingParams.AllowedChoices` (constrained-choice sampling) was implemented
only inside `InferenceSession` — `HotSession`'s batched path never applied it. Rather than drop the
capability, it was ported into `ContinuousBatchingEngine`'s batched decode loop (per-sequence
`TokenChoiceTrie.ChoiceState`, mirroring the existing grammar-`Constraint` masking pattern) before
the deletion proceeded; see `HotSessionChoiceConstraintTests.cs` for the regression coverage this
replaces from `ConstrainedChoiceSamplingTests.cs`.

## Pacing

Each phase is its own session, not a checklist to run through in one sitting — same reasoning as
the original seam-investigation pacing. Phase 0 is cheap and should happen next regardless of
which phase gets prioritized after it, since it determines how much of Phases 1-3's surrounding
scaffolding is real work versus already-duplicated effort.
