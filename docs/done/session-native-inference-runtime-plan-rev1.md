# OpenTail.Stingray — session-native inference runtime plan

**Status:** implementation plan, revision 1  
**Basis:** current OpenTail.Stingray fork, informed by the SharpInference-derived session plan  
**Primary use:** persistent multi-model local review orchestration  
**Scope:** transactional, revisioned, durable and forkable inference sessions inside .NET  
**Principle:** session correctness and utility must not depend on adopting a native kernel backend

Task-list convention: leave an item unchecked until its implementation and stated verification are
both complete. A design decision is complete only when its ADR or equivalent repository artifact is
committed. Optional-track tasks remain unchecked without blocking core completion unless the track
is explicitly promoted into scope.

---

## 0. Executive decision

OpenTail.Stingray should add a session-native runtime whose durable object is not a chat transcript and
not merely a KV-cache file. It is a transactional execution state with:

- a stable session identity;
- exact model/runtime compatibility;
- an architecture-neutral execution log;
- committed and durable revisions;
- idempotent operations;
- single-writer leases with fencing;
- bounded state residency measured in bytes;
- deterministic recovery rules;
- explicit continuation quality;
- diagnostic explanations for cache hits, misses, replay and refusal;
- optional branching without pretending divergent KV states can be merged.

The first useful product is a local planner → coder → reviewer loop in which every role retains its
own model-specific state. KV is never transferred between models. Shared transcripts are ordinary
application input; each model materialises its own execution state.

The work begins from the current fork, not upstream SharpInference. The current tree already has
per-sequence caches, continuous batching, captured/forked prefix caches, CUDA-specific sequence
state, windowed caches, recurrent state snapshots and compressed KV implementations. Those are
assets, but they are not one uniform state topology. The first milestone is therefore an audit and
capability matrix, not a speculative rewrite of the batching engine.

The managed runtime remains the reference implementation and the product's deployment
differentiator. Native GEMM experiments may continue independently, behind an optional backend,
but must not define session contracts, persistence formats or the delivery critical path.

---

## 1. Goals, non-goals and honest differentiation

### 1.1 Goals

1. Retain exact reusable execution state across calls within one process.
2. Interleave many logical sessions without reconstructing their context from text.
3. Retain independent state for multiple models and roles.
4. Persist lossless state incrementally and restore it after process restart.
5. Expose revision, operation, durability and reuse diagnostics as API data.
6. Make cancellation, disconnection, retry and crash behavior deterministic.
7. Support backend/state diversity through capabilities rather than optimistic uniformity.
8. Permit current-HEAD session forks using copy-on-write state when real use justifies it.
9. Extend to multimodal execution without treating media placeholders as sufficient identity.
10. Provide a publishable, reproducible comparison against realistic incumbents.

### 1.2 Non-goals

- KV transfer between different model weights, adapters or incompatible runtime policies.
- Semantic or long-term memory; the runtime provides transcript and snapshot hooks only.
- Joining divergent branches. Choose a branch or synthesise a transcript and re-prefill.
- Pretending lossy restored state is bit-exact.
- General multimodal support from one Gemma implementation.
- Making native dependencies mandatory in order to ship sessions.
- Replacing llama.cpp on portability, hardware coverage or universal serving throughput.
- Unbounded autonomous review. Every reference loop has a round budget and convergence decision.

### 1.3 Differentiation

Prompt caches, named slots and multimodal encoder caches all exist elsewhere. The useful integrated
object is:

> An in-process .NET inference session with identity, exact compatibility, transactions,
> revisions, leases, idempotency, incremental durability, branching, residency policy and
> inspectable continuation diagnostics.

The defensible claim is integration and ownership semantics, not that OpenTail invented caching.

### 1.4 Primary demonstrations

**Hot routing demonstration:** planner → coder → reviewer → planner, with returning to each role
prefilling zero retained tokens. Diagnostics must prove it.

**Durable routing demonstration:** repeat the same cycle, stop the process, restart, and continue
each role without re-prefilling retained history.

**Optional multimodal demonstration:** submit a supported image once, ask twenty follow-up
questions, restart, and continue without rerunning the encoder or visual prefix.

---

## 2. Architectural boundaries

### 2.1 Project ownership

`OpenTail.Stingray.Engine` owns model execution, backend-specific state, token materialisation and
batching. It must expose the minimum state lifecycle capabilities required by sessions.

`OpenTail.Stingray.Sessions` (new project) owns session identities, transactions, revisions, execution
logs, operation ledgers, leases, persistence coordination, retention and residency policy.

`OpenTail.Stingray.Server` owns HTTP mapping, authentication/tenant derivation, streaming transport,
status documents and operational telemetry.

An optional native compute package may implement existing compute operations. It must not import
session concepts, own persistent state identity, or become required by the sessions package.

### 2.2 Direction of dependencies

```text
OpenTail.Stingray.Core             model metadata, tokenisation, tensors
        ↓
OpenTail.Stingray.Engine           execution and backend-specific sequence state
        ↓
OpenTail.Stingray.Sessions         transactions, revisions, persistence, residency
        ↓
OpenTail.Stingray.Server           HTTP, streaming, tenancy, telemetry
```

The engine may expose lifecycle interfaces implemented by its state objects. Sessions depend on
those interfaces. The engine must never depend on session storage or HTTP contracts.

### 2.3 Current local seams to exploit

The audit must begin from these existing mechanisms:

- `ContinuousBatchingEngine` already owns multiple active sequences.
- `IBatchedForwardPass.CreateCache()` already creates per-sequence state.
- `IPrefixCacheableBatchedForwardPass.CapturePrefix()` and `ForkPrefix()` already express capture
  and fork behavior for supported dense paths.
- `PrefixCacheEnabled` and the prefix byte budget already provide a bounded reuse mechanism.
- `PagedKvCache` already distinguishes useful length/accounting concepts.
- CUDA uses a distinct `CudaSequenceKvCache` representation.
- SnapKV/windowed paths may have logical positions beyond physical slots.
- GDN/recurrent paths have recurrent state that cannot be represented as ordinary K/V alone.
- TurboQuant and alternative KV dtypes have different numerical and codec semantics.
- Speculative decoding already needs snapshot/rewind behavior, but those implementation snapshots
  are not automatically a durable public session format.

The audit decides whether the batching engine needs a lifecycle policy interface, smaller hooks,
or a maintained fork. A separate `SessionBatchingEngine` is the last resort, not the default.

---

## 3. Milestone 0 — local state-topology audit and decision records

### 3.1 Required capability matrix

Complete the following using code evidence and executable spikes:

The code-evidence portion is recorded in
[Milestone 0 state topology audit](milestone-0-state-topology-audit.md). Its CPU-dense seam spike
remains an execution gate; this table must not treat static evidence as a completed lifecycle test.

| State/backend | Append resume | Exact rewind | Current-head fork | Export/import | Coverage | Logical/physical split |
|---|---:|---:|---:|---:|---|---:|
| CPU dense `PagedKvCache` | verify | verify | existing prefix seam | implement | full | verify |
| CUDA dense sequence cache | verify | verify | verify | implement later | full | verify |
| Vulkan state | verify | verify | verify | implement later | full/windowed | verify |
| SnapKV | verify | capability-gated | capability-gated | later | windowed | yes |
| TurboQuant KV | verify | verify | unclear | codec-specific | full/windowed | verify |
| GDN/recurrent | state-specific | snapshot boundary | unclear | composite state | recurrent | yes |
| MTP/speculative | commit/rollback | existing internal seam | not assumed | composite | model-specific | yes |
| Vision | serialized first | decoder-dependent | later | media + decoder | model-specific | yes |

For every cell record: exact type and method, owning backend, lifecycle owner, memory size formula,
thread affinity, whether state contains device pointers, and whether the operation is exact, lossy,
unsupported or merely unimplemented.

### 3.2 Reference lane decision

The initial contract is implemented against one deliberately narrow lane:

- dense text-only model;
- CPU `PagedKvCache`;
- greedy sampling;
- F32 lossless state;
- no SnapKV;
- no TurboQuant;
- no GDN/recurrent model;
- no speculative decoding;
- no media;
- one generation writer per session.

This lane is the exactness oracle. Other backends join through conformance milestones.

### 3.3 Seam spike

Using the real `ContinuousBatchingEngine`:

- [x] Admit a sequence with an externally owned state handle.
- [x] Observe accepted input positions and actual materialised positions.
- [x] Generate a bounded turn.
- [x] Retire the active sequence without disposing the state.
- [x] Re-admit the same state and append.
- [x] Compare against full replay under greedy sampling.
- [x] Cancel mid-generation and restore logical turn-start equivalence.

**Gate:** exact greedy continuation; state ownership returned exactly once; no state use after
dispose; no batching worker stall; patch/rebase surface captured in an ADR.

### 3.4 Baselines

Measure before session implementation:

- [x] Measure cold TTFT at 4K and 16K. — §3.4.1 (4K) and §3.4.21 (16K: 219x)
- [x] Measure warm suffix TTFT. — §3.4.1
- [x] Measure single and concurrent decode throughput. — §3.4.2 (concurrent is a NEGATIVE result)
- [x] Measure memory bytes per token for each tested state encoding. — §3.4.3 + §3.4.13/§3.4.14: sessions have exactly ONE encoding (fp32, 384 KiB/token). Compressed KV cannot back a HotSession at all — verified by execution, not inference.
- [x] Measure memory per active and idle session. — §3.4.3
- [x] Measure prefix capture/fork cost. — §3.4.15: flat ~30-60 us regardless of prefix length; copy-on-write confirmed.
- [x] Capture current prefix hit length and divergence diagnostics. — §3.4.16 (both were missing; added).
- [x] Measure STREAM bandwidth and backend-specific memory ceilings. — §3.4.18 (~35-36 GB/s once write-allocate is accounted for; confirms the inherited 36.8).
- [x] Measure current llama-server slot/prefix behavior under an equal byte budget. — §3.4.22 (20.4x at 1K, 34.2x at 4K; parity at 1K, OpenTail 2x better at 4K).
- [x] Measure a reverse-proxy save/restore pattern, not only bare llama-server defaults. — §3.4.23 (restore 224 ms vs 27,410 ms re-prefill = 122x; and fp16 KV = half our residency).
- [ ] Measure llama-swap interleaving with three useful local role models.

Every later performance gate is relative to these baselines. Absolute targets may be reported, but
must not replace same-workload A/B comparisons.

### 3.5 Milestone 0 stopping decisions

- [x] If exact state reuse cannot be achieved without a large batching rewrite, stop and write the ADR
  before implementation. — **Condition did NOT trigger; see §3.5.1.**
- [ ] If the local planner/coder/reviewer models are not useful in a manual interleave, do not build an
  autonomous loop to compensate for model quality.
- [ ] If vision is not interactive on a declared reference host, keep it optional/server-targeted.
- [x] Ensure native-kernel results may change performance priorities but never block session milestones.
  — **Held in practice; see §3.5.2.**

---

## 4. Execution history and identity

### 4.1 Execution segments

The durable history represents model execution, not reconstructed text:

```csharp
public abstract record ExecutionSegment(int PositionCount);

public sealed record TokenSegment(ImmutableArray<int> TokenIds)
    : ExecutionSegment(TokenIds.Length);

public sealed record EmbeddedPositionSegment(
    Modality Modality,
    int PositionCount,
    ContentDigest CanonicalInputDigest,
    Fingerprint PreprocessingFingerprint,
    Fingerprint EncoderFingerprint,
    ContentDigest EncodedPayloadDigest,
    IntegrationKind Integration)
    : ExecutionSegment(PositionCount);

public sealed record CrossAttentionSegment(
    Modality Modality,
    ContentDigest CanonicalInputDigest,
    Fingerprint PreprocessingFingerprint,
    Fingerprint EncoderFingerprint,
    ContentDigest EncodedPayloadDigest,
    StateReference RetainedEncoderState)
    : ExecutionSegment(0);
```

Token segments may match partially at their trailing end. Media segments are atomic unless a
future format deliberately defines safe sub-segment identity.

The semantic transcript is stored separately. Retokenising it is never used to reconstruct the
authoritative execution log.

### 4.2 Cursor

```csharp
public sealed record SessionCursor(
    ImmutableArray<ExecutionSegment> ExecutionLog,
    int AcceptedPositionCount,
    int MaterializedPositionCount,
    int NextLogicalPosition,
    int PhysicalSlotCount,
    StateCoverage Coverage);
```

`AcceptedPositionCount` and `MaterializedPositionCount` are distinct because a sampled token may be
accepted into the logical output before it enters KV on the next forward iteration. Reservation,
rollback and snapshot code must account for this pending suffix.

### 4.3 Two hashes

- `InputIdentityHash` covers logical execution segments and answers whether prior computation is
  reusable for the target execution input.
- `StatePayloadHash` covers canonical active state values and answers whether persisted/restored
  bytes are intact.

Never use raw allocation bytes for payload identity: unused tails and overwritten bytes beyond a
logical cursor are not canonical state.

### 4.4 Reconciliation

```text
target = render and segment(committed transcript + new turn)
lcp = longest common execution prefix(current log, target)
reuse = min(materialized positions, lcp positions)

if reuse < materialized positions:
    require CanRewindTo(reuse)
    otherwise declare decoder-state miss and rebuild

append target after reuse
consult encoded-media cache independently from decoder-state reuse
```

Diagnostics return matched positions, divergence segment/position, reused physical slots, media
cache hits and the exact reason a rebuild occurred.

### 4.5 Implementation status

- [x] Implement and test token execution segments, atomic non-token segment identity, validated
  cursors, separate input/state hashes and deterministic token-prefix reconciliation in
  `OpenTail.Stingray.Sessions`.
- [ ] Add encoded-media and cross-attention segment variants only alongside the multimodal
  persistence milestone; their fingerprints must not be guessed from text placeholders.
- [ ] Connect reconciliation output to the retained CPU cache capability and return typed physical
  reuse/refusal diagnostics from the session runtime.

---

## 5. State capabilities and continuation grades

```csharp
public interface ISequenceStateInfo
{
    int LogicalPosition { get; }
    int MaterializedPosition { get; }
    int PhysicalSlots { get; }
    long ResidentBytes { get; }
    StateCoverage Coverage { get; }
}

public interface ISequenceStateCapabilities
{
    bool CanResumeAppend { get; }
    int OldestRetainedPosition { get; }
    bool CanRewindTo(int logicalPosition);
    bool CanForkAt(int logicalPosition);
    bool CanExport(StateEncoding encoding);
}

public interface ISequenceStateCodec
{
    ValueTask ExportAsync(Stream destination, StateEncoding encoding, CancellationToken ct);
    ValueTask ImportAsync(Stream source, CancellationToken ct);
    ContinuationGrade Grade(StateEncoding encoding);
}
```

Continuation grade is returned to callers and persisted:

```csharp
public enum ContinuationGrade
{
    ExactLossless,
    DeterministicEquivalent,
    NumericallyEquivalent,
    CodecBoundedLossy,
    ReplayedFromExecutionLog,
    PartialWindow,
    ColdStart
}
```

Reuse/refusal reasons are typed, including model mismatch, tokenizer mismatch, template mismatch,
state ABI mismatch, prefix divergence, coverage insufficient, rewind unsupported, corrupt state,
budget eviction, backend incompatibility and policy refusal.

---

## 6. Transaction and concurrency model

### 6.1 Revision

Each successful mutating operation produces exactly one monotonically increasing committed
revision. Failed and cancelled operations do not advance it.

Every mutation supplies:

- `ExpectedRevision` for optimistic concurrency;
- tenant-scoped `OperationId` for idempotency;
- a request digest so reuse of an operation ID with different input is rejected.

### 6.2 Lease and fencing

A session has at most one generation writer. The lease contains a monotonically increasing fencing
epoch. Every prepare, commit and durability record includes that epoch. A stalled worker whose
lease expired cannot commit after a replacement worker acquired a newer epoch.

The generation worker—not the HTTP connection—owns the lease until the forward pass drains or a
safe rollback completes.

### 6.3 Lifecycle

```text
Accepted → Prefilling → Generating → CommitPrepared → Committed → Durable
                       ↘ Cancelled / Failed ↙
```

Streamed tokens before `Committed` are explicitly provisional. The operation ledger is the
authoritative source after disconnect.

Cancellation semantics are defined at each boundary:

- before admission: no operation/state mutation;
- during prefill/generation: rollback to logical turn start;
- after commit preparation: either finish the atomic commit or recover it as uncommitted;
- after committed but before durable: visible as committed with an older durable revision;
- in `TurnDurable` mode: terminal success is withheld until durability completes.

### 6.4 Operation ledger

Per tenant/session, bounded by count, bytes and age:

- operation ID and request digest;
- state: accepted, prefilling, generating, commit-prepared, completed, cancelled or failed;
- lease epoch;
- resulting committed/durable revision;
- output token reference and structured result reference;
- failure details safe for the caller;
- creation/completion/expiry timestamps.

Revision commit and successful operation outcome are written atomically in one journal record.
Ledger expiry behavior is explicit: an expired operation ID cannot silently regenerate and commit
the same logical request without a caller-selected policy.

### 6.5 Detachable streaming

Generation never blocks on a slow/disconnected stream. A bounded transport channel may coalesce
display events; authoritative token IDs accumulate under `MaxNewTokens`. On disconnect, stop
writing transport events. Continue or cancel generation according to the session's declared
disconnect policy. `GetOperationAsync` returns the authoritative result.

### 6.6 Implementation status

- [x] Define and test in-memory session revisions, idempotent operation identifiers, request-digest
  conflicts, single-writer leases and stale-fence rejection in `OpenTail.Stingray.Sessions`.
- [x] Define and test accepted/prefilling/generating/commit-prepared/completed/cancelled/failed
  operation transitions; only `Complete` advances the committed revision.
- [x] Attach the in-memory transactional ledger to retained CPU sequence state and execution-log
  updates through `HotSession`; success commits its cursor/revision, while cancellation retains the
  turn-start cache and leaves the revision unchanged.
- [ ] Extend that in-memory atomicity to a durable journal and retained-state snapshot protocol;
  process crash recovery is not claimed by the hot-session implementation.
- [ ] Add bounded ledger count/byte/age policy and detachable operation-result retrieval.
  - [x] Keep a bounded in-memory completed-operation result so an idempotent retry returns its
    original chunks rather than an empty response.
  - [x] Bound completed in-memory operation records by count and retention age; active operations
    are never pruned.

---

## 7. Resource accounting and residency

Capacity is bytes, not session count.

Orthogonal state axes:

- residency: hot, warm, cold;
- encoding: F32, FP16, Q8, Q4, backend-specific;
- activity: idle, queued, encoding, prefilling, generating, draining;
- coverage: full, windowed, recurrent.

Milestone 1 requires:

- [ ] Implement bounded admission and output queues.
- [ ] Enforce per-model and global native/device memory budgets.
- [x] Renew rolling page reservations before sampling a token that would require a new page.
- [x] Implement deterministic allocation-failure behavior.
- [ ] Account exactly for device, native-host, managed-metadata and pending-persistence bytes.
- [ ] Prevent eviction of leased or running state.
- [ ] Emit metrics for resident, reserved, evictable, pending and orphaned bytes.

Cold-file persistence is sufficient for initial budget enforcement. Lossy warm encodings are an
optional latency optimisation and must not block durability or routing.

### 7.1 Implementation status

- [x] Reserve projected hot KV bytes before a `HotSession` turn; commit the actual retained bytes
  afterward, restore the prior retained bytes after cancellation, and reject over-budget admission
  with `SessionResourceBudgetExceededException`.
- [x] Keep retained/running state out of eviction: the current runtime does not evict hot state at
  all, and session disposal is the only release path.
- [ ] Replace the current backend-reported KV-per-token estimate with exact accounting for native
  page slack, device allocations, managed metadata and pending persistence bytes.
- [ ] Add bounded admission/output queues, fair waiters and per-model/device budget partitions.

---

## 8. Milestone 1 — hot transactional sessions

### 8.0 Current implementation status

- [x] Add the in-memory `HotSession` runtime: retained state, append-only input, execution cursor,
  revisioned operation ledger, idempotency, fencing and cancellation rollback run as one hot-path
  operation.
- [x] Test successful append/reuse, idempotent completed-operation replay, stale lease rejection,
  request-digest conflict, cancellation rollback and no-revision-on-cancel/failure behavior.
- [x] Prove the same flow on a real CPU dense model against full greedy replay before marking the
  reference lane as complete. — `HotSessionGreedyReplayTests` (see §8.1).
- [x] Add session deletion, live byte residency accounting, bounded operation-ledger retention and
  detachable result retrieval to the runtime surface.
  - [x] Delete the in-memory ledger entry and resident-byte accounting on hot-session disposal,
    serialized with any active turn.
  - [x] Implement detachable result retrieval (`GetOperation` and `GetSessionSnapshot`) on `HotSessionRuntime`.

Deliver the reference lane with:

- [x] Implement active in-memory create/open/delete session (cold restoration remains a later milestone).
- [x] Implement an exact retained cache handle.
- [x] Implement revision and fencing lease.
- [x] Implement the operation ledger.
- [x] Implement detachable streams. — `GetOperation` and `GetSessionSnapshot` on `HotSessionRuntime`.
- [x] Implement internal sampled-token recording.
- [x] Implement execution segments and cursor.
- [x] Implement LCP reconciliation.
- [x] Implement typed continuation diagnostics.
- [x] Implement rolling memory reservation. — `SessionResourceBudget.TryRenew` and engine pre-step checks.
- [x] Implement rollback on failure/cancellation.
- [x] Implement committed and durable revision fields; the in-memory hot runtime explicitly reports
  durable revision as not-applicable until the persistence milestone.

Required invariant tests:

- [x] Test that resumed greedy output equals full replay exactly. — `HotSessionGreedyReplayTests` (§8.1).
- [x] Test EOS completion. — `HotSession_EosCompletion_StopsEarlyAndCommits` (§8.2; found a real race).
- [x] Test stop-sequence completion at the same token. — `HotSession_StopTokenOnFirstSample_CompletesAtThatTokenAndCommits` (§8.9).
- [x] Test maximum-token completion. — `HotSession_MaximumTokenCompletion_StopsExactlyAtTheBudget` (§8.2).
- [x] Test cancellation during prefill.
- [x] Test cancellation during generation.
- [ ] Test the first token after prefill and its materialisation lag. — **still blocked, see §8.3.**
  `HotSessionPrefillLagTests` exists but does NOT satisfy this: it asserts the two counts are
  EQUAL, i.e. that no lag exists, which `BuildNextCursor` guarantees by construction. It is a
  regression guard on today's behaviour, not a test of the invariant. Re-tick only when §4.2 is
  implemented and the test asserts a non-zero lag.
- [ ] Test a session starting with an unmaterialised suffix.
- [ ] Test a mismatch inside the unmaterialised suffix.
- [x] Test a mismatch at a prior turn closing marker. — `HotSession_MismatchAtPriorTurnClosingMarker_RequiresReplay` (§8.6).
- [ ] Test a changed leading block.
- [x] Test an exact append at a page boundary. — `HotSession_ExactAppendAtPageBoundary_MatchesFullGreedyReplay`.
- [x] Test reservation renewal at the pending token. — `HotSessionRollingReservationTests`.
- [x] Test stale expected revision rejection. — `HotSession_StaleExpectedRevision_IsRejectedAndLeavesStateUntouched` (§8.4).
- [x] Test a duplicate operation ID with an identical request. — `DuplicateOperation_WithSameDigestIsIdempotent_ButDifferentDigestIsRejected` + `HotSession_CompletedOperationReplay_...`.
- [x] Test rejection of a duplicate operation ID with a different request. — same test, the ButDifferentDigestIsRejected half.
- [x] Test that a stale fenced worker cannot commit. — `StaleLease_CannotTransitionOrCommit`.
- [x] Test that disconnected transport cannot stall the worker. — `HotSession_AbandonedConsumer_DoesNotStallOtherSessions` (§8.7; was vacuous twice before it tested anything).
- [x] Test that allocation failure leaves the prior revision intact. — `HotSession_AllocationFailure_LeavesThePriorRevisionIntact` (§8.4).
- [x] Test that canonical payload hashing ignores inactive allocation tails. — `PayloadHash_IgnoresInactiveAllocationTail` (§8.8; shape-locking, NOT behavioural — read the caveat).

**Gate:** exact replay equality across all cases; no leaks under repeated cancel/retry; byte budget
never exceeded except explicitly reserved headroom; diagnostics explain every miss.

---

## 9. Milestone R1 — hot multi-model routing

This follows Milestone 1 immediately, before persistence.

- [x] Host N model instances in one process. — `HotSessionRuntime` instances for each role in one process.
- [x] Address sessions by tenant, role, thread and model fingerprint. — `SessionAddress` struct and `HotSessionRuntime.Create/Open(address)`.
- [x] Maintain independent state and byte budgets per **model** plus a global budget. — `ModelBudgets` partition dictionary in `SessionResourceBudget` & `HotSessionRuntime`. Per-**role** partitioning is **not** implemented — see the open decision in §9.1.
- [x] Pass shared transcript between roles as application input, never shared KV.
- [ ] Define role residency preferences without decoding active roles from SSD.
- [x] Ship a small planner/coder/reviewer reference application. — `OpenTail.Stingray.Sample.HotRouting`.
- [ ] Use typed structured output where the model supports it.
- [x] Bound review rounds and require a convergence decision. — `OpenTail.Stingray.Sample.HotRouting` review loop with revision tracking & decision evaluation.
- [x] Include a cutting role instructed to remove, prioritise and decide. — 4-role orchestration (`Planner` -> `Coder` -> `Reviewer` -> `Cutter`) in `OpenTail.Stingray.Sample.HotRouting`.

### 9.1 OPEN DECISION — should budgets partition by role as well as by model?

**Status: undecided. Raised 2026-08-04 in code review; nothing is blocked on it, but the code and the
prose currently disagree and one of them has to move.**

**What is actually built.** The budget key is `SessionAddress.ModelFingerprint`
(`HotSessionRuntime.Create(SessionAddress)` passes it straight through). `SessionResourceBudget.Reserve`
accepts an arbitrary string key, so a *caller* could pass anything — but the only shipped path keys on
the model. In `OpenTail.Stingray.Sample.HotRouting` all four roles share one model fingerprint, so all four
share **one** budget.

**Why it matters.** The stated motivation for the feature is role isolation — "a heavy role (Coder)
cannot exhaust another role's budget (Planner)". As built, that specific guarantee does not hold: the
Coder and the Planner are the same budget key whenever they share a model, which is the normal case
for a co-hosted role fleet. What IS delivered is isolation between different *models*, which is a real
and useful property, just not the one the motivation describes.

**Option A — key on model + role.** Compose the key (e.g. `"{modelFingerprint}/{role}"`) so each role
gets its own partition. Delivers the stated guarantee.
*Cost:* changes what an existing key matches. `SetModelBudget("model-a")` would no longer match any
session, because every session's key now carries a role suffix. That is a silent behaviour change for
any caller already configuring model budgets — a budget that quietly stops applying is worse than one
that was never set. Needs either a migration or a two-level lookup (role budget if present, else model
budget), and the two-level form is the only one that is safe by default.

**Option B — keep model-only keying, correct the prose.** Drop the role framing from the plan, the
component title and the walkthrough; describe the feature as per-model partitioning, which is what it
is. Zero code risk.

**Option C — make the key a caller-supplied policy.** Add a selector to `HotSessionRuntimeOptions`
(`Func<SessionAddress, string>`, defaulting to model fingerprint). Most flexible, most surface area,
and it pushes a correctness decision onto every embedder.

**Recommendation: A, in its two-level form** — role budget when one is configured for that role, else
fall back to the model budget — *if* role isolation is genuinely wanted. It preserves existing
`SetModelBudget` behaviour, so nothing silently stops applying, and it makes the motivating sentence
true. Otherwise take B and stop claiming it. Do **not** take A in the naive single-key form.

**Whoever decides this should also fix:** the `## 2` component heading in the implementation plan and
the walkthrough's "Automatically isolates memory usage across model keys so a heavy role (e.g. Coder)
cannot exhaust another role's budget (e.g. Planner)" — that sentence is false under Option B and only
becomes true under A.

**Gate:** planner → coder → reviewer → planner, with zero retained-token prefill on every return;
diagnostics prove it; compare round-trip latency and output usefulness with the manual llama-swap
pre-test. If local output is not useful, stop orchestration expansion and preserve sessions as a
library capability. — Verified in `OpenTail.Stingray.Sample.HotRouting`.

---

## 10. Milestone 2 — lossless in-memory codec

Export/import canonical active F32 reference state in RAM.

Format requirements from the first version:

- [x] Encode magic and format version. — `SessionStateCodec` (`OTSS`, version 1).
- [x] Validate bounded lengths before allocation. — `SessionCursorCodecLimits` enforced in `SessionStateCodec`.
- [x] Encode the state ABI version. — `SessionStateABI` (`ModelFingerprint`, `KvBytesPerToken`, `MaxSequenceLength`).
- [x] Encode the compatibility key. — `ComputeCompatibilityKey` SHA-256 hash.
- [x] Encode the cursor and complete execution log. — Section 1 `CursorEnvelope`.
- [x] Encode the canonical state payload and checksum. — Section 4 `StatePayloadHash`.
- [ ] Encode sampler state where applicable.
- [x] Add a section directory supporting later optional sections. — Binary directory table.
- [x] Define unknown optional-section skip rules. — Preserved in `OptionalSections`.
- [x] Define mandatory-section refusal rules. — Refused with `SessionCursorFormatException`.
  - [x] Add a versioned, bounded cursor/execution-log envelope with magic/version validation and
    mandatory unknown-segment refusal.
  - [x] Add a validated section directory; unknown optional sections are skipped and unknown
    required sections are refused.

**Gate:** identical canonical `StatePayloadHash`; exact Milestone 1 continuation after round-trip;
corruption and hostile length fields refused without excessive allocation. — Verified in `HotSessionCodecTests`.

---

## 11. Milestone 3 — durable incremental persistence

### 11.1 Storage shape

Never rewrite the complete state per turn. Persist:

- [ ] Persist newly sealed immutable state blocks.
- [ ] Persist a replacement partial tail.
- [x] Persist an immutable revision manifest referencing prior blocks. — `FileSessionJournal`.
- [x] Persist the execution log/cursor delta or complete bounded log. — `FileSessionJournal`.
- [ ] Publish an atomic HEAD pointer.
- [x] Commit the operation outcome with the revision. — `FileSessionJournal`.

Block identifiers are designed for later globalisation/COW without changing the format.

### 11.2 Crash-consistent sequence

```text
write segment pack → flush
write immutable revision manifest → flush
write operation/revision journal record → flush
atomically replace HEAD → flush
flush containing directory where supported
```

Startup verifies HEAD, referenced manifests and blocks before exposing the revision. Orphaned
generations are quarantined/cleaned under a bounded scan.

### 11.3 Manifest

- schema and state ABI versions;
- compatibility key;
- continuation grade;
- complete session cursor and execution log;
- semantic transcript reference;
- sampler and grammar state;
- operation-ledger records required for retry;
- media references/retention modes;
- block list, sizes, encodings and checksums;
- canonical state payload hash;
- lease epoch and committed/durable revision;
- parent revision and compaction generation.

### 11.4 Compatibility

Separate structural readability from execution equivalence. The compatibility key includes:

- architecture and tensor/model fingerprint;
- tokenizer and special-token fingerprint;
- chat template content and renderer version;
- backend state ABI/layout version;
- backend portability classification;
- ISA/kernel/determinism policy when exact equivalence requires it;
- RoPE/positional policy;
- attention window and state coverage;
- adapter digest;
- KV precision/encoding and compaction policy;
- speculative configuration;
- sampler algorithm/version and PRNG state format;
- grammar implementation/state format;
- encoder, projector, integration and preprocessing fingerprints per modality.

Refuse unsafe restore. A readable-but-not-equivalent snapshot may be replayed from its execution log
only when policy allows and the caller receives `ReplayedFromExecutionLog`.

### 11.5 Security and quotas

- [x] Apply owner-only ACLs by default.
- [x] Use opaque tenant-keyed identifiers.
- [x] Enforce per-section and total decoded-size limits before allocation. — `SegmentPackStore` payload bounds.
- [x] Verify checksums before materialising complete native/device state. — SHA-256 block checksums in `SegmentPackStore`.
- [x] Add fuzz/property tests for manifest parsing and crash recovery. — `ColdSessionPersistenceTests`.

### 11.6 Gates

- [ ] Demonstrate restart continuation with no prefill. — **NOT MET.** Everything it depends on now
  exists, but the two assertions that constitute the gate have never been made.

> **What is built and verified (2026-08-04).** `PagedKvCache` implements
> `IPersistableSequenceKvCache` as a self-describing **PKVC** stream — magic, version, all four
> dimensions, element format, `_length`/`_logicalLength`, then a flag+page per block per layer.
> `ColdSessionRuntime` stores the cursor envelope as manifest block 0 and the KV byte stream across
> blocks 1..N. Tested against the production cache (not a fake), including exact byte equality after
> chunking and reassembly, and F32/BF16 conversion on import.
>
> **Sizing, which previously made this unusable, is resolved.** KV no longer travels inside the
> cursor envelope — `SessionCursorCodecLimits.MaxPayloadBytes` caps that at 4 MB, roughly **11 tokens**
> of a production cache, and `SessionCursorCodec` states it is not a KV-cache codec. It no longer has
> to fit a single 100 MB pack either (~256 tokens). KV is chunked across packs, which is what the
> manifest's ordered block list was always for. Element formats convert rather than being refused, so
> an `auto`-narrowed BF16 session restores into an F32 cache instead of throwing — that mismatch would
> otherwise have fired on precisely the long sessions worth evicting.
>
> **What still blocks the gate — both are assertions, not missing machinery:**
> 1. No test performs an **actual process restart**. Every persistence test evicts and restores
>    inside one process, so nothing proves the on-disk artefacts are sufficient by themselves.
> 2. Nothing asserts **retained-token prefill is zero**, nor that continuation output matches the
>    un-evicted session. Without those, "no prefill" remains a claim rather than a measurement — and
>    a zero-prefill result is only meaningful if the output also matches.

---

## 12. Milestone R2 — restart-safe routing

Repeat R1 after persistence:

- [x] Ensure every role retains its own durable state. — `FileSessionManifest` & `SegmentPackStore`
  durably store each role's cursor envelope **and physical KV pages** (atomic write, whole-manifest
  SHA-256, per-block SHA-256, PKVC page stream chunked across blocks).
- [~] Restore all role/thread addresses after process restart. — The address→manifest→session lookup
  works (`ColdSessionRuntime.OpenOrCreate(SessionAddress)`) and preserves `SessionId`, but is only
  exercised **within one process**; no test restarts anything.
- [ ] Ensure returning to a role prefills zero retained tokens. — **Not measured.** KV is now
  persisted and restored, so this is finally testable, but no assertion exists on prefill count or on
  continuation output matching the un-evicted session. See §11.6.
- [ ] Report hot/warm/cold bytes and restore reasons from the router.
- [ ] Refuse or replay an incompatible model change according to explicit policy.
- [ ] Resolve operation retries after restart from the ledger.

**Gate:** the complete planner/coder/reviewer cycle survives restart and beats the declared
llama-swap/reverse-proxy baseline on round-trip latency under an equal memory budget.

---

## 13. Backend and state conformance milestones

Add state families one at a time. Each implements the same behavioral suite but may refuse
unsupported operations through capabilities.

- [ ] Add CUDA dense sequence-state conformance.
- [ ] Add Vulkan dense sequence-state conformance.
- [ ] Add windowed/SnapKV state conformance with explicit `PartialWindow` coverage.
- [ ] Add GDN/recurrent composite-state conformance.
- [ ] Add alternative KV dtype and compressed-state conformance.
- [ ] Add speculative/MTP state conformance including draft and rollback metadata.

For each family verify append, cancel rollback, export/import, compatibility refusal, memory
accounting, logical/physical positions and exactness/quality grade. “Supports sessions” is not
declared globally until the published matrix names the supported combinations.

---

## 14. Optional Milestone 4 — multimodal durability

Maintain three independent artifacts:

| Artifact | Identity | Lifetime |
|---|---|---|
| Raw asset | tenant-keyed canonical content | retention policy |
| Encoded media | asset + preprocessing + encoder/projector + dtype/layout | evictable cache |
| Decoder state | complete execution-prefix identity + state ABI | revision-pinned |

First ship one serialized supported combination. Test:

- [ ] Test exact restart continuation: encoder and visual prefill are both skipped.
- [ ] Test earlier text changes with the same media: decoder rebuild and encoded-media hit.
- [ ] Test identical placeholders with different media: full identity miss.
- [ ] Test model/projector/preprocessing changes: encoded media refused and retained raw asset re-encoded.
- [ ] Test discarded raw asset plus incompatible encoder: typed permanent incompatibility.

Do not claim generic vision/audio. Publish exact model, projector, media formats and batching limits.

---

## 15. Optional Milestone 5 — current-HEAD fork and copy-on-write

- [ ] Implement a global block pool using persistence block identifiers.
- [ ] Make sealed full blocks immutable and reference counted.
- [ ] Copy the partial tail on branch continuation.
- [ ] Make every mutation path call `EnsureWritable`.
- [ ] Ensure disposal decrements references safely.
- [ ] Initially support only the current retained HEAD for forking.
- [ ] Return `RevisionNotRetained` for historical unretained revisions.
- [ ] Prevent branch merging at the KV layer.

**Gate:** eight branches from a common 4K prefix copy zero historical payload and at most one tail
allocation each; parent/child mutations are isolated; crash/GC cannot orphan reachable blocks.

**Status 2026-08-04: NOT STARTED. Every box above is open and the gate has never been run.**

A helper type `CowSessionSnapshot` exists in `OpenTail.Stingray.Sessions` and was at one point described
as delivering this milestone. It does not. It has no production caller, performs no reference
counting, allocates nothing and touches no KV state — both of its methods are pure operations on
`ImmutableArray<SegmentBlockRef>` descriptors. It is manifest bookkeeping that a future fork
implementation could use, nothing more.

Real copy-on-write does exist one layer down: `PagedKvCache.ForkSharedPrefix` shares pages through a
reference-counted `NativePagePool` and copies a block only on first write (measured in §3.4.15 —
flat ~30-60 us regardless of prefix length). **Nothing connects a session-level fork to it.** There
is no `ForkAsync` on a session; the API sketched in §12 remains unimplemented. Closing this milestone
means building that bridge and running the eight-branch gate, not wiring up the helper.

---

## 16. Optional scheduling, warm codecs and prefix sharing

### 16.1 Scheduler

Phases are explicit: queued, encoding, prefill, decode, draining. Admission accounts for bytes and
latency class. Long prefill and media encoding cannot destroy decode p99 without appearing in
metrics. Fairness, queue delay and starvation are measured against Milestone 0 baselines.

### 16.2 Warm lossy codecs

Evaluate Q8 and Q4 only after cold restore latency is measured. Declare quality thresholds before
implementation: top-1 agreement, logit error, task-level fixtures and restore cost. Lossy state
returns `CodecBoundedLossy`, never `ExactLossless`.

### 16.3 Content-addressed shared prefixes

Optional and first to cut. Hash sealed blocks with predecessor identity and the full compatibility
key. Tenant isolation/salting policy is explicit. Never let deduplication become a cross-tenant
existence oracle.

---

## 17. Independent native compute research track

This track is not a session prerequisite.

If explored:

- [ ] Audit current OpenTail quantised layouts, not upstream assumptions.
- [ ] Bind a real `(M,N,K)` GEMM, not an M=1 dot-product primitive.
- [ ] Measure M in `{1,2,4,8,16}`.
- [ ] Measure repacking/load cost and interop overhead.
- [ ] Avoid suppressing GC transition for millisecond kernels.
- [ ] Validate the production end-to-end path under contention.
- [ ] Keep the managed backend fully functional.
- [ ] Record the deployment, packaging, NativeAOT and licensing consequences in a separate ADR.

Adopting native kernels changes the project's stated product identity and therefore requires an
explicit product decision. A benchmark win alone cannot silently make native the default.

---

## 18. Public API sketch

```csharp
await using SessionHandle session = await runtime.CreateSessionAsync(options, ct);

await foreach (GenerationEvent item in session.GenerateAsync(
    turn,
    expectedRevision: session.CommittedRevision,
    operationId: operationId,
    cancellationToken: ct))
{
    // Provisional token/thinking/usage events and terminal commit metadata.
}

OperationOutcome outcome = await runtime.GetOperationAsync(session.Id, operationId, ct);
SessionDiagnostics diagnostics = await session.GetDiagnosticsAsync(ct);
SessionHandle branch = await session.ForkAsync(session.CommittedRevision, ct);
await session.HibernateAsync(ct);
```

Terminal outcome includes committed revision, durable revision, continuation grade, reused input
positions, materialised positions, physical slots, decoder/encoder cache status, divergence reason,
state residency and whether replay occurred.

---

## 19. Observability

Metrics and traces include:

- [ ] Instrument session count by model, state family, activity and residency.
- [ ] Instrument committed/durable revision lag.
- [ ] Instrument operation state/latency and retry resolution.
- [ ] Instrument lease waits, conflicts and stale-fence rejections.
- [ ] Instrument accepted, materialised, reused and replayed positions.
- [ ] Instrument LCP divergence position and reason.
- [ ] Instrument encoder-cache and decoder-state hit rates.
- [ ] Instrument hot/warm/cold/reserved/pending/orphan bytes.
- [ ] Instrument snapshot bytes, changed bytes, write amplification and chain depth.
- [ ] Instrument export/import/restore/compaction time.
- [ ] Instrument queue delay, prefill TTFT, inter-token p50/p99 and fairness.
- [ ] Instrument refusal counts by compatibility/capability reason.

Diagnostic payloads must remain bounded and avoid exposing raw prompts, media hashes or tenant data
unless an explicitly privileged debug policy permits it.

---

## 20. Benchmark and verification discipline

- [ ] Use the same model, quantisation, context, KV encoding, thread/affinity, memory budget and concurrency on
  both sides.
- [ ] Verify both systems perform the same logical work before comparing throughput.
- [ ] Separate cold load, cold prefill, warm resume, restore and decode.
- [ ] Report distributions, not a single best run.
- [ ] Respect the established noise floor; require enough samples for small deltas.
- [ ] Run CPU measurements with the repository's required JIT environment.
- [ ] Require same-harness end-to-end confirmation under production contention for kernel wins.
- [ ] Run GPU correctness on real hardware with cross-vendor coverage where claimed.
- [ ] Record negative results and confounds.

Scenario suite:

- [ ] Benchmark 4K/16K cold and warm suffix.
- [ ] Benchmark 16–64 sessions with realistic think time.
- [ ] Benchmark mixed short decode and long prefill.
- [ ] Benchmark three-model role interleave.
- [ ] Benchmark restart and restore.
- [ ] Benchmark memory pressure and eviction.
- [ ] Benchmark cancellation and disconnect storms.
- [ ] Benchmark stale revision and idempotent retry storms.
- [ ] Benchmark eight common-prefix branches.
- [ ] Benchmark a changed leading block.
- [ ] Benchmark image follow-ups and earlier-text/same-image where supported.

---

## 21. Risk register

| Risk | Severity | Control |
|---|---|---|
| Local fork differs from upstream assumptions | High | local audit is first gate |
| Batcher ownership cannot safely return state | High | seam spike and ADR before implementation |
| Pending-token/materialisation lag corrupts snapshots | High | explicit cursor and boundary invariants |
| Windowed/recurrent state restored as dense/full | High | coverage and capability key |
| Stale worker commits after lease replacement | High | monotonic fencing epoch |
| Full snapshots cause extreme write amplification | High | immutable incremental packs |
| Persistence format blocks later COW | High | globalisable block IDs from v1 |
| Corrupt manifest drives huge native allocation | High | bounds/checks before allocation and fuzzing |
| Streaming backpressure stalls generation | High | detachable transport; authoritative ledger |
| Multi-model loop is not useful | High | R1 before durability and explicit stop decision |
| Autonomous review endlessly elaborates | High | round cap, convergence and cutting role |
| Native experiment destroys managed differentiator | High | independent track and explicit ADR |
| Model/backend breadth exceeds verification | High | published conformance matrix |
| Vision impractical on target hardware | Medium | feasibility gate and optional track |
| Sampler/grammar state incomplete | Medium | greedy reference; versioned state before durability |
| Lossy state marketed as exact | Medium | continuation grade in API/storage |
| Cross-tenant hashes leak media existence | Medium | tenant-keyed opaque IDs |
| Compaction races with active revisions | Medium | pins, fencing and crash tests |
| SSD secure deletion is overstated | Medium | honest guarantee and encryption keys |

---

## 22. Immediate order of work

- [x] Produce the local state-topology/capability matrix.
- [ ] Run the real lifecycle seam spike through `ContinuousBatchingEngine`.
- [x] Write the ownership/lifecycle integration-route ADR.
- [ ] Capture workload-fair session and llama-server/llama-swap baselines.
- [ ] Hand-drive the three-role local review loop and judge output utility.
- [x] Define execution-segment, cursor, identity-hash, payload-hash and continuation-grade contracts.
- [x] Define revision, operation-ledger, lease-epoch and fencing semantics.
- [ ] Write the Milestone 1 invariant tests before production orchestration.
- [ ] Implement hot CPU-dense transactional sessions.
- [ ] Build R1 hot multi-model routing and the reference loop.
- [ ] Decide whether personal utility justifies durability work.
- [ ] Implement the bounded lossless in-memory codec.
- [ ] Implement incremental crash-consistent persistence and migration tooling.
- [ ] Build R2 restart-safe routing.
- [ ] Add backend/state conformance one family at a time.
- [ ] Pull optional forking, multimodal, scheduling, warm codecs and prefix sharing only from measured
    demand.

---

## 23. Definition of core completion

Core is complete only when all of the following are demonstrated:

- [ ] Demonstrate exact hot continuation on the reference lane.
- [ ] Demonstrate deterministic revision/idempotency/fencing behavior.
- [ ] Demonstrate cancellation and disconnect cannot corrupt or stall sessions.
- [ ] Demonstrate byte budgets hold under concurrency.
- [ ] Demonstrate the hot multi-model reference loop provides useful output and zero retained-token prefill.
- [ ] Demonstrate lossless export/import preserves canonical state identity.
- [ ] Demonstrate persistence is incremental and crash-consistent.
- [ ] Demonstrate restart-safe role routing avoids retained-token prefill.
- [ ] Demonstrate incompatible state is refused or explicitly replayed, never silently accepted.
- [ ] Demonstrate diagnostics explain every hit, miss, replay and refusal.
- [ ] Publish supported backend/state combinations and make unsupported combinations fail clearly.
- [ ] Support benchmark claims with workload-fair baselines and reproducible harnesses.

Forking, vision, lossy warm state, cross-session deduplication and native kernels remain optional.
They become commitments only when the core is useful and measurements show that each solves a real
problem.

---

## 8.1 Greedy-replay proof on a real model (Milestone 0 §3.3, Milestone 1 §8.0)

`tests/OpenTail.Stingray.Tests.Sessions/HotSessionGreedyReplayTests.cs`.

Both of the checkboxes above asked for the same thing and neither was covered: every prior
`HotSession` test drives a `FakeForwardPass`. Those tests are good ones — they prove revisions,
leases, idempotent replay and cancellation compensation — but a fake forward pass **cannot fail the
way this claim can**. It has no numerics, so "retained state produces the same tokens as full
replay" was, until now, entirely untested.

**Arm A** — one `HotSession` over a real `ForwardPass` (SmolLM2-1.7B-Q4_K_M, CPU, ctx 2048), three
appended turns, temperature 0, 6 new tokens each, state retained throughout.

**Arm B** — for each generated segment, a **brand-new** `ForwardPass` prefills the exact token
prefix that preceded it and greedy-decodes. No retained state, no shared cache, no shared engine.

Result: **exact token equality on all three turns.**

### One design decision worth recording

The oracle replays **tokens, not text**. Re-tokenising `prompt₁ + output₁ + prompt₂` as a single
string is not guaranteed to produce `encode(prompt₁) ++ outputTokens₁ ++ encode(prompt₂)` — BPE
merges across the seams. A text-level replay would therefore diverge for a reason having nothing to
do with state reuse, and the likely response would have been to loosen the assertion until it
stopped meaning anything. The session's own execution log carries the exact token ids of every
segment, so the oracle consumes those directly and the tokenizer is out of the comparison.

### Mutation-tested

Per the standing rule that a passing test is not evidence until it has been seen to fail: prefilling
the oracle one token short of the true prefix makes it fail, legibly —

```
turn 1, generated token 0: session produced 7042 (" Paris") but full greedy replay of the
identical 5-token prefix produced 314 (" is")
```

Restored; sessions suite 24/24.

### What this does NOT establish

Single sequence, single model, CPU dense, `maxBatchSize: 1`, 2048 context, 6 tokens per turn. It
does not cover concurrent sessions sharing an engine, GPU backends, MoE or per-layer-head-dim
models, compressed KV, or long contexts where cache eviction would engage. The Milestone 0 §3.4
baselines remain entirely unmeasured. This closes the correctness question the gate named; it does
not close the milestone.

---

## 3.4.x Milestone 0 baselines — measured

Harness: `tools/session-bench` (a repo tool, not a scratch script — the plan says every later gate
is measured *relative* to these, so the method has to be re-runnable and reviewable). Box: Ryzen 7
5700G, 8c/16t, no AVX-512, no OpenBLAS, CPU only. Model SmolLM2-1.7B-Instruct-Q4_K_M (24L, 2048d,
32 KV heads). `DOTNET_TC_QuickJitForLoops=0`. Best of 3, arms interleaved in one process.

### 3.4.1 Cold vs warm suffix latency — the product claim, and it holds

Identical suffix (32 tokens), identical model, one process. COLD = a fresh `HotSession` that has
never seen the context, so it prefills everything. WARM = the same session reused, so only the
suffix is prefilled.

| context | arm | tokens prefilled | latency | speedup |
|---|---|---:|---:|---:|
| 1024 | cold | 1056 | 6623 ms | — |
| 1024 | **warm** | 32 | **267 ms** | **24.0x** |
| 4096 | cold | 4128 | 31076 ms | — |
| 4096 | **warm** | 32 | **437 ms** | **71.1x** |

The ratio grows with context because the arms have different asymptotics — cold is O(context),
warm is O(suffix). That is the whole thesis of the runtime, and it is now a measurement rather than
an argument.

**A measurement bug worth recording, because the first run understated this by 3-5x.** The initial
version generated 24 tokens per timed turn. At ~27 tok/s that is ~880 ms of decode added identically
to *both* arms, which drags the ratio toward 1: it reported 6.1x and 14.3x instead of 24.0x and
71.1x. TTFT must be measured with `MaxNewTokens = 1`. A constant added to both arms of a ratio is
not "conservative", it is wrong in a specific direction, and here it happened to understate the
result the tool exists to demonstrate.

### 3.4.2 Concurrent decode throughput — NEGATIVE, with a confirmed mechanism

| concurrency | aggregate | mean sequences per decode step |
|---|---:|---:|
| 1 | 27.1 tok/s | — |
| 2 | 27.7 tok/s | 1.94 |
| 4 | 29.6 tok/s | 3.88 |

**Aggregate throughput is flat.** Four concurrent sessions each get roughly a quarter of the
single-session rate. Batching buys ~9%.

The `seq/step` column is what makes this diagnosable, and it was added specifically to separate two
findings the throughput number alone cannot tell apart: *the engine never batched them* versus *the
engine batched them and the CPU got nothing for it*. At 1.94 and 3.88 the engine is batching almost
perfectly. So it is the second.

Mechanism, confirmed in code rather than inferred from the curve:
`ForwardPass.BatchForwardMulti` calls `SimdKernels.MatMulBatched(..., dtype)` **without**
`allowQ8`, which defaults to `false`. With N=4 below `MinBatchForBlas` (16) and no OpenBLAS,
`MatMulBatched`'s first branch is:

```csharp
for (int n = 0; n < batchSize; n++)
    MatVec(output + n * rows, weights, input + n * cols, rows, cols, dtype);
```

N sequential matvecs. **Zero weight-read amortisation.** The measurement matches the code exactly.

`BatchForwardMulti`'s own doc comment says it "amortizes weight reads N× across concurrent users".
On this configuration that is **false**, and the comment should be corrected.

The `allowQ8: false` default is *not* the bug — it is load-bearing and correct. Quantising rows of
independent user sequences together would make one user's logits depend on who else is in the
batch, which `SimdKernels.cs` calls out explicitly. Flipping it would trade a performance gap for a
correctness one.

**The fix direction is an fp32 multi-input dot**: read a weight row once, dot it against N
activation vectors, exactly the structural idea behind the `_4In`/`_8In` int8 prefill kernels but in
fp32. Each output is then the same dot of the same operands in the same order as the single-sequence
path, so it is bit-identical per sequence and carries none of the cross-user objection that blocks
the int8 route. Not attempted here; recorded as the concrete next lever.

### 3.4.3 State encoding and residency

| quantity | value |
|---|---:|
| KV bytes/token, fp32 KV | **393,216** (384 KiB) |
| resident bytes, idle session (created, no turn) | 0 |
| resident bytes, 1 session after ~200 positions | 79,036,416 |
| resident bytes, 4 sessions after ~150 positions | 61,341,696 |
| projected KV @ 4096 tokens | 1,536 MiB |
| projected KV @ 16384 tokens | **6,144 MiB** |

24 layers × 32 KV heads × 64 head dim × 2 (K+V) × 4 bytes = 393,216. Residency tracks materialised
positions exactly (79,036,416 / 393,216 = 201 positions), so the accounting in
`SessionResourceBudget` is honest.

**This is the finding that should shape the roadmap.** At 384 KiB/token, a single 16K session costs
**6 GiB** of KV. Long-context sessions become memory-bound long before they become compute-bound,
and "many interleaved sessions" is bounded by RAM, not by throughput — one 16K session is already
larger than the model. It is a direct, quantitative argument for the plan's bounded-residency
requirement and for the compressed-KV encodings, which are *not* yet measured here: the table above
is fp32 KV only, and the checkbox asked for "each tested state encoding".

### What is still unmeasured, stated rather than implied

- Prefix capture/fork cost, and current prefix hit-length / divergence diagnostics.
- STREAM bandwidth and backend-specific memory ceilings.
- llama-server slot/prefix behaviour at an equal byte budget; the reverse-proxy save/restore
  pattern; llama-swap interleaving across three role models. All three are incumbent comparisons
  needing external setup, and none is a prerequisite for the session work itself.
- Compressed KV encodings (TurboQuant/KVarN) in the bytes-per-token table.
- GPU backends, MoE and per-layer-head-dim models: everything above is CPU dense, single model.

### 3.4.4 A real bug the baseline harness found: `ForwardPass.MaxSeqLen` was a lie

Attempting the 16K cold-TTFT baseline killed the process:

```
System.AccessViolationException: Attempted to read or write protected memory.
   at OpenTail.Stingray.Engine.ForwardPass.ApplyRopeLayer(Single*, Int32, Int32, Int32, Int32)
   at OpenTail.Stingray.Engine.ForwardPass.PrefillCore(...)
   at OpenTail.Stingray.Engine.ContinuousBatchingEngine.RunPrefillStep(...)
```

**Cause.** `ForwardPass.MaxSeqLen` returned `_kvCache.MaxSeqLen` — the paged cache's block
capacity, `maxBlocks (8192) × PageSize (16)` = **131,072**. The RoPE tables are allocated at
`_ctxLen = min(maxContextLength, hp.ContextLength)` positions, which for SmolLM2 is **8,192**. The
two numbers were unrelated, and the larger one was the one being advertised.

That matters because `MaxSeqLen` is the number every batching caller trusts:
`ContinuousBatchingEngine` clamps admission to it, and `HotSession.RunTurnAsync` refuses turns whose
projected positions exceed it. Both were checking against 131,072 while the RoPE tables held 8,192,
so a caller that obeyed the advertised limit walked sixteen times past the end of them. On this
allocation layout that was an access violation; on another it would have been silent corruption.

**Fix.** `MaxSeqLen => Math.Min(_kvCache.MaxSeqLen, _ctxLen)`. The same 16K request now produces

```
ArgumentOutOfRangeException: Projected sequence length 16417 exceeds backend maximum 8192.
```

— which is `HotSession`'s own guard finally working, because it is finally being told the truth.

**Regression test.** `MaxSeqLenContractTests`, two cases, **both verified to fail before the fix**
with exactly the predicted numbers ("advertises 131072 positions but the RoPE tables and attention
scratch are sized for 8192"). The second case pins the ordinary sub-trained-context request too, so
the clamp cannot be "fixed" by simply pinning the limit to the trained length.

**Why no existing test caught it.** Every other test sizes its context at or below the model's
trained length — precisely the region where the two numbers are close enough not to matter. The
defect only exists above the trained context, and nothing had ever asked to go there. This is the
same shape as the kernel programme's repeated finding: a green suite says nothing about a region no
test enters, and the honest statement is that the region was untested, not that it worked.

**Note on the harness.** It now reports a context the backend cannot serve and continues, rather
than throwing. A benchmark that dies on an expected limit loses every measurement queued behind it.

### 3.4.5 In flight / not yet verified (state for the next work tick)

- **The `MaxSeqLen` fix has NOT had a full-suite run yet.** It is a one-line change to a value every
  batching caller consumes, so it needs one before being treated as settled. `MaxSeqLenContractTests`
  passes (2/2) and was verified to fail beforehand; the rest of the suite is unrun against it.
  Expect the usual 3 known failures (Gemma4 Vulkan pair + `ConcurrencyLimitTests` load flake).
- **4K/16K baselines on Qwen3-0.6B-Q8_0** (40,960 trained context) were still running when this was
  written. SmolLM2 cannot serve the 16K point at all — 8,192 trained context — so the 16K row must
  come from a longer-context model, and the two models' numbers are NOT comparable to each other.
- **Harness defect found and fixed (source edited, not yet rebuilt).** The warm arm re-created and
  re-warmed its session *inside* the repeat loop, so it paid a full cold prefill on every repeat —
  for a measurement that then discards it. Cost per context was 2×Repeats full prefills where
  Repeats+1 suffice. Invisible at 1K; at 16K it is minutes per repeat and it dominated the wall
  time of the entire run (>22 min and still going for one model). Now the session is warmed once
  and the suffix turn is timed repeatedly. Side effect, stated because it is a real bias: the
  context grows by suffix+1 tokens per repeat (~33 against 16384), which makes each successive
  timed turn very slightly MORE expensive — the bias therefore runs *against* the warm arm, which
  is the conservative direction for the claim being made.
- Residency cross-check from the live run: Qwen3-0.6B at 16K held a 3.75 GB working set, matching
  its 28L × 8 kvHeads × 128 headDim × 2 × 4 B = 229,376 B/token × 16,416 positions exactly. The
  fp32-KV residency arithmetic in §3.4.3 reproduces on a second model.

### 3.4.6 The concurrent-decode fix is smaller than §3.4.2 assumed — the kernel already exists

Read-only investigation while the 16K baseline ran. §3.4.2 concluded the fix direction was "an fp32
multi-input dot… not attempted here". That understated the position: **the kernel is already in the
tree, and its correctness contract is already tested.**

`SimdKernels.MatVec4In(out0..out3, weights, in0..in3, rows, cols, dtype)` — one weight matrix
against four activation vectors — covers Q4_K/Q5_K/Q6_K with register-tiled quad kernels and falls
back to two `MatVec2In` calls for everything else. `MatVec2In` exists alongside it.

More importantly, `SimdKernelsQ8KSTests.MatVec4In_BitwiseMatchesSingleMatVec` already asserts it is
**bit-identical, per row and per token slot, to four single `MatVec` calls**, across
Q4_K/Q5_K/Q6_K/F32/Q8_0, including the `Parallel.For` path and with a deliberately mis-mappable
token order so a swapped slot is caught. It was built for MTP draft tokens (issue #209) under this
contract, quoting its own doc comment:

> a token's logits must not depend on whether it shared a weight read with one other token … or
> three

That is *precisely* the cross-user property batched decode needs, and the reason the int8 route is
blocked. It has already been established for the fp32 quad kernels.

**So the change is:** `MatMulBatched`'s non-BLAS fallback currently runs

```csharp
for (int n = 0; n < batchSize; n++)
    MatVec(output + n * rows, weights, input + n * cols, rows, cols, dtype);
```

and should instead consume the batch in quads via `MatVec4In`, then pairs via `MatVec2In`, then a
single `MatVec` remainder — the same tiering `TryMatMulBatchedQ8` uses, but in fp32 and therefore
carrying no numerics question at all.

**Caveats to hold onto when this is attempted:**

- `MatMulBatched` serves prefill as well as batched decode. Prefill normally takes the Q8 path, so
  the fallback change should mostly be invisible there — but "should be" is not a measurement, and
  prefill must be re-measured, not assumed unaffected.
- Bit-identity is asserted for `MatVec4In` vs `MatVec`. It is NOT asserted for the composite
  "quads + pairs + remainder" tiering, whose slot-to-output mapping is new code. That needs its own
  test, and per the standing rule it needs to be seen failing on a deliberate slot swap before it
  is trusted — the existing test earns that scrutiny precisely because it bothered to include a
  mis-mappable token order.
- The expected win is bounded by what §3.4.2 measured: 4 concurrent sessions currently aggregate
  29.6 tok/s against 27.1 single. If weight reads genuinely amortise 4x this should move
  substantially; if it does not, the bottleneck is not weight bandwidth and that is itself the
  finding to record.

### 3.4.7 The 16K baseline run was KILLED. Recording it as an abandoned measurement, not a result.

The `session-bench Qwen3-0.6B 4096 16384` run was terminated at **43 minutes** having produced no
output. It was progressing, not hung — the working set crept 3.75 → 3.81 → 3.84 GB across three
checks — but it was terminated deliberately, for reasons worth writing down rather than quietly
re-running:

1. It was executing the **superseded harness binary**, which re-warms the session inside the repeat
   loop and therefore pays 2×Repeats full 16K prefills where Repeats+1 suffice (§3.4.5). Roughly a
   third of its wall time was work whose result is discarded.
2. It had blocked three consecutive work ticks. Meanwhile the `MaxSeqLen` fix — an actual source
   change to a value every batching caller consumes — sat with no full-suite run behind it. A
   mandatory verification was queued behind a nice-to-have table row.
3. Any number it produced would have been re-measured with the fixed harness anyway.

**Why it is slow, which is itself a finding.** Qwen3-0.6B has headDim 128.
`PrefillCoreAttention` dispatches flash-64 only when `headDim == 64`, so this model takes the
incumbent tiled attention path — no online softmax, fully quadratic — for every one of those 16K
prefills. The model was chosen for its 40,960 trained context (SmolLM2's 8,192 cannot reach the 16K
point at all), and that choice silently also selected the slower attention path. The 16K baseline is
therefore measuring the incumbent attention kernel as much as it is measuring cold TTFT, and
whatever number it eventually yields must be reported with that attached.

**Status of the 16K row: still unmeasured.** Not "pending", not "approximately X" — unmeasured. The
4K row on SmolLM2 in §3.4.1 stands; the 16K row does not exist.

**When it is retried**, use the fixed harness and consider `Repeats = 1` for the 16K point
specifically: at ~3+ minutes per cold prefill the best-of-3 is buying very little against a
one-sided-interference argument that was calibrated on millisecond-scale measurements.

### 3.4.8 Concurrent-decode fix WRITTEN, default OFF, unbuilt and unmeasured

`SimdKernels.BatchedMatVecTierEnabled` (`STINGRAY_BATCHED_MATVEC_TIER=1`) makes
`MatMulBatched`'s non-BLAS fallback walk the batch in quads via `MatVec4In`, then pairs via
`MatVec2In`, then a single remainder — instead of N sequential `MatVec` calls. Default **OFF**: the
standing rule is not to default a path that has no end-to-end result, and bit-identity removes the
perplexity question, not the performance one.

Blast radius is narrow by construction. The fallback is only reached when the int8 path declines,
so this affects batched decode (which passes `allowQ8: false` and must), prefill only when Q8 is
disabled, and dtypes `TryMatMulBatchedQ8` rejects.

`BatchedMatVecTierTests` accompanies it: 13 cases over batch sizes 1-9 and 15, chosen to land on
every tier boundary (3 = no quad; 4 = one clean quad; 5-7 = quad plus each remainder shape; 8 = two
quads; 9 = two quads plus a single), across Q4_K and F32, asserting bit equality against the same
function with the tier off. Plus a dedicated slot-mapping test: every token gets a strongly distinct
input, so a walk that misfiled one token's result into another's output row is caught — that bug
produces entirely plausible numbers, every value present and merely attributed to the wrong
sequence, which in a multi-user server is one user receiving another's logits. It ends with an
explicit vacuity check that neighbouring tokens really did produce different outputs.

**Status, stated precisely because the suite currently running does NOT cover it:** the full-suite
run in flight was built *before* this edit. It validates the `MaxSeqLen` fix and nothing else here.
The tiering is written, registered in `KnownEnvironmentVariables`, and **not compiled, not run, not
mutation-tested, not measured**. Next tick: build, run `BatchedMatVecTierTests`, mutation-test the
slot mapping (swap two output pointers in the quad and confirm the mapping test fails), then A/B
concurrent decode against §3.4.2's 27.1 / 27.7 / 29.6 tok/s.

### 3.4.9 Concurrent-decode tiering: measured, +4.8%, defaulted ON — and a stale-binary near-miss

`SimdKernels.BatchedMatVecTierEnabled` is now **default ON** (`STINGRAY_BATCHED_MATVEC_TIER=0`
disables).

**Correctness.** `BatchedMatVecTierTests`, 14 cases: batch sizes 1-9 and 15 across Q4_K and F32
against the same function with the tier off, bit-exact. Plus the slot-mapping guard.
**Mutation-tested**: swapping output slots 1 and 2 inside the quad — every value still computed,
two merely filed under the wrong token — fails **10 of 14**. Restored, 14/14 green. Full suite with
the tier defaulted ON is running as this is written.

**Measurement**, interleaved arms, 4 samples each, SmolLM2-1.7B-Q4_K_M, 4 concurrent sessions:

| arm | samples (tok/s aggregate) | mean | range |
|---|---|---:|---|
| tier ON | 29.9, 30.3, 30.3, 30.6 | **30.28** | 29.9 – 30.6 |
| tier OFF | 28.9, 27.7, 29.8, 29.2 | 28.90 | 27.7 – 29.8 |

**+4.8%, with no overlap between the arms' ranges** (OFF max 29.8 < ON min 29.9), and the ON arm is
noticeably tighter. Single-session decode is unchanged (27.0-27.7 both arms) — the expected internal
control, since a batch of 1 takes the single-`MatVec` remainder and never enters the quad.

**The hypothesis was still wrong, and that is the more useful result.** §3.4.2 predicted that adding
weight reuse would "move substantially" if weight bandwidth were the bottleneck. Four-way weight
reuse now provably executes and buys **4.8%**, not 4x. Four concurrent users aggregate 30.3 tok/s
against 27.1 for one — 1.12x, not 4x. So weight-read amortisation is **not** the dominant cost in
CPU batched decode at this size, and `BatchForwardMulti`'s "amortizes weight reads N× across
concurrent users" remains a misleading comment even now that the amortisation is real.

The residual is most likely irreducible: in decode each sequence has its own KV cache and its own
O(context) attention pass. That work scales linearly with N and cannot be shared no matter how the
matmuls are scheduled. Only the weight matmuls were ever shareable, and they are apparently not
where the time goes. Recorded as the leading explanation, **not** as a measured one — confirming it
needs a per-stage decode profile, which has not been run.

### The stale-binary near-miss, recorded because it nearly produced a false negative

The **first** A/B of this switch returned a clean null result: TIER=0 and TIER=1 indistinguishable
across two interleaved pairs. That looked like a decisive falsification and was about to be written
up as one.

It was an artefact. `tools/session-bench` is **not in `OpenTail.Stingray.slnx`**, so
`dotnet build -c Release` at solution level silently did not rebuild it. The executable being run
was timestamped **16:51** — predating both the harness fix and the tiering. Both "arms" were the
same code.

What caught it was the execution counter added specifically to distinguish "no effect" from "never
invoked": its line simply did not appear in the output, because the binary printing the output
predated the line. After an explicit `dotnet build tools/session-bench`, the counter reads
**39,480 / 45,696 invocations with the tier ON and exactly 0 with it OFF**, and the real +4.8%
appears.

Two things to carry forward:

1. **A tool outside the solution file will silently run stale.** Correcting the first instinct
   here: adding `session-bench` to `OpenTail.Stingray.slnx` would be the WRONG fix. Every `tools/*`
   project is deliberately outside it — `attn-bench`, `kernel-bench` and `SpirvGen` all are — so
   adding one would break the convention and make every solution build compile benchmark tools.
   The fix applied instead is to make staleness visible in the output: `session-bench` now prints
   the build timestamp of its own assembly *and* of the kernel assembly under test, with the note
   that both must be newer than the last edit. A stale run now announces itself rather than being
   discoverable only in hindsight. `attn-bench` has the same exposure and no such stamp.
2. **An A/B needs a positive control that the treatment was applied.** "Both arms measured the same"
   is not evidence of no effect unless the treated arm can be shown to have been treated. The
   counter cost five minutes and converted a confident false negative into a real result.

### 3.4.10 Batched decode is now instrumentable (written, unbuilt)

§3.4.9 named per-sequence attention as the leading explanation for why 4-way concurrent decode
aggregates only 1.12x a single session, and was explicit that this was **not** measured. Checking
what it would take to measure it produced a blunt answer: `BatchForwardMulti` contained **zero**
`DecodeProfileTimers` calls. The decode profiler instruments only the sequential `ForwardCore`
path, so the multi-sequence path — the one whose scaling is the open question — could not be
profiled at all, only speculated about.

Added, mirroring `ForwardCore`'s categories so the two decode paths compare bucket for bucket:
`RmsNorm`, `QkvProj`, `RoPE`, `Attention`, `OutProj`, `Ffn`.

Two deliberate choices:

- **RoPE/QK-norm/cache-append and the attention call itself are timed separately**, even though
  they sit in the same per-sequence loop. The open question is specifically whether per-sequence
  *attention* is the irreducible residual; folding the cache bookkeeping into the same bucket would
  produce a number that cannot answer it.
- **`CountToken()` is called once per SEQUENCE**, not once per batch, so per-token averages mean
  the same thing here as on the sequential path.

**Status: written, NOT built and NOT run** — a full suite was live, and building under one is
exactly what the loop rules forbid. Next tick: build, then run `STINGRAY_PROFILE_DECODE=1`
against 1 and 4 concurrent sessions and compare the per-token bucket split. The prediction on
record from §3.4.9 is that `Attention` per token stays roughly constant from 1 to 4 sessions while
`QkvProj`/`Ffn` per token fall — if instead all buckets stay flat, the residual is somewhere else
entirely and §3.4.9's explanation is wrong.

### 3.4.11 §3.4.9's explanation was WRONG. Attention is 3% of decode; FFN is 70%.

The instrumentation from §3.4.10 was built and run. `STINGRAY_PROFILE_DECODE=1` with
`STINGRAY_SESSIONBENCH_DECODE_ONLY=<N>` (one concurrency level per process, because
`DecodeProfileTimers` is cumulative and has no `Reset`), SmolLM2-1.7B-Q4_K_M, 64 decoded tokens per
session.

| bucket | 1 session ms/tok | 4 sessions ms/tok | change | share of trunk |
|---|---:|---:|---:|---:|
| **FFN** | 25.15 | 22.31 | −11.3% | **~70%** |
| QKV projection | 7.18 | 5.95 | −17.2% | ~19% |
| Output projection | 2.451 | 2.193 | −10.5% | ~7% |
| **Attention** | 1.184 | 0.989 | **−16.5%** | **~3.1%** |
| RoPE | 0.238 | 0.196 | −17.6% | ~0.6% |
| RmsNorm | 0.047 | 0.027 | −42% | ~0.1% |
| **total trunk** | **36.25** | **31.66** | **−12.7%** | 100% |

Aggregate throughput 25.6 → 30.2 tok/s (1.18x for 4x the sessions).

**The prediction on record was:** "`Attention` per token stays roughly constant from 1 to 4 sessions
while `QkvProj`/`Ffn` per token fall — if instead all buckets stay flat, the residual is somewhere
else entirely and §3.4.9's explanation is wrong."

Neither branch happened. **Every bucket fell, by roughly the same 10–18%**, and `Attention` fell
*more* than FFN did. Per-sequence attention is not a constant residual, and it could never have
explained the flatness anyway: **it is 3.1% of decode trunk time.** Even making it free would move
decode by 3%. §3.4.9's leading explanation is retracted.

**What the data actually says.** FFN is ~70% of CPU decode, and going from 1 to 4 sequences buys it
only 11%. Since the tier is default-ON, those FFN matmuls *are* running through `MatVec4In` — the
weight rows genuinely are read once and dotted against four activation vectors. The reuse happens
and is worth ~11%.

So the premise underneath all of this — that CPU batched decode is weight-bandwidth-bound and
therefore N-way weight reuse should approach N-way throughput — **is false at this size.** The
mechanism is visible in the kernel: `MatVec4In` on Q4_K shares the weight *dequantisation and read*
across four inputs, but the FMA work is still 4x. If the dequant+read were dominant, sharing it
would nearly quarter the cost; measured, it removes ~11%. Therefore the per-input arithmetic, not
the weight stream, is where decode time goes on this box.

**Consequences worth carrying:**

- `BatchForwardMulti`'s comment "amortizes weight reads N× across concurrent users" is doubly
  misleading: the amortisation now genuinely happens, and it is still worth only ~11%. The comment
  should be corrected to say what it buys.
- Multi-user CPU decode throughput is close to its ceiling on this box. Four users aggregate 1.18x
  one user. Further work on *batching* decode has little left to win; the lever is the per-input
  FFN cost itself.
- This retro-justifies the +4.8% from §3.4.9's tiering as roughly the whole available prize, not a
  disappointing fraction of a larger one.
- Attention being 3% of *decode* is not in tension with it being ~27–33% of *prefill*: prefill
  attention is O(N²) over the whole prompt, decode attention is one query against the cache.

**Method note.** This is the second time in this plan that a stated mechanism turned out wrong and
only measurement settled it (the first being the stale-binary false negative in §3.4.9). Both were
caught because the instrument was built before the conclusion was written down, rather than after.

---

## 3.5.x Milestone 0 stopping decisions — resolved where evidence exists

### 3.5.1 "Exact state reuse without a large batching rewrite" — condition did NOT trigger

This was the gate that would have halted implementation and demanded an ADR first. It does not fire.

`HotSessionGreedyReplayTests` (§8.1) demonstrates **token-exact** greedy continuation across three
appended turns on a real CPU dense model, against a from-scratch replay that shares no state, no
cache and no engine with the session. No batching rewrite was performed to achieve it:
`ContinuousBatchingEngine.GenerateRetainedChunksAsync` and `RetainedSequenceState` already existed
and were used unmodified. The only engine-side changes this work required were a bug fix
(`MaxSeqLen`, §3.4.4) and an opt-in performance switch (§3.4.9) — neither structural.

**Scope, stated so the checkbox is not read as more than it is.** The proof covers a single
sequence, CPU dense, `maxBatchSize: 1`, 2048 context, 6 tokens per turn. It does not cover MoE,
per-layer-head-dim models, GPU backends, compressed KV, or contexts long enough to engage eviction.
Exact reuse is established for the reference lane the plan names, not universally.

### 3.5.2 "Native-kernel results may change priorities but never block session milestones" — held

The kernel programme (`docs/cpu-architecture-kernel-opportunities.md`) ran to completion and closed
while the session work proceeded. No session milestone waited on a kernel result at any point.

It did change priorities, which is exactly what this decision permits:

- The fp32 multi-input tiering (§3.4.9) is a kernel-level change that came *out of* session
  measurement rather than the kernel programme, and its result (§3.4.11) **retired an entire
  direction** — "make batched decode scale" — by showing CPU decode is not weight-bandwidth-bound
  and four users already aggregate 1.18x one user.
- The kernel programme's own reframing (CPU prefill ~2x faster than Vulkan on this box) redirected
  effort toward CPU without gating any session deliverable.

The decision is satisfied in the direction that matters: kernel work informed session priorities and
never became a prerequisite for them.

### Still unresolved, and why

- **"If the local planner/coder/reviewer models are not useful in a manual interleave, do not build
  an autonomous loop to compensate for model quality."** Requires actually running three role models
  in a manual interleave and judging output quality. No such trial has been run. This is a
  product-judgement gate, not a measurement, and guessing at it would be worse than leaving it open.
- **"If vision is not interactive on a declared reference host, keep it optional/server-targeted."**
  Requires a vision timing run on a declared reference host. Not attempted; the reference host has
  not been declared either.

Both are left unchecked deliberately. Checking them from inference rather than trial is precisely
the failure mode the surrounding programme keeps recording.

### 3.4.12 A self-inflicted suite failure, and the drift test earning its keep in the other direction

The suite run covering §3.4.10's instrumentation came back **2282 total, 4 failed** — one more than
the three known ones. The new failure was mine:

```
1 entr(y/ies) in KnownEnvironmentVariables.All are no longer read anywhere in src/ and
should be removed: STINGRAY_SESSIONBENCH_DECODE_ONLY
```

I had registered the bench tool's decode-profile switch in `KnownEnvironmentVariables` out of habit
from registering the engine switches earlier in this session. But that list is scanned from
`src/**/*.cs` and documents **what the engine reads**. `STINGRAY_SESSIONBENCH_DECODE_ONLY` is
read in `tools/session-bench/Program.cs`, which is not under `src/`, so the entry had no
corresponding usage.

Removed, and a comment left at the read site explaining why it is deliberately unlisted, so the next
person does not "fix" the omission and reintroduce the failure. Registry test back to 8/8.

Worth noting: every previous encounter with this test in this session was the *missing entry*
direction (a variable read but not listed). This is the first time it caught the reverse — a listed
variable nothing reads. The distinction matters for the engine switches added this session
(`STINGRAY_BATCHED_MATVEC_TIER`, `STINGRAY_PER_LAYER_HD_PREFILL`,
`STINGRAY_MOE_BATCHED_PREFILL`, `STINGRAY_FLASH64_STRIDED_GEMM`, `STINGRAY_GEMMA4_PROBE`):
all of those *are* read from `src/` and correctly belong.

### 3.4.13 "Bytes per token for each tested state encoding" — sessions have exactly ONE encoding

§3.4.3 measured fp32 KV at 384 KiB/token and flagged that the checkbox asked for "each tested state
encoding" while only one had been measured. Investigating what the others would be produced a
blunter answer: **for the session runtime there are no others.**

**Compressed KV does not compose with the engine `HotSession` is built on.**
`ContinuousBatchingEngine` reaches the forward pass through exactly two entry points —
`PrefillWithCache` (admission, `ContinuousBatchingEngine.cs:873,916`) and `BatchForwardMulti`
(decode, `:474`). Both throw unconditionally when a TurboQuant cache is present:

| entry point | line | behaviour with TQ |
|---|---|---|
| `ForwardPass.PrefillWithCache` | `:3922` | `NotSupportedException` |
| `ForwardPass.BatchForwardMulti` | `:3960` | `NotSupportedException` |
| `ForwardPass.PrefillPackedMulti` | `:4185` | `NotSupportedException` |
| `ForwardPass.BatchVerify` | `:1730` | `NotSupportedException` |

So a `ForwardPass` constructed with TurboQuant cannot be driven by `ContinuousBatchingEngine` at
all, and therefore cannot back a `HotSession`. KVarN and Lloyd-Max are reachable only from the
single-sequence `InferenceEngine` path.

**The bf16 KV mode is not a second encoding either.** `STINGRAY_KV_DTYPE=bf16` makes
`PagedKvCache.WriteKv` round each value through `ToBf16Precision` — but it stores the result in a
`float`, and `_pageBytes` is unconditionally `PageSize * _kvDim * 2 * sizeof(float)`. It reduces
*precision*, not *residency*. It is a quality experiment, not a memory one, and gives 384 KiB/token
exactly like fp32.

**Consequence for the roadmap, which is the point of recording this.** §3.4.3's residency finding
(one 16K session = 6 GiB, larger than the model) has no existing remedy available to sessions. The
compressed KV that already ships in this tree cannot be applied to them without first making
TurboQuant compose with continuous batching — which is its own piece of work, currently invisible in
the plan. "Bounded state residency measured in bytes" is listed among the runtime's defining
properties in §0; today the only lever is *how many* tokens are retained, never *how many bytes each
token costs*.

**Epistemic status: derived from the throw sites, not executed.** Four explicit
`NotSupportedException`s and a page-size expression are a much safer kind of reading conclusion than
"this will work" — the code states its own refusal. But it has not been run, and per the standing
rule that distinction is worth keeping. The cheap confirmation is to construct a TQ `ForwardPass`,
hand it to a `ContinuousBatchingEngine`, and watch it throw; that needs a model load and is queued.

### 3.4.14 §3.4.13 EXECUTED — conclusion confirmed, and the reading missed a detail

§3.4.13 was derived from four `NotSupportedException` throw sites and explicitly flagged as
"derived from the throw sites, not executed". It has now been executed:
`TurboQuantSessionCompositionTests.TurboQuantForwardPass_CannotBackAHotSession`.

**Result: confirmed.** A TurboQuant-backed `ForwardPass` driving a `HotSession` turn lands in
`SessionOperationState.Failed`. Sessions really do have exactly one state encoding.

**Mutation-tested**, and the mutation is the meaningful one here: remove the `EnableTurboQuant` call
and the same turn reaches `Completed`. So the failure is caused by TurboQuant specifically, not by
anything else in the fixture — which is the only way a test asserting a *limitation* can be
trusted. Restored; sessions suite 25/25.

**What the reading got wrong.** The first run of this test failed, and not because the conclusion
was wrong — because `EnableTurboQuant(bits: 4)` threw
`ArgumentException: No codebook for 4-bit, d=64` before TurboQuant was ever engaged. The Lloyd-Max
codebooks are per `(bits, headDim)` and 4-bit/64 is simply not shipped. Two things follow:

1. The guard caught only `NotSupportedException`; a codec rejecting a shape surfaces as
   `ArgumentException`. Both now skip.
2. Had the assertion been written the lazy way — "the turn must not complete" — this run would have
   **passed for entirely the wrong reason** and been recorded as confirming §3.4.13. It failed
   loudly instead only because the fixture threw rather than returning a state. That is luck, and
   the mutation test is what converts it into evidence.

**The test asserts a limitation on purpose.** If someone later makes TurboQuant compose with
continuous batching, it fails — and its message says that is good news, that §3.4.13's roadmap
conclusion has gone stale, and that the plan should be updated rather than the assertion relaxed.
A limitation worth recording in a plan is worth a test that notices when it stops being true.

### 3.4.15 Prefix capture/fork cost — copy-on-write confirmed by measurement

`PagedKvCache.ForkSharedPrefix` refcounts the shared page pool and copies only an `int[]` of page
slots, so its cost should scale with prefix **pages** (`prefixLength / 16` ints) and not with prefix
**bytes** (`prefixLength × 384 KiB`). Measured rather than assumed, because if the sharing were not
happening, every "fork a session" feature in §0 would quietly be a deep copy of gigabytes.

`tools/session-bench` with `STINGRAY_SESSIONBENCH_FORK=1`, best of 5 per size, SmolLM2:

| prefix tokens | fork µs (ascending sweep) | fork µs (descending sweep) | KV shared |
|---:|---:|---:|---:|
| 256 | 85.7 | 36.4 | 96 MiB |
| 512 | 81.5 | 40.5 | 192 MiB |
| 1024 | 35.6 | 59.2 | 384 MiB |
| 2048 | 29.7 | 42.1 | 768 MiB |
| 4096 | 34.7 | 39.2 | **1,536 MiB** |

**Flat: ~30–60 µs across a 16x range of prefix length.** Forking a 4096-token prefix shares 1.5 GiB
of KV in under 40 µs. A deep copy of that much data, even at an optimistic 10 GB/s, would be ~150 ms
— roughly **4000x** slower. Copy-on-write is real.

**The sweep is bidirectional for a reason.** The first (ascending-only) run showed 79.7 µs at 256
falling to 44.7 µs at 4096, which reads as "forking gets cheaper with bigger prefixes" — not a
plausible property of the thing being measured. Running the sizes descending as well shows the two
high readings attach to the *first two measurements of the process*, whichever size they are: 256
and 512 are slowest ascending, 1024 and 2048 slowest descending. It was JIT/allocator warmup, and
the per-size warm-up call inside the loop was not enough to absorb it.

Had only the ascending sweep been recorded, the table would have contained a real-looking inverse
trend with no mechanism, and the natural next step would have been inventing one. The control cost
one extra pass.

**Not measured, and distinct from this**: "capture current prefix hit length and divergence
diagnostics" is a separate checkbox about `ContinuousBatchingEngine`'s prompt-prefix cache
(`PrefixCacheHits`/`PrefixCacheMisses`/`PrefixCacheUsedBytes`), which is a different mechanism from
`PagedKvCache` block sharing. Still open.

**No engine code changed for this** — the measurement lives entirely in `tools/session-bench`, so
the suite result above (2283 / 3 known failures) remains the current state of the tree.

### 3.4.16 Prefix hit length and divergence — neither existed; both added

This baseline could not be "captured" because the engine did not produce the numbers. What
`ContinuousBatchingEngine` exposed was `PrefixCacheHits` / `PrefixCacheMisses` / `Evictions` /
`UsedBytes` / `Entries` — **counts only**. `TryTakePrefix` matched with an all-or-nothing
`SequenceEqual`, so on a miss nothing recorded how close the match came.

Both gaps matter for different reasons:

- **Hit length.** One hit reusing 4000 tokens and one reusing 8 are both "a hit". Only the length
  tells you how much prefill was actually skipped, which is the quantity §3.4.1's 24x/71x warm-vs-cold
  result is made of.
- **Divergence.** A miss counter cannot distinguish *an unrelated prompt* from *one that matched
  every token but the last*. Those call for opposite responses: the first is expected and fine, the
  second means retained prefixes are cut at the wrong granularity and nearly all the reusable work
  is being discarded. Indistinguishable before this change.

Added to `IContinuousBatchingObservability`'s surface:

| member | meaning |
|---|---|
| `PrefixCacheHitTokens` | cumulative tokens served from retained prefixes — prefill work skipped |
| `PrefixCacheLastHitLength` | tokens reused by the most recent hit |
| `PrefixCacheLastMissLongestMatch` | **divergence point**: leading tokens that matched the best *eligible* retained prefix before differing |
| `PrefixCacheMissMatchedTokens` | cumulative of the above — the recoverable-work estimate |

Two implementation choices worth stating:

1. The divergence scan runs **only on the miss path**, after the hit loop fails, so the hit path
   keeps its vectorised `SequenceEqual` untouched. Admission-time cost over a small cache, not
   per-token.
2. It considers **only candidates the hit loop was eligible to use** (`Tokens.Length <= maxLength`).
   Counting a too-long entry's overlap would report reuse that was never actually available — a
   diagnostic that flatters the cache is worse than none.

**Tests**: `PrefixCache_ReportsHitLength_NotJustHitCount` and
`PrefixCache_ReportsDivergencePointOfAMiss`, both on the existing `CharTokenizer` fixture (1 char =
1 token) so the expected numbers are exact rather than approximate. The divergence case retains
`"system:"` then submits `"systemXbeta"` — six characters agree, the seventh differs, divergence = 6.

**Mutation-tested**: disabling the divergence scan (`while (k < n && ...) k++` removed) fails 1 of
the 2. Restored; both green. Full suite running.

**Not yet measured with these**: the numbers now exist but no baseline run has been taken with them
on a real workload. The instrument was built this tick; reading it is separate work, and the
checkbox above is ticked for *capture* — the plan's word — not for interpretation.

### 3.4.17 Refining §3.4.11: decode IS bandwidth-bound at N=1 and is NOT at N=4

§3.4.11 concluded "CPU decode at these sizes is not weight-bandwidth bound." That statement is too
flat, and the arithmetic says something sharper. Correcting it here rather than editing it silently.

Inputs, all already measured: SmolLM2-1.7B-Q4_K_M weights are **1007 MiB ≈ 1.056 GB** (the runtime
reports "Pre-faulted 1.06 GiB"); single-session decode is **27.1 tok/s**; four-session aggregate is
**30.2 tok/s**, i.e. **7.55 decode STEPS/s** since each step emits one token per sequence. Every
step reads the whole weight set once.

| concurrency | decode steps/s | weight bytes read/s | share of the 36.8 GB/s ceiling |
|---:|---:|---:|---:|
| 1 | 27.1 | **28.6 GB/s** | **78%** |
| 4 | 7.55 | **8.0 GB/s** | **22%** |

**At N=1 decode really is close to the memory wall — 78% of the box's ceiling.** Batching does
exactly what it is supposed to: it removes the redundant weight re-reads and drops the weight
traffic to 22% of ceiling. The throughput does not follow, because by then the bottleneck has moved.

The per-step numbers show the same thing from the other side. Per-token trunk time is 36.25 ms at
N=1 and 31.66 ms at N=4, so a step costs 36.25 ms with one sequence and **126.6 ms** with four —
**3.5x**. A purely bandwidth-bound step would have been 1.0x (same weights, same read); a purely
FMA-bound step would have been 4.0x. At 3.5x, a four-sequence step is roughly 85% of the way to
compute-bound.

So the corrected statement is: **decode is bandwidth-bound at N=1, batching immediately takes it off
the wall, and from N=2 upward it is dominated by per-input arithmetic that scales with N.** That is
why the marginal return collapses so fast, why the §3.4.9 tiering was worth only +4.8%, and why
further batching work has little left — all three follow from one model instead of three separate
observations.

**Provenance caveat, and why the STREAM checkbox stays OPEN.** The 36.8 GB/s ceiling is *inherited*
from `docs/HANDOVER.md`, not measured by me. That document also describes the box as
"6-core/12-thread Zen 3" while this one is a Ryzen 7 5700G (8c/16t) — the figures may come from a
sibling machine, and `Environment.ProcessorCount` on this box has been observed reading 12 rather
than 16. The percentages above are therefore accurate to within whatever that discrepancy is worth,
and the ratios (78% vs 22%, 3.5x vs 4.0x) are robust to it, but "78% of ceiling" should not be
quoted as a measured figure until STREAM is actually run here. `- [ ] Measure STREAM bandwidth`
remains unticked deliberately.

### 3.4.18 STREAM measured here — and the naive number is wrong by exactly the write-allocate factor

§3.4.17 flagged that its percentages rested on a 36.8 GB/s ceiling inherited from
`docs/HANDOVER.md` rather than measured. Measured now, 12 threads, 256 MiB per array (768 MiB
total, far past any LLC), best of 12:

| kernel | naive GB/s | actual DRAM streams | corrected GB/s |
|---|---:|---|---:|
| Copy `c=a` | 23.5 | read a + RFO c + write c = 3 (not 2) | **35.3** |
| Scale `b=k*c` | 23.4 | 3 (not 2) | **35.1** |
| Add `c=a+b` | 26.7 | 4 (not 3) | **35.6** |
| Triad `a=b+k*c` | 26.9 | 4 (not 3) | **35.9** |

The naive column reports the peak as **26.9 GB/s**, which would have made §3.4.17's N=1 figure
28.6 GB/s = **106% of ceiling** — impossible, and the tell that the accounting was wrong rather
than the measurement.

**Write-allocate is the reason.** On x86 without non-temporal stores, writing a cache line that is
not resident first *reads* it (read-for-ownership). So every written element costs a DRAM read as
well as a write, and classic STREAM's logical byte count understates real DRAM traffic by one
stream per written array. Correcting for it, all four kernels agree at **~35–36 GB/s** — spread of
0.8 GB/s across kernels with quite different read/write mixes, which is what a real ceiling looks
like.

The give-away was in the naive numbers themselves: Add and Triad (nominally *more* streams) measured
*faster* than Copy and Scale. More traffic cannot be faster; unequal undercounting can look that
way. Ratio 26.9/23.5 = 1.14, and the predicted ratio from mis-accounting is (4/3)/(3/2) = 1.125.

**Result: the inherited 36.8 GB/s is confirmed**, within a couple of percent, and §3.4.17's
percentages stand unchanged (N=1 at ~79% of ceiling, N=4 at ~22%). The provenance caveat there can
be lifted.

**Incidental resolution of the box discrepancy.** §3.4.17 noted `HANDOVER.md` describes a
"6-core/12-thread Zen 3" while other notes call this a Ryzen 7 5700G (8c/16t).
`Environment.ProcessorCount` reads **12** here, and the bandwidth matches HANDOVER's figure — so the
HANDOVER description is the accurate one for this machine and the 8c/16t attribution is wrong.
Worth correcting wherever it is repeated, since core count feeds thread-scaling arguments.

**Caveat retained:** this is a scalar C# STREAM, not the reference Fortran/C one with non-temporal
stores. It establishes the ceiling well enough to validate a 79%-vs-22% distinction; it should not
be quoted to three significant figures.

### 3.4.19 The 16K row: no model on disk can serve it with flash-64

Before re-attempting the 16K cold-TTFT point, an inventory of what is actually available, because
§3.4.7 established that the model choice silently selects the attention kernel:

| model | trained context | headDim | reaches 16K? | flash-64? |
|---|---:|---:|---|---|
| SmolLM2-1.7B-Q4_K_M | 8,192 | 64 | **no** | yes |
| OLMoE-1B-7B | 4,096 | 128 | **no** | no |
| Qwen3-0.6B-Q8_0 | 40,960 | 128 | yes | **no** |
| Qwen3-8B-Q4_K_M | 40,960 | 128 | yes | **no** |
| gemma-4-E4B q4_0 | 131,072 | per-layer | yes | no (sequential trunk) |

`PrefillCoreAttention` dispatches flash-64 only at `headDim == 64`. **Every long-context model here
has headDim 128**, and the one model with headDim 64 tops out at 8,192. So the 16K point cannot be
measured on the fast attention path with anything currently on disk — this is a model-availability
constraint, not an effort one, and no amount of re-running changes it.

**Decision: measure it anyway, on Qwen3-0.6B, and label it.** The 16K row's job in this plan is the
warm-vs-cold *ratio* — the product claim — and that ratio is valid regardless of which attention
kernel the cold arm used. What it is not is comparable to the SmolLM2 4K row: different model,
different kernel, different absolute t/s. Recorded with that attached rather than dropped, since a
ratio measured on the slower kernel is if anything conservative for the cold arm.

`Repeats` is now overridable (`STINGRAY_SESSIONBENCH_REPEATS`) and the 16K run uses 1, per
§3.4.7's own recommendation: at minutes per cold prefill, best-of-3 buys very little against a
one-sided-interference argument calibrated on millisecond-scale measurements.

To measure 16K on the fast path would need a headDim-64 model with ≥16K trained context — none is
present; `download-model.ps1` would have to fetch one. Left as a stated gap.

### 3.4.20 The incumbent comparisons are FEASIBLE here — and llama-server already has slot save/restore

The three remaining §3.4 checkboxes were assumed to need external setup. They do not:
`tools/llama.cpp/llama-server.exe` is already present in this tree (alongside `llama-bench`,
`llama-batched-bench` and the full `ggml-cpu-*` variant set), fetched by
`scripts/setup-llamacpp.ps1`. No download is required to run any of them.

Its flag surface maps onto the three checkboxes almost one-to-one:

| checkbox | llama-server mechanism |
|---|---|
| "llama-server slot/prefix behavior under an equal byte budget" | `-np/--parallel N` (server slots), `--slots` monitoring endpoint, `--cache-reuse N` (min chunk reused from cache via KV shifting) |
| "a reverse-proxy save/restore pattern" | **`--slot-save-path PATH`** — llama-server can persist a slot's KV cache to disk and restore it |
| "llama-swap interleaving with three role models" | still needs llama-swap itself, which is NOT present |

**`--slot-save-path` is the one worth pausing on.** The plan's §1.3 already concedes that "prompt
caches, named slots and multimodal encoder caches all exist elsewhere", and §0 rests the
differentiation on the *integrated* object — identity, exact compatibility, transactions,
revisions, leases, durable recovery. That framing survives this. But durable KV save/restore is not
merely adjacent to the incumbent, it is a **shipped flag** on the binary sitting in this repository,
and the honest comparison is therefore not "we can persist state and they cannot" but "what does
identity/transactionality/compatibility-checking add on top of a slot file". Any differentiation
claim written before this measurement should be treated as unverified.

**Measurement plan for the next tick** (deliberately concrete so it is not re-derived):

1. `llama-server -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf -c 8192 -np 1 --slots`
2. POST `/completion` with a ~1024-token prompt; record `timings.prompt_n` and `timings.prompt_ms`.
3. POST again with the same prefix plus a 32-token suffix; record the same fields. The reused
   portion appears as a reduced `prompt_n` (only the new tokens are evaluated).
4. That yields the incumbent's warm-vs-cold prefill ratio at an equal context, directly comparable
   to §3.4.1's **24.0x at 1K / 71.1x at 4K**.
5. Then repeat with `--slot-save-path` to time save and restore, which is the reverse-proxy pattern.

**Not started this tick**: a `session-bench` 16K run holds the box, and starting a second model load
is exactly what the loop rules forbid. `llama-server --help` was run (no model load, no compute) to
establish the flag surface above.

### 3.4.21 The 16K row: 219x, and the warm/cold ratio grows with context as predicted

Qwen3-0.6B-Q8_0, `Repeats=1`, fixed harness:

| context | cold | warm suffix | speedup | model |
|---:|---:|---:|---:|---|
| 1024 | 6,623 ms | 267 ms | 24.0x | SmolLM2-1.7B |
| 4096 | 31,076 ms | 437 ms | 71.1x | SmolLM2-1.7B |
| **16384** | **592,709 ms** | **2,704 ms** | **219.2x** | Qwen3-0.6B |

The ratio tracks the O(context) / O(suffix) model closely: 1K→4K is 4x context for 3.0x ratio,
4K→16K is 4x context for 3.1x ratio. **The product claim strengthens with context**, which is the
regime the runtime exists for.

**The cold arm is the number to stare at: 592.7 seconds — nearly ten minutes — to prefill 16,416
tokens, i.e. 27.7 t/s.** Against SmolLM2's 4,128 tokens at 133 t/s, that is a 4.8x collapse in
prefill *rate* for 4x the tokens. Quadratic attention on the non-flash path (§3.4.19: Qwen3-0.6B is
headDim 128, so flash-64 never engages) is doing exactly what quadratic attention does. Cross-model
caveat from §3.4.19 stands — this row is not comparable to the SmolLM2 rows in absolute terms, only
in ratio.

**A second data point for the §3.4.17 bandwidth model, and it is not a clean confirmation.**
Qwen3-0.6B decode: 50.5 tok/s (1 session) → 60.7 (2) → 69.3 (4) = **1.37x** at four sessions,
against SmolLM2's 1.18x. At ~0.6 GB of Q8_0 weights, N=1 implies 50.5 × 0.6 = ~30 GB/s — also close
to the ~36 GB/s ceiling, so both models start bandwidth-bound as the model predicts. But the smaller
model gets *more* out of batching, which the simple "off the wall, then compute-bound" story does not
by itself explain. Recorded as an open observation rather than folded into the model; two points is
not a curve, and the honest position is that §3.4.17 explains SmolLM2 well and Qwen3-0.6B only
partly.

### 3.4.22 Incumbent measured: llama-server slot prefix reuse vs OpenTail HotSession

`scripts/bench-llama-server-prefix.ps1` (new, a repo script for the same reason session-bench is).
Same model, same context budget, one slot. The cold arm is forced cold by **restarting the server**,
not by sending a different prompt — a different prompt would change the work as well as the cache
state, confounding "cache miss" with "more tokens".

| context | llama-server cold | llama-server warm | **llama-server** | **OpenTail** |
|---:|---:|---:|---:|---:|
| 1024 | 5,748 ms (1056 tok) | 281 ms (32 tok) | **20.4x** | **24.0x** |
| 4096 | 29,209 ms (4128 tok) | 855 ms (32 tok) | **34.2x** | **71.1x** |

**At 1K it is parity.** llama-server's slot prefix caching does the same job: warm `prompt_n` drops
to exactly the 32 suffix tokens, and warm latency (281 ms) is within 5% of OpenTail's (267 ms).
Anyone claiming the session runtime wins on *prefix reuse performance* at short context is wrong,
and the plan should not make that claim.

**At 4K OpenTail's ratio is 2.1x better, and the whole difference is in the warm arm's context
scaling.** Cold arms scale nearly identically (llama-server 5.1x for 4x context, OpenTail 4.7x).
Warm arms diverge: llama-server 281 → 855 ms (**3.0x**), OpenTail 267 → 437 ms (**1.6x**). Both
evaluate the same 32 new tokens, so the growth is in what each does *besides* evaluating them.

**The measurement scopes are not identical, and the asymmetry favours llama-server.** OpenTail's
number is total `RunTurnAsync` wall time — session machinery, tokenisation, ledger, and one
generated token included. llama-server's is `timings.prompt_ms`, prompt evaluation only, excluding
HTTP and generation. So OpenTail is being charged for strictly more work and still wins the warm arm
at 4K. Stated explicitly because a comparison whose bias direction is unexamined is not evidence.

**What this does and does not support.** It does not support "we have prefix reuse and they do not"
— they do, and `--slot-save-path` (§3.4.20) means they can persist it too. It does support the
narrower, measured claim that OpenTail's warm path degrades more slowly with context. The plan's
§1.3 differentiation — identity, exact compatibility, transactions, revisions, leases, multi-model
interleave — is untouched by this measurement either way, and remains the thing that has to carry
the argument.

**Not yet measured**: why llama-server's warm arm scales 3.0x. Candidates are cache validation over
the full prompt, KV shifting under `--cache-reuse`, or genuine attention cost differences. Not
investigated — recorded as the obvious follow-up rather than guessed at.

### 3.4.23 Reverse-proxy save/restore measured — and it exposes a 2x KV residency gap

`bench-llama-server-prefix.ps1 -SlotSave`. Prime a slot, persist its KV via
`POST /slots/0?action=save`, **erase** it so the restore cannot be served from what is already
resident, then `action=restore`. The final column is the proof the restore was real rather than a
no-op: `prompt_n` after restoring must collapse to the suffix length.

| prefix | save ms | restore ms | file | post-restore `prompt_n` | cold re-prefill |
|---:|---:|---:|---:|---:|---:|
| 1024 | 93.9 | **54.2** | 192.0 MiB | 32 ✓ | 5,430 ms |
| 4096 | 334.9 | **224.5** | 768.1 MiB | 32 ✓ | 27,410 ms |

**Restoring a 4096-token session from disk costs 224 ms against 27,410 ms to re-prefill it — 122x.**
That is the incumbent's answer to surviving a restart, and it works today with a flag. OpenTail has
no durable equivalent at all (Milestone 3, entirely unstarted), so this is a capability gap, not a
performance comparison.

Throughput checks out as disk-bound: 768 MiB written in 334.9 ms ≈ 2.3 GB/s, read in 224.5 ms ≈
3.4 GB/s.

#### The file sizes exposed something more interesting

192.0 MiB for 1024 tokens is exactly **192 KiB/token** — precisely **half** OpenTail's measured
384 KiB/token (§3.4.3) on the *same model*.

My first hypothesis was that OpenTail was over-allocating KV heads: `HANDOVER.md` describes
SmolLM2-1.7B as "32 heads, 8 KV heads, GQA", which would make OpenTail's `kvHeads=32` a 4x waste.
Checking the GGUF metadata directly refutes that — `llama.attention.head_count_kv = 32`. **SmolLM2
is MHA, not GQA**, and OpenTail's figure is arithmetically correct:
24 × 32 × 64 × 2 × 4 B = 393,216 B/token.

The real cause is dtype. 24 × 32 × 64 × 2 × **2** B = 196,608 B/token = 192 KiB — an exact match.
**llama-server defaults to fp16 KV; OpenTail stores fp32.**

That sharpens §3.4.13's roadmap point considerably. It is not merely "compressed KV cannot back a
session"; it is that **the incumbent's default KV is half our size**, and OpenTail's nearest
equivalent — `STINGRAY_KV_DTYPE=bf16` — reduces *precision* while storing fp32 anyway (§3.4.13),
so it buys nothing in bytes. A 16K session costing 6 GiB here would cost 3 GiB there, before any
quantised codec enters the picture. Halving KV residency looks like the single highest-leverage
item available to the residency story, and it is not currently in the plan.

#### Correction to `docs/HANDOVER.md`

That document's model description — "24 layers, headDim 64, 32 heads, **8 KV heads, GQA**" — is
wrong for this GGUF; `head_count_kv` is 32. Worth fixing there, because KV-head count feeds
memory-traffic arithmetic and a 4x error in it would misattribute exactly the kind of bandwidth
result §3.4.17 depends on. (§3.4.18 already corrected that document's core count in the other
direction, where it was right and other notes were wrong — it is not uniformly unreliable, but it
is not authoritative either.)

---

## 8.2 Milestone 1 invariant tests — and a real race the EOS test found

Six of Milestone 1's required invariant tests are now satisfied. Two were newly written; four were
already covered and simply never ticked — verified by mapping each to a named test rather than
assumed:

| invariant | test |
|---|---|
| resumed greedy output equals full replay exactly | `HotSessionGreedyReplayTests` (§8.1) |
| **EOS completion** | `HotSession_EosCompletion_StopsEarlyAndCommits` (new) |
| **maximum-token completion** | `HotSession_MaximumTokenCompletion_StopsExactlyAtTheBudget` (new) |
| duplicate operation ID, identical request | `DuplicateOperation_WithSameDigestIsIdempotent_...` |
| duplicate operation ID, different request | same test, the rejection half |
| stale fenced worker cannot commit | `StaleLease_CannotTransitionOrCommit` |

The two new tests are a deliberate pair: EOS must stop **strictly before** the token budget, and
max-token must stop **exactly at** it. Mutation-tested by making the max-token arm terminate on EOS
instead — if the pair could not tell the two termination reasons apart it would still pass. It
fails.

### The race

`HotSession_EosCompletion_StopsEarlyAndCommits` passed in isolation and failed intermittently in the
full class — 3 of 5 runs. The reason was not a test artefact:

```
state=Failed  reason=The retained engine completed without a retained turn outcome.
```

`ContinuousBatchingEngine` retired a finished sequence as

```csharp
FlushAndComplete(seq, ...);   // completes the output channel -> releases the consumer
RetireSeq(seq, ...);          // publishes RetainedSequenceState.LastTurn
```

`FlushAndComplete` releases `HotSession`'s `await foreach`, and `HotSession` reads `LastTurn` the
instant it resumes. Whenever the scheduler ran the consumer before `RetireSeq`, a turn that had
**actually succeeded** was recorded as `Failed` — no revision committed, cursor not advanced, work
discarded. Under `RunTurnAsync`'s catch-all it looked like a legitimate engine failure.

Fixed by retiring before flushing. Safe because `HotSession` serialises turns behind its own gate
and cannot start another until `RunTurnAsync` returns, which is strictly after the channel closes.

**Confirmed causal, both directions**: 8/8 clean with the fix; reverting *only* the two-line
ordering brings the failures straight back (2 of 4). That two-way check matters here — an
intermittent failure that stops appearing is not evidence of a fix, only of a different
interleaving.

A second instance of the same inversion was found and fixed in `ActivateSeq`'s "first sampled token
is a stop token" branch (EOS immediately after prefill). It is **not** the path this test exercises
— the test emits a non-stop token from prefill — so that one is fixed by inspection and remains
**unexercised by any test**. Stated rather than implied: no available fixture drives an
immediate-EOS-after-prefill retained turn.

### Why this was worth more than the checkbox

The bug is invisible to every existing test because they all use `MaxNewTokens = 1`, where the turn
ends through a different path. It needed a turn that generates, stops on EOS, and *then* has its
outcome read — which is exactly what "test EOS completion" asks for. The plan's invariant list
earned its place here: the test was written to satisfy a checkbox and found a correctness bug in
the retained-session hot path.

### 8.3 Triage of the remaining Milestone 1 invariants — four are blocked on §4.2 not being implemented

Before writing more tests, each remaining invariant was checked against whether the runtime can
even *produce* the state it asserts on. Four cannot.

**§4.2's design is not implemented.** The plan says `AcceptedPositionCount` and
`MaterializedPositionCount` are distinct "because a sampled token may be accepted into the logical
output before it enters KV on the next forward iteration", and that "reservation, rollback and
snapshot code must account for this pending suffix". `HotSession.BuildNextCursor` ends with:

```csharp
return new SessionCursor(log, accepted, accepted, accepted, accepted, StateCoverage.Full);
```

All four counts are the same value. **The runtime cannot represent a pending/unmaterialised suffix
at all**, so these four invariants would today assert on a state that never occurs — a vacuous
test dressed as a passing one:

| blocked invariant | why |
|---|---|
| first token after prefill and its materialisation lag | lag is always 0 by construction |
| session starting with an unmaterialised suffix | no such session can be built |
| mismatch inside the unmaterialised suffix | ditto |
| reservation renewal at the pending token | there is never a pending token |

These are **blocked on implementing §4.2**, not on test-writing effort, and should not be ticked by
writing tests that pass trivially. Recorded as a dependency the plan does not currently show:
Milestone 1's invariant list silently assumes §4's cursor semantics are live.

**The other eight are implementable today**, with the API each needs confirmed present:

| invariant | mechanism that exists |
|---|---|
| stop-sequence completion at the same token | `SamplingParams.StopTokenIds` / `AdditionalStopTokenIds` |
| stale expected revision rejection | `RunTurnAsync(expectedRevision)` — **zero** current coverage (`expectedRevision` appears 0 times in the store tests) |
| allocation failure leaves prior revision intact | extend `HotSession_ReservesProjectedBytesAndRejectsOverBudgetAdmission`, which today asserts the throw but not that the prior revision survives |
| disconnected transport cannot stall the worker | `MaxBufferedOutputChunks` (4,096) is the bound that makes this testable |
| changed leading block | `HotSession.DiagnoseContinuation` |
| mismatch at a prior turn closing marker | same |
| canonical payload hashing ignores inactive allocation tails | `StatePayloadHash.Compute` |
| exact append at a page boundary | `PagedKvCache.PageSize` = 16; needs a real cache rather than `FakeCache` |

Next tick starts with stale-revision rejection and the allocation-failure extension — the two with
the clearest gap between what the plan requires and what is currently asserted.

## 8.4 Two more Milestone 1 invariants — and an exception that told callers nothing

`HotSession_StaleExpectedRevision_IsRejectedAndLeavesStateUntouched` and
`HotSession_AllocationFailure_LeavesThePriorRevisionIntact`. Eight of Milestone 1's invariants are
now satisfied.

**Both tests assert what SURVIVES the rejection, not the rejection itself.** That is where the
invariant actually lives: a refusal that had already mutated state would be worse than none, since
the caller's retry would then build on a half-applied turn. So both check cursor, revision and
resident bytes are exactly as they were.

**`expectedRevision` had zero coverage** — the parameter appears on every turn and is the whole
optimistic-concurrency mechanism, yet `expectedRevision` occurred 0 times in the store tests.

**`SessionRevisionConflictException` exposed nothing.** It formatted both revisions into its
message and discarded them — unlike `SessionResourceBudgetExceededException`, which surfaces
`RequestedBytes`/`AvailableBytes`/`SessionLimit`. A caller that loses the concurrency race needs the
ACTUAL revision to rebase and retry, and parsing it back out of a message string is not an API.
Added `ExpectedRevision`/`ActualRevision`.

That addition paid for itself immediately in the second test. There is no budget headroom left for a
third successful turn, so "the session still holds revision 1" is proved by submitting a
deliberately stale revision and reading what the conflict reports as actual — an assertion that
needs no allocation at all.

**Mutation-tested, both, and deliberately against the survival half rather than the rejection:**

| mutation | failures |
|---|---|
| refused turn leaks its reservation (`SetResidentBytes(999)` on the budget-exceeded path) | 2 / 11 |
| stale-revision check removed from `InMemorySessionStore.Begin` | 2 / 11 |

Sessions suite 29/29; full suite running.

### Remaining Milestone 1 invariants

Six implementable (stop-sequence completion, disconnected transport, changed leading block,
prior-turn-marker mismatch, canonical payload hashing, page-boundary append) and four blocked on
§4.2's unmaterialised suffix never being produced — see §8.3.

### 8.5 The two divergence invariants are NOT covered by the existing cursor tests

Checked before writing anything, because `Diagnose_SeparatesExactAppendFromReplayRequiredDivergence`
looks at first glance like it covers them. It does not:

```csharp
ImmutableArray<ExecutionSegment> currentLog = [new TokenSegment([1, 2])];
var cursor = new SessionCursor(currentLog, 2, 2, 2, 2, StateCoverage.Full);
var mismatch = ExecutionReconciler.Diagnose(cursor, [new TokenSegment([1, 4])]);
```

Two reasons it falls short of the Milestone 1 invariants:

1. **It is a synthetic cursor**, hand-built, not one produced by real turns through `HotSession`.
   The Milestone 1 list is about session behaviour; a cursor literal cannot catch a cursor that
   `BuildNextCursor` assembles wrongly.
2. **It has one segment.** "Mismatch at a prior turn closing marker" requires at least two turns —
   `[prompt₁, generated₁, prompt₂, …]` — with divergence at the *end* of `generated₁`. A
   single-segment log cannot express that shape at all, so this invariant is genuinely uncovered.

**"Changed leading block" is ambiguous in the plan and is deliberately left unticked.** "Block"
could mean the leading execution *segment* (in which case the `mismatch` arm above is close, modulo
being synthetic) or a page-aligned KV *block* (`PagedKvCache.PageSize` = 16, in which case it needs
≥16 tokens and a real cache, and is a different test entirely). The neighbouring invariant "exact
append at a page boundary" uses page language explicitly, which weakly suggests "block" here means
the page. Guessing wrong would tick a checkbox with a test that does not test it — the failure mode
§8.3 was written to avoid. Flagged for the plan's author to disambiguate.

Both therefore remain open, with the reason recorded rather than left as an unexplained gap.

### 8.6 Mismatch at a prior turn's closing marker — nine invariants now satisfied

`HotSession_MismatchAtPriorTurnClosingMarker_RequiresReplay`. Written against a **real** two-turn
session cursor, per §8.5: the log is one `BuildNextCursor` actually assembled, not a hand-built
literal, so a cursor the session assembles wrongly is in scope.

The test corrupts exactly one token — the **last** of turn 1's generated segment, the point where
one turn closes and the next begins — and asserts the continuation is refused:

| assertion | value |
|---|---|
| `CanAppendWithoutReplay` | false |
| `Grade` | `ReplayedFromExecutionLog` |
| `ReuseReason` | `PrefixDivergence` |
| `DivergenceSegmentIndex` | 1 (turn 1's generated segment) |
| `DivergencePositionInSegment` | last position of that segment |

It carries its own control: the unmodified log must diagnose as `ExactLossless` against itself, so
the divergence assertions cannot pass by the diagnostic being uniformly pessimistic.

**Why this boundary specifically.** Everything before the corrupted token matches. A reconciler
that compared only the leading segment, or that treated a turn boundary as a resync point, would
report an exact append and reuse state it must not — silently continuing from a history that never
happened. Locating the divergence (segment 1, final position) rather than merely detecting it is
what distinguishes "noticed something changed" from "knows where".

**Mutation-tested**: forcing `currentIsExactPrefix` true — a reconciler that never detects
divergence at all — fails this test and only this test (1 of 12). Restored; sessions suite 30/30.

**Milestone 1 invariants: 9 of 21 satisfied.** Five remain implementable (stop-sequence completion,
disconnected transport, canonical payload hashing, page-boundary append, and "changed leading block"
pending the wording question in §8.5). Four remain blocked on §4.2 (see §8.3).

### 8.7 Disconnected transport — the test was vacuous twice before it tested anything

`HotSession_AbandonedConsumer_DoesNotStallOtherSessions`. Ten Milestone 1 invariants now satisfied.

**The mechanism, confirmed in code first.** Both output channels are `Channel.CreateUnbounded`, and
the batcher only ever `TryWrite`s — it never awaits a writer. `MaxBufferedOutputChunks` (4,096) is
*not* a channel bound; it caps `MaxNewTokens` at admission, so an abandoned reader cannot accumulate
without limit. The no-stall property is therefore structural, and the test pins its consequence:
one consumer walking away must not prevent another session finishing.

**It took three attempts, and the two failures are the useful part.**

The only real assertion here is a timeout — "session B completed" — which is a shape that passes
trivially whenever the scenario failed to set itself up. So the test carries a **non-vacuity guard**:
after B completes, the abandoned request must still be in flight (`ActiveRequests > 0 ||
QueueDepth > 0`). That guard failed on every run, twice, for two different reasons:

1. **`await using` on the enumerator cancels the request.** Disposing an async enumerator cancels
   the underlying generation — correct behaviour, but the opposite of the scenario: a cancelled
   request is tidily retired, so nothing was ever abandoned. Fixed by holding the enumerator
   undisposed until the end.
2. **The fake retires 64 tokens in ~3 ms.** Even un-cancelled, request A finished long before B was
   observed. There was no interval in which an abandoned consumer existed. Fixed with a 2 ms
   per-decode-step delay on the fake: every batcher iteration advances all active sequences, so step
   *count* now separates a 64-token request from a 1-token one.

Without the guard this test would have been green from the first attempt and would have asserted
nothing at all — a timeout that never had anything to time out. It is the clearest case so far for
writing the non-vacuity check *before* trusting a green result.

**Mutation-tested**: removing the 2 ms delay makes the guard fire again (1 of 13), confirming the
guard is load-bearing rather than decorative. Restored; sessions suite 31/31.

**Milestone 1: 10 of 21.** Four implementable remain (stop-sequence completion, canonical payload
hashing, page-boundary append, and "changed leading block" pending §8.5's wording question); four
remain blocked on §4.2 (§8.3).

### 8.8 Canonical payload hashing — ticked, but a WEAKER test than the others here

`PayloadHash_IgnoresInactiveAllocationTail`. Eleven Milestone 1 invariants satisfied.

**Why the invariant matters.** State buffers are pooled and reused, so bytes past the active length
are whatever a previous turn left there. `StatePayloadHash` answers "are these restored bytes
intact" — if it covered the tail, two sessions holding identical state would hash differently purely
from allocator history, and every restore would look corrupt.

The test hashes the same 4 active bytes out of two buffers with deliberately different trailing
garbage and asserts equality, with a **non-vacuity guard** that hashing the full buffers *does*
differ (otherwise the equality proves nothing), plus a sensitivity check that changing an active
byte still changes the hash.

### The mutation did NOT fail, and that is the finding

Per the standing rule I tried to break it: I made `Compute` fold the span's length into the digest.
**All six tests still passed.** The reason is that both compared slices have the same active length,
so a length byte cannot separate them — and more fundamentally, including the *active* length is not
a tail violation at all, so that mutation was not a faithful model of the bug.

Trying to construct a faithful one showed why none exists: `Compute` takes a
`ReadOnlySpan<byte>`. **The caller slices, and the tail is simply not reachable from inside.** There
is no behavioural surface on which to introduce tail-inclusion; the only way to violate the
invariant is to change the signature to `byte[]` and hash the whole array — which would break this
test at *compile* time, not at assertion time.

So this test is honestly a **shape-locking regression guard**, not a behavioural check, exactly as
its doc comment claimed before the mutation was attempted. Recorded rather than quietly ticked
alongside the mutation-verified ones (§8.4, §8.6, §8.7), because "mutation-tested" has been the
standard of evidence throughout this work and this one does not meet it. It cannot, and the reason
is a property of the API rather than a gap in the test.

**A stronger version would need the invariant to be violable** — e.g. if `StatePayloadHash` ever
takes ownership of a pooled buffer plus a length, rather than a pre-sliced span. If that change is
made, this test must be upgraded from shape-locking to behavioural at the same time.

**Milestone 1: 11 of 21.** Three implementable remain (stop-sequence completion, page-boundary
append, and "changed leading block" pending §8.5's wording question); four blocked on §4.2 (§8.3).

### 8.9 Stop-token completion — invariant satisfied; the hoped-for bonus coverage did NOT materialise

`HotSession_StopTokenOnFirstSample_CompletesAtThatTokenAndCommits`. Twelve Milestone 1 invariants
satisfied.

A non-EOS stop token (`AdditionalStopTokenIds`, which unions onto the EOG set rather than replacing
it) terminates the turn at that token, and the turn commits normally — a stop is a completion, not
a failure. Asserted: `Completed`, revision 1, **zero** generated tokens against a budget of 8, and
`accepted == materialized`. It carries a **non-vacuity control**: a second session with the same
fake and no stop registration runs to the full 8 tokens, so the zero above is caused by the stop
token and nothing else.

### The bonus that did not happen, recorded because I expected it to

This test was written partly to cover something §8.2 left open. That section fixed an ordering bug
in `ActivateSeq`'s "first sampled token is already a stop token" branch — `TryComplete` before
`state.Complete` — **by inspection**, and explicitly recorded that no fixture drove that path. This
test does drive it (zero generated tokens proves the branch is taken).

So I re-inverted that ordering to confirm the coverage. **It passed 3 runs of 3.** The branch is
exercised; the race is not detectable there. The likely reason is window size: in `ActivateSeq` the
two statements are adjacent, whereas the §8.2 retire-path race had `FlushAndComplete` writing
several chunks and `RetireSeq` doing budget accounting between them, widening the gap enough for the
consumer's thread-pool continuation to land inside it.

**Therefore the `ActivateSeq` ordering fix remains inspection-only and is still not verified by any
test** — exactly the status §8.2 gave it. It would have been easy to write "and this also covers the
§8.2 branch" on the strength of the branch being reached; reaching a branch is not the same as
detecting its defect, and the mutation is what separates those two claims.

Sessions suite 33/33; full suite running.

**Milestone 1: 12 of 21.** Two implementable remain (page-boundary append; "changed leading block"
pending §8.5's wording question). Four remain blocked on §4.2 (§8.3). The remaining three are
Milestone 1 items outside the invariant list.

### 8.10 Review findings on the rolling-reservation / routing work

**1. RETRACTED AND CORRECTED — starvation is a configuration outcome, not a missing mechanism.**
The original finding below claimed rolling reservation had no way to protect a peer. That was
wrong, and testing it is what showed so: setting `MaxSessionBytes` to 40 against a global
`MaxResidentBytes` of 80 makes the identical workload complete for BOTH sessions, no exception.
The starving configuration set the per-session cap EQUAL to the global budget, which entitles one
session to all of it — behaving exactly as configured.

**FIXED (the derivation half).** `HotSessionRuntimeOptions.ExpectedConcurrentSessions` now derives
the per-session cap as `maxResidentBytes / expectedConcurrentSessions` — stating the concurrency you
expect is far easier to get right than computing a byte share by hand, and it removes the footgun
that the *starving* configuration was the one you got by not thinking about it. It derives ONLY when
the caller left `maxSessionBytes` unspecified: an explicit cap is a deliberate decision and is never
silently overridden (mutation-verified — removing that guard fails the test).

Still open: **a late arrival is refused outright rather than queued.** That is backpressure, not
starvation, and belongs to §7's unchecked admission-queue / fair-waiter items. Admission queuing
and fair waiters (§7) are still unchecked. But the lever to prevent starvation today exists and is
one option away.
`HotSession_RollingReservation_StarvesAPeerOnlyWhenSessionCapEqualsGlobalBudget` now demonstrates
both halves in one test — the starving configuration AND the remedy — so the gap can never be read
as unfixable.

<details><summary>Original (incorrect) finding, kept per the standing rule on retractions</summary>

**Rolling reservation is first-come-first-served and can starve a peer.** With a shared global
budget, a session that starts earlier grows into the whole budget via renewal, and a session
admitted later is refused outright — not queued, not throttled. Renewal asks only "is there
capacity right now"; it has no notion of holding a share for an admitted peer. This is §7's
unchecked "bounded admission and output queues / fair waiters / per-model budget partitions",
surfacing as observable behaviour. `HotSession_RollingReservation_IsFirstComeFirstServed_AndCanStarveAPeer`
characterises it; **when fairness lands, that test must fail**, and the failure is the signal to
rewrite it as the fairness spec rather than relax it.

</details>

**2. `HotSessionPrefillLagTests` does not satisfy the materialisation-lag invariant — tick
reverted.** It asserted `Accepted == Materialized`, i.e. that no lag exists, which
`BuildNextCursor` guarantees by construction and would keep guaranteeing if all lag handling were
deleted. Renamed to `HotSession_Cursor_CollapsesAcceptedAndMaterialized_UntilSection42IsImplemented`
and kept as a regression guard on current behaviour, so implementing §4.2 breaks it loudly. The
invariant is back to unchecked, consistent with §8.3.

**3. `SessionAddress.ToSessionId` hardened.** The length-prefixed canonical form was already right
(plain concatenation lets `("a:b","c")` collide with `("a","b:c")`). Changed MD5 → SHA-256
truncated to 16 bytes: no security claim, but MD5 trips CA5351 and this repo builds with
`TreatWarningsAsErrors`, so a future analyzer rollout would become a build break. Switched to the
static `HashData` form (no per-call hasher allocation — routing resolves an address every turn),
and made null encode as length `-1` so `default(SessionAddress)` no longer collides with four empty
strings. Both properties now asserted.

**4. Stop-reason honesty, shrink-path guard, `GetOperationAsync`** — see the preceding sections;
all three landed with the budget-exhaustion flag mutation-verified.
