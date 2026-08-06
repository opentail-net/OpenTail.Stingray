# ADR-0001: retain session state through the existing batching engine

**Status:** accepted for the CPU-dense seam spike  
**Date:** 2026-08-02

## Context

OpenTail.Stingray already separates per-sequence state from batching mechanics through
`IBatchedForwardPass` and `ISequenceKvCache`. The current `ContinuousBatchingEngine` nevertheless
owns every cache from admission to retirement and disposes it at every terminal path. Persistent
sessions need a completed cache to survive a turn and be re-admitted later.

Creating a second batching engine would duplicate admission limits, packed prefill, decode
scheduling, cancellation and prefix-cache behavior. Making a session storage project own engine
internals would reverse the dependency direction and couple persistence to HTTP or scheduling.

## Decision

Extend `ContinuousBatchingEngine` with a small, explicit cache-lifecycle contract. It will be
consumed by the future sessions layer but remain free of persistence, tenancy and HTTP types.

The first contract will support only the CPU-dense reference lane. It will:

- accept a supplied `ISequenceKvCache` only after backend/type/capability validation;
- attach a cache to one admitted request and track a single lifecycle owner at a time;
- expose a terminal retained-cache result only after the generation operation has committed;
- transfer ownership exactly once on successful retention; and
- dispose a cache on cancellation, failure or engine shutdown unless the operation explicitly
  rolled it back and transferred it to the caller.

The sessions layer will own transcript/cursor/revision/lease policy. The engine will own batcher
thread affinity, cache admission, compute and low-level disposal while it holds a cache.

The initial API is `ContinuousBatchingEngine.GenerateRetainedChunksAsync`. It accepts an
append-only prompt suffix and a `RetainedSequenceState`; the opaque handle owns the hot cache
between turns and enforces one queued or active writer. It is deliberately not an execution-log,
revision, persistence, or transport API. Only a cache implementing `IRewindableSequenceKvCache`
may be retained. `PagedKvCache` implements that capability only while its logical and physical
positions agree, which keeps SnapKV/windowed state outside this exact lane.

## Rejected alternatives

**Separate `SessionBatchingEngine`.** Rejected because it would copy complex scheduling behavior
which is already exercised in `ContinuousBatchingEngine`.

**Have sessions call `IBatchedForwardPass` directly.** Rejected because it bypasses the batcher and
makes cache thread affinity and contention the caller's problem.

**Persist `ISequenceKvCache` directly.** Rejected because the handle is deliberately opaque and
some implementations contain native or device resources. Persistence belongs behind a separate,
lossless capability introduced after hot continuation is proven.

## Guardrails

- The public session API cannot expose a concrete `PagedKvCache` type.
- A returned handle must be non-disposable by both engine and session at the same time.
- Cancellation never produces a partially advanced retained state; it returns the last committed
  turn boundary or fails without retention.
- SnapKV/windowed, CUDA, TurboQuant, GDN, MTP and vision must return typed unsupported/refusal
  results until each earns conformance coverage.
- The engine never imports the sessions or server project.

## Verification required before broadening scope

- [x] A fake-backend unit test covers ownership transfer, append re-admission and single-writer exclusion.
- [ ] A real CPU-dense seam spike proves detach, re-admission and greedy replay equivalence.
- [x] Mid-generation cancellation proves turn-start rollback before retention with a deterministic fake backend.
- [x] Existing continuous-batching and prefix-cache tests still pass.
- [x] The initial API, ownership boundary and patch surface are recorded here; persistence remains
  blocked on the real CPU-model equivalence gate.
