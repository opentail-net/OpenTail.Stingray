> **ARCHIVED, 2026-08-15.** Implemented as an `IInferenceSession`/`InferenceSession`-era
> Sessions-layer feature (confirmed against source, not just this document's own claim). That
> whole lineage is being superseded by the `HotSession` architecture — see
> [028](028-inference-session-to-hotsession-migration-plan.md) for current migration status
> (Phases 1-3 done) and [030](../030-delete-inferencesession-todo.md) for the still-open
> deletion of the legacy `InferenceSession`/`InferenceRuntime` types once HotSession fully
> replaces them. Carried forward: nothing beyond what 028/030 already track in
> [00-current-work.md](../00-current-work.md).

---

# Implementation Plan — Automatic KV Memory Governor & Idle Session Auto-Suspension (`Plan 006`)

## Objective

Implement an automatic KV-memory governor for OpenTail.Stingray that prevents uncontrolled physical KV-cache exhaustion by automatically suspending suitable idle sessions when KV memory reaches a configurable high-water mark.

The governor should build directly on the existing:

- `IKvCache` page accounting;
- physical page allocation/release;
- session lifecycle;
- `SuspendAsync()`;
- suspended-session state;
- session token/state persistence;
- session resume;
- existing locking and lifecycle safety.

This is a **memory-management feature**, not a scheduler redesign and not a performance optimisation.

The desired behaviour is:

```text
                  KV Cache
                     │
                     ▼
              memory utilisation
                     │
          ┌──────────┴──────────┐
          │                     │
       below HWM             ≥ HWM
          │                     │
          ▼                     ▼
       normal             find idle sessions
                                │
                                ▼
                         suspend LRU session
                                │
                                ▼
                       release physical KV pages
                                │
                                ▼
                         memory pressure falls
```

The central invariant is:

> **The governor may suspend idle sessions to reclaim physical KV pages, but must never suspend an actively executing session or corrupt session state.**

---

# 1. Important scope boundary

Do **not** implement:

- a new scheduler;
- request prioritisation;
- token scheduling;
- speculative decoding;
- a new KV cache;
- a new session persistence system;
- arbitrary session eviction;
- forced termination of sessions;
- performance tuning.

The governor should simply observe KV pressure and use the **existing safe session suspension mechanism**.

---

# 2. Existing architecture to reuse

Before changing code, inspect and reuse the existing implementations of:

```text
IKvCache
CpuKvCache
InferenceRuntime
InferenceSession
SessionState
SuspendAsync()
ResumeAsync()
```

Do not create duplicate memory accounting.

If the KV cache already exposes something equivalent to:

```csharp
UsedPages
TotalPages
```

use it.

If it exposes bytes rather than pages, derive utilisation from the existing authoritative accounting.

---

# 3. New abstraction: IKvMemoryGovernor

Introduce a small abstraction so the runtime is not tightly coupled to one governor implementation.

Possible design:

```csharp
public interface IKvMemoryGovernor
{
    ValueTask<MemoryGovernorResult> EnsureCapacityAsync(
        int requiredPages,
        CancellationToken cancellationToken = default);
}
```

However, **the existing architecture may suggest a better contract**.

The implementing AI is explicitly allowed to improve this API if the repository already has a more natural abstraction.

The important concept is:

> The governor should be asked to make enough KV capacity available, rather than being blindly told to suspend sessions.

This makes it useful at allocation boundaries.

---

# 4. Configuration

Introduce a small immutable configuration object.

For example:

```csharp
public sealed record KvMemoryGovernorOptions
{
    public bool Enabled { get; init; } = true;

    public double HighWaterMark { get; init; } = 0.90;

    public double TargetWaterMark { get; init; } = 0.75;

    public int MinimumIdleAgeSeconds { get; init; } = 30;

    public int MaximumSessionsToSuspendPerPass { get; init; } = 8;
}
```

These values are examples, **not mandatory API design**.

The implementing AI should choose names/types consistent with existing Stingray configuration conventions.

### Important distinction

Use two thresholds:

```text
90% → begin reclamation
75% → stop reclamation
```

rather than:

```text
90% → suspend one session → stop
```

This prevents repeated suspend/wake cycles around a single threshold.

The governor should reclaim until:

```text
utilisation <= TargetWaterMark
```

or until no suitable idle sessions remain.

---

# 5. Do not rely on a background polling loop initially

The first implementation should preferably be **allocation-pressure driven**.

For example:

```text
allocation requested
       │
       ▼
KV cache says insufficient capacity
       │
       ▼
Governor.EnsureCapacityAsync(...)
       │
       ├── enough capacity already → continue
       │
       └── pressure high → suspend idle sessions
```

This is simpler and safer than introducing a background memory-monitoring thread.

A background monitor could be a future enhancement.

---

# 6. Session eligibility

The governor must distinguish between sessions that are safe to suspend and sessions that are currently active.

An eligible session should satisfy something conceptually like:

```text
Session exists
AND
Session is idle
AND
Session is not executing
AND
Session is not already suspended
AND
Session is not disposing/disposed
AND
Session has reclaimable KV pages
AND
minimum idle age has elapsed
```

Use the existing session lifecycle state rather than inventing another independent state machine.

---

# 7. Never suspend an active session

This is a critical invariant.

The governor must not do:

```text
Session A executing
       ↓
Governor sees old LastUsed timestamp
       ↓
SuspendAsync()
```

There must be a race-safe check.

The desired conceptual sequence is:

```text
select candidate
      ↓
atomically verify still idle
      ↓
SuspendAsync()
      ↓
physical KV released
```

If the session became active between selection and suspension:

```text
SuspendAsync()
      ↓
must refuse / safely no-op
```

The governor must handle that result gracefully.

---

# 8. LRU selection

The governor should select the **least-recently-used eligible idle session**.

Use an existing session activity timestamp if one already exists.

Prefer existing lifecycle information such as:

```text
LastActivity
LastUsed
LastExecution
```

rather than creating a second competing concept of session recency.

Candidate ordering:

```text
oldest idle session
        ↓
next oldest
        ↓
...
```

Only eligible sessions should enter the candidate set.

---

# 9. Reclamation algorithm

Conceptually:

```csharp
while (KvUtilisation > TargetWaterMark)
{
    var candidate = FindOldestEligibleIdleSession();

    if (candidate is null)
        break;

    var result = await candidate.SuspendAsync(
        cancellationToken);

    if (!result.Suspended)
        continue;

    // Re-read authoritative KV usage.
}
```

The important point is:

> **Re-read actual KV usage after every suspension.**

Do not estimate how many pages suspension will free.

The session may have:

- zero KV pages;
- shared pages;
- prefix-cache references;
- partially retained state;
- already released pages.

The cache itself is authoritative.

---

# 10. Interaction with prefix caching

This feature must work correctly with Plan 005b.

A suspended session may release its session-owned KV references while the prefix cache still owns physical pages.

Therefore:

```text
Session A
   │
   └── page 42 ← session reference

Prefix cache
   │
   └── page 42 ← cache reference
```

After suspension:

```text
Session A
   │
   └── no physical reference

Prefix cache
   │
   └── page 42 ← still retained
```

The governor must **never directly release physical pages**.

It only asks the session to suspend.

Session lifecycle owns the release.

This separation is essential.

---

# 11. Interaction with shared / CoW pages

The governor must not assume that suspending a session frees every page associated with it.

For example:

```text
Session A ─────┐
               ├── Page 10
Session B ─────┘
```

Suspending A should release A's reference only.

Page 10 remains alive while B owns it.

Likewise, prefix-cache references remain independent.

The governor therefore measures **actual cache utilisation after suspension**, not expected utilisation.

---

# 12. MemoryGovernorResult

Provide enough information for diagnostics and testing.

For example:

```csharp
public sealed record MemoryGovernorResult
{
    public bool CapacityAvailable { get; init; }

    public int SessionsSuspended { get; init; }

    public int PagesReclaimed { get; init; }

    public int UsedPages { get; init; }

    public int TotalPages { get; init; }

    public bool NoEligibleSessions { get; init; }
}
```

Again, the implementing AI may improve this design.

The important thing is that the result should make it possible to understand:

```text
Did the governor reclaim memory?
Why did it stop?
Is there now sufficient capacity?
```

---

# 13. What happens if no idle session can be suspended?

Do **not** silently pretend capacity was created.

If:

```text
KV utilisation = 97%

No eligible idle sessions
```

the governor should return/report:

```text
CapacityAvailable = false
NoEligibleSessions = true
```

The existing allocation path should then produce its normal out-of-memory/capacity failure.

The governor is a safety mechanism, not permission to violate cache invariants.

---

# 14. What happens if one session isn't enough?

Example:

```text
Used = 92%
Target = 75%

Suspend A → 88%
Suspend B → 82%
Suspend C → 74%
```

The governor should stop after C.

Do not suspend D unnecessarily.

This protects session responsiveness and avoids unnecessary churn.

---

# 15. Prevent suspension thrashing

Use hysteresis.

Example:

```text
HighWaterMark = 90%
TargetWaterMark = 75%
```

Do not suspend merely because utilisation is 76%.

Only begin reclamation at the high-water mark.

Once reclamation begins, continue until the target watermark is reached or there are no eligible sessions.

---

# 16. Minimum idle age

Do not suspend a session immediately after it becomes idle.

For example:

```text
MinimumIdleAge = 30 seconds
```

This prevents:

```text
request finishes
     ↓
session becomes idle
     ↓
memory pressure check
     ↓
immediate suspension
     ↓
next request
     ↓
resume
```

for short gaps between requests.

Use the existing activity timestamps where possible.

---

# 17. Session activity update

Review the existing session lifecycle carefully.

Ensure the timestamp used for LRU selection is updated when appropriate:

- session created;
- generation begins;
- generation completes;
- prompt prefill occurs;
- session resumes;
- session receives activity.

Do not update it on every token unless the existing architecture already does that.

The goal is **session activity recency**, not token-level profiling.

---

# 18. Runtime integration

`InferenceRuntime` should own/configure the governor.

Conceptually:

```csharp
public IKvMemoryGovernor MemoryGovernor { get; }
```

The governor should receive the authoritative:

```csharp
IKvCache
```

and session registry/provider.

Possible architecture:

```text
InferenceRuntime
       │
       ├── IKvCache
       ├── SessionRegistry
       │
       └── IKvMemoryGovernor
               │
               ├── observes cache pressure
               └── suspends sessions
```

Do not make `CpuKvCache` aware of sessions.

The cache should remain a lower-level memory substrate.

---

# 19. Allocation integration

The preferred integration point is where Stingray knows:

> "I need more KV pages."

At that point:

```text
Need N pages
      ↓
Can cache provide them?
      │
      ├── yes → allocate
      │
      └── no / pressure high
             ↓
        governor
             ↓
       suspend idle
             ↓
       retry allocation
```

Avoid adding governor calls at dozens of arbitrary locations.

Find the central allocation/admission point if one already exists.

---

# 20. Retry semantics

After governor reclamation:

```text
retry allocation
```

only a bounded number of times.

Do not create an infinite loop.

For example:

```csharp
for (var attempt = 0; attempt < 2; attempt++)
{
    if (TryAllocate(...))
        return ...;

    await governor.EnsureCapacityAsync(...);
}
```

But use the repository's existing allocation/error semantics if they already provide a cleaner mechanism.

---

# 21. Tests

Create:

```text
KvMemoryGovernorTests.cs
```

with at least the following mandatory tests.

### Test 1 — BelowHighWaterMark_DoesNothing

```text
utilisation = 50%
HWM = 90%

→ no session suspended
```

---

### Test 2 — HighWaterMark_SuspendsOldestIdleSession

Create:

```text
A = oldest
B = newer
C = active
```

Trigger pressure.

Expected:

```text
A suspended
B remains active/idle
C remains active
```

---

### Test 3 — ActiveSession_NeverSuspended

Make the oldest session active.

Verify the governor chooses another eligible session.

---

### Test 4 — SuspendedSession_NotSelected

Already suspended sessions should not be selected again.

---

### Test 5 — MinimumIdleAge_Respected

A session that became idle 5 seconds ago should not be suspended when:

```text
MinimumIdleAge = 30 seconds
```

---

### Test 6 — ReclaimsUntilTargetWaterMark

Example:

```text
HWM = 90%
Target = 75%

A → 88%
B → 82%
C → 74%

Stop.
```

Verify D is not suspended.

---

### Test 7 — NoEligibleSessions_ReturnsFailure

If all sessions are:

```text
active
```

or otherwise ineligible:

```text
governor cannot reclaim enough memory
```

must be reported correctly.

---

### Test 8 — SuspensionActuallyReleasesPages

Verify:

```text
Before suspension:
UsedPages = X

After suspension:
UsedPages < X
```

using authoritative cache accounting.

---

### Test 9 — SharedPagesRemainAlive

Two sessions share physical KV pages.

Suspend one.

Verify the shared page remains valid for the other.

---

### Test 10 — PrefixCacheOwnershipUnaffected

A page owned by both:

```text
session
prefix cache
```

must remain alive after session suspension.

---

### Test 11 — ResumeAfterGovernorSuspension

Suspend through the governor, then resume.

Verify:

- session state remains valid;
- token state is preserved;
- KV state is reconstructed/reloaded correctly;
- subsequent generation succeeds.

---

### Test 12 — ActiveRace_DoesNotSuspend

Create a race where:

```text
governor selects session A
        ↓
session A becomes active
        ↓
governor attempts suspension
```

Verify A is not incorrectly suspended.

This test is particularly important.

---

# 22. Metrics / observability

Expose lightweight metrics.

Suggested:

```text
MemoryGovernor.Suspensions
MemoryGovernor.PagesReclaimed
MemoryGovernor.LastSuspensionUtc
MemoryGovernor.SuspensionFailures
MemoryGovernor.NoEligibleSessionEvents
```

Do not build a telemetry framework.

Use existing Stingray metrics/logging conventions.

---

# 23. Logging

When a governor suspension occurs, a diagnostic message should make it clear:

```text
KV memory pressure: 94%.
Suspending idle session abc123.
Idle for 143s.
Reclaimed 37 pages.
KV usage now 76%.
```

Avoid logging every cache utilisation check.

Only log meaningful events.

---

# 24. Configuration defaults

A sensible initial default might be:

```text
Enabled = true
HighWaterMark = 90%
TargetWaterMark = 75%
MinimumIdleAge = 30s
```

However, **do not hard-code these values if Stingray already has a configuration/options pattern**.

The implementing AI may choose better defaults based on the existing architecture.

---

# 25. Safety requirements

The following invariants must always hold:

### Invariant A

```text
Active session
    → NEVER automatically suspended
```

### Invariant B

```text
Suspension
    → uses existing Session.SuspendAsync()
```

The governor must not manipulate session KV pages directly.

### Invariant C

```text
Prefix cache ownership
    → independent of session ownership
```

### Invariant D

```text
Shared physical page
    → released only when ALL owners release it
```

### Invariant E

```text
Governor failure
    → cannot corrupt session/cache state
```

### Invariant F

```text
Governor disabled
    → existing Stingray behaviour unchanged
```

This last one is particularly important for backward compatibility.

---

# 26. Optional future enhancement — NOT part of this plan

A future version could add a proactive background governor:

```text
Timer
  ↓
monitor KV utilisation
  ↓
if pressure high
  ↓
suspend idle sessions
```

Do **not** implement this now unless the existing architecture makes it essentially free.

The first version should be allocation-pressure driven.

---

# 27. Acceptance criteria

This plan is complete when:

- [ ] `IKvMemoryGovernor` (or an equivalent existing-architecture abstraction) exists.
- [ ] Governor uses authoritative KV-cache accounting.
- [ ] Governor can identify eligible idle sessions.
- [ ] Sessions are selected LRU-first.
- [ ] Active sessions are never automatically suspended.
- [ ] Minimum idle age is respected.
- [ ] High/target watermarks provide hysteresis.
- [ ] Governor suspends only as many sessions as required.
- [ ] Actual KV usage is re-read after suspension.
- [ ] Governor never manipulates KV page ownership directly.
- [ ] Prefix-cache references remain safe.
- [ ] Shared/CoW pages remain safe.
- [ ] Suspended sessions can resume correctly.
- [ ] No-eligible-session condition is reported correctly.
- [ ] Governor can be disabled.
- [ ] Metrics/logging are available.
- [ ] Mandatory tests pass.
- [ ] Full Stingray test suite passes.
- [ ] Release build passes.

---

# 28. Definition of done

The feature should make this scenario safe:

```text
Stingray starts
      │
      ▼
many sessions accumulate
      │
      ▼
KV reaches 90%
      │
      ▼
Governor activates
      │
      ▼
oldest eligible idle session suspended
      │
      ▼
physical KV released
      │
      ▼
KV remains above target?
      │
      ├── yes → suspend next eligible session
      │
      └── no → stop
```

Meanwhile:

```text
Active session
      │
      └── untouched

Prefix cache
      │
      └── untouched

Shared pages
      │
      └── ref-count protected

Suspended session
      │
      └── resumable
```

The final result should be:

> **Stingray automatically manages KV memory pressure without requiring the caller to manually decide which idle sessions to suspend, while preserving all existing session, CoW, prefix-cache, and physical-page lifetime invariants.**

---

## Implementation guidance

The above API sketches are **concepts, not rigid requirements**.

If inspection of the current repository reveals a cleaner design — for example, an existing session registry, memory-pressure abstraction, allocation admission point, or lifecycle interface that makes one of the proposed types unnecessary — **use the better existing abstraction**.

The priority is:

1. reuse existing Stingray architecture;
2. avoid duplicate state/accounting;
3. preserve existing lifecycle invariants;
4. keep the implementation small;
5. make the governor easy to disable and test.

Do not introduce a large framework for what should remain a small memory-pressure component.