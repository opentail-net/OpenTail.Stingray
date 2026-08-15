> **ARCHIVED, 2026-08-15.** Implemented as an `IInferenceSession`/`InferenceSession`-era
> Sessions-layer feature (confirmed against source, not just this document's own claim). That
> whole lineage is being superseded by the `HotSession` architecture — see
> [028](028-inference-session-to-hotsession-migration-plan.md) for current migration status
> (Phases 1-3 done) and [030](../030-delete-inferencesession-todo.md) for the still-open
> deletion of the legacy `InferenceSession`/`InferenceRuntime` types once HotSession fully
> replaces them. Carried forward: nothing beyond what 028/030 already track in
> [00-current-work.md](../00-current-work.md).

---

# Implementation Plan — Plan 016: Adaptive KV Memory Governor & Pressure-Aware Session Suspension

## Objective

Implement an optional `KvMemoryGovernor` that automatically manages physical KV-cache pressure by suspending idle sessions when memory becomes contested.

The governor must be **adaptive rather than timer-driven**.

A session should NOT be suspended merely because it has been idle for 30 seconds.

Instead:

> **Idle time determines eligibility; physical KV pressure determines whether suspension is actually necessary.**

This keeps Stingray lightweight for normal single-user workloads while allowing heavily concurrent deployments to reclaim KV memory automatically.

---

# 1. Design Principles

### 1.1 Zero pressure = don't evict

If the KV cache has plenty of free capacity:

```text
KV utilisation = 35%

Session idle for:
    10 seconds
    30 seconds
    5 minutes
    30 minutes
```

Do nothing.

Idle sessions are cheap from the governor's perspective and remain resident for fast resume.

---

### 1.2 Pressure activates reclamation

When physical KV usage reaches a configurable pressure threshold:

```text
Normal:
    < 75%       → no action

Elevated:
    75–85%      → monitor more aggressively

Pressure:
    >= 85%      → begin reclaiming idle sessions

Critical:
    >= 95%      → aggressively reclaim eligible sessions
```

These should be configurable rather than hard-coded.

The exact default values may be adjusted to match the actual KV cache metrics exposed by the current implementation.

---

### 1.3 Idle time is an eligibility signal

Do not use:

```text
if idle > 30s → suspend
```

Instead:

```text
if pressure exists
AND session is idle
AND idle duration >= minimum idle duration
    → candidate for suspension
```

The minimum idle duration should be configurable.

For example:

```text
MinimumIdleDuration = 30 seconds
```

but this is only the **minimum eligibility age**, not a mandatory suspension timer.

---

# 2. Adaptive Pressure Policy

Introduce a small policy object rather than scattering thresholds throughout the governor.

Conceptually:

```csharp
public sealed record KvGovernorOptions
{
    public bool Enabled { get; init; } = true;

    public double ElevatedThreshold { get; init; } = 0.75;
    public double PressureThreshold { get; init; } = 0.85;
    public double CriticalThreshold { get; init; } = 0.95;

    public TimeSpan MinimumIdleDuration { get; init; }
        = TimeSpan.FromSeconds(30);

    public TimeSpan PollInterval { get; init; }
        = TimeSpan.FromSeconds(1);

    public int MaxSessionsSuspendedPerCycle { get; init; } = 4;
}
```

These are examples only.

**The coding agent may choose a better configuration shape if it fits existing conventions.**

Do not over-engineer this into a policy framework.

---

# 3. Pressure-Aware Behaviour

The governor should behave approximately as follows.

### Below elevated pressure

```text
< 75%

Do nothing.
```

No session enumeration or suspension is required beyond lightweight monitoring.

---

### Elevated pressure

```text
75–85%

Do not immediately suspend.

Prefer allowing active workloads to continue.

Sessions may be ranked so that the governor is ready to reclaim quickly if pressure increases.
```

---

### Pressure

```text
>= 85%

Find eligible idle sessions.

Rank candidates by:
    1. idle duration
    2. physical KV pages held
    3. whether the session is waiting on an external operation
    4. recent activity

Suspend the least valuable sessions until memory returns below a target threshold.
```

Importantly, don't suspend everything back to 85%.

Use hysteresis.

For example:

```text
Start reclaiming: 85%

Stop reclaiming: 70%
```

This prevents:

```text
85% → suspend
84% → stop
85% → suspend
84% → stop
```

repeated oscillation.

---

### Critical pressure

At approximately 95%+:

```text
Immediately reclaim eligible idle sessions.

Continue until sufficient safety margin exists.

Never suspend an actively generating session merely because it is old.
```

If there are no eligible sessions, the governor should report that capacity is exhausted rather than pretending it can solve the problem.

---

# 4. Session Activity Tracking

The governor needs a lightweight way to know whether a session is active.

Prefer existing session state/activity information if available.

Otherwise introduce the smallest possible mechanism:

```csharp
DateTimeOffset LastActivityUtc
```

updated on meaningful session operations:

- user generation starts
- token generation
- tool result submitted
- resume
- explicit session interaction
- generation completes

Do **not** update activity on every internal KV operation unless necessary.

The objective is:

> "When did the application last meaningfully use this session?"

---

# 5. Never Suspend Unsafe Sessions

The governor must never suspend a session while it is in a state where suspension would violate inference correctness.

At minimum, exclude sessions that are:

```text
Generating
Prefilling
Executing a state transition
Already suspended
Being disposed
Being resumed
```

If existing `SessionState` safety guards already encode this, **reuse them**.

Do not duplicate state-machine logic in the governor.

Conceptually:

```csharp
if (!session.CanSuspend)
    continue;
```

The exact API should follow the current implementation.

---

# 6. Candidate Scoring

Do not simply sort by idle time.

A better lightweight score is:

```text
SuspensionScore =
    idle duration
    × physical pages held
```

This means:

```text
Session A:
    idle 5 minutes
    2 pages

Session B:
    idle 2 minutes
    100 pages
```

Session B may be the better reclamation candidate.

The exact scoring formula is deliberately left open to the coding agent.

A simple deterministic comparator is also acceptable.

The important principle is:

> Reclaim the sessions that provide the greatest useful KV memory recovery for the least disruption.

Avoid adding a machine-learning or elaborate weighted-policy system.

---

# 7. Suspend Operation

When a candidate is selected:

```text
Governor
    ↓
verify session is still eligible
    ↓
SuspendAsync()
    ↓
FileSessionStore / existing persistence mechanism
    ↓
KV pages released
```

The governor must tolerate races.

A session may become active between candidate selection and suspension.

Therefore:

```text
select candidate
      ↓
re-check eligibility
      ↓
SuspendAsync()
```

If the session is no longer suspendable:

```text
skip it
continue with next candidate
```

Do not treat this as an error.

---

# 8. Resume / Transparent Reactivation

The governor should **not own normal resume behaviour**.

If an application interacts with a suspended session:

```text
session.GenerateAsync(...)
session.AppendToolResultAsync(...)
```

the existing session/runtime layer should reactivate it.

Conceptually:

```text
GenerateAsync()
    ↓
IsSuspended?
    ↓ yes
ResumeAsync()
    ↓
restore session/KV state
    ↓
continue generation
```

If this automatic activation does not currently exist, add the smallest session-level hook required.

Do not put resume orchestration into `KvMemoryGovernor`.

The governor's responsibility is:

> reclaim memory.

The session's responsibility is:

> become usable again.

---

# 9. Avoid Disk I/O When Possible

The existing `FileSessionStore` may be used for durable suspension, but the governor should not assume that every deployment wants disk persistence.

If the existing architecture supports an in-memory session snapshot, prefer it for temporary pressure suspension.

The important distinction is:

```text
User explicitly saves session
    → durable persistence

Governor temporarily suspends session
    → memory reclamation / resumable snapshot
```

If the current `SuspendAsync()` already has the correct semantics, reuse it rather than introducing another storage architecture.

---

# 10. Governor Lifetime

Introduce:

```csharp
IKvMemoryGovernor
```

only if an interface is actually useful for testing/injection.

Otherwise a concrete:

```csharp
KvMemoryGovernor
```

is preferable.

The governor should be hosted as a background service/task owned by the runtime.

Conceptually:

```text
InferenceRuntime
    │
    ├── sessions
    ├── KV cache
    └── KvMemoryGovernor
             │
             └── periodic pressure check
```

Do not put the governor inside `CpuKvCache`.

The cache should expose **memory facts**.

The governor makes **policy decisions**.

That separation is important.

---

# 11. KV Cache Metrics

The governor needs lightweight metrics such as:

```text
TotalPages
UsedPages
FreePages
ReservedPages
```

If these already exist, reuse them.

Prefer a single snapshot:

```csharp
KvMemorySnapshot
{
    TotalPages
    UsedPages
    FreePages
    ReservedPages
}
```

rather than repeatedly querying mutable values during one governor cycle.

Conceptually:

```text
snapshot = kvCache.GetMemorySnapshot()

pressure =
    snapshot.UsedPages / snapshot.TotalPages
```

Be careful not to confuse:

```text
Used physical pages
```

with:

```text
Reserved but not yet allocated pages
```

The governor must use the actual physical-memory pressure metric.

---

# 12. Hysteresis

This is important.

Do not constantly suspend/resume around one threshold.

Use:

```text
PressureThreshold = 85%
RecoveryThreshold = 70%
```

Example:

```text
82% → do nothing
86% → reclaim
84% → continue reclaiming
75% → continue until target achieved
69% → stop
```

This should be explicitly tested.

---

# 13. Single-User / Low-Concurrency Optimisation

The governor should effectively disappear when unnecessary.

For a typical desktop user:

```text
One session
8 GB available
KV utilisation = 40%
```

there should be:

- no session suspension
- no unnecessary serialization
- no resume latency
- minimal background overhead

The governor is primarily a **safety valve**, not a mandatory lifecycle manager.

This is one of the key design goals of this plan.

---

# 14. Optional Disablement

Allow:

```csharp
Enabled = false
```

so applications that manage their own session lifecycle can disable the governor completely.

This is important for:

- benchmarking
- debugging
- deterministic testing
- applications with external memory management
- simple single-session inference

---

# 15. Observability

Add lightweight metrics/events.

At minimum:

```text
GovernorCycles
PressureEvents
SessionsSuspended
PagesReclaimed
SuspensionFailures
SkippedBusySessions
```

A useful event might be:

```csharp
SessionSuspendedDueToMemoryPressure
{
    SessionId
    IdleDuration
    PagesReleased
    PressureBefore
    PressureAfter
}
```

Do not create a large telemetry framework.

Use existing logging/metrics conventions.

---

# 16. Failure Safety

Suspension must be best-effort.

If:

```text
SuspendAsync()
```

fails:

- do not corrupt the session
- do not assume its pages were released
- do not subtract pages that weren't actually released
- log/report the failure
- continue evaluating other candidates

The governor itself must never bring down inference because one session could not be suspended.

---

# 17. Mandatory Tests

Create:

### `KvMemoryGovernorTests.cs`

#### Test 1 — NoPressure_DoesNotSuspend

```text
KV = 40%
idle session = 10 minutes

→ session remains active
```

#### Test 2 — Pressure_SuspendsIdleSession

```text
KV >= pressure threshold
idle session eligible

→ SuspendAsync called
```

#### Test 3 — Pressure_DoesNotSuspendActiveSession

```text
KV >= pressure threshold
session generating

→ session remains active
```

#### Test 4 — LargerSessionPreferred

```text
Session A:
    idle 5 minutes
    2 pages

Session B:
    idle 2 minutes
    100 pages

→ B is preferred
```

Adjust this test if the final scoring policy uses a different but equally sensible ranking.

#### Test 5 — Hysteresis_PreventsOscillation

Verify that reclamation continues until the recovery threshold rather than stopping immediately after crossing below the pressure threshold.

#### Test 6 — SessionBecomesActiveBeforeSuspend

```text
candidate selected
session becomes active
SuspendAsync attempted

→ suspension skipped safely
```

#### Test 7 — SuspendFailureDoesNotBreakGovernor

A failed suspension does not terminate the governor loop.

#### Test 8 — PagesActuallyReleased

Verify:

```text
before suspension:
    N physical pages

after suspension:
    N - released pages
```

#### Test 9 — ResumeRestoresSession

Suspend → interact with session → automatic/normal resume → generation continues correctly.

#### Test 10 — DisabledGovernorDoesNothing

Verify that:

```text
Enabled = false
```

results in no suspension activity.

#### Test 11 — SingleSessionLowPressure

A single normal session under low pressure remains resident indefinitely.

#### Test 12 — ConcurrentSessions

Create multiple sessions with different:

- activity states
- idle durations
- KV footprints

and verify the governor reclaims sensible candidates without corrupting active sessions.

---

# 18. Verification

Run:

```bash
dotnet test OpenTail.Stingray.slnx
```

Then:

```bash
dotnet build OpenTail.Stingray.slnx -c Release
```

Also specifically run:

```text
KvMemoryGovernorTests
```

under repeated execution to catch lifecycle races.

---

# 19. Important Non-Goals

Do NOT add:

- a general scheduler
- workload scheduling
- token-level admission control
- GPU memory compaction
- KV migration between devices
- a new persistence system
- agent orchestration
- MCP functionality
- a complex policy engine
- automatic session destruction
- forced suspension of active generation

This feature is specifically:

> **Adaptive physical KV memory reclamation through safe suspension of eligible idle sessions.**

---

# 20. Definition of Done

The implementation is complete when:

1. The governor monitors actual physical KV pressure.
2. No-pressure workloads leave idle sessions resident.
3. Pressure causes eligible idle sessions to be suspended.
4. Active sessions are never suspended by the governor.
5. Reclamation uses hysteresis.
6. Candidate selection considers both idleness and memory footprint.
7. Suspension races are handled safely.
8. Suspension failures cannot crash the governor.
9. Released KV pages return to the physical pool.
10. Suspended sessions resume correctly through the existing session lifecycle.
11. The feature can be disabled.
12. Low-concurrency/single-user workloads incur negligible overhead.
13. All tests pass.
14. Release build passes.

### Architectural goal

The final behaviour should feel like:

```text
                 KV Memory
                    │
             ┌──────┴──────┐
             │             │
         Plenty free     Pressure
             │             │
          Do nothing      ↓
                    Find idle sessions
                           │
                    Rank by usefulness
                           │
                    Suspend selectively
                           │
                    Release KV pages
                           │
                    Memory recovers
                           │
                    ───────────────
                    Session returns
                           │
                    Resume transparently
```

**The key principle is: don't evict because a timer says so. Evict because memory pressure says so, using idle time to determine who is safe to reclaim.**

The coding agent is free to improve individual implementation details if they better fit the existing Stingray architecture, but should preserve these behavioural and correctness invariants and avoid introducing unnecessary abstractions.