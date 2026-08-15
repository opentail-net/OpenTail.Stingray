# Implementation Plan — Plan 018: Session Usage & Performance Metrics (`ISessionMetrics`)

## Objective

Add a lightweight, read-only metrics surface to `IInferenceSession` exposing useful per-session inference and physical KV usage statistics.

The metrics must be:

- cheap to maintain;
- thread-safe;
- read-only to consumers;
- cumulative for the lifetime of the session;
- compatible with streaming generation;
- compatible with speculative decoding;
- compatible with tool continuation;
- compatible with session forking;
- compatible with suspension/resumption;
- independent of any particular host/UI/telemetry framework.

This is a **metrics surface**, not a telemetry framework.

Do not add OpenTelemetry, Prometheus, logging exporters, background metric services, or similar infrastructure.

---

# 1. Public API

Add:

```csharp
public interface ISessionMetrics
{
    long PromptTokens { get; }
    long GeneratedTokens { get; }

    TimeSpan TotalPrefillTime { get; }
    TimeSpan TotalGenerationTime { get; }

    double TokensPerSecond { get; }

    int KvPagesHeld { get; }
}
```

Expose it from `IInferenceSession`:

```csharp
ISessionMetrics Metrics { get; }
```

The exact placement/naming may be adjusted if Stingray's existing API conventions suggest a better fit.

The metrics object must be **read-only from the consumer's perspective**.

---

# 2. Keep Metrics Lightweight

Do not create a large metrics hierarchy.

Avoid:

```text
ISessionMetrics
    ├── IGenerationMetrics
    ├── IPrefillMetrics
    ├── IKvMetrics
    ├── ITokenMetrics
    └── ...
```

The purpose of this feature is to give OpenTail and other hosts a simple snapshot:

```csharp
var metrics = session.Metrics;

Console.WriteLine(metrics.GeneratedTokens);
Console.WriteLine(metrics.TokensPerSecond);
Console.WriteLine(metrics.KvPagesHeld);
```

One small interface is sufficient.

---

# 3. Snapshot vs Mutable Counter Object

Prefer a read-only metrics implementation owned by the session.

For example:

```csharp
internal sealed class SessionMetrics : ISessionMetrics
{
    // Internal atomic counters/timers.
}
```

Consumers see only:

```csharp
ISessionMetrics
```

Do not expose mutable counters.

If the existing architecture already has a suitable immutable snapshot mechanism, that is also acceptable.

The coding agent may choose the cleaner implementation provided the public API remains read-only.

---

# 4. PromptTokens Semantics

`PromptTokens` should represent tokens processed as prompt/prefill input.

This should include the initial prompt/context supplied to the session.

It should also correctly account for subsequent user/tool-result continuations according to Stingray's existing prompt/prefill semantics.

Do not simply return:

```csharp
session.TokenCount
```

because that mixes prompt/context and generated output.

Define and document the distinction clearly.

For example:

```text
PromptTokens
    = tokens processed through prompt/prefill operations

GeneratedTokens
    = tokens actually committed as model-generated output
```

The exact accounting should follow the existing Stingray inference pipeline.

---

# 5. GeneratedTokens Semantics

`GeneratedTokens` counts **committed output tokens**.

This is particularly important because Stingray supports speculative decoding.

If speculative decoding produces:

```text
Draft:       8 tokens
Accepted:    5 tokens
Rejected:    3 tokens
```

then:

```text
GeneratedTokens += 5
```

not 8.

Never count discarded speculative tokens as generated output.

Likewise, rolled-back tokens must not remain in the metric.

---

# 6. Prefill Timing

Track cumulative time spent performing prompt/prefill processing:

```csharp
TimeSpan TotalPrefillTime { get; }
```

Use the existing inference execution boundary rather than adding Stopwatches around individual low-level kernels.

Conceptually:

```text
Prefill begins
    ↓
timer starts
    ↓
prefill completes
    ↓
elapsed added to TotalPrefillTime
```

If a prefill fails, do not count failed work as successfully processed prompt tokens.

The exact timing treatment of failed operations should follow existing transaction semantics.

---

# 7. Generation Timing

Track cumulative time spent performing actual generation:

```csharp
TimeSpan TotalGenerationTime { get; }
```

This should include the time required to generate committed output through the normal generation pipeline.

It should work correctly with:

- greedy sampling;
- stochastic sampling;
- speculative decoding;
- tool continuation;
- streaming;
- cancellation.

Avoid double-counting time when speculative decoding internally performs multiple model operations.

The metric should represent **wall-clock generation time observed by the session**, not a sum of arbitrary internal kernel timings.

---

# 8. TokensPerSecond

Expose:

```csharp
double TokensPerSecond { get; }
```

Recommended definition:

```text
GeneratedTokens / TotalGenerationTime
```

with:

```text
0 tokens or 0 elapsed time → 0
```

Do not return `NaN` or infinity.

For example:

```csharp
if (generated == 0 || generationTime <= TimeSpan.Zero)
    return 0;

return generated / generationTime.TotalSeconds;
```

This should be a cumulative lifetime/session rate.

Do not make it an instantaneous rolling-window metric in this first implementation.

A future feature could add windowed throughput if needed.

---

# 9. KV Pages Held

Expose:

```csharp
int KvPagesHeld { get; }
```

This must represent the number of physical KV pages currently held by the session.

It should reflect Stingray's actual page ownership/reference semantics.

Do not estimate:

```text
TokenCount / PageSize
```

because that can be wrong with:

- partial pages;
- shared prefix pages;
- CoW;
- forked sessions;
- released pages;
- prefix-cache references.

Use the authoritative `CpuKvCache` / `IKvCache` / `CpuKvSequence` ownership information already present in Stingray.

---

# 10. Shared Prefix Pages

Be explicit about shared physical pages.

If:

```text
Session A ──┐
            ├── Physical Page 42
Session B ──┘
```

then each session's `KvPagesHeld` should report according to the semantics chosen by the existing sequence ownership model.

Recommended meaning:

> Number of physical KV pages currently retained by this session's sequence/reference set.

Do not report the global cache's total page count.

Do not count the same physical page multiple times merely because the RadixPrefixTree also retains it.

If the existing architecture distinguishes **session-held references** from **cache-held references**, use the session reference count.

Document the chosen semantics in the API.

---

# 11. Fork Semantics

Forking must be handled explicitly.

Example:

```text
Parent
  ├── Page 1
  ├── Page 2
  └── Page 3

Fork
  ↓

Parent → 1,2,3
Child  → 1,2,3
```

Both sessions may initially report the same number of pages held.

As branches generate:

```text
Parent → 1,2,3,4
Child  → 1,2,3,5
```

their metrics should update independently.

Do not treat a fork as generating new prompt tokens.

Do not count shared physical pages as newly allocated memory merely because the child references them.

---

# 12. Suspension Semantics

When a session is suspended and its physical KV pages are released:

```text
KvPagesHeld
```

must fall accordingly.

For example:

```text
Before suspend:
KvPagesHeld = 100

Suspend:
KvPagesHeld = 0

Resume:
KvPagesHeld = 100
```

assuming the resumed session restores the same KV state.

This is particularly important for the existing `KvMemoryGovernor`.

Do not maintain a stale cached page count across suspension.

---

# 13. Prefix Cache Interaction

Do not count RadixPrefixTree's own retention as session-owned pages unless those pages are also retained by the session.

For example:

```text
PrefixCache
    └── Page 42

Session A
    └── Page 42
```

`KvPagesHeld` for Session A should count the page once.

Evicting the prefix cache should not cause the session metric to suddenly drop if the session still owns its reference.

Likewise, disposing the session should not affect a prefix-cache-only reference.

Use the existing reference-count architecture rather than inventing a second page accounting system.

---

# 14. Tool Continuation

Tool results must be handled correctly.

Example:

```text
User prompt
    ↓
Model generation
    ↓
Tool call
    ↓
Tool executes externally
    ↓
Tool result appended
    ↓
Model continues
```

Metrics should account for:

- tool-result prompt/prefill tokens under `PromptTokens`;
- subsequently generated model tokens under `GeneratedTokens`;
- actual prefill/generation time in the corresponding timers.

Do not count time spent waiting for an external tool as `TotalGenerationTime`.

Likewise, do not count OpenTail/tool execution time as Stingray prefill time.

---

# 15. Streaming

Metrics must update correctly during streaming.

For example:

```text
GenerateAsync()
    ↓
token 1
Metrics.GeneratedTokens == 1

token 2
Metrics.GeneratedTokens == 2

token 3
Metrics.GeneratedTokens == 3
```

A host should not need to wait until the entire generation completes to inspect metrics.

This is particularly useful for:

- live TUI displays;
- OpenTail FlightDeck;
- REST streaming;
- debugging;
- benchmark output.

---

# 16. Thread Safety

Metrics may be read concurrently with inference.

For example:

```text
Thread A:
    session.GenerateAsync()

Thread B:
    session.Metrics.GeneratedTokens
```

No torn values or inconsistent integer reads should occur.

Use appropriate atomic operations for counters.

For timings, use a safe representation such as:

```text
long elapsedTicks
```

and convert to `TimeSpan` when read.

Avoid locking the inference hot path unnecessarily.

The implementation should be cheap enough that metrics do not become a meaningful inference overhead.

---

# 17. Avoid Stopwatch Allocation Per Token

Do not create a `Stopwatch` for every generated token.

Timing should happen around logical operations:

```text
Prefill operation
Generation operation
```

rather than:

```text
token → Stopwatch
token → Stopwatch
token → Stopwatch
```

For example:

```csharp
var start = Stopwatch.GetTimestamp();

try
{
    // prefill
}
finally
{
    metrics.AddPrefillTime(Stopwatch.GetElapsedTime(start));
}
```

Use the appropriate .NET timing APIs available in the project's target framework.

---

# 18. Failed Operations

Metrics must preserve sensible semantics when inference fails.

For example:

```text
Generate
    ↓
partial speculative work
    ↓
exception
    ↓
rollback
```

Do not leave:

```text
GeneratedTokens += speculativeTokens
```

if those tokens were never committed.

Similarly, failed prefill should not increase `PromptTokens` as though the prompt was successfully processed.

Timing may optionally include failed execution time, but token counters must follow committed-state semantics.

If there is ambiguity, prefer **successful committed work** for token counts.

---

# 19. Reset Semantics

Do not add a general:

```csharp
Metrics.Reset()
```

to the public API.

These are lifetime/session counters.

If the session itself is forked, suspended, resumed, or continued, metrics remain associated with that logical session according to the semantics defined above.

If a future benchmark needs resettable counters, add a separate host/diagnostics mechanism rather than making the public session metrics mutable.

---

# 20. Tests

Add:

```text
SessionMetricsTests.cs
```

with at least:

### Test 1 — InitialMetricsAreZero

New session:

```text
PromptTokens == 0
GeneratedTokens == 0
PrefillTime == 0
GenerationTime == 0
TokensPerSecond == 0
```

---

### Test 2 — PromptTokensTracked

Process a known prompt and verify the correct count.

---

### Test 3 — GeneratedTokensTracked

Generate a known number of tokens and verify committed output count.

---

### Test 4 — TokensPerSecond

Verify:

```text
GeneratedTokens / TotalGenerationTime
```

and zero behaviour when no generation has occurred.

Use deterministic/fake timing if the existing test infrastructure allows it; don't make tests depend on an exact real-world execution speed.

---

### Test 5 — PrefillTimingRecorded

Verify successful prefill increases `TotalPrefillTime`.

---

### Test 6 — GenerationTimingRecorded

Verify generation increases `TotalGenerationTime`.

---

### Test 7 — SpeculativeTokensOnlyCountWhenCommitted

Draft 8 / accept 5 should result in:

```text
GeneratedTokens += 5
```

---

### Test 8 — StreamingMetricsUpdate

Inspect metrics while generation is still streaming.

---

### Test 9 — ToolContinuationMetrics

Verify tool-result tokens are accounted for as prompt/prefill work and subsequent generated tokens as generation.

---

### Test 10 — ForkMetricsAreIndependent

Fork a session and verify each branch maintains its own generated-token counters.

---

### Test 11 — KvPagesHeldTracksOwnership

Verify allocation/release changes the reported page count.

---

### Test 12 — KvPagesHeldWithFork

Verify CoW/shared pages are reported according to the defined session ownership semantics.

---

### Test 13 — SuspensionReleasesPages

Verify:

```text
suspend → KvPagesHeld decreases
resume → KvPagesHeld restored
```

---

### Test 14 — PrefixCacheDoesNotDoubleCount

Verify prefix-cache retention does not inflate session-owned page count.

---

### Test 15 — FailedGenerationDoesNotCommitTokens

Force a failed/rolled-back generation and verify generated-token metrics remain correct.

---

### Test 16 — ConcurrentMetricReads

Read metrics concurrently while inference runs and verify no invalid values/errors occur.

---

# 21. Documentation

Document the exact semantics of each property.

Especially clarify:

```text
PromptTokens
    = successfully processed prompt/prefill tokens

GeneratedTokens
    = committed model-generated tokens

TotalPrefillTime
    = cumulative Stingray prompt/prefill execution time

TotalGenerationTime
    = cumulative Stingray generation execution time

TokensPerSecond
    = cumulative generated tokens / cumulative generation time

KvPagesHeld
    = physical KV pages currently retained by this session
```

Also document that:

- counters are cumulative;
- metrics are read-only;
- values may change while generation is running;
- `TokensPerSecond` is a lifetime/session rate, not instantaneous throughput;
- tool waiting time is not generation time;
- discarded speculative tokens are not generated tokens.

---

# 22. Non-Goals

Do NOT add:

- Prometheus;
- OpenTelemetry;
- EventCounters;
- logging exporters;
- metrics HTTP endpoints;
- dashboards;
- rolling throughput windows;
- per-token timing;
- GPU kernel timing;
- hardware performance counters;
- model-wide global statistics;
- distributed telemetry;
- a metrics database.

Those belong above Stingray.

---

# 23. Definition of Done

Plan 012 is complete when:

1. `ISessionMetrics` exists.
2. `IInferenceSession.Metrics` exposes it.
3. Metrics are read-only.
4. Prompt tokens are tracked correctly.
5. Committed generated tokens are tracked correctly.
6. Prefill timing is tracked.
7. Generation timing is tracked.
8. Tokens/sec is correctly derived.
9. Physical session KV page ownership is exposed.
10. Speculative rejected tokens are never counted as generated.
11. Streaming updates metrics correctly.
12. Tool continuation updates the appropriate counters.
13. Forked sessions maintain independent metrics.
14. Suspend/resume maintains correct KV page reporting.
15. Prefix-cache references are not double-counted.
16. Failed/rolled-back operations do not leave incorrect token counts.
17. Concurrent reads are safe.
18. Metrics impose negligible overhead on the inference path.
19. No telemetry framework or unrelated abstraction is introduced.
20. All new tests pass.
21. Full Stingray test suite passes.
22. Release build passes.

---

## Implementation Principle

Keep this feature **boringly simple**.

The coding agent is encouraged to choose better implementation details if they fit the existing Stingray architecture, especially where existing counters, page ownership APIs, checkpoint semantics, or timing infrastructure can be reused.

Do not redesign session state or KV ownership to accommodate metrics.

The desired result is simply:

```csharp
var m = session.Metrics;

Console.WriteLine(
    $"{m.GeneratedTokens} tokens @ {m.TokensPerSecond:F1} tok/s " +
    $"({m.KvPagesHeld} KV pages)");
```

without OpenTail having to wrap or instrument the inference engine itself.