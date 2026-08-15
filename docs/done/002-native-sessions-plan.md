> **ARCHIVED, 2026-08-15.** Implemented as an `IInferenceSession`/`InferenceSession`-era
> Sessions-layer feature (confirmed against source, not just this document's own claim). That
> whole lineage is being superseded by the `HotSession` architecture — see
> [028](028-inference-session-to-hotsession-migration-plan.md) for current migration status
> (Phases 1-3 done) and [030](../030-delete-inferencesession-todo.md) for the still-open
> deletion of the legacy `InferenceSession`/`InferenceRuntime` types once HotSession fully
> replaces them. Carried forward: nothing beyond what 028/030 already track in
> [00-current-work.md](../00-current-work.md).

---

# OpenTail.Stingray — Native Sessions Implementation Plan

## Objective

Implement **first-class native inference sessions** inside OpenTail.Stingray.

A native session represents the complete state required to continue inference without re-processing the entire prompt from scratch.

The session must own and coordinate:

- token history
- current position
- KV-cache state
- generation state
- model association
- sampling state
- session lifecycle
- cancellation
- checkpoints
- optional persistence

The implementation must remain **Stingray-native**.

Do not introduce dependencies on OpenTail's higher-level conversation, agent, memory, MCP, or tool systems.

---

# 1. Target architecture

The desired architecture is:

```text
                    Stingray
                       │
                ┌──────┴──────┐
                │             │
             Model         Sessions
                              │
                 ┌────────────┼────────────┐
                 ▼            ▼            ▼
             Session A    Session B    Session C
                 │            │            │
                 ▼            ▼            ▼
               KV A         KV B         KV C
```

The model is shared.

Each session owns its own inference state.

The key distinction is:

```text
Model = shared immutable/inference weights

Session = mutable inference state
```

---

# 2. What a session means

A session represents:

> "This model has already processed these tokens and is ready to continue from this exact inference position."

For example:

```text
Session
│
├── Model
│
├── Tokens
│    ├── 1
│    ├── 2
│    ├── 3
│    └── ...
│
├── Position
│
├── KV cache
│
├── Generation state
│
└── Sampling state
```

If the user generates another token, Stingray should continue from the existing state.

It must **not** need to reconstruct the entire prompt.

---

# 3. First inspect the existing implementation

Before changing code, inspect the existing Stingray architecture.

Identify exactly:

```text
Current model abstraction
Current inference engine
Current session abstraction
Current KV cache
Current token storage
Current generation loop
Current sampling state
Current scheduler
Current cancellation model
Current disposal model
```

Specifically locate:

```text
ISequenceKvCache
KvBytesPerToken
PagedKvCache
ContinuousBatchingEngine
```

and determine which parts already represent session semantics.

Produce an architecture note before implementing.

Do not duplicate existing abstractions unnecessarily.

---

# 4. Establish the session boundary

The first implementation task is to make the boundary explicit.

Conceptually:

```csharp
IInferenceSession
```

should own:

```text
tokens
position
KV state
generation state
```

while the model owns:

```text
weights
architecture
tokenizer/model configuration
backend
```

Avoid:

```text
Engine owns everything
```

because that makes independent sessions difficult.

---

# 5. Session identifier

Create a stable session ID.

For example:

```csharp
public readonly record struct SessionId(Guid Value);
```

or use the project's existing identifier convention.

Requirements:

- unique
- immutable
- cheap to compare
- serializable
- independent of object reference identity

Expose:

```csharp
SessionId Id { get; }
```

---

# 6. Session state

Define explicit lifecycle state.

Suggested:

```csharp
public enum SessionState
{
    Created,
    Ready,
    Generating,
    Suspended,
    Faulted,
    Disposed
}
```

Adapt names if the existing codebase has a better convention.

The important part is that lifecycle transitions are explicit.

---

# 7. State transition rules

Initially implement:

```text
Created
   ↓
Ready
   ↓
Generating
   ↓
Ready
```

Suspension:

```text
Ready
   ↓
Suspended
   ↓
Ready
```

Fatal error:

```text
Ready
   ↓
Faulted
```

Disposal:

```text
Ready / Suspended / Faulted
   ↓
Disposed
```

Reject invalid operations.

For example:

```text
Generate()
```

must not be allowed after:

```text
Disposed
```

---

# 8. Define IInferenceSession

Create the smallest useful interface.

Conceptually:

```csharp
public interface IInferenceSession : IAsyncDisposable
{
    SessionId Id { get; }

    SessionState State { get; }

    long TokenCount { get; }

    ValueTask AppendAsync(
        ReadOnlyMemory<int> tokens,
        CancellationToken cancellationToken = default);

    IAsyncEnumerable<GeneratedToken> GenerateAsync(
        GenerationOptions options,
        CancellationToken cancellationToken = default);
}
```

Do not immediately put every possible operation into this interface.

Prefer a small stable core.

---

# 9. Keep token ownership explicit

A session needs to know which tokens have already been processed.

Maintain:

```text
Token history
Committed token count
KV token count
Current position
```

The key invariant is:

```text
CommittedTokenCount == KvTokenCount
```

after every completed operation.

---

# 10. Separate committed and speculative tokens

Eventually speculative decoding and checkpoints will need temporary state.

Therefore distinguish:

```text
Committed tokens
```

from:

```text
Speculative tokens
```

Conceptually:

```text
Committed:
[A B C D E]

Speculative:
[F G H]
```

If committed:

```text
[A B C D E F G H]
```

If rejected:

```text
[A B C D E]
```

This distinction should be designed into the session architecture now, even if speculative decoding comes later.

---

# 11. Session creation

Provide a factory or manager.

Conceptually:

```csharp
var session = await sessionManager.CreateAsync(model);
```

A new session must contain:

```text
unique ID
empty token state
empty KV state
Ready state
default generation state
```

It must not duplicate model weights.

---

# 12. Multiple sessions

The same model must support:

```text
Model
 ├── Session A
 ├── Session B
 ├── Session C
 └── Session D
```

Each session has:

```text
different token history
different KV
different generation state
```

but all reference the same model.

---

# 13. Model lifetime

Do not make a session responsible for unloading the model.

Prefer:

```text
ModelManager
      │
      ▼
    Model
      │
 ┌────┼────┐
 ▼    ▼    ▼
 S1   S2   S3
```

A session should hold a safe reference to the model/runtime.

Disposing a session must not unload the model used by other sessions.

---

# 14. Append tokens

Implement:

```csharp
AppendAsync(tokens)
```

The operation should:

1. validate session state
2. validate token IDs
3. append tokens logically
4. run the model forward pass
5. populate KV
6. advance position
7. commit the tokens

Do not update committed logical state before the operation can be successfully completed unless rollback is guaranteed.

---

# 15. Prompt ingestion

A prompt should become:

```text
tokens
   ↓
session.AppendAsync()
   ↓
KV populated
   ↓
session Ready
```

Generation then starts from the already-populated session.

This creates the fundamental distinction:

```text
Prompt ingestion
```

versus:

```text
Generation
```

---

# 16. Generation

Generation becomes:

```text
session.GenerateAsync(options)
```

The generation loop should:

```text
read current position
      ↓
run model
      ↓
sample token
      ↓
append token
      ↓
update KV
      ↓
yield token
      ↓
repeat
```

The session remains the owner of the evolving state.

---

# 17. Generation state

Generation-specific state should be associated with the session where appropriate.

Examples:

```text
random seed
sampling RNG
repetition history
stop conditions
generated-token count
```

However, distinguish:

```text
Persistent session state
```

from:

```text
Per-generation options
```

For example:

```text
temperature = 0.7
```

should normally be a generation option, not permanently embedded into the session.

---

# 18. Sampling state

Be particularly careful with RNG state.

If deterministic continuation is desired, a session/checkpoint may need to capture:

```text
RNG algorithm
RNG seed
RNG state
```

Otherwise:

```text
save → restore → generate
```

may produce a different continuation even with identical model state.

---

# 19. Cancellation

Generation must support:

```csharp
CancellationToken
```

Cancellation must leave the session valid.

After cancellation:

```text
SessionState == Ready
```

unless a genuinely unrecoverable error occurred.

Do not leave the session permanently in:

```text
Generating
```

after cancellation.

---

# 20. Partial generation semantics

Define exactly what happens when cancellation occurs.

Recommended initial behaviour:

> Every successfully sampled token is committed.

Therefore:

```text
Generate
 ↓
token A committed
 ↓
token B committed
 ↓
token C committed
 ↓
cancel
```

results in:

```text
session contains A B C
```

This is simpler and more useful for normal generation.

Checkpoint/transaction APIs can provide alternative all-or-nothing semantics later.

---

# 21. Session locking

Only one mutation operation should execute against a session at a time.

For example:

```text
Session A
    │
    ├── Generate()   ← running
    │
    └── Append()     ← reject/queue
```

Do not allow concurrent mutation of the same KV state.

Use an appropriate lightweight async synchronization primitive.

Avoid locking the model globally.

---

# 22. Session statistics

Expose useful state:

```text
TokenCount
KvTokenCount
GenerationCount
GeneratedTokenCount
PromptTokenCount
```

Eventually:

```text
KvMemoryBytes
GenerationTime
PromptProcessingTime
TokensPerSecond
```

These are useful for both diagnostics and OpenTail's UI.

---

# 23. KV cache integration

The session must depend on an abstraction such as:

```csharp
IKvCache
```

rather than a concrete KV implementation.

Conceptually:

```text
Session
   │
   ▼
IKvCache
   │
   ├── CpuKvCache
   ├── PagedKvCache
   ├── QuantizedKvCache
   └── future GPU implementations
```

The session should know:

```text
append
position
release
checkpoint
restore
```

but should not know:

```text
page allocation internals
tensor layout
memory pointers
backend-specific details
```

---

# 24. Session ↔ KV ownership

The simplest initial rule:

> One session owns one logical KV-cache instance.

The KV allocator/pool may physically share memory.

Therefore:

```text
Session A → KV handle A
Session B → KV handle B
```

but internally:

```text
KV Page Pool
 ├── pages used by A
 └── pages used by B
```

---

# 25. Session disposal

Disposing a session must:

1. prevent further operations
2. release session-owned KV resources
3. release unmanaged resources
4. unregister from the session manager
5. preserve the model for other sessions

It must be safe to call disposal more than once if the existing project conventions permit idempotent disposal.

---

# 26. Session manager

Introduce:

```csharp
ISessionManager
```

responsible for:

```text
create
lookup
remove
dispose
```

Example:

```csharp
ValueTask<IInferenceSession> CreateAsync(...);

IInferenceSession? Get(SessionId id);

ValueTask RemoveAsync(SessionId id);
```

Do not put global session registry logic inside the session itself.

---

# 27. In-memory implementation first

Implement:

```text
InMemorySessionManager
```

first.

It should maintain:

```text
SessionId → IInferenceSession
```

Do not implement disk persistence at this stage.

The objective is to prove the lifecycle architecture.

---

# 28. Checkpoints

After the basic session is working, introduce:

```csharp
SessionCheckpoint
```

A checkpoint represents:

> The exact committed inference state at a particular point.

It should capture enough state to restore:

```text
token position
KV position
generation/RNG state where required
```

---

# 29. Checkpoint API

Conceptually:

```csharp
var checkpoint = session.CreateCheckpoint();

await session.GenerateAsync(...);

session.Rollback(checkpoint);
```

The rollback must restore:

```text
tokens
position
KV
generation state
```

to the checkpoint.

---

# 30. Do not initially serialize KV checkpoints

For the first implementation:

```text
Checkpoint = in-memory state
```

Do not immediately implement:

```text
Checkpoint = serialized KV
```

That introduces substantial complexity before the semantics are proven.

---

# 31. Forking

Add:

```csharp
IInferenceSession Fork();
```

The first correct implementation may use a deep copy if necessary.

The API should be introduced before optimising its internals.

Required behaviour:

```text
parent = original
child = parent.Fork()
```

Initially:

```text
parent state == child state
```

Then:

```text
child generates
```

must not change:

```text
parent
```

---

# 32. Optimise fork with paged KV

Once paged KV exists:

```text
Parent
   │
   ├── Page 0 ── shared
   ├── Page 1 ── shared
   ├── Page 2 ── shared
   └── Page 3 ── shared
```

Fork:

```text
Parent ──┐
         ├── Page 0
Child ───┘
```

Reference counts increase.

When either session modifies a shared page:

```text
if refcount > 1
    allocate new page
    copy
    decrement old refcount
```

This is copy-on-write.

---

# 33. Native sessions and paged KV are deliberately connected

Do not implement these as unrelated features.

The relationship should be:

```text
Native Sessions
       │
       ▼
Session owns logical KV state
       │
       ▼
Paged KV owns physical allocation
       │
       ▼
Fork can share pages
```

This is where the architecture starts paying off.

---

# 34. Suspension

Add a future:

```csharp
session.SuspendAsync()
```

Suspension should:

```text
retain logical session state
release resident KV
```

The session becomes:

```text
Suspended
```

---

# 35. Resume

Resume should:

```text
load/recreate KV
replay committed tokens
restore generation state
return Ready
```

Initially, replay from tokens is acceptable.

Do not optimise this prematurely.

---

# 36. Persistence

After native sessions, checkpoints and lifecycle work correctly, add:

```csharp
ISessionStore
```

Conceptually:

```csharp
SaveAsync(SessionSnapshot)
LoadAsync(SessionId)
DeleteAsync(SessionId)
```

The first persistent format should store:

```text
session ID
model fingerprint
tokenizer fingerprint
tokens
position
generation metadata
session format version
```

---

# 37. Why persist tokens rather than raw KV initially

Raw KV persistence is tightly coupled to:

```text
model architecture
KV layout
quantisation
backend
runtime version
hardware
```

Tokens are much more portable.

Therefore:

```text
Persistent session
      =
metadata + tokens
```

while:

```text
KV
=
runtime acceleration
```

On restore:

```text
tokens
   ↓
forward pass
   ↓
KV reconstructed
```

---

# 38. Model compatibility

Persist:

```text
ModelId
ModelFingerprint
TokenizerFingerprint
```

On restore:

```text
same compatible model
    → restore

different/incompatible model
    → reject
```

Never silently restore a session against a different architecture.

---

# 39. Session snapshot

Create a versioned DTO such as:

```csharp
public sealed record SessionSnapshot
{
    public int Version { get; init; }

    public SessionId Id { get; init; }

    public string ModelId { get; init; }

    public string ModelFingerprint { get; init; }

    public string TokenizerFingerprint { get; init; }

    public IReadOnlyList<int> Tokens { get; init; }

    public long Position { get; init; }
}
```

Do not serialize runtime object graphs.

Do not use .NET binary serialization.

---

# 40. Native session tests

Create a dedicated test suite.

## Creation

```text
Create session
→ valid ID
→ Ready
→ zero tokens
→ zero KV tokens
```

## Append

```text
Append prompt
→ token count correct
→ KV count correct
```

## Generate

```text
Generate N tokens
→ N tokens committed
→ KV matches
```

## Repeat generation

```text
Generate
Generate again
→ second generation starts from previous state
```

---

# 41. Cancellation tests

Test:

```text
start generation
cancel
```

Verify:

```text
session not corrupted
session not permanently locked
state = Ready
KV/token consistency preserved
```

---

# 42. Fork tests

Test:

```text
Parent
  ↓
Fork
  ↓
Child
```

Verify:

```text
same token history
same position
same model
```

Then:

```text
Child generates
```

Verify:

```text
Parent unchanged
Child advanced
```

---

# 43. Checkpoint tests

Test:

```text
A
checkpoint
B
rollback
```

Verify:

```text
state == A
```

Then generate again and verify the continuation is correct.

---

# 44. Persistence tests

Test:

```text
Create
Append
Generate
Save
Dispose
Load
Generate
```

The loaded session must continue correctly.

For deterministic generation, compare output against the original continuation.

---

# 45. Multi-session tests

Create:

```text
Session A
Session B
Session C
```

with different prompts.

Verify:

```text
A cannot affect B
B cannot affect C
C cannot affect A
```

This is particularly important for detecting accidental shared KV state.

---

# 46. Model-sharing tests

Load one model.

Create multiple sessions.

Verify:

```text
model loaded once
sessions reference same model
disposing Session A does not affect B
```

---

# 47. Memory tests

Measure:

```text
1 session
10 sessions
100 sessions
```

where practical.

Look for:

```text
KV leakage
unmanaged memory leakage
session registry leakage
disposed session retention
```

---

# 48. Performance tests

Compare:

### Old behaviour

```text
prompt
→ process prompt
→ generate
```

against:

### Native session

```text
prompt
→ create session
→ process once
→ generate
→ generate again
```

The second generation should not reprocess the complete prompt.

---

# 49. Important correctness invariant

At every stable session boundary:

```text
Logical token count
==
KV token count
==
current inference position
```

If this invariant is violated, the session is corrupted.

Make this invariant visible in tests.

---

# 50. Another critical invariant

A session must never accidentally share mutable state with another session.

Shared:

```text
model weights
immutable configuration
tokenizer
```

Not shared:

```text
KV state
token history
RNG state
generation state
```

unless explicitly implemented through a safe sharing mechanism such as paged copy-on-write.

---

# 51. API compatibility

Do not break existing Stingray users unnecessarily.

If the current API already has:

```text
Generate()
Run()
Infer()
```

retain it initially.

Implement those APIs internally using native sessions where practical.

For example:

```text
Existing Generate(prompt)
       ↓
Create temporary session
       ↓
Append prompt
       ↓
Generate
       ↓
Dispose session
```

This gives backwards compatibility while introducing the new architecture.

---

# 52. This is an important migration strategy

Eventually:

```text
Old API
   ↓
temporary session
   ↓
Native session
   ↓
Inference
```

and:

```text
New API
   ↓
persistent session
   ↓
Inference
```

both use the same engine.

This avoids maintaining two independent inference paths.

---

# 53. NativeAOT requirements

The implementation must remain compatible with the project's NativeAOT goals.

Avoid unnecessary:

```text
reflection-heavy serialization
dynamic runtime type discovery
runtime-generated proxies
```

Prefer explicit DTOs and serializers already compatible with the project.

Run NativeAOT builds as part of validation.

---

# 54. Thread-safety requirements

Define the contract clearly:

> A session is logically single-writer.

Multiple readers of immutable metadata are acceptable.

Only one operation may mutate:

```text
tokens
KV
generation state
```

at a time.

This makes correctness much easier than trying to make the entire session fully concurrent.

---

# 55. What the first release should NOT contain

Do not attempt to implement all of these simultaneously:

```text
❌ distributed sessions
❌ network session server
❌ encrypted session database
❌ automatic summarisation
❌ agent memory
❌ multi-agent orchestration
❌ automatic context compaction
❌ GPU KV persistence
❌ distributed KV
```

These can come later.

The first goal is an excellent local session primitive.

---

# 56. Recommended implementation order

Follow this order strictly:

```text
M0  Existing architecture audit
 ↓
M1  SessionId + SessionState
 ↓
M2  IInferenceSession
 ↓
M3  Move logical token state into session
 ↓
M4  Session-owned KV
 ↓
M5  Generate through session
 ↓
M6  Cancellation + lifecycle correctness
 ↓
M7  Multiple independent sessions
 ↓
M8  Checkpoints
 ↓
M9  Fork
 ↓
M10 Paged-KV copy-on-write fork
 ↓
M11 Suspension
 ↓
M12 Persistence
 ↓
M13 Resume/KV rebuild
 ↓
M14 Session manager + memory management
```

Do not jump directly to M10.

---

# 57. Definition of done

Native Sessions are complete when all of the following are true:

- [ ] Stingray has a first-class `IInferenceSession`.
- [ ] Sessions have stable IDs.
- [ ] Sessions have explicit lifecycle state.
- [ ] A model can have multiple independent sessions.
- [ ] Sessions own their logical token history.
- [ ] Sessions own their logical inference position.
- [ ] Sessions own their KV state.
- [ ] Sessions can ingest tokens.
- [ ] Sessions can generate repeatedly.
- [ ] Generation does not reprocess existing context.
- [ ] Cancellation leaves the session valid.
- [ ] Session mutation is single-writer safe.
- [ ] Session disposal releases KV resources.
- [ ] Model weights remain shared between sessions.
- [ ] Session state cannot leak between sessions.
- [ ] Checkpoints can be created.
- [ ] Checkpoints can be rolled back.
- [ ] Sessions can be forked.
- [ ] Forking preserves parent state.
- [ ] Paged KV can eventually make forks copy-on-write.
- [ ] Sessions can be suspended.
- [ ] Sessions can release resident KV.
- [ ] Sessions can resume.
- [ ] Sessions can persist token state.
- [ ] Persisted sessions validate model compatibility.
- [ ] Persisted sessions can rebuild KV.
- [ ] Existing stateless APIs remain functional.
- [ ] Existing inference tests remain green.
- [ ] NativeAOT builds remain functional.
- [ ] Performance does not regress materially for existing workloads.

---

# 58. Final target

The end result should make this possible:

```csharp
var session = await runtime.CreateSessionAsync(model);

await session.AppendAsync(promptTokens);

await foreach (var token in session.GenerateAsync(options))
{
    Console.Write(token);
}

// Later...

await foreach (var token in session.GenerateAsync(options))
{
    Console.Write(token);
}
```

The second generation should continue from the existing inference state.

Then:

```csharp
var branch = session.Fork();
```

should create an independent continuation.

And eventually:

```csharp
await session.SuspendAsync();
```

can release its KV resources while retaining its logical state.

Later:

```csharp
await session.ResumeAsync();
```

reconstructs the inference state.

The architectural progression becomes:

```text
                 Native Session
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Persistent    Forkable     Suspendable
          │            │            │
          ▼            ▼            ▼
       Restart      Branching    Memory mgmt
          │            │            │
          └────────────┼────────────┘
                       ▼
                   Paged KV
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
           Sharing  COW fork  Eviction
              │
              ▼
       Speculative decoding
```

The important architectural decision is therefore:

> **Make the session the owner of logical inference state, and make the KV cache the replaceable physical storage mechanism underneath it.**

That gives Stingray a clean foundation for the next generation of features without coupling it to OpenTail itself.