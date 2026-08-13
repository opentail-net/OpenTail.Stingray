# OpenTail.Stingray — Inference State Architecture & Transition Plan

## Purpose

This plan defines the architectural transition required to move OpenTail.Stingray from its current inference-engine structure toward:

```text
Model
  │
  ├── immutable model state
  │
  ▼
Runtime
  │
  ├── Session A
  │     └── KV state
  │
  ├── Session B
  │     └── KV state
  │
  └── Session C
        └── KV state
```

This is **not** another major feature.

It is the structural work required to make the following features fit together cleanly:

1. Native Sessions
2. Paged KV Cache
3. Session Forking / Copy-on-Write
4. Checkpoints / Rollback
5. Multi-session batching
6. Speculative decoding
7. KV quantisation
8. KV eviction / suspend / resume
9. Prefix sharing

The goal is to make those features possible **without repeatedly redesigning the inference engine**.

---

# 1. What has NOT yet been explicitly planned

The previous plans describe:

```text
Native Sessions
Paged KV
Speculative Decoding
```

but there is an important missing layer between them:

> **What exactly constitutes inference state, who owns it, and which components are allowed to mutate it?**

That needs to be defined before implementation.

The following areas need an explicit architecture.

---

# 2. Missing architecture area #1 — Model vs Runtime vs Session

Stingray needs three distinct conceptual levels.

## Model

The model contains things that do not change during inference:

```text
weights
architecture
layer configuration
vocabulary/model metadata
quantisation configuration
model dimensions
```

Conceptually:

```text
IModel
```

The model should be shareable.

---

## Runtime

The runtime manages execution resources:

```text
backend
thread pools
memory pools
KV allocator
scheduler
model execution
```

Conceptually:

```text
IInferenceRuntime
```

A runtime may own one or more models and many sessions.

---

## Session

A session owns mutable inference state:

```text
token sequence
current position
KV state
generation state
sampling state
```

Conceptually:

```text
IInferenceSession
```

The fundamental relationship is:

```text
Runtime
   │
   ├── Model
   │
   ├── Session A
   ├── Session B
   └── Session C
```

---

# 3. Missing architecture area #2 — Define ownership

Every important piece of state must have one clear owner.

Create an ownership table before coding.

| State | Owner |
|---|---|
| Model weights | Model |
| Architecture | Model |
| Tokenizer/model vocabulary | Model/runtime layer |
| Token history | Session |
| Current position | Session |
| KV logical sequence | Session |
| KV physical pages | KV subsystem |
| KV allocator | Runtime |
| RNG state | Session/generation state |
| Sampling configuration | Generation request |
| Generation cancellation | Generation operation |
| Scheduler state | Runtime |
| Session registry | Session manager |
| Persistence | Session store |

The implementation must avoid ambiguous ownership.

---

# 4. Missing architecture area #3 — Mutable vs immutable state

Explicitly classify every major object.

### Immutable/shared

```text
Model
Weights
Architecture
Model configuration
Tokenizer configuration
```

### Session-owned mutable

```text
Token history
Current position
KV sequence handle
Generation state
RNG state
```

### Runtime-owned mutable

```text
Memory pools
KV page allocator
Scheduler
Execution queues
```

This distinction is essential for safe multi-session operation.

---

# 5. Missing architecture area #4 — Inference state object

Introduce a conceptual grouping for session state.

Do not necessarily expose this as a public API immediately.

For example:

```csharp
internal sealed class InferenceState
{
    TokenSequence Tokens;
    long Position;
    IKvSequence Kv;
    GenerationState Generation;
}
```

The purpose is to prevent state from being scattered across:

```text
engine fields
model fields
KV fields
scheduler fields
```

The exact class structure can differ.

The architectural requirement is:

> There must be one identifiable place where session-specific mutable inference state lives.

---

# 6. Missing architecture area #5 — Logical KV vs physical KV

This distinction must be made explicit before Paged KV.

A session needs to say:

```text
"I have KV for sequence positions 0..N."
```

The KV implementation decides:

```text
"Those positions are stored in pages 12, 18, 21..."
```

Therefore:

```text
Session
   │
   ▼
Logical KV sequence
   │
   ▼
IKvCache
   │
   ▼
Physical pages
```

The session must never directly manipulate page addresses.

---

# 7. Missing architecture area #6 — KV sequence handle

Introduce a logical handle between session and KV storage.

Conceptually:

```csharp
public interface IKvSequence
{
    long TokenCount { get; }

    void Append(...);

    void Release();

    IReadOnlyList<KvPage> GetPages(...);
}
```

The exact API should be determined after inspecting the current `ISequenceKvCache`.

The key idea is:

```text
Session → KV sequence handle → KV implementation
```

rather than:

```text
Session → concrete PagedKvCache internals
```

---

# 8. Existing `ISequenceKvCache`

Before creating another abstraction, inspect the current:

```text
ISequenceKvCache
```

Determine whether it already represents:

```text
logical sequence ownership
```

or merely:

```text
physical KV storage
```

If it already performs the required logical role:

> **Do not replace it just for architectural purity.**

Refactor/evolve it.

The objective is to avoid:

```text
ISequenceKvCache
IKvSequence
IKvCache
IKvStore
IKvHandle
```

all representing essentially the same thing.

Prefer one clear abstraction.

---

# 9. Missing architecture area #7 — Generation state

Separate generation state from model state.

Generation state may include:

```text
current generated token count
stop conditions
RNG state
sampling history
repetition information
EOS state
```

It must not be stored globally in the model or engine.

The same model must be able to support:

```text
Session A → temperature 0.2
Session B → temperature 0.8
Session C → greedy
```

simultaneously.

---

# 10. Missing architecture area #8 — Generation request vs session state

Distinguish:

### Session state

```text
what has already happened
```

from:

### Generation request

```text
what we want to do now
```

For example:

```text
GenerationOptions
```

should contain:

```text
temperature
top-k
top-p
max tokens
stop conditions
seed/options
```

where appropriate.

Do not permanently mutate session configuration merely because one generation used different sampling settings.

---

# 11. Missing architecture area #9 — Commit boundary

Define exactly when inference state becomes committed.

For normal generation:

```text
forward pass
   ↓
logits
   ↓
sample token
   ↓
KV/token commit
```

The session should only consider the new token committed once the corresponding state is consistent.

This becomes critical later for:

```text
checkpoint
rollback
speculative decoding
```

---

# 12. Missing architecture area #10 — Speculative state

Even before speculative decoding exists, reserve the conceptual distinction:

```text
Committed State
       │
       ▼
Temporary State
```

For example:

```text
Committed:
tokens 0..100

Temporary:
tokens 101..108
```

This allows later features to operate transactionally.

Do not implement speculative decoding yet.

Just make sure the architecture does not make temporary state impossible.

---

# 13. Missing architecture area #11 — Session mutation boundary

Establish:

> A session has one logical writer.

Only one operation may mutate:

```text
tokens
position
KV
generation state
```

at a time.

This prevents subtle races when sessions later become:

```text
forkable
batchable
speculative
```

---

# 14. Missing architecture area #12 — Engine responsibilities

Audit the current inference engine.

Move toward:

```text
Engine
    ↓
execute model operations
```

rather than:

```text
Engine
    ↓
owns every session's state
    ↓
owns every KV cache
    ↓
owns generation state
```

The engine should become an execution mechanism rather than the universal owner of inference state.

---

# 15. Missing architecture area #13 — Scheduler responsibilities

The scheduler should know about:

```text
work
requests
execution opportunities
batching
```

It should not become the permanent owner of:

```text
session tokens
session KV
session identity
```

A scheduler should be able to schedule work belonging to Session A without becoming Session A.

---

# 16. Missing architecture area #14 — Session-independent model execution

Refactor model execution so that it can receive explicit state.

Conceptually:

```text
Execute(
    model,
    sessionState,
    inputTokens)
```

rather than relying on hidden engine fields.

The actual API can be internal.

The important requirement is:

> Model execution should not depend on a single global mutable context.

---

# 17. Missing architecture area #15 — Position handling

Centralise position management.

Avoid having:

```text
session position
KV position
engine position
attention position
scheduler position
```

all independently tracking progress.

There should be one authoritative logical position.

The KV subsystem should be able to validate against it.

---

# 18. Missing architecture area #16 — Context length

Define explicitly where context limits live.

For example:

```text
Model
    MaxContextLength

Session
    CurrentTokenCount
```

The session validates:

```text
CurrentTokenCount + requestedTokens
    <=
Model.MaxContextLength
```

unless sliding-window/context eviction is explicitly supported.

---

# 19. Missing architecture area #17 — Error boundaries

Define which failures belong to which layer.

### Model failure

```text
invalid model
unsupported architecture
```

### KV failure

```text
allocation failure
corrupt page
```

### Session failure

```text
invalid lifecycle operation
state inconsistency
```

### Generation failure

```text
cancelled
sampling failure
```

This will make recovery and rollback substantially easier.

---

# 20. Missing architecture area #18 — Disposal boundaries

Define:

```text
Model.Dispose()
Session.Dispose()
KV.Dispose()
Runtime.Dispose()
```

and their relationships.

The key rule:

> Disposing one session must never destroy shared model resources or another session's KV.

Similarly:

> Disposing a model/runtime must not occur while active sessions still depend on it unless explicitly coordinated.

---

# 21. Missing architecture area #19 — Session manager

Native Sessions require a central registry.

Conceptually:

```text
SessionManager
    │
    ├── Session A
    ├── Session B
    └── Session C
```

The manager should eventually own:

```text
lookup
creation
destruction
suspension
resumption
eviction
```

Do not put registry responsibilities into the model.

---

# 22. Missing architecture area #20 — KV allocator ownership

Paged KV will need a physical memory allocator.

The ownership should become:

```text
Runtime
   │
   ▼
KV Page Pool
   │
   ├── Page 1
   ├── Page 2
   ├── Page 3
   └── ...
```

Sessions request logical KV capacity.

They should not directly allocate raw pages.

---

# 23. Missing architecture area #21 — Memory accounting

Introduce explicit accounting concepts.

At minimum:

```text
KV bytes
KV pages
tokens
sessions
```

Eventually:

```text
resident KV
evicted KV
shared pages
private pages
```

This is required for:

```text
KV eviction
session management
performance diagnostics
```

---

# 24. Missing architecture area #22 — Observability

Define session/runtime metrics before adding lots of optimisation.

Useful measurements:

```text
prompt processing time
generation time
tokens/sec
KV bytes
KV pages
session count
fork count
checkpoint count
cache hits
cache misses
```

This will allow future architectural decisions to be measured rather than guessed.

---

# 25. Missing architecture area #23 — Compatibility layer

Do not immediately break existing Stingray APIs.

Instead:

```text
Existing API
     │
     ▼
Temporary Native Session
     │
     ▼
New session-based engine
```

For example:

```text
Generate(prompt)
```

can internally:

```text
Create session
Append prompt
Generate
Dispose session
```

This allows migration without maintaining two inference engines.

---

# 26. Transition principle

The transition must be incremental.

Do not attempt:

```text
Current Engine
      ↓
massive rewrite
      ↓
perfect architecture
```

Instead:

```text
Current Engine
      ↓
introduce explicit state
      ↓
move ownership
      ↓
introduce session
      ↓
abstract KV
      ↓
paged KV
```

Each step must remain executable and testable.

---

# 27. Transition phase 0 — Freeze behaviour

Before architectural changes:

- run all existing tests
- establish baseline performance
- establish baseline memory usage
- record current generation outputs
- record current prompt-processing behaviour

Create a baseline report.

Do not optimise yet.

---

# 28. Transition phase 1 — Map current state

Create a document:

```text
CURRENT_INFERENCE_ARCHITECTURE.md
```

Document:

```text
Model
Engine
ContinuousBatchingEngine
ISequenceKvCache
KV storage
sampling
token state
scheduler
memory
```

For every mutable field, identify:

```text
who creates it
who mutates it
who disposes it
```

This is one of the most valuable pieces of work in the entire transition.

---

# 29. Transition phase 2 — Classify fields

For every mutable field in the inference engine, classify:

```text
MODEL
RUNTIME
SESSION
REQUEST
TEMPORARY
CACHE
```

Example:

```text
_fieldA → SESSION
_fieldB → MODEL
_fieldC → RUNTIME
_fieldD → REQUEST
```

Do not proceed until every important mutable field has an owner.

---

# 30. Transition phase 3 — Extract session state

Create an internal session-state object.

Move session-specific fields into it.

Initially:

```text
Engine
 └── SessionState
```

is acceptable.

The purpose is to centralise the state before changing the public API.

---

# 31. Transition phase 4 — Make execution explicit

Change internal execution paths from:

```text
Engine uses hidden fields
```

to:

```text
Engine receives explicit SessionState
```

This is a key transition.

Example:

```csharp
Execute(SessionState state, ReadOnlySpan<int> tokens)
```

The exact API should follow the current architecture.

---

# 32. Transition phase 5 — Introduce Native Session

Once state is explicit:

```text
SessionState
```

can become:

```text
InferenceSession
```

Expose the public session API.

Existing APIs can become wrappers.

---

# 33. Transition phase 6 — Separate logical KV

Make the session hold:

```text
IKvSequence / existing equivalent
```

rather than concrete KV implementation details.

At this point:

```text
Session
   ↓
logical KV
   ↓
existing KV implementation
```

must work.

Do not introduce paged storage yet.

---

# 34. Transition phase 7 — Stabilise the seam

At this point stop and test.

The architecture should now be:

```text
Model
Runtime
Session
Logical KV
Existing physical KV
```

Run:

- correctness tests
- multi-session tests
- cancellation tests
- memory tests
- performance tests

Only proceed when this boundary is stable.

---

# 35. Transition phase 8 — Introduce paged KV

Now replace the physical implementation:

```text
Existing KV
      ↓
Paged KV
```

without changing:

```text
Session API
Logical KV API
```

This is the payoff for doing the previous work.

The session should not know that the KV implementation changed.

---

# 36. Transition phase 9 — Add page sharing

Once paged KV works:

```text
Session A
    ↓
shared pages
    ↓
Session B
```

becomes possible.

Implement:

```text
reference counting
copy-on-write
page ownership
```

This enables efficient session fork.

---

# 37. Transition phase 10 — Add checkpoints

Use the new KV abstraction to implement:

```text
checkpoint
rollback
```

The checkpoint should reference logical/session state rather than knowing about physical page addresses.

---

# 38. Transition phase 11 — Add speculative decoding

Only after:

```text
Native Sessions
+
stable logical KV
+
Paged KV
+
checkpoint/rollback
```

should speculative decoding be implemented.

At that point its fundamental operations become:

```text
checkpoint
   ↓
draft
   ↓
verify
   ↓
commit/rollback
```

rather than requiring a new state architecture.

---

# 39. Target architecture after transition

The final intended structure is:

```text
                    Runtime
                       │
            ┌──────────┴──────────┐
            │                     │
          Model              SessionManager
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
                Session A     Session B     Session C
                    │             │             │
                    ▼             ▼             ▼
               KV Sequence   KV Sequence   KV Sequence
                    │             │             │
                    └─────────────┼─────────────┘
                                  ▼
                             KV Page Pool
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
                   CPU          CUDA          Vulkan
```

---

# 40. Dependency direction

The dependency direction should be:

```text
Session
   ↓
KV abstraction
   ↓
KV implementation
```

and:

```text
Session
   ↓
Model execution abstraction
   ↓
Model/backend
```

Avoid:

```text
KV implementation
   ↓
Session
```

or:

```text
Model
   ↓
Session registry
```

The lower-level components must not depend on higher-level session management.

---

# 41. Important architectural rule

Do not let the session become a giant "god object".

It should coordinate:

```text
token state
KV state
generation state
```

but delegate:

```text
model execution → model/runtime
sampling → sampler
KV allocation → KV subsystem
scheduling → scheduler
persistence → session store
```

The session is an orchestrator of its own state, not the entire engine.

---

# 42. What should remain outside the session

Do not put these inside the session:

```text
Model weights
Global thread pools
KV page allocator
Global scheduler
Model loading
Hardware discovery
Tokenizer implementation
Global memory manager
```

The session references these services.

It does not own them.

---

# 43. Testing architecture

Create tests that explicitly verify ownership.

### Model sharing

```text
2 sessions
1 model
```

### KV isolation

```text
2 sessions
different prompts
no state contamination
```

### Session lifecycle

```text
create
generate
cancel
resume
dispose
```

### Session migration

Verify existing APIs produce identical results after the transition.

---

# 44. Performance gate

Every transition phase must compare against the baseline.

Track:

```text
prompt tokens/sec
generation tokens/sec
memory
allocation rate
GC activity
KV memory
startup time
```

A refactoring that makes architecture cleaner but causes a significant hot-path regression must be investigated before continuing.

---

# 45. Hot-path rule

Do not introduce abstractions that cause unnecessary per-token overhead.

The following must remain cheap:

```text
token append
KV lookup
position lookup
sampling
generation loop
```

Interfaces are acceptable where they enable architecture, but avoid:

```text
allocations per token
virtual dispatch chains per tensor operation
LINQ in hot loops
boxing
unnecessary Span → array conversions
```

The architecture must preserve Stingray's performance goals.

---

# 46. Avoid premature GPU abstractions

The abstraction should be capable of supporting:

```text
CPU
CUDA
Vulkan
```

but the transition should initially target the current CPU implementation.

Do not create three partially implemented backends merely to prove extensibility.

The contract matters more than the number of implementations.

---

# 47. Avoid premature persistence

Native Sessions should work entirely in memory first.

Then:

```text
checkpoint
fork
suspend
```

can be validated.

Only after those semantics are stable should durable session storage be added.

---

# 48. Avoid premature speculative decoding

Speculative decoding should consume the architecture.

It should not dictate the architecture.

First establish:

```text
session
KV
checkpoint
rollback
```

Then implement speculation.

---

# 49. Documentation deliverables

Create:

```text
docs/
    inference-state-architecture.md
    native-sessions.md
    kv-cache-architecture.md
```

The first document should explain:

```text
Model
Runtime
Session
KV
Scheduler
Generation
```

and their ownership boundaries.

This will also make the repository substantially easier for other developers and AI coding agents to understand.

---

# 50. Recommended code structure

Do not blindly create this exact structure; adapt it to the existing repository.

A useful conceptual organisation is:

```text
Stingray/
│
├── Models/
│
├── Runtime/
│
├── Sessions/
│   ├── IInferenceSession
│   ├── InferenceSession
│   ├── SessionId
│   ├── SessionState
│   └── SessionManager
│
├── Generation/
│
├── KvCache/
│   ├── IKvCache
│   ├── IKvSequence
│   ├── ...
│
├── Scheduling/
│
└── Persistence/
```

Do not reorganise namespaces/files merely for aesthetics.

Follow existing project conventions.

---

# 51. Definition of transition complete

The transition architecture is complete when:

- [ ] Every major mutable state field has an explicit owner.
- [ ] Model state is separated from session state.
- [ ] Runtime state is separated from session state.
- [ ] Generation-request state is separated from persistent session state.
- [ ] Session state is represented explicitly.
- [ ] Model weights are shareable across sessions.
- [ ] Session KV is logically owned by the session.
- [ ] Physical KV allocation is owned by the KV subsystem.
- [ ] Session code does not depend on concrete KV page storage.
- [ ] Model execution does not depend on hidden global session state.
- [ ] Position tracking has a single authoritative source.
- [ ] Session mutation is single-writer.
- [ ] Existing APIs can operate through temporary sessions.
- [ ] Existing correctness tests remain green.
- [ ] Performance has been measured against baseline.
- [ ] NativeAOT still works.
- [ ] The logical KV seam is stable.
- [ ] Paged KV can be introduced without redesigning the Session API.

---

# 52. The most important outcome

The objective is **not** to produce more interfaces.

The objective is to establish this invariant:

```text
                 MODEL
                   │
        immutable/shared state
                   │
                   ▼
                RUNTIME
                   │
          execution resources
                   │
                   ▼
                SESSION
                   │
          mutable inference state
                   │
                   ▼
             LOGICAL KV
                   │
                   ▼
            PHYSICAL KV
                   │
                   ▼
             MEMORY PAGES
```

Once that is true, the future architecture becomes dramatically easier.

---

# 53. Final implementation sequence

The coding agent should follow this sequence:

```text
STEP 1
Audit existing architecture
        ↓
STEP 2
Document ownership of all mutable state
        ↓
STEP 3
Classify Model / Runtime / Session / Request state
        ↓
STEP 4
Extract explicit internal SessionState
        ↓
STEP 5
Make inference execution consume explicit state
        ↓
STEP 6
Introduce Native Session API
        ↓
STEP 7
Route existing APIs through temporary sessions
        ↓
STEP 8
Separate logical KV from physical KV
        ↓
STEP 9
Stabilise and benchmark
        ↓
STEP 10
Proceed to Paged KV implementation
```

**Do not skip STEP 2.**

The most dangerous thing at this point is asking an AI to "refactor the architecture" without first forcing it to identify where the current state actually lives.

---

# 54. Strategic result

After this transition, Stingray should no longer fundamentally think:

```text
"I am an engine that generates text."
```

It should think:

```text
"I am a runtime capable of executing many independent
inference states against shared model resources."
```

That distinction is the foundation for:

```text
Native Sessions
       ↓
Paged KV
       ↓
Forking
       ↓
Checkpoints
       ↓
Batching
       ↓
Speculative Decoding
       ↓
KV Quantisation
       ↓
Eviction
       ↓
Prefix Sharing
```

The architectural work here is therefore **not overhead**. It is the transition that prevents each of those later capabilities from becoming another special case inside the inference engine.