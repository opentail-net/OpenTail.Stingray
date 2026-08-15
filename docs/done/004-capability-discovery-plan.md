> **ARCHIVED, 2026-08-15.** Implemented as designed (confirmed against source). No open
> remainder tracked separately from [00-current-work.md](../00-current-work.md).

---

# OpenTail.Stingray — Capability Discovery & Unified Inference API Plan

## 1. Objective

Create a unified, authoritative capability-discovery API for OpenTail.Stingray.

The purpose is to allow callers such as OpenTail to ask:

> "What can this particular Stingray runtime/model actually do?"

without having to inspect individual interfaces, model metadata, forward-pass implementations, configuration objects, or make assumptions based on architecture names.

This is **not** a plan to implement new inference functionality.

The current codebase already contains substantial functionality including:

- paged KV cache;
- session management;
- fork/CoW;
- checkpoint/rollback;
- snapshot/replay;
- suspend/resume;
- prefix caching;
- continuous batching;
- chunked prefill;
- speculative decoding;
- speculative sampling;
- prompt-lookup speculation;
- structured/grammar constraints;
- embedding input;
- MTP/GDN/hybrid-related capabilities;
- CUDA-related capabilities;
- model-specific execution support.

This plan should **surface those existing capabilities cleanly and consistently**.

Do not redesign or reimplement those systems.

---

# 2. Desired outcome

At the end of this plan, OpenTail should be able to do something conceptually like:

```csharp
var capabilities = runtime.GetCapabilities();

if (capabilities.SpeculativeDecoding.IsSupported)
{
    // Enable speculative generation UI/configuration.
}

if (capabilities.StructuredGeneration.IsSupported)
{
    // Offer JSON/grammar constrained generation.
}

if (capabilities.EmbeddingInput.IsSupported)
{
    // Allow multimodal/embedding input.
}
```

And, importantly:

```csharp
var capabilities = runtime.GetCapabilities();

Console.WriteLine(capabilities.Architecture);
Console.WriteLine(capabilities.ContextLength);
Console.WriteLine(capabilities.KvCache);
Console.WriteLine(capabilities.SpeculativeDecoding);
```

The caller should not need to know which internal implementation provides the capability.

---

# 3. Design principles

## 3.1 One authoritative capability surface

Avoid having OpenTail determine capabilities by checking five different interfaces.

Bad:

```csharp
if (forwardPass.SupportsEmbeddingInput &&
    runtime is IContinuousBatchingRuntime &&
    modelConfig.HasMtp &&
    cache is IPagedKvCache)
{
    ...
}
```

Good:

```csharp
var capabilities = runtime.GetCapabilities();

if (capabilities.Supports(Capability.StructuredGeneration))
{
    ...
}
```

The capability system should be the authoritative answer.

---

## 3.2 Capabilities describe reality, not aspirations

Do not report a capability simply because the architecture theoretically supports it.

For example:

```text
Llama architecture
        ↓
theoretically supports X
```

does not mean:

```text
this loaded model/runtime
        ↓
X = supported
```

Capabilities must describe what the **currently instantiated runtime/model combination** can actually execute.

---

## 3.3 No duplicate feature implementation

This project should not implement:

- a second speculative decoder;
- a second KV cache;
- a second batching system;
- a second grammar engine;
- a second embedding path.

The capability API wraps/discovers existing functionality.

---

## 3.4 Capability information should be immutable

Once a runtime/model is loaded, capability information should preferably be immutable.

Conceptually:

```csharp
public sealed record InferenceCapabilities(...)
{
}
```

rather than a mutable bag of flags.

This prevents confusing situations where callers cache a capability object and some unrelated component later changes its contents.

---

# 4. Proposed API

Introduce a central capability model.

A possible starting point:

```csharp
public sealed record InferenceCapabilities
{
    public required ModelCapabilities Model { get; init; }

    public required ExecutionCapabilities Execution { get; init; }

    public required StateCapabilities State { get; init; }

    public required GenerationCapabilities Generation { get; init; }

    public required DeviceCapabilities Device { get; init; }
}
```

The exact naming can be adapted to the existing Stingray conventions.

Do not blindly copy this structure if the repository already has a better-established terminology.

---

# 5. Model capabilities

Model capabilities describe what the loaded model itself represents.

Suggested:

```csharp
public sealed record ModelCapabilities
{
    public required string Architecture { get; init; }

    public required int ContextLength { get; init; }

    public bool IsDecoderOnly { get; init; }

    public bool IsEncoderDecoder { get; init; }

    public bool SupportsEmbeddingInput { get; init; }

    public bool SupportsMtp { get; init; }

    public bool SupportsGdn { get; init; }

    public bool SupportsMoE { get; init; }

    public int? ExpertCount { get; init; }

    public int? ActiveExpertCount { get; init; }
}
```

Do not add fields simply because they sound useful.

Only expose information that can be reliably determined from the current model/runtime.

---

# 6. Execution capabilities

This describes how the runtime can execute the model.

Suggested:

```csharp
public sealed record ExecutionCapabilities
{
    public bool SupportsContinuousBatching { get; init; }

    public bool SupportsChunkedPrefill { get; init; }

    public bool SupportsPackedPrefill { get; init; }

    public bool SupportsSpeculativeDecoding { get; init; }

    public bool SupportsPromptLookupSpeculation { get; init; }

    public bool SupportsCancellation { get; init; }
}
```

Again, these should be populated from actual implementation support.

For example, if speculative decoding exists but the currently loaded model cannot participate in it, report:

```csharp
SupportsSpeculativeDecoding = false
```

rather than merely checking whether `SpeculativeDecoder` exists in the assembly.

---

# 7. State capabilities

This is particularly important because Stingray now has a substantial inference-state architecture.

Suggested:

```csharp
public sealed record StateCapabilities
{
    public bool SupportsPagedKvCache { get; init; }

    public bool SupportsKvForking { get; init; }

    public bool SupportsKvCopyOnWrite { get; init; }

    public bool SupportsCheckpointRollback { get; init; }

    public bool SupportsSessionFork { get; init; }

    public bool SupportsSuspendResume { get; init; }

    public bool SupportsSnapshotRestore { get; init; }

    public bool SupportsPrefixCaching { get; init; }
}
```

Do not confuse these with model capabilities.

For example:

```text
Model:
    Qwen

Runtime:
    Stingray

State:
    paged KV + CoW + prefix cache
```

The capability API should describe all three layers cleanly.

---

# 8. Generation capabilities

These describe what the generation pipeline supports.

Suggested:

```csharp
public sealed record GenerationCapabilities
{
    public bool SupportsSampling { get; init; }

    public bool SupportsGreedyDecoding { get; init; }

    public bool SupportsSpeculativeSampling { get; init; }

    public bool SupportsGrammarConstraints { get; init; }

    public bool SupportsStructuredGeneration { get; init; }

    public bool SupportsToolArgumentConstraints { get; init; }

    public bool SupportsStreaming { get; init; }

    public bool SupportsReasoningStreamHandling { get; init; }
}
```

Be careful about terminology.

If the repository already has a canonical term such as `Grammar`, `Constraint`, `StructuredOutput`, or `ToolCall`, use that terminology instead of creating competing names.

---

# 9. Device capabilities

The API should also expose relevant execution-device information.

For example:

```csharp
public sealed record DeviceCapabilities
{
    public bool SupportsCpu { get; init; }

    public bool SupportsCuda { get; init; }

    public bool SupportsGpuArgmax { get; init; }

    public string? Backend { get; init; }
}
```

Do not turn this into a complete hardware-information API.

The purpose is capability discovery, not hardware telemetry.

---

# 10. Capability enum

In addition to strongly typed capability groups, provide a simple machine-checkable capability identifier.

For example:

```csharp
public enum InferenceCapability
{
    PagedKvCache,
    KvForking,
    KvCopyOnWrite,
    CheckpointRollback,
    SessionFork,
    SuspendResume,
    SnapshotRestore,
    PrefixCaching,

    ContinuousBatching,
    ChunkedPrefill,
    PackedPrefill,
    Cancellation,

    SpeculativeDecoding,
    SpeculativeSampling,
    PromptLookupSpeculation,

    Streaming,
    GrammarConstraints,
    StructuredGeneration,
    ToolArgumentConstraints,
    ReasoningStreams,

    EmbeddingInput,
    Mtp,
    Gdn,

    Cpu,
    Cuda,
    GpuArgmax
}
```

The exact list should be derived from the actual repository.

**Do not invent capabilities which aren't currently supported merely to fill out the enum.**

---

# 11. `Supports()` convenience API

The top-level capability object should provide:

```csharp
public bool Supports(InferenceCapability capability)
```

Example:

```csharp
if (capabilities.Supports(InferenceCapability.PagedKvCache))
{
    ...
}
```

This gives OpenTail a simple generic mechanism while retaining strongly typed properties for important information.

---

# 12. Where capability discovery should live

Do not make callers assemble capabilities themselves.

There should be one authoritative factory/builder.

For example:

```csharp
public interface IInferenceCapabilityProvider
{
    InferenceCapabilities GetCapabilities();
}
```

or, if the existing runtime architecture has a more appropriate home:

```csharp
public interface IInferenceRuntime
{
    InferenceCapabilities Capabilities { get; }
}
```

Prefer the existing Stingray architecture if it already has an appropriate runtime abstraction.

**Do not introduce a second top-level runtime abstraction simply to host this feature.**

---

# 13. Capability composition

Some capabilities come from different layers.

For example:

```text
Model
 └── supports MTP

Forward implementation
 └── implements MTP execution

Runtime
 └── exposes MTP capability
```

The final capability should only be `true` if all required layers support it.

Conceptually:

```csharp
SupportsMtp =
    model.SupportsMtp &&
    forwardPass.SupportsMtp;
```

Likewise:

```csharp
SupportsSpeculativeDecoding =
    targetModelSupportsSpeculation &&
    draftModelSupportsSpeculation &&
    runtimeSupportsSpeculation;
```

The API should therefore represent **effective capability**, not merely individual component capability.

---

# 14. Capability reasons / diagnostics

Where practical, support an explanation for unavailable capabilities.

This is extremely useful for OpenTail.

Instead of:

```text
Speculative decoding: false
```

allow something like:

```text
Speculative decoding: unavailable
Reason: loaded model does not expose a compatible draft configuration
```

A possible model:

```csharp
public sealed record CapabilityStatus
{
    public required bool IsSupported { get; init; }

    public string? Reason { get; init; }
}
```

Then:

```csharp
public CapabilityStatus SpeculativeDecoding { get; init; }
```

However, **do not over-engineer this**.

If the current architecture makes a simple boolean API much cleaner, start with booleans and add diagnostics only where there is a real need.

---

# 15. Avoid capability explosion

Do not expose every internal implementation detail.

For example, these should probably remain implementation details:

```text
UsesArrayPool
UsesXxxAllocator
HasFastPath
UsesSpecificKernel
InternalPagePoolImplementation
```

Those aren't user-facing inference capabilities.

The test for inclusion should be:

> "Could OpenTail reasonably make a decision based on this?"

If yes, expose it.

If not, keep it internal.

---

# 16. Model information versus capability

Do not mix these unnecessarily.

For example:

```text
Architecture = Qwen2
ContextLength = 32768
ParameterCount = ...
Quantisation = Q4_K_M
```

are **model metadata**.

Whereas:

```text
SupportsSpeculativeDecoding = true
SupportsPrefixCaching = true
SupportsStructuredGeneration = true
```

are **capabilities**.

The API may expose both, but keep their semantic distinction clear.

---

# 17. Capability negotiation

Add a simple mechanism allowing OpenTail to ask for requirements.

For example:

```csharp
public bool SupportsAll(
    params InferenceCapability[] capabilities)
```

and:

```csharp
public bool SupportsAny(
    params InferenceCapability[] capabilities)
```

Usage:

```csharp
if (capabilities.SupportsAll(
    InferenceCapability.PagedKvCache,
    InferenceCapability.SessionFork,
    InferenceCapability.SpeculativeDecoding))
{
    ...
}
```

This will become particularly useful as OpenTail chooses execution strategies.

---

# 18. Capability-aware execution selection

Do not implement this as a full scheduler.

Just establish the foundation for callers to choose an execution mode.

Example:

```csharp
var caps = runtime.Capabilities;

if (caps.Supports(InferenceCapability.SpeculativeDecoding))
{
    // choose speculative path
}
else
{
    // standard generation
}
```

The actual policy belongs above Stingray.

Stingray reports what it can do.

OpenTail decides what it wants to do.

---

# 19. Testing requirements

This feature needs a strong test suite, but the tests should primarily verify **truthfulness**.

## Test 1 — Known runtime capability set

Construct a representative runtime and verify:

```text
Paged KV             = true
Session fork         = true
CoW                  = true
Prefix cache         = true
Speculative decode   = true
Structured output    = true
```

using the actual implementation.

---

## Test 2 — Unsupported capability

Construct a runtime/model combination which genuinely lacks a capability.

Verify:

```csharp
Supports(X) == false
```

Do not fake unsupported capabilities just to test the boolean.

---

## Test 3 — Capability consistency

If:

```csharp
Supports(InferenceCapability.PagedKvCache)
```

is true, then the corresponding typed property must also be true.

For example:

```csharp
capabilities.State.SupportsPagedKvCache
```

must agree with:

```csharp
capabilities.Supports(
    InferenceCapability.PagedKvCache)
```

---

## Test 4 — Effective capability composition

If a model supports MTP but the active execution path does not:

```text
Model MTP       = true
Execution MTP   = false
Effective MTP   = false
```

This is an important test.

---

## Test 5 — Capability immutability

Verify that retrieving capabilities doesn't expose mutable internal state.

---

## Test 6 — No capability falsely advertised

For every capability exposed by the current runtime, have at least one test demonstrating that the corresponding operation actually exists and is usable.

This is the most important philosophy of the suite:

> **A capability declaration is a promise.**

---

# 20. Integration tests

Add one or more tests which simulate what OpenTail will actually do.

For example:

```csharp
var runtime = CreateRuntime(...);
var caps = runtime.Capabilities;

Assert.True(caps.Supports(
    InferenceCapability.PagedKvCache));

Assert.True(caps.Supports(
    InferenceCapability.SpeculativeDecoding));

Assert.True(caps.Supports(
    InferenceCapability.StructuredGeneration));
```

Then use the capabilities to select an execution path.

This verifies that the capability API isn't merely decorative.

---

# 21. Documentation

Add a concise capability reference.

Something similar to:

```text
docs/
    capabilities.md
```

Document:

1. What capability discovery means.
2. How OpenTail should consume it.
3. The difference between model metadata and runtime capabilities.
4. The current capability list.
5. What happens when a capability is unavailable.
6. Examples.

Keep this concise.

---

# 22. Backward compatibility

Do not unnecessarily break existing public APIs.

The capability API should be additive.

Existing:

```csharp
forwardPass.SupportsEmbeddingInput
```

may remain temporarily if it is part of an established contract.

The new capability layer should become the preferred external discovery mechanism.

If a capability is currently exposed through multiple APIs, avoid creating conflicting sources of truth.

Where possible:

```text
Existing implementation
        ↓
Capability provider
        ↓
Unified public capability API
```

rather than:

```text
Existing implementation ───────┐
                               ├── conflicting answers
New capability implementation ─┘
```

---

# 23. Do NOT do these things in this plan

This is important.

Do **not** use this plan as an excuse to implement:

- MoE;
- LoRA;
- multimodal inference;
- encoder-decoder models;
- Mamba/SSM;
- new quantisation formats;
- performance optimisation;
- kernel changes;
- scheduler redesign;
- new KV cache functionality;
- new speculative decoding algorithms;
- distributed inference.

Those belong to future plans.

The purpose of this plan is to **describe and expose what Stingray already supports**.

---

# 24. Suggested implementation sequence

## M1 — Inventory

Inspect the current codebase and produce an authoritative list of currently supported capabilities.

Do not start coding until this inventory is complete.

Categorise each capability:

```text
Model
Execution
State
Generation
Device
```

Mark each as:

```text
Implemented
Partially implemented
Unsupported
Unknown
```

Only `Implemented` capabilities should initially be exposed as `true`.

---

## M2 — Capability contracts

Introduce the smallest appropriate public types:

```csharp
InferenceCapabilities
ModelCapabilities
ExecutionCapabilities
StateCapabilities
GenerationCapabilities
DeviceCapabilities
```

Adjust names to match existing repository conventions.

---

## M3 — Capability provider

Implement a single provider/factory using the existing runtime/model/forward-pass information.

Do not duplicate implementation logic.

---

## M4 — Generic `Supports()`

Add:

```csharp
Supports(InferenceCapability capability)
```

plus `SupportsAll()` / `SupportsAny()` if they fit naturally.

---

## M5 — Wire into runtime

Expose capabilities from the appropriate top-level Stingray runtime/model execution object.

Prefer:

```csharp
runtime.Capabilities
```

over requiring callers to construct the capability provider themselves.

---

## M6 — Tests

Implement the capability correctness tests described above.

Prioritise:

1. effective capability correctness;
2. unsupported capability correctness;
3. consistency between typed properties and enum lookup;
4. real-operation verification.

---

## M7 — Documentation

Document the public API and current capability matrix.

---

# 25. Acceptance criteria

This plan is complete when all of the following are true:

### API

- [ ] Stingray exposes one coherent capability surface.
- [ ] OpenTail can query capabilities without inspecting internal implementations.
- [ ] Capabilities are immutable.
- [ ] Generic capability lookup exists.
- [ ] Capability groups are logically separated.

### Accuracy

- [ ] Capabilities reflect the actual loaded runtime/model.
- [ ] Unsupported capabilities are not advertised.
- [ ] Composite capabilities require all necessary underlying support.
- [ ] There is one authoritative source of truth.

### Existing functionality

The capability system accurately exposes, where applicable:

- [ ] Paged KV
- [ ] KV CoW
- [ ] KV/session forking
- [ ] checkpoint/rollback
- [ ] suspend/resume
- [ ] snapshot/restore
- [ ] prefix caching
- [ ] continuous batching
- [ ] chunked prefill
- [ ] packed prefill
- [ ] cancellation
- [ ] speculative decoding
- [ ] speculative sampling
- [ ] prompt lookup speculation
- [ ] grammar constraints
- [ ] structured generation
- [ ] tool argument constraints
- [ ] streaming
- [ ] reasoning stream handling
- [ ] embedding input
- [ ] MTP
- [ ] GDN/hybrid capabilities where actually supported
- [ ] CUDA capabilities where actually supported
- [ ] GPU argmax where actually supported

The exact list must be verified against the current repository rather than assumed from this plan.

### Tests

- [ ] Capability discovery tests pass.
- [ ] Unsupported capability tests pass.
- [ ] Capability composition tests pass.
- [ ] Integration tests pass.
- [ ] Existing Stingray tests remain green.

### Documentation

- [ ] Capability API documented.
- [ ] Current capability matrix documented.
- [ ] OpenTail integration example documented.

---

# 26. Definition of "done"

This plan is intentionally complete when **capability discovery is reliable**, not when every imaginable capability has been added.

The final architectural relationship should look approximately like:

```text
                   OpenTail
                       │
                       │ asks:
                       │ "What can you do?"
                       ▼
              ┌──────────────────┐
              │ Stingray Runtime │
              └────────┬─────────┘
                       │
                       ▼
              InferenceCapabilities
                       │
       ┌───────────────┼────────────────┐
       │               │                │
       ▼               ▼                ▼
     Model          Execution         State
       │               │                │
       │               │                │
       └───────────────┼────────────────┘
                       │
                       ▼
                  Generation
                       │
                       ▼
                   Device
```

OpenTail should be able to make decisions from this API without knowing how Stingray implements those capabilities.

---

# 27. Important final instruction to the implementing AI

**Do not over-engineer this.**

The repository already contains the majority of the underlying functionality.

This plan is primarily a **consolidation and exposure task**.

Before making changes:

1. inspect the existing public interfaces;
2. inspect existing capability flags/properties;
3. identify duplicated capability information;
4. identify the appropriate top-level runtime object;
5. produce the capability inventory;
6. only then implement the smallest coherent API.

If an existing API already provides the correct abstraction, **reuse it rather than creating another abstraction**.

If a proposed capability cannot currently be established reliably, report it as unsupported/unknown rather than pretending it exists.

The goal is a clean, truthful capability surface — **not another large subsystem.**