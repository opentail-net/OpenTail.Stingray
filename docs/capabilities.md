# OpenTail.Stingray — Capability Discovery API Reference

## Overview

OpenTail.Stingray provides a unified, authoritative capability-discovery API via `IInferenceRuntime.Capabilities`.

Callers (such as OpenTail) query the capability surface to inspect what the currently loaded runtime and model combination can actually execute without making assumptions based on component names or reflection.

```csharp
var capabilities = runtime.Capabilities;

if (capabilities.Supports(InferenceCapability.SpeculativeDecoding))
{
    // Enable speculative decoding execution strategy
}

if (capabilities.SupportsAll(InferenceCapability.PagedKvCache, InferenceCapability.SessionFork))
{
    // Enable agent sub-session branching
}
```

---

## Capability Structure

The `InferenceCapabilities` object contains 5 strongly typed, immutable capability groups:

1. **`ModelCapabilities` (`Model`)**:
   - `Architecture` (string): Model architecture name (e.g. `"Qwen2"`, `"Llama"`).
   - `ContextLength` (int): Maximum sequence length supported.
   - `IsDecoderOnly` / `IsEncoderDecoder` (bool): Model architecture category.
   - `SupportsEmbeddingInput` (bool): Multimodal/embedding tensor input support.
   - `SupportsMtp` (bool): Multi-Token Prediction kernel support.
   - `SupportsMoE` (bool): Mixture-of-Experts routing support.

2. **`ExecutionCapabilities` (`Execution`)**:
   - `SupportsContinuousBatching` (bool): Continuous batching runtime support.
   - `SupportsChunkedPrefill` (bool): Chunked prefill support for long prompt inputs.
   - `SupportsPackedPrefill` (bool): Multi-sequence packed prefill support.
   - `SupportsSpeculativeDecoding` (bool): Target/draft speculative decoding support.
   - `SupportsPromptLookupSpeculation` (bool): Prompt lookup speculation support.
   - `SupportsCancellation` (bool): Cooperative cancellation token support.

3. **`StateCapabilities` (`State`)**:
   - `SupportsPagedKvCache` (bool): Zero-copy paged physical page allocation.
   - `SupportsKvForking` / `SupportsKvCopyOnWrite` (bool): Page table forking and Copy-on-Write page duplication.
   - `SupportsCheckpointRollback` (bool): Rich session checkpointing and token rollback.
   - `SupportsSessionFork` (bool): Isolated single-writer child session spawning.
   - `SupportsSuspendResume` (bool): Memory-evicting suspend and pre-fill resume.
   - `SupportsSnapshotRestore` (bool): Token-history persistence and replay restoration.
   - `SupportsPrefixCaching` (bool): Radix/prefix KV cache page sharing.

4. **`GenerationCapabilities` (`Generation`)**:
   - `SupportsSampling` / `SupportsGreedyDecoding` (bool): Temperature/Top-P and greedy decoding.
   - `SupportsSpeculativeSampling` (bool): Leviathan/Chen distribution-preserving sampling.
   - `SupportsGrammarConstraints` / `SupportsStructuredGeneration` (bool): GBNF grammar and JSON schema generation.
   - `SupportsToolArgumentConstraints` (bool): Tool parameter schema validation.
   - `SupportsStreaming` (bool): Token chunk streaming.

5. **`DeviceCapabilities` (`Device`)**:
   - `SupportsCpu` (bool): CPU SIMD/AVX execution backend.
   - `SupportsCuda` (bool): CUDA GPU hardware acceleration backend.
   - `SupportsGpuArgmax` (bool): On-device GPU argmax kernel dispatch.
   - `Backend` (string): Primary active execution backend identifier (e.g. `"CPU"`, `"CUDA"`).

---

## Machine-Checkable Enum & Query API

In addition to strongly typed group properties, `InferenceCapabilities` provides generic lookup methods:

```csharp
public bool Supports(InferenceCapability capability)
public bool SupportsAll(params InferenceCapability[] capabilities)
public bool SupportsAny(params InferenceCapability[] capabilities)
```

### Usage Example:
```csharp
var caps = runtime.Capabilities;

if (caps.SupportsAll(InferenceCapability.PagedKvCache, InferenceCapability.SessionFork, InferenceCapability.CheckpointRollback))
{
    // Execute tree-search agent exploration
}
```
