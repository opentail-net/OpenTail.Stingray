namespace OpenTail.Stingray.Core.Capabilities;

/// <summary>
/// Machine-checkable identifier for runtime capabilities supported by OpenTail.Stingray.
/// Describes truthfulness of currently instantiated model, runtime, and execution components.
/// </summary>
public enum InferenceCapability
{
    // State Capabilities
    PagedKvCache,
    KvForking,
    KvCopyOnWrite,
    CheckpointRollback,
    SessionFork,
    SuspendResume,
    SnapshotRestore,
    PrefixCaching,

    // Execution Capabilities
    ContinuousBatching,
    ChunkedPrefill,
    PackedPrefill,
    Cancellation,

    // Speculative Capabilities
    SpeculativeDecoding,
    SpeculativeSampling,
    PromptLookupSpeculation,

    // Generation Capabilities
    Streaming,
    GrammarConstraints,
    StructuredGeneration,
    ToolArgumentConstraints,

    // Model Capabilities
    EmbeddingInput,
    Mtp,
    Gdn,

    // Device Capabilities
    Cpu,
    Cuda,
    GpuArgmax
}
