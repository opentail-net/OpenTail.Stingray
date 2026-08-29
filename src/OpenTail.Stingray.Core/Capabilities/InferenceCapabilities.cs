
namespace OpenTail.Stingray.Core.Capabilities;

/// <summary>
/// Describes model-level representation capabilities.
/// </summary>
public sealed record ModelCapabilities
{
    public required string Architecture { get; init; }
    public required int ContextLength { get; init; }
    public bool IsDecoderOnly { get; init; } = true;
    public bool IsEncoderDecoder { get; init; }
    public bool SupportsEmbeddingInput { get; init; }
    public bool SupportsToolCalling { get; init; }
    public bool SupportsVision { get; init; }
    public bool SupportsStructuredOutput { get; init; } = true;
    public bool SupportsMtp { get; init; }
    public bool SupportsGdn { get; init; }
    public bool SupportsMoE { get; init; }
    public int? ExpertCount { get; init; }
    public int? ActiveExpertCount { get; init; }
}

/// <summary>
/// Describes runtime execution capabilities.
/// </summary>
public sealed record ExecutionCapabilities
{
    public bool SupportsContinuousBatching { get; init; }
    public bool SupportsChunkedPrefill { get; init; }
    public bool SupportsPackedPrefill { get; init; }
    public bool SupportsSpeculativeDecoding { get; init; }
    public bool SupportsPromptLookupSpeculation { get; init; }
    public bool SupportsCancellation { get; init; } = true;
}

/// <summary>
/// Describes inference state, session, and KV cache capabilities.
/// </summary>
public sealed record StateCapabilities
{
    public bool SupportsPagedKvCache { get; init; } = true;
    public bool SupportsKvForking { get; init; } = true;
    public bool SupportsKvCopyOnWrite { get; init; } = true;
    public bool SupportsCheckpointRollback { get; init; } = true;
    public bool SupportsSessionFork { get; init; } = true;
    public bool SupportsSuspendResume { get; init; } = true;
    public bool SupportsSnapshotRestore { get; init; } = true;
    public bool SupportsPrefixCaching { get; init; } = true;
}

/// <summary>
/// Describes text generation and constraint pipeline capabilities.
/// </summary>
public sealed record GenerationCapabilities
{
    public bool SupportsSampling { get; init; } = true;
    public bool SupportsGreedyDecoding { get; init; } = true;
    public bool SupportsSpeculativeSampling { get; init; } = true;
    public bool SupportsGrammarConstraints { get; init; } = true;
    public bool SupportsStructuredGeneration { get; init; } = true;
    public bool SupportsToolArgumentConstraints { get; init; } = true;
    public bool SupportsStreaming { get; init; } = true;
}

/// <summary>
/// Describes execution device and backend capabilities.
/// </summary>
public sealed record DeviceCapabilities
{
    public bool SupportsCpu { get; init; } = true;
    public bool SupportsCuda { get; init; }
    public bool SupportsGpuArgmax { get; init; }
    public string Backend { get; init; } = "CPU";
}

/// <summary>
/// Authoritative, immutable capability surface exposing effective capabilities
/// for a loaded OpenTail.Stingray model/runtime combination.
/// </summary>
public sealed record InferenceCapabilities
{
    public required ModelCapabilities Model { get; init; }
    public required ExecutionCapabilities Execution { get; init; }
    public required StateCapabilities State { get; init; }
    public required GenerationCapabilities Generation { get; init; }
    public required DeviceCapabilities Device { get; init; }

    /// <summary>
    /// Checks whether a specific capability is supported by this runtime/model instance.
    /// </summary>
    public bool Supports(InferenceCapability capability)
    {
        return capability switch
        {
            // State
            InferenceCapability.PagedKvCache => State.SupportsPagedKvCache,
            InferenceCapability.KvForking => State.SupportsKvForking,
            InferenceCapability.KvCopyOnWrite => State.SupportsKvCopyOnWrite,
            InferenceCapability.CheckpointRollback => State.SupportsCheckpointRollback,
            InferenceCapability.SessionFork => State.SupportsSessionFork,
            InferenceCapability.SuspendResume => State.SupportsSuspendResume,
            InferenceCapability.SnapshotRestore => State.SupportsSnapshotRestore,
            InferenceCapability.PrefixCaching => State.SupportsPrefixCaching,

            // Execution
            InferenceCapability.ContinuousBatching => Execution.SupportsContinuousBatching,
            InferenceCapability.ChunkedPrefill => Execution.SupportsChunkedPrefill,
            InferenceCapability.PackedPrefill => Execution.SupportsPackedPrefill,
            InferenceCapability.Cancellation => Execution.SupportsCancellation,

            // Speculative
            InferenceCapability.SpeculativeDecoding => Execution.SupportsSpeculativeDecoding,
            InferenceCapability.SpeculativeSampling => Generation.SupportsSpeculativeSampling && Execution.SupportsSpeculativeDecoding,
            InferenceCapability.PromptLookupSpeculation => Execution.SupportsPromptLookupSpeculation,

            // Generation
            InferenceCapability.Streaming => Generation.SupportsStreaming,
            InferenceCapability.GrammarConstraints => Generation.SupportsGrammarConstraints,
            InferenceCapability.StructuredGeneration => Generation.SupportsStructuredGeneration,
            InferenceCapability.ToolArgumentConstraints => Generation.SupportsToolArgumentConstraints,

            // Model
            InferenceCapability.EmbeddingInput => Model.SupportsEmbeddingInput,
            InferenceCapability.Mtp => Model.SupportsMtp,
            InferenceCapability.Gdn => Model.SupportsGdn,

            // Device
            InferenceCapability.Cpu => Device.SupportsCpu,
            InferenceCapability.Cuda => Device.SupportsCuda,
            InferenceCapability.GpuArgmax => Device.SupportsGpuArgmax,

            _ => false
        };
    }

    /// <summary>
    /// Returns true if ALL specified capabilities are supported.
    /// </summary>
    public bool SupportsAll(params InferenceCapability[] capabilities)
    {
        if (capabilities == null || capabilities.Length == 0) return true;
        for (int i = 0; i < capabilities.Length; i++)
        {
            if (!Supports(capabilities[i])) return false;
        }
        return true;
    }

    /// <summary>
    /// Returns true if AT LEAST ONE of the specified capabilities is supported.
    /// </summary>
    public bool SupportsAny(params InferenceCapability[] capabilities)
    {
        if (capabilities == null || capabilities.Length == 0) return false;
        for (int i = 0; i < capabilities.Length; i++)
        {
            if (Supports(capabilities[i])) return true;
        }
        return false;
    }
}
