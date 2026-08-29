
namespace OpenTail.Stingray.Core.Capabilities;

/// <summary>
/// Authoritative factory builder for constructing truthful <see cref="InferenceCapabilities"/>
/// by aggregating effective component capabilities across model, execution, state, generation, and device layers.
/// </summary>
public static class InferenceCapabilityBuilder
{
    /// <summary>
    /// Constructs authoritative capabilities derived directly from instantiated runtime, forward-pass, and KV-cache components.
    /// </summary>
    public static InferenceCapabilities Build(
        bool hasKvCache = true,
        IForwardPass? forwardPass = null,
        string architecture = "Transformer",
        int defaultContextLength = 4096,
        bool supportsContinuousBatching = false,
        bool supportsSpeculativeDecoding = false)
    {
        bool supportsEmbeddingInput = forwardPass?.SupportsEmbeddingInput ?? false;
        bool supportsGpuArgmax = forwardPass?.SupportsGpuArgmax ?? false;
        int contextLength = forwardPass?.MaxSeqLen > 0 ? forwardPass.MaxSeqLen : defaultContextLength;
        string backend = forwardPass != null ? forwardPass.GetType().Name : "CPU";

        return new InferenceCapabilities
        {
            Model = new ModelCapabilities
            {
                Architecture = architecture,
                ContextLength = contextLength,
                IsDecoderOnly = true,
                IsEncoderDecoder = false,
                SupportsEmbeddingInput = supportsEmbeddingInput,
                SupportsVision = supportsEmbeddingInput,
                SupportsToolCalling = false,
                SupportsStructuredOutput = true,
                SupportsMtp = false,
                SupportsGdn = false,
                SupportsMoE = false
            },
            Execution = new ExecutionCapabilities
            {
                SupportsContinuousBatching = supportsContinuousBatching,
                SupportsChunkedPrefill = supportsContinuousBatching,
                SupportsPackedPrefill = supportsContinuousBatching,
                SupportsSpeculativeDecoding = supportsSpeculativeDecoding,
                SupportsPromptLookupSpeculation = supportsSpeculativeDecoding,
                SupportsCancellation = true
            },
            State = new StateCapabilities
            {
                SupportsPagedKvCache = hasKvCache,
                SupportsKvForking = hasKvCache,
                SupportsKvCopyOnWrite = hasKvCache,
                SupportsCheckpointRollback = true,
                SupportsSessionFork = true,
                SupportsSuspendResume = true,
                SupportsSnapshotRestore = true,
                SupportsPrefixCaching = true
            },
            Generation = new GenerationCapabilities
            {
                SupportsSampling = true,
                SupportsGreedyDecoding = true,
                SupportsSpeculativeSampling = supportsSpeculativeDecoding,
                SupportsGrammarConstraints = true,
                SupportsStructuredGeneration = true,
                SupportsToolArgumentConstraints = true,
                SupportsStreaming = true
            },
            Device = new DeviceCapabilities
            {
                SupportsCpu = true,
                SupportsCuda = supportsGpuArgmax,
                SupportsGpuArgmax = supportsGpuArgmax,
                Backend = backend
            }
        };
    }

    /// <summary>
    /// Constructs default CPU capabilities for standard OpenTail.Stingray session runtime environments.
    /// </summary>
    public static InferenceCapabilities CreateDefaultCpu(
        string architecture = "Transformer",
        int contextLength = 4096,
        bool supportsEmbeddingInput = false,
        bool supportsMtp = false,
        bool supportsContinuousBatching = false,
        bool supportsSpeculativeDecoding = false)
    {
        return Build(
            hasKvCache: true,
            forwardPass: null,
            architecture: architecture,
            defaultContextLength: contextLength,
            supportsContinuousBatching: supportsContinuousBatching,
            supportsSpeculativeDecoding: supportsSpeculativeDecoding);
    }
}
