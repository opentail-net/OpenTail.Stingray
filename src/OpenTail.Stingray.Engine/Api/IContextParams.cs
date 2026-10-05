#nullable enable

namespace OpenTail.Stingray;

/// <summary>
/// Contract for parameters required to allocate an execution context and KV cache over a loaded model.
/// Inspired by LLamaSharp's <c>IContextParams</c>, tailored for Stingray's execution backends.
/// </summary>
public interface IContextParams
{
    /// <summary>Maximum context length in tokens allocated for the KV cache. <c>0</c> selects model default.</summary>
    uint ContextSize { get; }

    /// <summary>Prompt processing physical/logical batch size in tokens.</summary>
    uint BatchSize { get; }

    /// <summary>Number of worker threads allocated for CPU compute kernels. <c>0</c> selects system auto-detection.</summary>
    int ThreadCount { get; }

    /// <summary>Whether to enable flash attention optimization when supported by backend and hardware.</summary>
    bool FlashAttention { get; }

    /// <summary>Optional TurboQuant KV cache compression mode (e.g. "auto", "kvarn", "lloydmax").</summary>
    string? TurboQuantMode { get; }

    /// <summary>Optional head dimension override for TurboQuant validation.</summary>
    int? TurboQuantHeadDim { get; }

    /// <summary>Optional RoPE base frequency override (e.g. 10000.0 or 500000.0).</summary>
    float? RopeFrequencyBase { get; }

    /// <summary>Optional RoPE frequency scale override.</summary>
    float? RopeFrequencyScale { get; }

    /// <summary>Whether the context is created in embedding extraction mode rather than text generation.</summary>
    bool Embeddings { get; }
}

/// <summary>
/// Configuration for allocating an execution context and KV cache over a loaded model.
/// </summary>
public record ContextParams : IContextParams
{
    /// <summary>Maximum context length in tokens allocated for the KV cache. <c>0</c> selects model default.</summary>
    public uint ContextSize { get; init; } = 0;

    /// <summary>Prompt processing physical/logical batch size in tokens. Default is 512.</summary>
    public uint BatchSize { get; init; } = 512;

    /// <summary>Number of worker threads allocated for CPU compute kernels. <c>0</c> selects system auto-detection.</summary>
    public int ThreadCount { get; init; } = 0;

    /// <summary>Whether to enable flash attention optimization when supported by backend and hardware.</summary>
    public bool FlashAttention { get; init; } = true;

    /// <summary>Optional TurboQuant KV cache compression mode (e.g. "auto", "kvarn", "lloydmax").</summary>
    public string? TurboQuantMode { get; init; }

    /// <summary>Optional head dimension override for TurboQuant validation.</summary>
    public int? TurboQuantHeadDim { get; init; }

    /// <summary>Optional RoPE base frequency override (e.g. 10000.0 or 500000.0).</summary>
    public float? RopeFrequencyBase { get; init; }

    /// <summary>Optional RoPE frequency scale override.</summary>
    public float? RopeFrequencyScale { get; init; }

    /// <summary>Whether the context is created in embedding extraction mode rather than text generation.</summary>
    public bool Embeddings { get; init; } = false;
}
