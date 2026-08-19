namespace OpenTail.Stingray.Diffusion.Flux2;

/// <summary>
/// Architectural hyperparameters for the FLUX.2 (Klein &amp; Kontext) multi-reference foundation model.
/// Supports multi-image conditioning, contextual editing, and 3D Context RoPE.
/// </summary>
public sealed record Flux2Params
{
    /// <summary>Image latent channels per 2x2 patch (default: 64, corresponding to 16-channel VAE).</summary>
    public int InChannels { get; init; } = 64;

    /// <summary>Output velocity channels (default: 64).</summary>
    public int OutChannels { get; init; } = 64;

    /// <summary>Pooled text embedding dimension from CLIP (default: 768).</summary>
    public int VecInDim { get; init; } = 768;

    /// <summary>Sequence text embedding dimension from T5 (default: 4096).</summary>
    public int ContextInDim { get; init; } = 4096;

    /// <summary>Transformer hidden embedding dimension D (default: 3072).</summary>
    public int HiddenSize { get; init; } = 3072;

    /// <summary>MLP expansion ratio (default: 4.0).</summary>
    public float MlpRatio { get; init; } = 4.0f;

    /// <summary>Number of attention heads (default: 24, HeadDim = 128).</summary>
    public int NumHeads { get; init; } = 24;

    /// <summary>Number of double-stream transformer blocks (default: 19).</summary>
    public int DepthDoubleBlocks { get; init; } = 19;

    /// <summary>Number of single-stream unified blocks (default: 38).</summary>
    public int DepthSingleBlocks { get; init; } = 38;

    /// <summary>3D Context RoPE axial dimensions: [ImageIndexDim, HeightDim, WidthDim] (default: [16, 56, 56], sum = 128).</summary>
    public int[] AxesDim { get; init; } = [16, 56, 56];

    /// <summary>Base theta for RoPE frequency calculation (default: 10000.0).</summary>
    public float Theta { get; init; } = 10000.0f;

    /// <summary>Whether linear layers use QKV bias (default: true).</summary>
    public bool QkvBias { get; init; } = true;

    /// <summary>Whether guidance embedding MLP is active (default: true).</summary>
    public bool GuidanceEmbed { get; init; } = true;

    public int HeadDim => HiddenSize / Math.Max(1, NumHeads);
}
