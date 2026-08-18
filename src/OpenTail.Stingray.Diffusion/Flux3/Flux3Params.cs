namespace OpenTail.Stingray.Diffusion.Flux3;

/// <summary>
/// Architectural hyperparameters for the FLUX 3 multimodal foundation model.
/// Supports unified video, native synchronized audio, and text conditioning.
/// </summary>
public sealed record Flux3Params
{
    /// <summary>Video latent channels per spatiotemporal patch (default: 64, corresponding to 16-channel 3D VAE 2x2x2 patch).</summary>
    public int InVideoChannels { get; init; } = 64;

    /// <summary>Output video velocity channels (default: 64).</summary>
    public int OutVideoChannels { get; init; } = 64;

    /// <summary>Audio latent channels per acoustic patch (default: 32).</summary>
    public int InAudioChannels { get; init; } = 32;

    /// <summary>Output audio velocity channels (default: 32).</summary>
    public int OutAudioChannels { get; init; } = 32;

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

    /// <summary>Number of multimodal triple-stream blocks (default: 19).</summary>
    public int DepthDoubleBlocks { get; init; } = 19;

    /// <summary>Number of unified single-stream blocks (default: 38).</summary>
    public int DepthSingleBlocks { get; init; } = 38;

    /// <summary>RoPE axial dimensions for Video: [TimeDim, HeightDim, WidthDim] (default: [32, 48, 48], sum = 128).</summary>
    public int[] VideoAxesDim { get; init; } = [32, 48, 48];

    /// <summary>RoPE axial dimensions for Audio: [TimeDim, FreqDim] (default: [64, 64], sum = 128).</summary>
    public int[] AudioAxesDim { get; init; } = [64, 64];

    /// <summary>Base theta for RoPE frequency calculation (default: 10000).</summary>
    public float Theta { get; init; } = 10000.0f;

    /// <summary>Whether linear layers use QKV bias (default: true).</summary>
    public bool QkvBias { get; init; } = true;

    /// <summary>Whether guidance embedding MLP is active (default: true for dev/preview).</summary>
    public bool GuidanceEmbed { get; init; } = true;

    public int HeadDim => HiddenSize / Math.Max(1, NumHeads);
}
