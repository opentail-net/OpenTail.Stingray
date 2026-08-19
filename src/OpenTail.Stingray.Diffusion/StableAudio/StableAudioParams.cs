namespace OpenTail.Stingray.Diffusion.StableAudio;

/// <summary>
/// Architectural hyperparameters for Stable Audio 3.
/// Supports variable-length text-to-audio and music generation with continuous MMDiT flow matching.
/// </summary>
public sealed record StableAudioParams
{
    /// <summary>Number of latent channels per acoustic frame (default: 64).</summary>
    public int LatentChannels { get; init; } = 64;

    /// <summary>Hidden embedding dimension D for the MMDiT transformer (default: 768 for Small, 1536 for Medium).</summary>
    public int HiddenSize { get; init; } = 768;

    /// <summary>Number of transformer layers / depth (default: 12 for Small, 24 for Medium).</summary>
    public int Depth { get; init; } = 12;

    /// <summary>Number of attention heads (default: 12, HeadDim = 64).</summary>
    public int NumHeads { get; init; } = 12;

    /// <summary>Text conditioner dimension from T5 (default: 4096).</summary>
    public int TextContextDim { get; init; } = 4096;

    /// <summary>Dimension of timestep and continuous timing Fourier feature embeddings (default: 256).</summary>
    public int TimingFeaturesDim { get; init; } = 256;

    /// <summary>Target output audio sample rate in Hz (default: 44,100 Hz).</summary>
    public int SampleRate { get; init; } = 44100;

    /// <summary>Target number of audio channels (default: 2 for Stereo).</summary>
    public int AudioChannels { get; init; } = 2;

    /// <summary>Acoustic latent frame rate in Hz (default: ~43 Hz, ~1024x temporal compression from 44.1kHz).</summary>
    public float LatentFrameRate { get; init; } = 43.0664f;

    public int HeadDim => HiddenSize / Math.Max(1, NumHeads);
}
