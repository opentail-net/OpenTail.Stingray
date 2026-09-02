namespace OpenTail.Stingray.Diffusion.StableAudio;

/// <summary>
/// Architectural hyperparameters for Stable Audio 3.
/// Supports variable-length text-to-audio and music generation with continuous MMDiT flow matching.
/// </summary>
public sealed record StableAudioParams
{
    /// <summary>Number of latent channels per acoustic frame (real value from `small-music-base`'s checkpoint: 256).</summary>
    public int LatentChannels { get; init; } = 256;

    /// <summary>Hidden embedding dimension D for the DiT transformer (real value: 1024 for Small).</summary>
    public int HiddenSize { get; init; } = 1024;

    /// <summary>Number of transformer layers / depth (real value: 20 for Small).</summary>
    public int Depth { get; init; } = 20;

    /// <summary>Number of attention heads (real value: 16, HeadDim = 64).</summary>
    public int NumHeads { get; init; } = 16;

    /// <summary>Text conditioner dimension from the real T5Gemma encoder (google/t5gemma-b-b-ul2, hidden_size=768 — NOT T5-XXL's 4096).</summary>
    public int TextContextDim { get; init; } = 768;

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
