namespace OpenTail.Stingray.Diffusion.DiffusionGemma;

/// <summary>
/// Architectural configuration for DiffusionGemma (google/diffusiongemma-26B-A4B-it).
/// Block text-diffusion model based on Gemma-4 MoE backbone (26B total, ~4B active).
/// </summary>
public sealed record DiffusionGemmaConfig
{
    public int HiddenDim { get; init; } = 2816;
    public int NumLayers { get; init; } = 30;

    // Sliding Attention Layers (25 layers: all except 5, 11, 17, 23, 29)
    public int SlidingNumQHeads { get; init; } = 16;
    public int SlidingNumKvHeads { get; init; } = 8;
    public int SlidingHeadDim { get; init; } = 256;
    public int SlidingWindowSize { get; init; } = 1024;

    // Full Attention Layers (5 layers: 5, 11, 17, 23, 29)
    public int FullNumQHeads { get; init; } = 16;
    public int FullNumKvHeads { get; init; } = 2;
    public int FullHeadDim { get; init; } = 512;

    // Embedding scaling: sqrt(2816) ~= 53.0659966f
    public float EmbedScale => MathF.Sqrt(HiddenDim);

    // Dense MLP
    public int DenseIntermediateDim { get; init; } = 2112;

    // MoE
    public int NumExperts { get; init; } = 128;
    public int NumExpertsUsed { get; init; } = 8;
    public int ExpertIntermediateDim { get; init; } = 704;

    // Self-Conditioning MLP
    public int SelfCondIntermediateDim { get; init; } = 2112;

    // Canvas & Sampler
    public int CanvasLength { get; init; } = 256;
    public int MaxDenoisingSteps { get; init; } = 48;
    public float TemperatureMax { get; init; } = 0.8f;
    public float TemperatureMin { get; init; } = 0.408f;
    public float EntropyBudgetNats { get; init; } = 0.1f;
    public float ConvergenceEntropyThreshold { get; init; } = 0.005f;

    public int VocabSize { get; init; } = 262144;
    public float RmsNormEps { get; init; } = 1e-6f;
    public float FinalLogitSoftcap { get; init; } = 30.0f;

    /// <summary>
    /// Full attention layers are layers 5, 11, 17, 23, 29 (i.e. (layer % 6) == 5).
    /// All other layers are sliding attention.
    /// </summary>
    public bool IsFullAttention(int layer) => (layer % 6) == 5;
}
