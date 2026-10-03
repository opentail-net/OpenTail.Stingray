namespace OpenTail.Stingray.Diffusion.MiniMaxH3;

/// <summary>
/// Architecture constants and configuration for the MiniMax-H3 omni-modal diffusion model.
/// Reference: TensorSharp docs/models/minimax-h3.md, Models/MiniMaxH3/*.
/// Upstream: MiniMaxAI/MiniMax-H3.
/// </summary>
public static class MiniMaxH3Config
{
    // DiT Architecture Dimensions
    public const int NumLayers = 50;
    public const int HiddenDim = 5376;
    public const int NumHeads = 56;
    public const int HeadDim = 128; // 56 * 128 = 7168 attention dimension
    public const int AttentionDim = NumHeads * HeadDim; // 7168

    // Latents and Patching
    public const int VideoLatentChannels = 24;
    public const int AudioLatentChannels = 32;

    public const int PatchTime = 1;
    public const int PatchHeight = 2;
    public const int PatchWidth = 2;

    /// <summary>
    /// Video patch dimension = PatchTime * PatchHeight * PatchWidth * VideoLatentChannels = 1 * 2 * 2 * 24 = 96.
    /// Video patch values are strictly channel-major, patch-minor.
    /// </summary>
    public const int VideoPatchDim = PatchTime * PatchHeight * PatchWidth * VideoLatentChannels; // 96

    /// <summary>
    /// Audio patch dimension = 32 channels.
    /// </summary>
    public const int AudioPatchDim = AudioLatentChannels; // 32

    // Dual Flow Schedules
    public const float VideoFlowShift = 12.0f;
    public const float AudioFlowShift = 3.0f;

    // Distilled Guidance (CFG)
    public const float GuidanceScale = 1.0f; // Distilled model; no negative prompt branch in base path

    // AdaLN Learned Timestep Curve Table
    public const int AdaLnTableRows = 8;
    public const int AdaLnTableCols = 1025;

    /// <summary>
    /// Per-block AdaLN projection size: 8 -> 96,768.
    /// Slices into 3 modalities (Video, Audio, Context/Text), each with 6 modulation vectors of length HiddenDim (5376):
    /// 3 * 6 * 5376 = 96,768.
    /// </summary>
    public const int AdaLnModulationDimPerModality = 6 * HiddenDim; // 32,256
    public const int AdaLnProjectionDim = 3 * AdaLnModulationDimPerModality; // 96,768
}

/// <summary>
/// Multimodal token modality identifier in the packed H3 sequence.
/// </summary>
public enum MiniMaxH3Modality
{
    Video = 0,
    Audio = 1,
    Context = 2,
}
