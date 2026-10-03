namespace OpenTail.Stingray.Diffusion.MiniMaxH3;

/// <summary>
/// MiniMax-H3 Omni-Modal Diffusion Transformer.
/// Denoises packed video and audio latents using 50 DiT blocks with dual schedules and learned AdaLN curve table.
/// </summary>
public sealed class MiniMaxH3DiT
{
    public int HiddenDim { get; }
    public int NumLayers { get; }

    // Latent Input Projections [HiddenDim, InDim]
    public float[] VideoInProj { get; }   // [5376, 96]
    public float[] AudioInProj { get; }   // [5376, 32]
    public float[] ContextInProj { get; } // [5376, ContextDim]
    public int ContextDim { get; }

    // Final Output Projections [OutDim, HiddenDim]
    public float[] VideoOutProj { get; }  // [96, 5376]
    public float[] AudioOutProj { get; }  // [32, 5376]

    // DiT Blocks
    public MiniMaxH3DiTBlockWeights[] Blocks { get; }

    public MiniMaxH3DiT(
        int numLayers = MiniMaxH3Config.NumLayers,
        int hiddenDim = MiniMaxH3Config.HiddenDim,
        int attentionDim = MiniMaxH3Config.AttentionDim,
        int ffnDim = 14336,
        int contextDim = 4096)
    {
        NumLayers = numLayers;
        HiddenDim = hiddenDim;
        ContextDim = contextDim;

        VideoInProj = new float[hiddenDim * MiniMaxH3Config.VideoPatchDim];
        AudioInProj = new float[hiddenDim * MiniMaxH3Config.AudioPatchDim];
        ContextInProj = new float[hiddenDim * contextDim];

        VideoOutProj = new float[MiniMaxH3Config.VideoPatchDim * hiddenDim];
        AudioOutProj = new float[MiniMaxH3Config.AudioPatchDim * hiddenDim];

        Blocks = new MiniMaxH3DiTBlockWeights[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            Blocks[i] = new MiniMaxH3DiTBlockWeights(hiddenDim, attentionDim, ffnDim);
        }
    }

    /// <summary>
    /// Forward pass predicting video and audio velocities over the packed multimodal sequence.
    /// </summary>
    public (float[] VideoVelocity, float[] AudioVelocity) Forward(
        MiniMaxH3Layout layout,
        ReadOnlySpan<float> videoPatchTokens,
        ReadOnlySpan<float> audioTokens,
        ReadOnlySpan<float> contextTokens,
        ReadOnlySpan<float> curveVec8,
        int numHeads = MiniMaxH3Config.NumHeads,
        int headDim = MiniMaxH3Config.HeadDim)
    {
        int totalTokens = layout.TotalTokens;
        var hiddenStates = new float[totalTokens * HiddenDim];

        // 1. Project Context tokens -> HiddenDim
        int numContextTokens = layout.NumTextTokens + layout.NumVisualCondTokens;
        for (int i = 0; i < numContextTokens; i++)
        {
            var ctx = contextTokens.Slice(i * ContextDim, ContextDim);
            var h = hiddenStates.AsSpan(i * HiddenDim, HiddenDim);
            MatVec(ContextInProj, ctx, h, HiddenDim, ContextDim);
        }

        // 2. Project Video tokens -> HiddenDim
        int videoStart = layout.VideoTargetStart;
        for (int i = 0; i < layout.NumVideoTokens; i++)
        {
            var vid = videoPatchTokens.Slice(i * MiniMaxH3Config.VideoPatchDim, MiniMaxH3Config.VideoPatchDim);
            var h = hiddenStates.AsSpan((videoStart + i) * HiddenDim, HiddenDim);
            MatVec(VideoInProj, vid, h, HiddenDim, MiniMaxH3Config.VideoPatchDim);
        }

        // 3. Project Audio tokens -> HiddenDim
        int audioStart = layout.AudioTargetStart;
        for (int i = 0; i < layout.NumAudioTokens; i++)
        {
            var aud = audioTokens.Slice(i * MiniMaxH3Config.AudioPatchDim, MiniMaxH3Config.AudioPatchDim);
            var h = hiddenStates.AsSpan((audioStart + i) * HiddenDim, HiddenDim);
            MatVec(AudioInProj, aud, h, HiddenDim, MiniMaxH3Config.AudioPatchDim);
        }

        // 4. Precompute Multimodal RoPE tables once for this packed sequence layout
        var (ropeCos, ropeSin) = MiniMaxH3RoPE.ComputeMultimodalRoPE(layout, headDim);

        // 5. Pass through all DiT blocks
        var hiddenSpan = hiddenStates.AsSpan();
        for (int b = 0; b < NumLayers; b++)
        {
            MiniMaxH3DiTBlock.Forward(
                hiddenSpan,
                totalTokens,
                layout,
                curveVec8,
                Blocks[b],
                numHeads,
                headDim,
                ropeCos,
                ropeSin);
        }

        // 5. Output projection for Video Target tokens -> [NumVideoTokens, 96]
        var videoVelocity = new float[layout.NumVideoTokens * MiniMaxH3Config.VideoPatchDim];
        for (int i = 0; i < layout.NumVideoTokens; i++)
        {
            var h = hiddenStates.AsSpan((videoStart + i) * HiddenDim, HiddenDim);
            var outSpan = videoVelocity.AsSpan(i * MiniMaxH3Config.VideoPatchDim, MiniMaxH3Config.VideoPatchDim);
            MatVec(VideoOutProj, h, outSpan, MiniMaxH3Config.VideoPatchDim, HiddenDim);
        }

        // 6. Output projection for Audio Target tokens -> [NumAudioTokens, 32]
        var audioVelocity = new float[layout.NumAudioTokens * MiniMaxH3Config.AudioPatchDim];
        for (int i = 0; i < layout.NumAudioTokens; i++)
        {
            var h = hiddenStates.AsSpan((audioStart + i) * HiddenDim, HiddenDim);
            var outSpan = audioVelocity.AsSpan(i * MiniMaxH3Config.AudioPatchDim, MiniMaxH3Config.AudioPatchDim);
            MatVec(AudioOutProj, h, outSpan, MiniMaxH3Config.AudioPatchDim, HiddenDim);
        }

        return (videoVelocity, audioVelocity);
    }

    private static void MatVec(ReadOnlySpan<float> matrix, ReadOnlySpan<float> vec, Span<float> dst, int rows, int cols)
    {
        for (int r = 0; r < rows; r++)
        {
            int rowStart = r * cols;
            float sum = 0.0f;
            for (int c = 0; c < cols; c++)
            {
                sum += matrix[rowStart + c] * vec[c];
            }
            dst[r] = sum;
        }
    }
}
