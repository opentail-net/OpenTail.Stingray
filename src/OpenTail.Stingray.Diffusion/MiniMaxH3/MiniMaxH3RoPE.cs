namespace OpenTail.Stingray.Diffusion.MiniMaxH3;

/// <summary>
/// Multimodal RoPE (Rotary Positional Embedding) for MiniMax-H3.
/// Generates coordinate-aware rotary tables across the packed sequence:
///   [text tokens | visual conditioning | video target | audio target]
/// Video tokens use 3D-RoPE (t, y, x) decomposing the 128 head dimension into {44, 42, 42}.
/// Audio and text tokens use 1D temporal/sequence RoPE over the full 128 head dimension.
/// </summary>
public static class MiniMaxH3RoPE
{
    public const int DefaultHeadDim = 128;
    public const float DefaultTheta = 10000.0f;

    // 3D Video axis decomposition summing to 128
    public const int VideoDimT = 44;
    public const int VideoDimH = 42;
    public const int VideoDimW = 42;

    /// <summary>
    /// Compute multimodal RoPE cos and sin tables for the entire packed sequence.
    /// Returns arrays of length TotalTokens * headDim.
    /// </summary>
    public static (float[] Cos, float[] Sin) ComputeMultimodalRoPE(
        MiniMaxH3Layout layout,
        int headDim = DefaultHeadDim,
        float theta = DefaultTheta)
    {
        int totalTokens = layout.TotalTokens;
        var cos = new float[totalTokens * headDim];
        var sin = new float[totalTokens * headDim];

        // 1. Text & Visual Condition RoPE (1D sequence positions)
        int numContext = layout.NumTextTokens + layout.NumVisualCondTokens;
        if (numContext > 0)
        {
            var invFreq1D = InterleavedRoPE.ComputeInvFreqs(headDim, theta);
            for (int i = 0; i < numContext; i++)
            {
                int baseOff = i * headDim;
                InterleavedRoPE.FillAxisFreqs(cos.AsSpan(baseOff, headDim), sin.AsSpan(baseOff, headDim), i, invFreq1D);
            }
        }

        // 2. Video RoPE (3D spatio-temporal positions)
        if (layout.NumVideoTokens > 0)
        {
            int videoStart = layout.VideoTargetStart;
            int dimH = 2 * (headDim / 6);
            int dimW = dimH;
            int dimT = headDim - dimH - dimW;

            var invFreqT = InterleavedRoPE.ComputeInvFreqs(dimT, theta);
            var invFreqH = InterleavedRoPE.ComputeInvFreqs(dimH, theta);
            var invFreqW = InterleavedRoPE.ComputeInvFreqs(dimW, theta);

            for (int v = 0; v < layout.NumVideoTokens; v++)
            {
                var (t, hp, wp) = layout.GetVideoCoordinates(v);
                int baseOff = (videoStart + v) * headDim;

                // Temporal axis: dimT (44 when headDim=128)
                InterleavedRoPE.FillAxisFreqs(cos.AsSpan(baseOff, dimT), sin.AsSpan(baseOff, dimT), t, invFreqT);

                // Height axis: dimH (42 when headDim=128)
                InterleavedRoPE.FillAxisFreqs(cos.AsSpan(baseOff + dimT, dimH), sin.AsSpan(baseOff + dimT, dimH), hp, invFreqH);

                // Width axis: dimW (42 when headDim=128)
                InterleavedRoPE.FillAxisFreqs(cos.AsSpan(baseOff + dimT + dimH, dimW), sin.AsSpan(baseOff + dimT + dimH, dimW), wp, invFreqW);
            }
        }

        // 3. Audio RoPE (1D temporal audio positions)
        if (layout.NumAudioTokens > 0)
        {
            int audioStart = layout.AudioTargetStart;
            var invFreqAudio = InterleavedRoPE.ComputeInvFreqs(headDim, theta);
            for (int a = 0; a < layout.NumAudioTokens; a++)
            {
                int baseOff = (audioStart + a) * headDim;
                InterleavedRoPE.FillAxisFreqs(cos.AsSpan(baseOff, headDim), sin.AsSpan(baseOff, headDim), a, invFreqAudio);
            }
        }

        return (cos, sin);
    }

    /// <summary>
    /// Apply RoPE rotation in-place to Q or K tensors: [numTokens, numHeads, headDim].
    /// </summary>
    public static void ApplyRoPE(
        Span<float> qk,
        ReadOnlySpan<float> cos,
        ReadOnlySpan<float> sin,
        int numTokens,
        int numHeads,
        int headDim = DefaultHeadDim)
    {
        InterleavedRoPE.ApplyRoPE(qk, cos, sin, numTokens, numHeads, headDim);
    }
}
