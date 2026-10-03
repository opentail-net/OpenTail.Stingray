namespace OpenTail.Stingray.Diffusion.MiniMaxH3;

/// <summary>
/// Weights for a single MiniMax-H3 DiT block.
/// Hidden dimension = 5376, Attention dimension = 7168 (56 heads * 128 head dim).
/// SwiGLU FFN.
/// </summary>
public sealed class MiniMaxH3DiTBlockWeights
{
    public int HiddenDim { get; }
    public int AttentionDim { get; }
    public int FfnDim { get; }

    // Attention weights [OutDim, InDim]
    public float[] Wq { get; }
    public float[] Wk { get; }
    public float[] Wv { get; }
    public float[] Wo { get; }

    // FFN weights [OutDim, InDim]
    public float[] WGate { get; }
    public float[] WUp { get; }
    public float[] WDown { get; }

    // AdaLN Linear Projection: 8 -> 96,768 (or 8 -> 3 * 6 * HiddenDim)
    public float[] AdaLnWeight { get; } // [AdaLnProjectionDim, 8]
    public float[]? AdaLnBias { get; }  // [AdaLnProjectionDim]

    public MiniMaxH3DiTBlockWeights(
        int hiddenDim = MiniMaxH3Config.HiddenDim,
        int attentionDim = MiniMaxH3Config.AttentionDim,
        int ffnDim = 14336)
    {
        HiddenDim = hiddenDim;
        AttentionDim = attentionDim;
        FfnDim = ffnDim;

        Wq = new float[attentionDim * hiddenDim];
        Wk = new float[attentionDim * hiddenDim];
        Wv = new float[attentionDim * hiddenDim];
        Wo = new float[hiddenDim * attentionDim];

        WGate = new float[ffnDim * hiddenDim];
        WUp = new float[ffnDim * hiddenDim];
        WDown = new float[hiddenDim * ffnDim];

        int adalnDim = 3 * 6 * hiddenDim;
        AdaLnWeight = new float[adalnDim * MiniMaxH3Config.AdaLnTableRows];
        AdaLnBias = new float[adalnDim];
    }
}

/// <summary>
/// Native C# forward pass kernel for a single MiniMax-H3 DiT block over the packed multimodal sequence.
/// </summary>
public static class MiniMaxH3DiTBlock
{
    /// <summary>
    /// Forward pass through one DiT block for a packed sequence of tokens.
    /// x: [numTokens, hiddenDim] row-major.
    /// curveVec8: 8-dim AdaLN curve vector interpolated from the learned table for this timestep.
    /// </summary>
    public static void Forward(
        Span<float> x,
        int numTokens,
        MiniMaxH3Layout layout,
        ReadOnlySpan<float> curveVec8,
        MiniMaxH3DiTBlockWeights weights,
        int numHeads = MiniMaxH3Config.NumHeads,
        int headDim = MiniMaxH3Config.HeadDim,
        ReadOnlySpan<float> ropeCos = default,
        ReadOnlySpan<float> ropeSin = default)
    {
        int hiddenDim = weights.HiddenDim;
        int attnDim = weights.AttentionDim;
        int ffnDim = weights.FfnDim;

        // 1. Compute per-block AdaLN modulation vector: 8 -> 96,768 (or 3 * 6 * hiddenDim)
        int adalnDim = 3 * 6 * hiddenDim;
        Span<float> modulation = stackalloc float[adalnDim <= 2048 ? adalnDim : 0];
        float[]? rentedModulation = null;
        if (modulation.IsEmpty)
        {
            rentedModulation = ArrayPool<float>.Shared.Rent(adalnDim);
            modulation = rentedModulation.AsSpan(0, adalnDim);
        }

        try
        {
            MiniMaxH3AdaLN.ProjectModulation(curveVec8, weights.AdaLnWeight, weights.AdaLnBias ?? ReadOnlySpan<float>.Empty, modulation);

            // Pre-slice modulation for all 3 modalities
            var videoMod = MiniMaxH3AdaLN.GetModalityModulation(modulation, MiniMaxH3Modality.Video, hiddenDim);
            var audioMod = MiniMaxH3AdaLN.GetModalityModulation(modulation, MiniMaxH3Modality.Audio, hiddenDim);
            var contextMod = MiniMaxH3AdaLN.GetModalityModulation(modulation, MiniMaxH3Modality.Context, hiddenDim);

            // Buffer allocations for attention Q, K, V and normalized state
            float[] qkv = ArrayPool<float>.Shared.Rent(numTokens * attnDim * 3);
            Span<float> q = qkv.AsSpan(0, numTokens * attnDim);
            Span<float> k = qkv.AsSpan(numTokens * attnDim, numTokens * attnDim);
            Span<float> v = qkv.AsSpan(numTokens * attnDim * 2, numTokens * attnDim);

            float[] normBuf = ArrayPool<float>.Shared.Rent(hiddenDim);
            Span<float> normSpan = normBuf.AsSpan(0, hiddenDim);

            try
            {
                // 2. Pre-attention LayerNorm + AdaLN modulation + QKV projection
                for (int i = 0; i < numTokens; i++)
                {
                    var tokenX = x.Slice(i * hiddenDim, hiddenDim);
                    var mod = layout.GetModality(i) switch
                    {
                        MiniMaxH3Modality.Video => videoMod,
                        MiniMaxH3Modality.Audio => audioMod,
                        _ => contextMod
                    };

                    RmsNorm(tokenX, normSpan);
                    MiniMaxH3AdaLN.ModulateNorm(normSpan, mod.Gamma1, mod.Beta1);

                    // Q, K, V projections
                    MatVec(weights.Wq, normSpan, q.Slice(i * attnDim, attnDim), attnDim, hiddenDim);
                    MatVec(weights.Wk, normSpan, k.Slice(i * attnDim, attnDim), attnDim, hiddenDim);
                    MatVec(weights.Wv, normSpan, v.Slice(i * attnDim, attnDim), attnDim, hiddenDim);
                }

                // 2b. Apply Multimodal RoPE to Q and K if provided
                if (!ropeCos.IsEmpty && !ropeSin.IsEmpty)
                {
                    MiniMaxH3RoPE.ApplyRoPE(q, ropeCos, ropeSin, numTokens, numHeads, headDim);
                    MiniMaxH3RoPE.ApplyRoPE(k, ropeCos, ropeSin, numTokens, numHeads, headDim);
                }

                // 3. Full Bidirectional Multi-Head Self-Attention
                float[] attnOutBuf = ArrayPool<float>.Shared.Rent(numTokens * attnDim);
                Span<float> attnOut = attnOutBuf.AsSpan(0, numTokens * attnDim);

                float[] scoresBuf = ArrayPool<float>.Shared.Rent(numTokens);
                Span<float> scores = scoresBuf.AsSpan(0, numTokens);

                float scale = 1.0f / MathF.Sqrt(headDim);

                try
                {
                    for (int h = 0; h < numHeads; h++)
                    {
                        int headOffset = h * headDim;
                        for (int i = 0; i < numTokens; i++)
                        {
                            var qi = q.Slice(i * attnDim + headOffset, headDim);

                            // Compute dot products with all k_j
                            for (int j = 0; j < numTokens; j++)
                            {
                                var kj = k.Slice(j * attnDim + headOffset, headDim);
                                float dot = 0.0f;
                                for (int d = 0; d < headDim; d++) dot += qi[d] * kj[d];
                                scores[j] = dot * scale;
                            }

                            Softmax(scores.Slice(0, numTokens));

                            // Accumulate weighted sum of v_j
                            var outSlice = attnOut.Slice(i * attnDim + headOffset, headDim);
                            outSlice.Clear();

                            for (int j = 0; j < numTokens; j++)
                            {
                                float s = scores[j];
                                var vj = v.Slice(j * attnDim + headOffset, headDim);
                                for (int d = 0; d < headDim; d++)
                                {
                                    outSlice[d] += s * vj[d];
                                }
                            }
                        }
                    }

                    // 4. Output projection Wo + Attention Gate + Residual
                    float[] projBuf = ArrayPool<float>.Shared.Rent(hiddenDim);
                    Span<float> proj = projBuf.AsSpan(0, hiddenDim);

                    try
                    {
                        for (int i = 0; i < numTokens; i++)
                        {
                            var tokenAttn = attnOut.Slice(i * attnDim, attnDim);
                            MatVec(weights.Wo, tokenAttn, proj, hiddenDim, attnDim);

                            var mod = layout.GetModality(i) switch
                            {
                                MiniMaxH3Modality.Video => videoMod,
                                MiniMaxH3Modality.Audio => audioMod,
                                _ => contextMod
                            };

                            MiniMaxH3AdaLN.ApplyGate(proj, mod.Alpha1);

                            var tokenX = x.Slice(i * hiddenDim, hiddenDim);
                            for (int d = 0; d < hiddenDim; d++) tokenX[d] += proj[d];
                        }
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(projBuf);
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(attnOutBuf);
                    ArrayPool<float>.Shared.Return(scoresBuf);
                }

                // 5. Pre-FFN LayerNorm + AdaLN modulation + SwiGLU FFN + Gate + Residual
                float[] gateBuf = ArrayPool<float>.Shared.Rent(ffnDim);
                float[] upBuf = ArrayPool<float>.Shared.Rent(ffnDim);
                float[] downBuf = ArrayPool<float>.Shared.Rent(hiddenDim);

                Span<float> gateSpan = gateBuf.AsSpan(0, ffnDim);
                Span<float> upSpan = upBuf.AsSpan(0, ffnDim);
                Span<float> downSpan = downBuf.AsSpan(0, hiddenDim);

                try
                {
                    for (int i = 0; i < numTokens; i++)
                    {
                        var tokenX = x.Slice(i * hiddenDim, hiddenDim);
                        var mod = layout.GetModality(i) switch
                        {
                            MiniMaxH3Modality.Video => videoMod,
                            MiniMaxH3Modality.Audio => audioMod,
                            _ => contextMod
                        };

                        RmsNorm(tokenX, normSpan);
                        MiniMaxH3AdaLN.ModulateNorm(normSpan, mod.Gamma2, mod.Beta2);

                        // Gate and Up projections
                        MatVec(weights.WGate, normSpan, gateSpan, ffnDim, hiddenDim);
                        MatVec(weights.WUp, normSpan, upSpan, ffnDim, hiddenDim);

                        // SwiGLU: swish(gate) * up
                        for (int d = 0; d < ffnDim; d++)
                        {
                            float g = gateSpan[d];
                            float swish = g * (1.0f / (1.0f + MathF.Exp(-g)));
                            gateSpan[d] = swish * upSpan[d];
                        }

                        // Down projection
                        MatVec(weights.WDown, gateSpan, downSpan, hiddenDim, ffnDim);

                        // Gate scaling + residual
                        MiniMaxH3AdaLN.ApplyGate(downSpan, mod.Alpha2);
                        for (int d = 0; d < hiddenDim; d++) tokenX[d] += downSpan[d];
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(gateBuf);
                    ArrayPool<float>.Shared.Return(upBuf);
                    ArrayPool<float>.Shared.Return(downBuf);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(qkv);
                ArrayPool<float>.Shared.Return(normBuf);
            }
        }
        finally
        {
            if (rentedModulation != null) ArrayPool<float>.Shared.Return(rentedModulation);
        }
    }

    private static void RmsNorm(ReadOnlySpan<float> src, Span<float> dst, float eps = 1e-6f)
    {
        float sumSq = 0.0f;
        for (int i = 0; i < src.Length; i++) sumSq += src[i] * src[i];
        float invRms = 1.0f / MathF.Sqrt(sumSq / src.Length + eps);
        for (int i = 0; i < src.Length; i++) dst[i] = src[i] * invRms;
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

    private static void Softmax(Span<float> x)
    {
        float maxVal = x[0];
        for (int i = 1; i < x.Length; i++)
        {
            if (x[i] > maxVal) maxVal = x[i];
        }

        float sumExp = 0.0f;
        for (int i = 0; i < x.Length; i++)
        {
            float e = MathF.Exp(x[i] - maxVal);
            x[i] = e;
            sumExp += e;
        }

        float invSum = 1.0f / (sumExp + 1e-9f);
        for (int i = 0; i < x.Length; i++)
        {
            x[i] *= invSum;
        }
    }
}
