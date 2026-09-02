namespace OpenTail.Stingray.Diffusion.AceStep.Primitives;

/// <summary>
/// DRY pass (CLAUDE.md rule 7): the GQA/RoPE/per-head-QK-norm/softmax/SiLU primitives were
/// byte-identical duplicates between <see cref="Transformer.AceStepDiT"/>'s self-attention path
/// and <see cref="Conditioning.AceStepConditionEncoder"/>'s lyric encoder (both real, individually
/// verified against real weights) -- extracted here now that a second real, verified caller exists,
/// per this project's own DRY-timing rule. Cross-attention-specific pieces (the DiT's ungated
/// residual, cached K/V reuse) and each caller's own per-layer glue (AdaLN modulation vs. plain
/// pre-norm) stay where they are -- only the shared math moved.
/// </summary>
public static class AceStepAttentionKernels
{
    /// <summary>Real Qwen3-style RoPE table: `inv_freq[i] = theta^(-2i/headDim)`, position `p` in `[0,seqLen)`. Real HF convention: cos/sin tables are duplicated across both halves of `headDim`.</summary>
    public static (float[] Cos, float[] Sin) BuildRope(int seqLen, int headDim, float theta)
    {
        int half = headDim / 2;
        var cos = new float[seqLen * headDim];
        var sin = new float[seqLen * headDim];
        for (int p = 0; p < seqLen; p++)
        {
            for (int i = 0; i < half; i++)
            {
                float invFreq = MathF.Pow(theta, -2f * i / headDim);
                float angle = p * invFreq;
                float c = MathF.Cos(angle), s = MathF.Sin(angle);
                cos[p * headDim + i] = c; cos[p * headDim + half + i] = c;
                sin[p * headDim + i] = s; sin[p * headDim + half + i] = s;
            }
        }
        return (cos, sin);
    }

    /// <summary>Real `apply_rotary_pos_emb` (rotate_half convention): `q_embed = q*cos + rotate_half(q)*sin`, `rotate_half(x) = cat(-x[half:], x[:half])`.</summary>
    public static void ApplyRope(float[] qOrK, int seqLen, int numHeads, int headDim, float[] cos, float[] sin)
    {
        int half = headDim / 2;
        int rowDim = numHeads * headDim;
        for (int t = 0; t < seqLen; t++)
        {
            int cosBase = t * headDim;
            for (int h = 0; h < numHeads; h++)
            {
                int off = t * rowDim + h * headDim;
                for (int i = 0; i < half; i++)
                {
                    float x1 = qOrK[off + i];
                    float x2 = qOrK[off + half + i];
                    float c1 = cos[cosBase + i], s1 = sin[cosBase + i];
                    float c2 = cos[cosBase + half + i], s2 = sin[cosBase + half + i];
                    qOrK[off + i] = x1 * c1 - x2 * s1;
                    qOrK[off + half + i] = x2 * c2 + x1 * s2;
                }
            }
        }
    }

    /// <summary>Real per-head Q/K RMSNorm (`Qwen3RMSNorm(head_dim)`), applied independently to each head's `head_dim`-wide slice.</summary>
    public static void RmsNormPerHead(float[] qOrK, int seqLen, int numHeads, int headDim, float[] weight, float eps = 1e-6f)
    {
        for (int t = 0; t < seqLen; t++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int off = t * numHeads * headDim + h * headDim;
                var span = qOrK.AsSpan(off, headDim);
                float sumSq = 0f;
                for (int i = 0; i < headDim; i++) sumSq += span[i] * span[i];
                float invRms = 1f / MathF.Sqrt(sumSq / headDim + eps);
                for (int i = 0; i < headDim; i++) span[i] = span[i] * invRms * weight[i];
            }
        }
    }

    /// <summary>Real Qwen3RMSNorm: `x * rsqrt(mean(x^2) + eps) * weight`.</summary>
    public static void RmsNorm(ReadOnlySpan<float> x, float[] weight, Span<float> output, float eps)
    {
        int n = x.Length;
        float sumSq = 0f;
        for (int i = 0; i < n; i++) sumSq += x[i] * x[i];
        float invRms = 1f / MathF.Sqrt(sumSq / n + eps);
        for (int i = 0; i < n; i++) output[i] = x[i] * invRms * weight[i];
    }

    public static void SoftmaxRange(float[] scores, int start, int end)
    {
        float max = float.NegativeInfinity;
        for (int i = start; i < end; i++) if (scores[i] > max) max = scores[i];
        float sum = 0f;
        for (int i = start; i < end; i++)
        {
            float e = MathF.Exp(scores[i] - max);
            scores[i] = e;
            sum += e;
        }
        float invSum = 1f / sum;
        for (int i = start; i < end; i++) scores[i] *= invSum;
    }

    public static float Silu(float x) => x / (1f + MathF.Exp(-x));
}
