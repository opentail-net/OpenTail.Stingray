namespace OpenTail.Stingray.Diffusion.StableAudio;

/// <summary>
/// Shared real Stable Audio 3 DiT primitives (partial GPT-J-style RoPE, per-head RMSNorm,
/// bidirectional dot-product attention, the real `ExpoFourierFeatures` timestep/duration
/// embedding, sigmoid) used by BOTH <see cref="StableAudioDiT"/> (Small) and
/// <see cref="StableAudioMediumDiT"/> -- extracted now that Medium is itself real-weight-verified
/// (CLAUDE.md rule 7: DRY once a second real, verified caller of the same exact formula exists;
/// `StableAudioMediumDiT`'s own doc comment flagged this as the deferred next step). `RopeRotDim`
/// (32), `RopeTheta` (10000), `ExpoMinFreq`/`ExpoMaxFreq` (0.5/10000), and `HeadDim` (64) are real,
/// confirmed-identical constants between Small and Medium; only `Heads`/`Dim` genuinely differ
/// between the two real checkpoints, so those are the only real parameters below.
/// </summary>
internal static class StableAudioAttentionKernels
{
    private const int RopeRotDim = 32;
    private const float RopeTheta = 10000f;
    private const float ExpoMinFreq = 0.5f;
    private const float ExpoMaxFreq = 10000f;
    private const int HeadDim = 64;

    /// <summary>Real `ExpoFourierFeatures.forward` (blocks.py): exponentially-spaced (not linear) frequency ramp between min_freq and max_freq, [cos, sin] concatenated.</summary>
    public static float[] ExpoFourierFeatures(float t, int dim)
    {
        int half = dim / 2;
        var outp = new float[dim];
        float logMin = MathF.Log(ExpoMinFreq);
        float logMax = MathF.Log(ExpoMaxFreq);
        for (int i = 0; i < half; i++)
        {
            float ramp = half == 1 ? 0f : (float)i / (half - 1);
            float freq = MathF.Exp(ramp * (logMax - logMin) + logMin);
            float arg = t * freq * 2f * MathF.PI;
            outp[i] = MathF.Cos(arg);
            outp[half + i] = MathF.Sin(arg);
        }
        return outp;
    }

    public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    public static void PerHeadRmsNorm(float[] qkOrV, int seq, int heads, int dim, float[] weight)
    {
        for (int t = 0; t < seq; t++)
        {
            for (int h = 0; h < heads; h++)
            {
                DiffusionOps.RmsNorm(qkOrV.AsSpan(t * dim + h * HeadDim, HeadDim), weight, HeadDim, eps: 1e-6f);
            }
        }
    }

    /// <summary>Real bidirectional (no causal mask) multi-head dot-product attention.</summary>
    public static float[] DotProductAttention(float[] q, float[] k, float[] v, int seqQ, int seqKv, int heads, int dim)
    {
        float scale = 1f / MathF.Sqrt(HeadDim);
        var outp = new float[seqQ * dim];

        for (int h = 0; h < heads; h++)
        {
            var scores = new float[seqQ * seqKv];
            for (int i = 0; i < seqQ; i++)
            {
                int qOff = i * dim + h * HeadDim;
                for (int j = 0; j < seqKv; j++)
                {
                    int kOff = j * dim + h * HeadDim;
                    float dot = 0f;
                    for (int d = 0; d < HeadDim; d++) dot += q[qOff + d] * k[kOff + d];
                    scores[i * seqKv + j] = dot * scale;
                }
            }
            DiffusionOps.Softmax(scores, seqKv);

            for (int i = 0; i < seqQ; i++)
            {
                int outOff = i * dim + h * HeadDim;
                for (int j = 0; j < seqKv; j++)
                {
                    float w = scores[i * seqKv + j];
                    if (w == 0f) continue;
                    int vOff = j * dim + h * HeadDim;
                    for (int d = 0; d < HeadDim; d++) outp[outOff + d] += w * v[vOff + d];
                }
            }
        }
        return outp;
    }

    /// <summary>Real `RotaryEmbedding(dim_heads//2)` + `apply_rotary_pos_emb`'s "partial rotary embeddings, Wang et al. GPT-J" scheme: only the first <see cref="RopeRotDim"/> (32) of each 64-wide head vector are rotated (as two contiguous 16-wide halves, standard split-half rotation), the remaining 32 channels pass through untouched.</summary>
    public static (float[] cos, float[] sin) BuildPartialRope(int seq)
    {
        int half = RopeRotDim / 2;
        var cos = new float[seq * half];
        var sin = new float[seq * half];
        for (int s = 0; s < seq; s++)
        {
            for (int i = 0; i < half; i++)
            {
                float invFreq = MathF.Pow(RopeTheta, -2.0f * i / RopeRotDim);
                float angle = s * invFreq;
                cos[s * half + i] = MathF.Cos(angle);
                sin[s * half + i] = MathF.Sin(angle);
            }
        }
        return (cos, sin);
    }

    public static void ApplyPartialRope(float[] qk, int seq, int heads, int dim, float[] cos, float[] sin)
    {
        int half = RopeRotDim / 2;
        for (int s = 0; s < seq; s++)
        {
            for (int h = 0; h < heads; h++)
            {
                int headOff = s * dim + h * HeadDim;
                for (int i = 0; i < half; i++)
                {
                    float c = cos[s * half + i];
                    float sn = sin[s * half + i];
                    float x1 = qk[headOff + i];
                    float x2 = qk[headOff + half + i];
                    qk[headOff + i] = x1 * c - x2 * sn;
                    qk[headOff + half + i] = x1 * sn + x2 * c;
                }
                // channels [RopeRotDim, HeadDim) are left untouched -- real partial-rotary behavior.
            }
        }
    }
}
