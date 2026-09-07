namespace OpenTail.Stingray.Audio.VoxCpm2;

/// <summary>
/// Native C# port of VoxCPM2's local encoder forward pass, from `generator.cpp`'s
/// `VoxCPM2LocalEncoderRuntime::Impl::build`/`encode_patch` and `minicpm_blocks.h`'s
/// `minicpm_layer`/`minicpm_transformer`/`apply_minicpm_rope` (not guessed). Real per-call
/// shape: one "patch" of `patch_size` (real: 4) rows of `feat_dim` (real: 64) continuous
/// features -&gt; `Linear(feat_dim-&gt;encoderHiddenDim)` per row -&gt; a real learned "special token"
/// row PREPENDED (CLS-token pattern) -&gt; a real BIDIRECTIONAL (non-causal) 12-layer MiniCPM-
/// architecture transformer over the resulting 5-row sequence, RoPE positions `0..patchSize` --
/// -&gt; only the special-token row (index 0) is kept -&gt; `Linear(encoderHiddenDim-&gt;lmHiddenDim)`
/// projects into the LM's own hidden space.
///
/// <para><b>Real RoPE, confirmed non-trivial and NOT plain unscaled RoPE</b>: NEOX (split-half)
/// rotation, real `longrope` per-dimension frequency-correction factors (`rope_scaling.
/// short_factor`, an array of 64 real learned/derived values, reused directly from this
/// checkpoint's `lm_config` since `local_transformer_config` inherits the base LM's rope
/// settings verbatim except width/depth/head-count) -- `long_factor` is never selected here
/// because `apply_minicpm_rope`'s real config carries `max_position_embeddings=32768 &lt;=
/// original_max_position_embeddings=32768` (equal, not strictly greater), so
/// `active_rope_factors` always picks `short_factor`. Also confirmed real (not guessed):
/// `rope_attn_factor`'s real formula evaluates to exactly `1.0` for this checkpoint (same
/// `&lt;=` condition), and `ext_factor=0.0`/`freq_scale=1.0` always for this call site, so the
/// reference's more general YaRN ramp-mixing logic never activates here -- this reduces to
/// per-dimension-frequency-corrected RoPE with no additional magnitude scaling, reusing
/// <see cref="SimdKernels.BuildRopeTable"/>'s existing `freqFactors` parameter exactly as
/// designed for this case.</para>
///
/// <para>Real per-layer math (same family as VoxCPM2's `base_lm`, but bidirectional and with its
/// own smaller width/depth): pre-RMSNorm -&gt; fused QKV projections (no bias) -&gt; NEOX+longrope
/// RoPE on Q/K -&gt; GQA key/value head repeat (`numHeads/numKvHeads`) -&gt; full (non-causal)
/// scaled-dot-product attention -&gt; `o_proj` -&gt; residual add (no mup scaling, `use_mup=false`
/// for this checkpoint) -&gt; pre-RMSNorm -&gt; SwiGLU MLP -&gt; residual add -&gt; final RMSNorm after
/// all layers.</para>
/// </summary>
public static class VoxCpm2LocalEncoder
{
    public const int PatchSize = 4;
    public const int FeatDim = 64;
    public const int EncoderHiddenDim = 1024;
    public const int NumHeads = 16;
    public const int NumKvHeads = 2;
    public const int HeadDim = 128;
    public const int FfnDim = 4096;
    public const float RopeTheta = 10000f;
    public const float RmsNormEps = 1e-5f;

    // Real `lm_config.rope_scaling.short_factor` (64 values, one per RoPE pair -- headDim/2),
    // extracted directly from the checkpoint's embedded config.json (not guessed).
    public static readonly float[] RopeShortFactor =
    [
        0.9977997200264581f, 1.014658295992452f, 1.0349680404997148f, 1.059429246056193f,
        1.0888815016813513f, 1.1243301355211495f, 1.166977103606075f, 1.2182568066927284f,
        1.2798772354275727f, 1.3538666751582975f, 1.4426259039919596f, 1.5489853358570191f,
        1.6762658237220625f, 1.8283407612492941f, 2.0096956085876183f, 2.225478927469756f,
        2.481536379650452f, 2.784415934557119f, 3.1413289096347365f, 3.560047844772632f,
        4.048719380066383f, 4.615569542115128f, 5.2684819496549835f, 6.014438591970396f,
        6.858830049237097f, 7.804668263503327f, 8.851768731513417f, 9.99600492938444f,
        11.228766118181639f, 12.536757560834843f, 13.902257701387796f, 15.303885189125953f,
        16.717837610115794f, 18.119465097853947f, 19.484965238406907f, 20.792956681060105f,
        22.02571786985731f, 23.16995406772833f, 24.217054535738416f, 25.16289275000465f,
        26.007284207271347f, 26.753240849586767f, 27.40615325712662f, 27.973003419175363f,
        28.461674954469114f, 28.880393889607006f, 29.237306864684626f, 29.540186419591297f,
        29.79624387177199f, 30.01202719065413f, 30.193382037992453f, 30.34545697551969f,
        30.47273746338473f, 30.579096895249787f, 30.66785612408345f, 30.741845563814174f,
        30.80346599254902f, 30.85474569563567f, 30.897392663720595f, 30.932841297560394f,
        30.962293553185553f, 30.986754758742034f, 31.007064503249293f, 31.02392307921529f,
    ];

    /// <summary>Encodes one real patch of `[PatchSize][FeatDim]` continuous features into the LM's
    /// `lmHiddenDim`-wide embedding space.</summary>
    public static unsafe float[] EncodePatch(VoxCpm2LocalEncoderWeights w, float[][] patchFeatures, int lmHiddenDim)
    {
        if (patchFeatures.Length != PatchSize) throw new ArgumentException($"Expected {PatchSize} rows.", nameof(patchFeatures));

        int seqLen = PatchSize + 1;
        var hidden = new float[seqLen][];
        hidden[0] = (float[])w.SpecialToken.Clone();
        for (int i = 0; i < PatchSize; i++)
            hidden[i + 1] = Linear(patchFeatures[i], w.InProjWeight, w.InProjBias, FeatDim, EncoderHiddenDim);

        int halfDim = HeadDim / 2;
        var cos = new float[seqLen * halfDim];
        var sin = new float[seqLen * halfDim];
        fixed (float* cosPtr = cos, sinPtr = sin, freqPtr = RopeShortFactor)
            SimdKernels.BuildRopeTable(cosPtr, sinPtr, seqLen, HeadDim, RopeTheta, freqPtr);

        foreach (var layer in w.Layers)
            hidden = Layer(hidden, layer, cos, sin, seqLen);

        hidden = RmsNormRows(hidden, w.FinalNorm);

        return Linear(hidden[0], w.EncToLmProjWeight, w.EncToLmProjBias, EncoderHiddenDim, lmHiddenDim);
    }

    private static float[][] Layer(float[][] input, VoxCpm2MiniCpmLayerWeights layer, float[] cos, float[] sin, int seqLen)
    {
        int qOut = NumHeads * HeadDim;
        int kvOut = NumKvHeads * HeadDim;
        int kvRepeats = NumHeads / NumKvHeads;
        float scale = 1f / MathF.Sqrt(HeadDim);
        int halfDim = HeadDim / 2;

        var normed = RmsNormRows(input, layer.InputNorm);
        var q = new float[seqLen][];
        var k = new float[seqLen][];
        var v = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            q[t] = Linear(normed[t], layer.QProjWeight, bias: [], EncoderHiddenDim, qOut);
            k[t] = Linear(normed[t], layer.KProjWeight, bias: [], EncoderHiddenDim, kvOut);
            v[t] = Linear(normed[t], layer.VProjWeight, bias: [], EncoderHiddenDim, kvOut);
            ApplyRopeNeox(q[t], NumHeads, HeadDim, cos, sin, t, halfDim);
            ApplyRopeNeox(k[t], NumKvHeads, HeadDim, cos, sin, t, halfDim);
        }

        var context = new float[seqLen][];
        for (int ti = 0; ti < seqLen; ti++)
        {
            var ctxOut = new float[qOut];
            for (int h = 0; h < NumHeads; h++)
            {
                int hOff = h * HeadDim;
                int kvHOff = (h / kvRepeats) * HeadDim;
                var scores = new float[seqLen];
                for (int tj = 0; tj < seqLen; tj++)
                {
                    float dot = 0f;
                    for (int d = 0; d < HeadDim; d++) dot += q[ti][hOff + d] * k[tj][kvHOff + d];
                    scores[tj] = dot * scale;
                }
                Softmax(scores);
                for (int tj = 0; tj < seqLen; tj++)
                {
                    float p = scores[tj];
                    for (int d = 0; d < HeadDim; d++) ctxOut[hOff + d] += p * v[tj][kvHOff + d];
                }
            }
            context[ti] = ctxOut;
        }

        var attnOut = new float[seqLen][];
        var x = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            attnOut[t] = Linear(context[t], layer.OProjWeight, bias: [], qOut, EncoderHiddenDim);
            x[t] = Add(input[t], attnOut[t]);
        }

        var ffnNormed = RmsNormRows(x, layer.PostNorm);
        var output = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            var gate = Linear(ffnNormed[t], layer.GateProjWeight, bias: [], EncoderHiddenDim, FfnDim);
            SiluInPlace(gate);
            var up = Linear(ffnNormed[t], layer.UpProjWeight, bias: [], EncoderHiddenDim, FfnDim);
            for (int i = 0; i < gate.Length; i++) gate[i] *= up[i];
            var down = Linear(gate, layer.DownProjWeight, bias: [], FfnDim, EncoderHiddenDim);
            output[t] = Add(x[t], down);
        }
        return output;
    }

    private static void ApplyRopeNeox(float[] x, int numHeads, int headDim, float[] cos, float[] sin, int position, int halfDim)
    {
        int cosBase = position * halfDim;
        for (int h = 0; h < numHeads; h++)
        {
            int hOff = h * headDim;
            for (int i = 0; i < halfDim; i++)
            {
                float c = cos[cosBase + i];
                float s = sin[cosBase + i];
                int idx0 = hOff + i;
                int idx1 = hOff + i + halfDim;
                float x0 = x[idx0];
                float x1 = x[idx1];
                x[idx0] = x0 * c - x1 * s;
                x[idx1] = x0 * s + x1 * c;
            }
        }
    }

    private static float[][] RmsNormRows(float[][] rows, float[] weight)
    {
        var output = new float[rows.Length][];
        for (int t = 0; t < rows.Length; t++) output[t] = RmsNorm(rows[t], weight);
        return output;
    }

    private static float[] RmsNorm(float[] x, float[] weight)
    {
        double sumSq = 0;
        for (int i = 0; i < x.Length; i++) sumSq += (double)x[i] * x[i];
        float invRms = (float)(1.0 / Math.Sqrt(sumSq / x.Length + RmsNormEps));
        var output = new float[x.Length];
        for (int i = 0; i < x.Length; i++) output[i] = x[i] * invRms * weight[i];
        return output;
    }

    private static float[] Add(float[] a, float[] b)
    {
        var output = new float[a.Length];
        for (int i = 0; i < a.Length; i++) output[i] = a[i] + b[i];
        return output;
    }

    private static float[] Linear(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        bool hasBias = bias.Length > 0;
        for (int o = 0; o < outDim; o++)
        {
            float sum = hasBias ? bias[o] : 0f;
            int wBase = o * inDim;
            for (int i = 0; i < inDim; i++) sum += weight[wBase + i] * input[i];
            output[o] = sum;
        }
        return output;
    }

    private static void Softmax(float[] scores)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < scores.Length; i++) if (scores[i] > max) max = scores[i];
        float sum = 0f;
        for (int i = 0; i < scores.Length; i++)
        {
            float e = MathF.Exp(scores[i] - max);
            scores[i] = e;
            sum += e;
        }
        float invSum = 1f / sum;
        for (int i = 0; i < scores.Length; i++) scores[i] *= invSum;
    }

    private static void SiluInPlace(float[] x)
    {
        for (int i = 0; i < x.Length; i++)
        {
            float v = x[i];
            x[i] = v / (1f + MathF.Exp(-v));
        }
    }
}
