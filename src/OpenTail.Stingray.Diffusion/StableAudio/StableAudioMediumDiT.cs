namespace OpenTail.Stingray.Diffusion.StableAudio;

/// <summary>
/// Real Stable Audio 3 Medium DiT. Structurally the same real `dit.py`/`transformer.py` source as
/// <see cref="StableAudioDiT"/> (Small), but with real, checkpoint-confirmed config differences --
/// see docs/057-stable-audio-3-implementation-plan.md's "Medium — real archaeology" section: 24
/// layers, embed dim 1536, 24 heads (head_dim still 64), and real DIFFERENTIAL attention on BOTH
/// self- and cross-attention (`attn_kwargs.differential=true`, confirmed absent for Small).
///
/// <para><b>Differential attention -- re-confirmed against the real, up-to-date GitHub `main`
/// source</b> (the installed PyPI `stable-audio-tools` release turned out to be stale for several
/// unrelated real config fields -- see docs/057's "CONFIDENCE RESTORED" section -- but `main`'s
/// `Attention.forward` differential branch is byte-identical to what this class already
/// implements): widen QKV/QK-V to add a second `(q_diff, k_diff)` pair sharing the SAME `v`,
/// `qk_norm`/RoPE applied identically to both pairs (stacked, elementwise over head_dim), two full
/// attention passes, `out = out_main - out_diff`, no learnable mixing. The tensor-shape widening
/// factors (5x self-attn, 2x/3x cross-attn Q/KV) are also independently confirmed directly against
/// this checkpoint's own safetensors header.</para>
///
/// <para><b>Duplicated from `StableAudioDiT` rather than refactored in place</b>, deliberately: that
/// class is already golden-verified for Small and used in production; converting its `const` fields
/// to configurable instance state to serve two real variants is exactly the kind of DRY pass
/// CLAUDE.md rule 7 says to defer until there is a second real, verified caller to unify against --
/// this class IS that second caller, once IT is verified, unifying both into one config-driven
/// class becomes a real (not speculative) next step, not done here to avoid regressing the
/// already-shipped Small path in the same change that adds Medium.</para>
/// </summary>
public sealed class StableAudioMediumDiT : IDisposable
{
    private const int IoChannels = 256;
    private const int Dim = 1536;
    private const int Depth = 24;
    private const int Heads = 24;
    private const int HeadDim = 64;
    private const int RopeRotDim = 32; // dim_heads // 2 -- partial rotary, real GPT-J-style, head_dim unchanged from Small
    private const int CondTokenDimRaw = 768; // T5Gemma hidden size, unchanged (identical text encoder checkpoint)
    private const int GlobalCondDimRaw = 768;
    private const int FfInner = 6144; // mult=4.0 * 1536
    private const int MemoryTokens = 64;
    private const int TimestepFeaturesDim = 256;
    private const float RopeTheta = 10000f;
    private const float ExpoMinFreq = 0.5f;
    private const float ExpoMaxFreq = 10000f;

    private readonly IWeightLoader _st;
    private readonly bool _ownsLoader;

    public StableAudioMediumDiT(string path)
    {
        _st = SafetensorsLoader.Open(path);
        _ownsLoader = true;
    }

    private StableAudioMediumDiT(IWeightLoader loader, bool ownsLoader)
    {
        _st = loader;
        _ownsLoader = ownsLoader;
    }

    /// <summary>Wraps an already-open loader -- caller retains ownership.</summary>
    public static StableAudioMediumDiT FromLoader(IWeightLoader loader) => new(loader, ownsLoader: false);

    /// <summary>Predicts the rectified-flow velocity for one Euler step. Same real semantics as
    /// <see cref="StableAudioDiT.Forward"/> (see its doc comment) -- only the config and the
    /// differential self-/cross-attention math differ.</summary>
    public float[] Forward(
        float[] latent, int seqLen,
        float[] condTokens, int nCond,
        float[] secondsTotalRaw,
        float timestep)
    {
        var condEmbed = ToCondEmbed(condTokens, nCond);

        var globalEmbed = ToGlobalEmbed(secondsTotalRaw);
        var timestepEmbed = ToTimestepEmbed(timestep);
        for (int i = 0; i < Dim; i++) globalEmbed[i] += timestepEmbed[i];

        var latentPre = Conv1x1Residual(latent, seqLen, IoChannels, "model.model.preprocess_conv.weight");

        var projInW = _st.ReadF32("model.model.transformer.project_in.weight");
        var x = DiffusionOps.Linear(latentPre, projInW, null, seqLen, IoChannels, Dim);

        int totalSeq = MemoryTokens + seqLen;
        var xFull = new float[totalSeq * Dim];
        var memTokens = _st.ReadF32("model.model.transformer.memory_tokens");
        memTokens.AsSpan().CopyTo(xFull.AsSpan(0, MemoryTokens * Dim));
        x.AsSpan().CopyTo(xFull.AsSpan(MemoryTokens * Dim, seqLen * Dim));

        var (cos, sin) = BuildPartialRope(totalSeq);

        var globalCond = GlobalCondEmbedder(globalEmbed);

        for (int layer = 0; layer < Depth; layer++)
        {
            xFull = TransformerLayer(xFull, totalSeq, layer, condEmbed, nCond, cos, sin, globalCond);
        }

        var stripped = new float[seqLen * Dim];
        xFull.AsSpan(MemoryTokens * Dim, seqLen * Dim).CopyTo(stripped);

        var projOutW = _st.ReadF32("model.model.transformer.project_out.weight");
        var outLow = DiffusionOps.Linear(stripped, projOutW, null, seqLen, Dim, IoChannels);

        return Conv1x1Residual(outLow, seqLen, IoChannels, "model.model.postprocess_conv.weight");
    }

    private float[] Conv1x1Residual(float[] x, int seqLen, int channels, string weightKey)
    {
        var w = _st.ReadF32(weightKey);
        var y = DiffusionOps.Linear(x, w, null, seqLen, channels, channels);
        for (int i = 0; i < y.Length; i++) y[i] += x[i];
        return y;
    }

    private float[] ToCondEmbed(float[] condTokens, int nCond)
    {
        var w0 = _st.ReadF32("model.model.to_cond_embed.0.weight");
        var w2 = _st.ReadF32("model.model.to_cond_embed.2.weight");
        var h = DiffusionOps.Linear(condTokens, w0, null, nCond, CondTokenDimRaw, Dim);
        DiffusionOps.SiluInPlace(h);
        return DiffusionOps.Linear(h, w2, null, nCond, Dim, Dim);
    }

    private float[] ToGlobalEmbed(float[] secondsTotalRaw)
    {
        var w0 = _st.ReadF32("model.model.to_global_embed.0.weight");
        var w2 = _st.ReadF32("model.model.to_global_embed.2.weight");
        var h = DiffusionOps.Linear(secondsTotalRaw, w0, null, 1, GlobalCondDimRaw, Dim);
        DiffusionOps.SiluInPlace(h);
        return DiffusionOps.Linear(h, w2, null, 1, Dim, Dim);
    }

    private float[] ToTimestepEmbed(float timestep)
    {
        var feats = ExpoFourierFeatures(timestep, TimestepFeaturesDim);
        var w0 = _st.ReadF32("model.model.to_timestep_embed.0.weight");
        var b0 = _st.ReadF32("model.model.to_timestep_embed.0.bias");
        var w2 = _st.ReadF32("model.model.to_timestep_embed.2.weight");
        var b2 = _st.ReadF32("model.model.to_timestep_embed.2.bias");
        var h = DiffusionOps.Linear(feats, w0, b0, 1, TimestepFeaturesDim, Dim);
        DiffusionOps.SiluInPlace(h);
        return DiffusionOps.Linear(h, w2, b2, 1, Dim, Dim);
    }

    private static float[] ExpoFourierFeatures(float t, int dim)
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

    private float[] GlobalCondEmbedder(float[] globalEmbed)
    {
        var w0 = _st.ReadF32("model.model.transformer.global_cond_embedder.0.weight");
        var b0 = _st.ReadF32("model.model.transformer.global_cond_embedder.0.bias");
        var w2 = _st.ReadF32("model.model.transformer.global_cond_embedder.2.weight");
        var b2 = _st.ReadF32("model.model.transformer.global_cond_embedder.2.bias");
        var h = DiffusionOps.Linear(globalEmbed, w0, b0, 1, Dim, Dim);
        DiffusionOps.SiluInPlace(h);
        return DiffusionOps.Linear(h, w2, b2, 1, Dim, 6 * Dim);
    }

    private float[] TransformerLayer(
        float[] x, int seq, int layerIdx,
        float[] condEmbed, int nCond,
        float[] cos, float[] sin, float[] globalCond)
    {
        string p = $"model.model.transformer.layers.{layerIdx}";

        var toScaleShiftGate = _st.ReadF32($"{p}.to_scale_shift_gate");
        var gates = new float[6 * Dim];
        for (int i = 0; i < gates.Length; i++) gates[i] = toScaleShiftGate[i] + globalCond[i];
        var scaleSelf = gates.AsSpan(0, Dim);
        var shiftSelf = gates.AsSpan(Dim, Dim);
        var gateSelf = gates.AsSpan(2 * Dim, Dim);
        var scaleFf = gates.AsSpan(3 * Dim, Dim);
        var shiftFf = gates.AsSpan(4 * Dim, Dim);
        var gateFf = gates.AsSpan(5 * Dim, Dim);

        var preNormW = _st.ReadF32($"{p}.pre_norm.gamma");
        var xNorm = x.ToArray();
        DiffusionOps.RmsNorm(xNorm, preNormW, Dim, eps: 1e-5f);
        for (int t = 0; t < seq; t++)
        {
            var row = xNorm.AsSpan(t * Dim, Dim);
            for (int i = 0; i < Dim; i++) row[i] = row[i] * (1f + scaleSelf[i]) + shiftSelf[i];
        }

        var attn = SelfAttention(xNorm, seq, p, cos, sin);
        for (int t = 0; t < seq; t++)
        {
            var row = attn.AsSpan(t * Dim, Dim);
            for (int i = 0; i < Dim; i++) row[i] *= Sigmoid(1f - gateSelf[i]);
        }
        for (int i = 0; i < x.Length; i++) x[i] += attn[i];

        var crossNormW = _st.ReadF32($"{p}.cross_attend_norm.gamma");
        var xCrossNorm = x.ToArray();
        DiffusionOps.RmsNorm(xCrossNorm, crossNormW, Dim, eps: 1e-5f);
        var cross = CrossAttention(xCrossNorm, seq, condEmbed, nCond, p);
        for (int i = 0; i < x.Length; i++) x[i] += cross[i];

        var ffNormW = _st.ReadF32($"{p}.ff_norm.gamma");
        var xFfNorm = x.ToArray();
        DiffusionOps.RmsNorm(xFfNorm, ffNormW, Dim, eps: 1e-5f);
        for (int t = 0; t < seq; t++)
        {
            var row = xFfNorm.AsSpan(t * Dim, Dim);
            for (int i = 0; i < Dim; i++) row[i] = row[i] * (1f + scaleFf[i]) + shiftFf[i];
        }

        var ff = FeedForward(xFfNorm, seq, p);
        for (int t = 0; t < seq; t++)
        {
            var row = ff.AsSpan(t * Dim, Dim);
            for (int i = 0; i < Dim; i++) row[i] *= Sigmoid(1f - gateFf[i]);
        }
        for (int i = 0; i < x.Length; i++) x[i] += ff[i];

        return x;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    /// <summary>Real DIFFERENTIAL self-attention: `to_qkv` widens to `5*Dim` (`q,k,v,q_diff,k_diff`,
    /// real chunk order confirmed from source). `qk_norm`/RoPE apply identically to the main and
    /// diff pairs (elementwise over head_dim). Two full attention passes share the SAME `v`; final
    /// output is `out_main - out_diff`, no learnable mixing.</summary>
    private float[] SelfAttention(float[] x, int seq, string p, float[] cos, float[] sin)
    {
        var qkvW = _st.ReadF32($"{p}.self_attn.to_qkv.weight");
        var qNormW = _st.ReadF32($"{p}.self_attn.q_norm.gamma");
        var kNormW = _st.ReadF32($"{p}.self_attn.k_norm.gamma");
        var outW = _st.ReadF32($"{p}.self_attn.to_out.weight");

        var qkv = DiffusionOps.Linear(x, qkvW, null, seq, Dim, 5 * Dim);
        var q = new float[seq * Dim];
        var k = new float[seq * Dim];
        var v = new float[seq * Dim];
        var qDiff = new float[seq * Dim];
        var kDiff = new float[seq * Dim];
        for (int t = 0; t < seq; t++)
        {
            int b = t * 5 * Dim;
            qkv.AsSpan(b, Dim).CopyTo(q.AsSpan(t * Dim, Dim));
            qkv.AsSpan(b + Dim, Dim).CopyTo(k.AsSpan(t * Dim, Dim));
            qkv.AsSpan(b + 2 * Dim, Dim).CopyTo(v.AsSpan(t * Dim, Dim));
            qkv.AsSpan(b + 3 * Dim, Dim).CopyTo(qDiff.AsSpan(t * Dim, Dim));
            qkv.AsSpan(b + 4 * Dim, Dim).CopyTo(kDiff.AsSpan(t * Dim, Dim));
        }

        PerHeadRmsNorm(q, seq, qNormW);
        PerHeadRmsNorm(k, seq, kNormW);
        PerHeadRmsNorm(qDiff, seq, qNormW);
        PerHeadRmsNorm(kDiff, seq, kNormW);

        ApplyPartialRope(q, seq, cos, sin);
        ApplyPartialRope(k, seq, cos, sin);
        ApplyPartialRope(qDiff, seq, cos, sin);
        ApplyPartialRope(kDiff, seq, cos, sin);

        var attnMain = DotProductAttention(q, k, v, seq, seq);
        var attnDiff = DotProductAttention(qDiff, kDiff, v, seq, seq);
        var attnOut = new float[attnMain.Length];
        for (int i = 0; i < attnOut.Length; i++) attnOut[i] = attnMain[i] - attnDiff[i];

        return DiffusionOps.Linear(attnOut, outW, null, seq, Dim, Dim);
    }

    /// <summary>Real DIFFERENTIAL cross-attention: `to_q` widens to `2*Dim` (`q,q_diff`), `to_kv`
    /// widens to `3*Dim` (`k,k_diff,v` -- real chunk order confirmed from source, `v` shared by both
    /// passes). No RoPE (matches Small's cross-attention). Same "real padding mask is always
    /// discarded" behavior as <see cref="StableAudioDiT.CrossAttention"/> -- see that method's doc
    /// comment for the real source citation.</summary>
    private float[] CrossAttention(float[] x, int seq, float[] condEmbed, int nCond, string p)
    {
        var qW = _st.ReadF32($"{p}.cross_attn.to_q.weight");
        var kvW = _st.ReadF32($"{p}.cross_attn.to_kv.weight");
        var qNormW = _st.ReadF32($"{p}.cross_attn.q_norm.gamma");
        var kNormW = _st.ReadF32($"{p}.cross_attn.k_norm.gamma");
        var outW = _st.ReadF32($"{p}.cross_attn.to_out.weight");

        var qBoth = DiffusionOps.Linear(x, qW, null, seq, Dim, 2 * Dim);
        var q = new float[seq * Dim];
        var qDiff = new float[seq * Dim];
        for (int t = 0; t < seq; t++)
        {
            qBoth.AsSpan(t * 2 * Dim, Dim).CopyTo(q.AsSpan(t * Dim, Dim));
            qBoth.AsSpan(t * 2 * Dim + Dim, Dim).CopyTo(qDiff.AsSpan(t * Dim, Dim));
        }

        var kv = DiffusionOps.Linear(condEmbed, kvW, null, nCond, Dim, 3 * Dim);
        var k = new float[nCond * Dim];
        var kDiff = new float[nCond * Dim];
        var v = new float[nCond * Dim];
        for (int t = 0; t < nCond; t++)
        {
            int b = t * 3 * Dim;
            kv.AsSpan(b, Dim).CopyTo(k.AsSpan(t * Dim, Dim));
            kv.AsSpan(b + Dim, Dim).CopyTo(kDiff.AsSpan(t * Dim, Dim));
            kv.AsSpan(b + 2 * Dim, Dim).CopyTo(v.AsSpan(t * Dim, Dim));
        }

        PerHeadRmsNorm(q, seq, qNormW);
        PerHeadRmsNorm(qDiff, seq, qNormW);
        PerHeadRmsNorm(k, nCond, kNormW);
        PerHeadRmsNorm(kDiff, nCond, kNormW);

        var attnMain = DotProductAttention(q, k, v, seq, nCond);
        var attnDiff = DotProductAttention(qDiff, kDiff, v, seq, nCond);
        var attnOut = new float[attnMain.Length];
        for (int i = 0; i < attnOut.Length; i++) attnOut[i] = attnMain[i] - attnDiff[i];

        return DiffusionOps.Linear(attnOut, outW, null, seq, Dim, Dim);
    }

    private static void PerHeadRmsNorm(float[] qkOrV, int seq, float[] weight)
    {
        for (int t = 0; t < seq; t++)
        {
            for (int h = 0; h < Heads; h++)
            {
                DiffusionOps.RmsNorm(qkOrV.AsSpan(t * Dim + h * HeadDim, HeadDim), weight, HeadDim, eps: 1e-6f);
            }
        }
    }

    private static float[] DotProductAttention(float[] q, float[] k, float[] v, int seqQ, int seqKv)
    {
        float scale = 1f / MathF.Sqrt(HeadDim);
        var outp = new float[seqQ * Dim];

        for (int h = 0; h < Heads; h++)
        {
            var scores = new float[seqQ * seqKv];
            for (int i = 0; i < seqQ; i++)
            {
                int qOff = i * Dim + h * HeadDim;
                for (int j = 0; j < seqKv; j++)
                {
                    int kOff = j * Dim + h * HeadDim;
                    float dot = 0f;
                    for (int d = 0; d < HeadDim; d++) dot += q[qOff + d] * k[kOff + d];
                    scores[i * seqKv + j] = dot * scale;
                }
            }
            DiffusionOps.Softmax(scores, seqKv);

            for (int i = 0; i < seqQ; i++)
            {
                int outOff = i * Dim + h * HeadDim;
                for (int j = 0; j < seqKv; j++)
                {
                    float w = scores[i * seqKv + j];
                    if (w == 0f) continue;
                    int vOff = j * Dim + h * HeadDim;
                    for (int d = 0; d < HeadDim; d++) outp[outOff + d] += w * v[vOff + d];
                }
            }
        }
        return outp;
    }

    private float[] FeedForward(float[] x, int seq, string p)
    {
        var w0 = _st.ReadF32($"{p}.ff.ff.0.proj.weight");
        var b0 = _st.ReadF32($"{p}.ff.ff.0.proj.bias");
        var w2 = _st.ReadF32($"{p}.ff.ff.2.weight");
        var b2 = _st.ReadF32($"{p}.ff.ff.2.bias");

        var proj = DiffusionOps.Linear(x, w0, b0, seq, Dim, 2 * FfInner);
        var h = new float[seq * FfInner];
        for (int t = 0; t < seq; t++)
        {
            var val = proj.AsSpan(t * 2 * FfInner, FfInner);
            var gate = proj.AsSpan(t * 2 * FfInner + FfInner, FfInner);
            var dst = h.AsSpan(t * FfInner, FfInner);
            for (int i = 0; i < FfInner; i++) dst[i] = val[i] * DiffusionOps.Silu(gate[i]);
        }

        return DiffusionOps.Linear(h, w2, b2, seq, FfInner, Dim);
    }

    private static (float[] cos, float[] sin) BuildPartialRope(int seq)
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

    private static void ApplyPartialRope(float[] qk, int seq, float[] cos, float[] sin)
    {
        int half = RopeRotDim / 2;
        for (int s = 0; s < seq; s++)
        {
            for (int h = 0; h < Heads; h++)
            {
                int headOff = s * Dim + h * HeadDim;
                for (int i = 0; i < half; i++)
                {
                    float c = cos[s * half + i];
                    float sn = sin[s * half + i];
                    float x1 = qk[headOff + i];
                    float x2 = qk[headOff + half + i];
                    qk[headOff + i] = x1 * c - x2 * sn;
                    qk[headOff + half + i] = x1 * sn + x2 * c;
                }
            }
        }
    }

    public void Dispose()
    {
        if (_ownsLoader) _st.Dispose();
    }
}
