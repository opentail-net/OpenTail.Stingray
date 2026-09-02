namespace OpenTail.Stingray.Diffusion.StableAudio;

/// <summary>
/// Real Stable Audio 3 DiT (`DiffusionTransformer`/`ContinuousTransformer` in the real reference).
/// Every formula here is transcribed directly from the real `stable_audio_3/models/dit.py` and
/// `transformer.py` sources (see docs/057-stable-audio-3-implementation-plan.md's "Real DiT
/// forward-pass spec" section for the full derivation) against the real
/// `stabilityai/stable-audio-3-small-music-base` checkpoint's resolved config: 20 layers, embed
/// dim 1024, 16 heads (head_dim 64), `global_cond_type=adaLN`, `qk_norm=rms`, real RoPE (partial
/// GPT-J-style rotary -- only the first 32 of each head's 64 channels are rotated, NOT the full
/// head_dim), real SwiGLU FFN (mult=4.0), 64 learned memory tokens, no conformer, no modular local
/// conditioning, and no differential attention (all confirmed off by the absence of their tensors
/// in the real checkpoint). Inpainting's `local_add_cond` branch (real tensors exist:
/// `to_local_embed.*`) is deliberately left unwired -- irrelevant for plain text-to-audio
/// generation, same simplification FLUX/LTX shipped with on their first working ports.
/// </summary>
public sealed class StableAudioDiT : IDisposable
{
    private const int IoChannels = 256;
    private const int Dim = 1024;
    private const int Depth = 20;
    private const int Heads = 16;
    private const int HeadDim = 64;
    private const int RopeRotDim = 32; // dim_heads // 2 -- partial rotary, real GPT-J-style
    private const int CondTokenDimRaw = 768;
    private const int GlobalCondDimRaw = 768;
    private const int FfInner = 4096; // mult=4.0 * 1024
    private const int MemoryTokens = 64;
    private const int TimestepFeaturesDim = 256;
    private const float RopeTheta = 10000f;
    private const float ExpoMinFreq = 0.5f;
    private const float ExpoMaxFreq = 10000f;

    private readonly IWeightLoader _st;
    private readonly bool _ownsLoader;

    public StableAudioDiT(string path)
    {
        _st = SafetensorsLoader.Open(path);
        _ownsLoader = true;
    }

    private StableAudioDiT(IWeightLoader loader, bool ownsLoader)
    {
        _st = loader;
        _ownsLoader = ownsLoader;
    }

    /// <summary>Wraps an already-open loader -- caller retains ownership.</summary>
    public static StableAudioDiT FromLoader(IWeightLoader loader) => new(loader, ownsLoader: false);

    /// <summary>
    /// Predicts the rectified-flow velocity for one Euler step.
    /// <paramref name="latent"/>: [seqLen, 256] token-major acoustic latent.
    /// <paramref name="condTokens"/>: [nCond, 768] real cross-attention context -- the real
    /// pipeline concatenates the (padding-substituted) prompt embeddings with the `seconds_total`
    /// NumberConditioner embedding along the sequence axis (`cross_attention_cond_ids: [prompt,
    /// seconds_total]`). There is no `condMask` parameter: the real reference unconditionally
    /// discards any cross-attention padding mask before it ever reaches the attention op (a
    /// permanent workaround for a flash-attention kernel issue, confirmed by reading `dit.py` --
    /// `mask_padding_attention: true` in the real `model_config.json` is misleading, this is NOT
    /// actually applied), so real cross-attention always attends to every row of
    /// <paramref name="condTokens"/> including padding, and this port matches that exactly.
    /// <paramref name="secondsTotalRaw"/>: [768] the raw (pre-`to_global_embed`) `seconds_total`
    /// NumberConditioner embedding -- used again here as the DiT's separate global (AdaLN)
    /// conditioning input, distinct from its row inside <paramref name="condTokens"/>.
    /// </summary>
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

    /// <summary>Real Conv1d(kernel=1, no bias) + residual: with kernel size 1 this is a per-token
    /// dense layer over the channel dim, identical whether the caller's own layout is
    /// channels-first or token-first, so no real conv machinery is needed.</summary>
    private float[] Conv1x1Residual(float[] x, int seqLen, int channels, string weightKey)
    {
        var w = _st.ReadF32(weightKey); // [channels, channels, 1] -> read as [channels, channels]
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

    /// <summary>Real `ExpoFourierFeatures.forward` (blocks.py): exponentially-spaced (not linear)
    /// frequency ramp between min_freq and max_freq, [cos, sin] concatenated.</summary>
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

        var toScaleShiftGate = _st.ReadF32($"{p}.to_scale_shift_gate"); // [6144]
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

    private float[] SelfAttention(float[] x, int seq, string p, float[] cos, float[] sin)
    {
        var qkvW = _st.ReadF32($"{p}.self_attn.to_qkv.weight");
        var qNormW = _st.ReadF32($"{p}.self_attn.q_norm.gamma");
        var kNormW = _st.ReadF32($"{p}.self_attn.k_norm.gamma");
        var outW = _st.ReadF32($"{p}.self_attn.to_out.weight");

        var qkv = DiffusionOps.Linear(x, qkvW, null, seq, Dim, 3 * Dim);
        var q = new float[seq * Dim];
        var k = new float[seq * Dim];
        var v = new float[seq * Dim];
        for (int t = 0; t < seq; t++)
        {
            qkv.AsSpan(t * 3 * Dim, Dim).CopyTo(q.AsSpan(t * Dim, Dim));
            qkv.AsSpan(t * 3 * Dim + Dim, Dim).CopyTo(k.AsSpan(t * Dim, Dim));
            qkv.AsSpan(t * 3 * Dim + 2 * Dim, Dim).CopyTo(v.AsSpan(t * Dim, Dim));
        }

        PerHeadRmsNorm(q, seq, qNormW);
        PerHeadRmsNorm(k, seq, kNormW);

        ApplyPartialRope(q, seq, cos, sin);
        ApplyPartialRope(k, seq, cos, sin);

        var attnOut = DotProductAttention(q, k, v, seq, seq, mask: null);

        return DiffusionOps.Linear(attnOut, outW, null, seq, Dim, Dim);
    }

    private float[] CrossAttention(float[] x, int seq, float[] condEmbed, int nCond, string p)
    {
        var qW = _st.ReadF32($"{p}.cross_attn.to_q.weight");
        var kvW = _st.ReadF32($"{p}.cross_attn.to_kv.weight");
        var qNormW = _st.ReadF32($"{p}.cross_attn.q_norm.gamma");
        var kNormW = _st.ReadF32($"{p}.cross_attn.k_norm.gamma");
        var outW = _st.ReadF32($"{p}.cross_attn.to_out.weight");

        var q = DiffusionOps.Linear(x, qW, null, seq, Dim, Dim);
        var kv = DiffusionOps.Linear(condEmbed, kvW, null, nCond, Dim, 2 * Dim);
        var k = new float[nCond * Dim];
        var v = new float[nCond * Dim];
        for (int t = 0; t < nCond; t++)
        {
            kv.AsSpan(t * 2 * Dim, Dim).CopyTo(k.AsSpan(t * Dim, Dim));
            kv.AsSpan(t * 2 * Dim + Dim, Dim).CopyTo(v.AsSpan(t * Dim, Dim));
        }

        PerHeadRmsNorm(q, seq, qNormW);
        PerHeadRmsNorm(k, nCond, kNormW);

        // Real reference behavior, confirmed by reading dit.py line-by-line (not the "V-zeroing"
        // this class originally implemented, which was a plausible-looking but wrong guess): the
        // real forward() UNCONDITIONALLY discards any cross_attn_cond_mask right after receiving it
        // (`cross_attn_cond_mask = None  # Temporarily disabling conditioning masks due to kernel
        // issue for flash attention`), on every code path including the CFG branch -- so real
        // cross-attention in this shipped checkpoint never masks padded context rows at all, despite
        // `mask_padding_attention: true` in the real model_config.json suggesting otherwise. No
        // masking is applied here to match that real (if surprising) behavior exactly.

        var attnOut = DotProductAttention(q, k, v, seq, nCond, mask: null);

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

    private static float[] DotProductAttention(float[] q, float[] k, float[] v, int seqQ, int seqKv, bool[]? mask)
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

    /// <summary>Real `RotaryEmbedding(dim_heads//2)` + `apply_rotary_pos_emb`'s "partial rotary
    /// embeddings, Wang et al. GPT-J" scheme: only the first <see cref="RopeRotDim"/> (32) of each
    /// 64-wide head vector are rotated (as two contiguous 16-wide halves, standard split-half
    /// rotation), the remaining 32 channels pass through untouched. This is a materially different
    /// width than every other RoPE user in this project (which rotate the FULL head_dim), so it is
    /// NOT implemented via the shared <c>Primitives.SplitHalfRoPE</c> helper.</summary>
    private static (float[] cos, float[] sin) BuildPartialRope(int seq)
    {
        int half = RopeRotDim / 2; // 16
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
        int half = RopeRotDim / 2; // 16
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
                // channels [RopeRotDim, HeadDim) are left untouched -- real partial-rotary behavior.
            }
        }
    }

    public void Dispose()
    {
        if (_ownsLoader) _st.Dispose();
    }
}
