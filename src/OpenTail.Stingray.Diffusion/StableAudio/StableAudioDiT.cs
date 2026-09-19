using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

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
    private readonly QuantizedWeightCache _quantizedCache;
    private readonly CachedWeightReader _reader;
    private readonly bool _ownsLoader;
    private readonly Dictionary<float[], float[]> _condEmbedCache = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<float[], float[]> _globalEmbedCache = new(ReferenceEqualityComparer.Instance);
    private StableAudioGpuWeights? _gpuWeights;

    public StableAudioDiT(string path)
    {
        _st = SafetensorsLoader.Open(path);
        _quantizedCache = new QuantizedWeightCache(_st);
        _reader = new CachedWeightReader(_st, "");
        _ownsLoader = true;
    }

    private StableAudioDiT(IWeightLoader loader, bool ownsLoader)
    {
        _st = loader;
        _quantizedCache = new QuantizedWeightCache(_st);
        _reader = new CachedWeightReader(_st, "");
        _ownsLoader = ownsLoader;
    }

    private float[] ReadWeight(string name) => _reader.Get(name);

    private float[] Linear(string weightName, string? biasName, ReadOnlySpan<float> x, int rows, int inDim, int outDim)
    {
        var outF = new float[rows * outDim];
        var bias = biasName is not null ? ReadWeight(biasName) : null;
        _quantizedCache.Linear(weightName, x, bias ?? ReadOnlySpan<float>.Empty, outF.AsSpan(), rows, inDim, outDim);
        return outF;
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
        if (!_condEmbedCache.TryGetValue(condTokens, out var condEmbed))
        {
            condEmbed = ToCondEmbed(condTokens, nCond);
            _condEmbedCache[condTokens] = condEmbed;
        }

        if (!_globalEmbedCache.TryGetValue(secondsTotalRaw, out var globalEmbedBase))
        {
            globalEmbedBase = ToGlobalEmbed(secondsTotalRaw);
            _globalEmbedCache[secondsTotalRaw] = globalEmbedBase;
        }

        var globalEmbed = new float[Dim];
        globalEmbedBase.AsSpan().CopyTo(globalEmbed);
        var timestepEmbed = ToTimestepEmbed(timestep);
        for (int i = 0; i < Dim; i++) globalEmbed[i] += timestepEmbed[i];

        var latentPre = Conv1x1Residual(latent, seqLen, IoChannels, "model.model.preprocess_conv.weight");

        var x = Linear("model.model.transformer.project_in.weight", null, latentPre, seqLen, IoChannels, Dim);

        int totalSeq = MemoryTokens + seqLen;
        var xFull = new float[totalSeq * Dim];
        var memTokens = ReadWeight("model.model.transformer.memory_tokens");
        memTokens.AsSpan().CopyTo(xFull.AsSpan(0, MemoryTokens * Dim));
        x.AsSpan().CopyTo(xFull.AsSpan(MemoryTokens * Dim, seqLen * Dim));

        var (cos, sin) = StableAudioAttentionKernels.BuildPartialRope(totalSeq);

        var globalCond = GlobalCondEmbedder(globalEmbed);

        for (int layer = 0; layer < Depth; layer++)
        {
            xFull = TransformerLayer(xFull, totalSeq, layer, condEmbed, nCond, cos, sin, globalCond);
        }

        var stripped = new float[seqLen * Dim];
        xFull.AsSpan(MemoryTokens * Dim, seqLen * Dim).CopyTo(stripped);

        var outLow = Linear("model.model.transformer.project_out.weight", null, stripped, seqLen, Dim, IoChannels);

        return Conv1x1Residual(outLow, seqLen, IoChannels, "model.model.postprocess_conv.weight");
    }

    /// <summary>Real Conv1d(kernel=1, no bias) + residual: with kernel size 1 this is a per-token
    /// dense layer over the channel dim, identical whether the caller's own layout is
    /// channels-first or token-first, so no real conv machinery is needed.</summary>
    private float[] Conv1x1Residual(float[] x, int seqLen, int channels, string weightKey)
    {
        var y = Linear(weightKey, null, x, seqLen, channels, channels);
        for (int i = 0; i < y.Length; i++) y[i] += x[i];
        return y;
    }

    private float[] ToCondEmbed(float[] condTokens, int nCond)
    {
        var h = Linear("model.model.to_cond_embed.0.weight", null, condTokens, nCond, CondTokenDimRaw, Dim);
        DiffusionOps.SiluInPlace(h);
        return Linear("model.model.to_cond_embed.2.weight", null, h, nCond, Dim, Dim);
    }

    private float[] ToGlobalEmbed(float[] secondsTotalRaw)
    {
        var h = Linear("model.model.to_global_embed.0.weight", null, secondsTotalRaw, 1, GlobalCondDimRaw, Dim);
        DiffusionOps.SiluInPlace(h);
        return Linear("model.model.to_global_embed.2.weight", null, h, 1, Dim, Dim);
    }

    private float[] ToTimestepEmbed(float timestep)
    {
        var feats = StableAudioAttentionKernels.ExpoFourierFeatures(timestep, TimestepFeaturesDim);
        var h = Linear("model.model.to_timestep_embed.0.weight", "model.model.to_timestep_embed.0.bias", feats, 1, TimestepFeaturesDim, Dim);
        DiffusionOps.SiluInPlace(h);
        return Linear("model.model.to_timestep_embed.2.weight", "model.model.to_timestep_embed.2.bias", h, 1, Dim, Dim);
    }

    private float[] GlobalCondEmbedder(float[] globalEmbed)
    {
        var h = Linear("model.model.transformer.global_cond_embedder.0.weight", "model.model.transformer.global_cond_embedder.0.bias", globalEmbed, 1, Dim, Dim);
        DiffusionOps.SiluInPlace(h);
        return Linear("model.model.transformer.global_cond_embedder.2.weight", "model.model.transformer.global_cond_embedder.2.bias", h, 1, Dim, 6 * Dim);
    }

    private float[] TransformerLayer(
        float[] x, int seq, int layerIdx,
        float[] condEmbed, int nCond,
        float[] cos, float[] sin, float[] globalCond)
    {
        string p = $"model.model.transformer.layers.{layerIdx}";

        var toScaleShiftGate = ReadWeight($"{p}.to_scale_shift_gate"); // [6144]
        var gates = new float[6 * Dim];
        for (int i = 0; i < gates.Length; i++) gates[i] = toScaleShiftGate[i] + globalCond[i];
        var scaleSelf = gates.AsSpan(0, Dim);
        var shiftSelf = gates.AsSpan(Dim, Dim);
        var gateSelf = gates.AsSpan(2 * Dim, Dim);
        var scaleFf = gates.AsSpan(3 * Dim, Dim);
        var shiftFf = gates.AsSpan(4 * Dim, Dim);
        var gateFf = gates.AsSpan(5 * Dim, Dim);

        var preNormW = ReadWeight($"{p}.pre_norm.gamma");
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
            for (int i = 0; i < Dim; i++) row[i] *= StableAudioAttentionKernels.Sigmoid(1f - gateSelf[i]);
        }
        for (int i = 0; i < x.Length; i++) x[i] += attn[i];

        var crossNormW = ReadWeight($"{p}.cross_attend_norm.gamma");
        var xCrossNorm = x.ToArray();
        DiffusionOps.RmsNorm(xCrossNorm, crossNormW, Dim, eps: 1e-5f);
        var cross = CrossAttention(xCrossNorm, seq, condEmbed, nCond, p);
        for (int i = 0; i < x.Length; i++) x[i] += cross[i];

        var ffNormW = ReadWeight($"{p}.ff_norm.gamma");
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
            for (int i = 0; i < Dim; i++) row[i] *= StableAudioAttentionKernels.Sigmoid(1f - gateFf[i]);
        }
        for (int i = 0; i < x.Length; i++) x[i] += ff[i];

        return x;
    }

    private float[] SelfAttention(float[] x, int seq, string p, float[] cos, float[] sin)
    {
        var qNormW = ReadWeight($"{p}.self_attn.q_norm.gamma");
        var kNormW = ReadWeight($"{p}.self_attn.k_norm.gamma");

        var qkv = Linear($"{p}.self_attn.to_qkv.weight", null, x, seq, Dim, 3 * Dim);
        var q = new float[seq * Dim];
        var k = new float[seq * Dim];
        var v = new float[seq * Dim];
        for (int t = 0; t < seq; t++)
        {
            qkv.AsSpan(t * 3 * Dim, Dim).CopyTo(q.AsSpan(t * Dim, Dim));
            qkv.AsSpan(t * 3 * Dim + Dim, Dim).CopyTo(k.AsSpan(t * Dim, Dim));
            qkv.AsSpan(t * 3 * Dim + 2 * Dim, Dim).CopyTo(v.AsSpan(t * Dim, Dim));
        }

        StableAudioAttentionKernels.PerHeadRmsNorm(q, seq, Heads, Dim, qNormW);
        StableAudioAttentionKernels.PerHeadRmsNorm(k, seq, Heads, Dim, kNormW);

        StableAudioAttentionKernels.ApplyPartialRope(q, seq, Heads, Dim, cos, sin);
        StableAudioAttentionKernels.ApplyPartialRope(k, seq, Heads, Dim, cos, sin);

        var attnOut = StableAudioAttentionKernels.DotProductAttention(q, k, v, seq, seq, Heads, Dim);

        return Linear($"{p}.self_attn.to_out.weight", null, attnOut, seq, Dim, Dim);
    }

    private float[] CrossAttention(float[] x, int seq, float[] condEmbed, int nCond, string p)
    {
        var qNormW = ReadWeight($"{p}.cross_attn.q_norm.gamma");
        var kNormW = ReadWeight($"{p}.cross_attn.k_norm.gamma");

        var q = Linear($"{p}.cross_attn.to_q.weight", null, x, seq, Dim, Dim);
        var kv = Linear($"{p}.cross_attn.to_kv.weight", null, condEmbed, nCond, Dim, 2 * Dim);
        var k = new float[nCond * Dim];
        var v = new float[nCond * Dim];
        for (int t = 0; t < nCond; t++)
        {
            kv.AsSpan(t * 2 * Dim, Dim).CopyTo(k.AsSpan(t * Dim, Dim));
            kv.AsSpan(t * 2 * Dim + Dim, Dim).CopyTo(v.AsSpan(t * Dim, Dim));
        }

        StableAudioAttentionKernels.PerHeadRmsNorm(q, seq, Heads, Dim, qNormW);
        StableAudioAttentionKernels.PerHeadRmsNorm(k, nCond, Heads, Dim, kNormW);

        var attnOut = StableAudioAttentionKernels.DotProductAttention(q, k, v, seq, nCond, Heads, Dim);

        return Linear($"{p}.cross_attn.to_out.weight", null, attnOut, seq, Dim, Dim);
    }

    private float[] FeedForward(float[] x, int seq, string p)
    {
        var proj = Linear($"{p}.ff.ff.0.proj.weight", $"{p}.ff.ff.0.proj.bias", x, seq, Dim, 2 * FfInner);
        var h = new float[seq * FfInner];
        for (int t = 0; t < seq; t++)
        {
            var val = proj.AsSpan(t * 2 * FfInner, FfInner);
            var gate = proj.AsSpan(t * 2 * FfInner + FfInner, FfInner);
            var dst = h.AsSpan(t * FfInner, FfInner);
            for (int i = 0; i < FfInner; i++) dst[i] = val[i] * DiffusionOps.Silu(gate[i]);
        }

        return Linear($"{p}.ff.ff.2.weight", $"{p}.ff.ff.2.bias", h, seq, FfInner, Dim);
    }

    public (float[] k, float[] v)[] PrecomputeConditioning(float[] condTokens, int nCond)
    {
        var condEmbed = ToCondEmbed(condTokens, nCond);
        var kvCache = new (float[] k, float[] v)[Depth];
        for (int layer = 0; layer < Depth; layer++)
        {
            string p = $"model.model.transformer.layers.{layer}";
            var kv = Linear($"{p}.cross_attn.to_kv.weight", null, condEmbed, nCond, Dim, 2 * Dim);
            var k = new float[nCond * Dim];
            var v = new float[nCond * Dim];
            for (int t = 0; t < nCond; t++)
            {
                kv.AsSpan(t * 2 * Dim, Dim).CopyTo(k.AsSpan(t * Dim, Dim));
                kv.AsSpan(t * 2 * Dim + Dim, Dim).CopyTo(v.AsSpan(t * Dim, Dim));
            }
            var kNormW = ReadWeight($"{p}.cross_attn.k_norm.gamma");
            StableAudioAttentionKernels.PerHeadRmsNorm(k, nCond, Heads, Dim, kNormW);
            kvCache[layer] = (k, v);
        }
        return kvCache;
    }

    public StableAudioGpuWeights EnsureGpuWeights(IComputeBackend backend)
    {
        return _gpuWeights ??= new StableAudioGpuWeights(backend, ReadWeight, Depth, Dim, IoChannels, HeadDim, FfInner, MemoryTokens);
    }

    /// <summary>
    /// Precomputes and caches cross-attention K and V projections on GPU for all 20 layers.
    /// Since prompt and duration text conditioning is constant across all 8 diffusion steps,
    /// this runs once before step 0, eliminating 320 redundant GEMM and QK-norm dispatches.
    /// </summary>
    public void PrecomputeConditioningGpu(
        float[] condTokens, int nCond, bool isCond,
        StableAudio3GpuWorkspace ws, StableAudioGpuWeights gpuWeights, IComputeBackend backend)
    {
        if (!_condEmbedCache.TryGetValue(condTokens, out var condEmbed))
        {
            condEmbed = ToCondEmbed(condTokens, nCond);
            _condEmbedCache[condTokens] = condEmbed;
        }

        var targetK = isCond ? ws.CondLayerK : ws.UncondLayerK;
        var targetV = isCond ? ws.CondLayerV : ws.UncondLayerV;

        for (int layer = 0; layer < Depth; layer++)
        {
            string p = $"model.model.transformer.layers.{layer}";
            var kvW = ReadWeight($"{p}.cross_attn.to_kv.weight");
            var kNormW = ReadWeight($"{p}.cross_attn.k_norm.gamma");

            var kv = DiffusionOps.Linear(condEmbed, kvW, null, nCond, Dim, 2 * Dim);
            var k = new float[nCond * Dim];
            var v = new float[nCond * Dim];
            for (int t = 0; t < nCond; t++)
            {
                kv.AsSpan(t * 2 * Dim, Dim).CopyTo(k.AsSpan(t * Dim, Dim));
                kv.AsSpan(t * 2 * Dim + Dim, Dim).CopyTo(v.AsSpan(t * Dim, Dim));
            }

            StableAudioAttentionKernels.PerHeadRmsNorm(k, nCond, Heads, Dim, kNormW);

            backend.WritePinned(targetK[layer], k);
            backend.WritePinned(targetV[layer], v);
        }
    }

    /// <summary>
    /// Executes one full 20-layer DiT forward pass entirely on the GPU.
    /// In integrated GPU residency mode, the input latent in <paramref name="ws"/>.Latent
    /// is transformed into <paramref name="outputVelocity"/> with zero CPU synchronizations.
    /// </summary>
    public void ForwardGpuCore(
        StableAudio3GpuWorkspace ws,
        int seqLen,
        float[] secondsTotalRaw,
        float timestep,
        bool isCond,
        int nCond,
        StableAudioGpuWeights gpuWeights,
        IVisionOpsBackend visionOps,
        IImageOpsBackend imageOps,
        CoreTensor outputVelocity)
    {
        if (!_globalEmbedCache.TryGetValue(secondsTotalRaw, out var globalEmbedBase))
        {
            globalEmbedBase = ToGlobalEmbed(secondsTotalRaw);
            _globalEmbedCache[secondsTotalRaw] = globalEmbedBase;
        }

        var globalEmbed = new float[Dim];
        globalEmbedBase.AsSpan().CopyTo(globalEmbed);
        var timestepEmbed = ToTimestepEmbed(timestep);
        for (int i = 0; i < Dim; i++) globalEmbed[i] += timestepEmbed[i];

        var globalCond = GlobalCondEmbedder(globalEmbed);

        // Precompute and write layer modulations into pinned memory
        var hostMod = new float[6 * Dim];
        for (int layer = 0; layer < Depth; layer++)
        {
            var toScaleShiftGate = gpuWeights.Layers[layer].ToScaleShiftGate;
            string p = $"model.model.transformer.layers.{layer}";
            var preNormW = ReadWeight($"{p}.pre_norm.gamma");
            var ffNormW = ReadWeight($"{p}.ff_norm.gamma");

            for (int i = 0; i < Dim; i++)
            {
                float g0 = toScaleShiftGate[i] + globalCond[i];
                float g1 = toScaleShiftGate[Dim + i] + globalCond[Dim + i];
                float g2 = toScaleShiftGate[2 * Dim + i] + globalCond[2 * Dim + i];
                float g3 = toScaleShiftGate[3 * Dim + i] + globalCond[3 * Dim + i];
                float g4 = toScaleShiftGate[4 * Dim + i] + globalCond[4 * Dim + i];
                float g5 = toScaleShiftGate[5 * Dim + i] + globalCond[5 * Dim + i];

                // s' = gamma * (1 + scale) - 1
                // sh' = shift
                // gate = sigmoid(1 - g)
                hostMod[i] = preNormW[i] * (1f + g0) - 1f;
                hostMod[Dim + i] = g1;
                hostMod[2 * Dim + i] = StableAudioAttentionKernels.Sigmoid(1f - g2);

                hostMod[3 * Dim + i] = ffNormW[i] * (1f + g3) - 1f;
                hostMod[4 * Dim + i] = g4;
                hostMod[5 * Dim + i] = StableAudioAttentionKernels.Sigmoid(1f - g5);
            }
            backend_WritePinned(imageOps, ws.LayerMods[layer], hostMod);
        }

        int totalSeq = MemoryTokens + seqLen;

        // Input preprocess conv 1x1 + residual
        imageOps.Sgemm(ws.LatentPre, ws.Latent, gpuWeights.PreprocessConvW, seqLen, IoChannels, IoChannels);
        imageOps.AddInPlace(ws.LatentPre, ws.Latent);

        // Project in
        imageOps.Sgemm(ws.XIn, ws.LatentPre, gpuWeights.ProjInW, seqLen, IoChannels, Dim);

        // Concat memory tokens
        visionOps.FluxConcatTxtImg(gpuWeights.MemoryTokens, ws.XIn, ws.XFull, MemoryTokens, seqLen, Dim);

        var targetK = isCond ? ws.CondLayerK : ws.UncondLayerK;
        var targetV = isCond ? ws.CondLayerV : ws.UncondLayerV;

        for (int layer = 0; layer < Depth; layer++)
        {
            var bw = gpuWeights.Layers[layer];
            var modTensor = ws.LayerMods[layer];

            // 1. Self-attention
            visionOps.AdaLNModulate(ws.Normed, ws.XFull, modTensor, totalSeq, Dim, shiftOffset: Dim, scaleOffset: 0, isRmsNorm: true, eps: 1e-5f);
            imageOps.Sgemm(ws.Qkv, ws.Normed, bw.SelfAttnQkvW, totalSeq, Dim, 3 * Dim);
            visionOps.FluxUnpackQkv(ws.Qkv, ws.Q, ws.K, ws.V, totalSeq, Dim, dstTokenOffset: 0);

            visionOps.RmsNormBatched(ws.Q, ws.Q, bw.SelfAttnQNormGamma, HeadDim, totalSeq * Heads, eps: 1e-5f);
            visionOps.RmsNormBatched(ws.K, ws.K, bw.SelfAttnKNormGamma, HeadDim, totalSeq * Heads, eps: 1e-5f);

            visionOps.RoPEPartialBatched(ws.Q, basePosition: 0, headDim: HeadDim, ropeDim: RopeRotDim, ropeTheta: RopeTheta, numHeads: Heads, nTok: totalSeq, neox: true);
            visionOps.RoPEPartialBatched(ws.K, basePosition: 0, headDim: HeadDim, ropeDim: RopeRotDim, ropeTheta: RopeTheta, numHeads: Heads, nTok: totalSeq, neox: true);

            imageOps.MultiHeadAttentionTiled(ws.AttnOut, ws.Q, ws.K, ws.V, totalSeq, totalSeq, Heads, HeadDim);
            imageOps.Sgemm(ws.AttnProj, ws.AttnOut, bw.SelfAttnToOutW, totalSeq, Dim, Dim);
            visionOps.ScaleGateAdd(ws.XFull, ws.AttnProj, modTensor, totalSeq, Dim, gateOffset: 2 * Dim);

            // 2. Cross-attention
            visionOps.RmsNormBatched(ws.CrossNormed, ws.XFull, bw.CrossAttendNormGamma, Dim, totalSeq, eps: 1e-5f);
            imageOps.Sgemm(ws.CrossQ, ws.CrossNormed, bw.CrossAttnQW, totalSeq, Dim, Dim);
            visionOps.RmsNormBatched(ws.CrossQ, ws.CrossQ, bw.CrossAttnQNormGamma, HeadDim, totalSeq * Heads, eps: 1e-5f);

            imageOps.MultiHeadAttentionTiled(ws.CrossAttnOut, ws.CrossQ, targetK[layer], targetV[layer], totalSeq, nCond, Heads, HeadDim);
            imageOps.Sgemm(ws.CrossAttnProj, ws.CrossAttnOut, bw.CrossAttnToOutW, totalSeq, Dim, Dim);
            imageOps.AddInPlace(ws.XFull, ws.CrossAttnProj);

            // 3. SwiGLU FeedForward
            visionOps.AdaLNModulate(ws.FfNormed, ws.XFull, modTensor, totalSeq, Dim, shiftOffset: 4 * Dim, scaleOffset: 3 * Dim, isRmsNorm: true, eps: 1e-5f);
            imageOps.Sgemm(ws.FfGate, ws.FfNormed, bw.Ff0GateW, totalSeq, Dim, FfInner);
            imageOps.AddRowBroadcastInPlace(ws.FfGate, bw.Ff0GateB, totalSeq, FfInner);

            imageOps.Sgemm(ws.FfUp, ws.FfNormed, bw.Ff0UpW, totalSeq, Dim, FfInner);
            imageOps.AddRowBroadcastInPlace(ws.FfUp, bw.Ff0UpB, totalSeq, FfInner);

            imageOps.SiLuMul(ws.FfGate, ws.FfUp);

            imageOps.Sgemm(ws.FfOut, ws.FfGate, bw.Ff2W, totalSeq, FfInner, Dim);
            imageOps.AddRowBroadcastInPlace(ws.FfOut, bw.Ff2B, totalSeq, Dim);

            visionOps.ScaleGateAdd(ws.XFull, ws.FfOut, modTensor, totalSeq, Dim, gateOffset: 5 * Dim);
        }

        // Slice out acoustic tokens
        visionOps.FluxSliceImg(ws.XFull, ws.Stripped, MemoryTokens, seqLen, Dim);

        // Project out
        imageOps.Sgemm(ws.OutLow, ws.Stripped, gpuWeights.ProjOutW, seqLen, Dim, IoChannels);

        // Postprocess conv 1x1 + residual
        imageOps.Sgemm(outputVelocity, ws.OutLow, gpuWeights.PostprocessConvW, seqLen, IoChannels, IoChannels);
        imageOps.AddInPlace(outputVelocity, ws.OutLow);
    }

    private static void backend_WritePinned(IImageOpsBackend imageOps, CoreTensor tensor, ReadOnlySpan<float> data)
    {
        imageOps.WritePinned(tensor, data);
    }

    /// <summary>
    /// Standalone GPU forward pass for unit tests and single-step evaluation.
    /// Uploads the host latent, dispatches all 20 GPU layers, and downloads the predicted velocity.
    /// </summary>
    public float[] ForwardGpu(
        float[] latent, int seqLen,
        float[] condTokens, int nCond,
        float[] secondsTotalRaw,
        float timestep,
        IComputeBackend backend)
    {
        if (backend is not IVisionOpsBackend visionOps || backend is not IImageOpsBackend imageOps)
            throw new NotSupportedException("ForwardGpu requires a backend implementing IVisionOpsBackend and IImageOpsBackend (e.g. VulkanBackend).");

        var gpuWeights = EnsureGpuWeights(backend);
        using var ws = new StableAudio3GpuWorkspace(backend, maxSeqLen: Math.Max(seqLen, 16), memoryTokens: MemoryTokens, dim: Dim, ioChannels: IoChannels, ffInner: FfInner, nCondMax: Math.Max(nCond, 32));

        backend.Upload(latent, ws.Latent.Shape, exact: false); // Upload latent into ws
        // Copy to ws.Latent via staging upload
        using (var hostLatentTensor = backend.Upload(latent, ws.Latent.Shape, exact: true))
        {
            backend.AddInPlace(ws.Latent, hostLatentTensor);
        }

        PrecomputeConditioningGpu(condTokens, nCond, isCond: true, ws, gpuWeights, backend);

        imageOps.BeginBatch();
        ForwardGpuCore(ws, seqLen, secondsTotalRaw, timestep, isCond: true, nCond, gpuWeights, visionOps, imageOps, ws.Velocity);
        imageOps.EndBatch();

        var velocity = new float[seqLen * IoChannels];
        imageOps.Download(ws.Velocity, velocity);
        return velocity;
    }

    public void Dispose()
    {
        if (_ownsLoader) _st.Dispose();
        _quantizedCache.Dispose();
        _gpuWeights?.Dispose();
    }
}
