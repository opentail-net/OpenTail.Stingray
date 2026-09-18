using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

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
    private readonly CachedWeightReader _reader;
    private readonly bool _ownsLoader;
    private readonly Dictionary<float[], float[]> _condEmbedCache = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<float[], float[]> _globalEmbedCache = new(ReferenceEqualityComparer.Instance);
    private StableAudioMediumGpuWeights? _gpuWeights;

    public StableAudioMediumDiT(string path)
    {
        _st = SafetensorsLoader.Open(path);
        _reader = new CachedWeightReader(_st, "");
        _ownsLoader = true;
    }

    private StableAudioMediumDiT(IWeightLoader loader, bool ownsLoader)
    {
        _st = loader;
        _reader = new CachedWeightReader(_st, "");
        _ownsLoader = ownsLoader;
    }

    private float[] ReadWeight(string name) => _reader.Get(name);

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

        var projInW = ReadWeight("model.model.transformer.project_in.weight");
        var x = DiffusionOps.Linear(latentPre, projInW, null, seqLen, IoChannels, Dim);

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

        var projOutW = ReadWeight("model.model.transformer.project_out.weight");
        var outLow = DiffusionOps.Linear(stripped, projOutW, null, seqLen, Dim, IoChannels);

        return Conv1x1Residual(outLow, seqLen, IoChannels, "model.model.postprocess_conv.weight");
    }

    private float[] Conv1x1Residual(float[] x, int seqLen, int channels, string weightKey)
    {
        var w = ReadWeight(weightKey);
        var y = DiffusionOps.Linear(x, w, null, seqLen, channels, channels);
        for (int i = 0; i < y.Length; i++) y[i] += x[i];
        return y;
    }

    private float[] ToCondEmbed(float[] condTokens, int nCond)
    {
        var w0 = ReadWeight("model.model.to_cond_embed.0.weight");
        var w2 = ReadWeight("model.model.to_cond_embed.2.weight");
        var h = DiffusionOps.Linear(condTokens, w0, null, nCond, CondTokenDimRaw, Dim);
        DiffusionOps.SiluInPlace(h);
        return DiffusionOps.Linear(h, w2, null, nCond, Dim, Dim);
    }

    private float[] ToGlobalEmbed(float[] secondsTotalRaw)
    {
        var w0 = ReadWeight("model.model.to_global_embed.0.weight");
        var w2 = ReadWeight("model.model.to_global_embed.2.weight");
        var h = DiffusionOps.Linear(secondsTotalRaw, w0, null, 1, GlobalCondDimRaw, Dim);
        DiffusionOps.SiluInPlace(h);
        return DiffusionOps.Linear(h, w2, null, 1, Dim, Dim);
    }

    private float[] ToTimestepEmbed(float timestep)
    {
        var feats = StableAudioAttentionKernels.ExpoFourierFeatures(timestep, TimestepFeaturesDim);
        var w0 = ReadWeight("model.model.to_timestep_embed.0.weight");
        var b0 = ReadWeight("model.model.to_timestep_embed.0.bias");
        var w2 = ReadWeight("model.model.to_timestep_embed.2.weight");
        var b2 = ReadWeight("model.model.to_timestep_embed.2.bias");
        var h = DiffusionOps.Linear(feats, w0, b0, 1, TimestepFeaturesDim, Dim);
        DiffusionOps.SiluInPlace(h);
        return DiffusionOps.Linear(h, w2, b2, 1, Dim, Dim);
    }

    private float[] GlobalCondEmbedder(float[] globalEmbed)
    {
        var w0 = ReadWeight("model.model.transformer.global_cond_embedder.0.weight");
        var b0 = ReadWeight("model.model.transformer.global_cond_embedder.0.bias");
        var w2 = ReadWeight("model.model.transformer.global_cond_embedder.2.weight");
        var b2 = ReadWeight("model.model.transformer.global_cond_embedder.2.bias");
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

        var toScaleShiftGate = ReadWeight($"{p}.to_scale_shift_gate");
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

    /// <summary>Real DIFFERENTIAL self-attention: `to_qkv` widens to `5*Dim` (`q,k,v,q_diff,k_diff`,
    /// real chunk order confirmed from source). `qk_norm`/RoPE apply identically to the main and
    /// diff pairs (elementwise over head_dim). Two full attention passes share the SAME `v`; final
    /// output is `out_main - out_diff`, no learnable mixing.</summary>
    private float[] SelfAttention(float[] x, int seq, string p, float[] cos, float[] sin)
    {
        var qkvW = ReadWeight($"{p}.self_attn.to_qkv.weight");
        var qNormW = ReadWeight($"{p}.self_attn.q_norm.gamma");
        var kNormW = ReadWeight($"{p}.self_attn.k_norm.gamma");
        var outW = ReadWeight($"{p}.self_attn.to_out.weight");

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

        StableAudioAttentionKernels.PerHeadRmsNorm(q, seq, Heads, Dim, qNormW);
        StableAudioAttentionKernels.PerHeadRmsNorm(k, seq, Heads, Dim, kNormW);
        StableAudioAttentionKernels.PerHeadRmsNorm(qDiff, seq, Heads, Dim, qNormW);
        StableAudioAttentionKernels.PerHeadRmsNorm(kDiff, seq, Heads, Dim, kNormW);

        StableAudioAttentionKernels.ApplyPartialRope(q, seq, Heads, Dim, cos, sin);
        StableAudioAttentionKernels.ApplyPartialRope(k, seq, Heads, Dim, cos, sin);
        StableAudioAttentionKernels.ApplyPartialRope(qDiff, seq, Heads, Dim, cos, sin);
        StableAudioAttentionKernels.ApplyPartialRope(kDiff, seq, Heads, Dim, cos, sin);

        var attnMain = StableAudioAttentionKernels.DotProductAttention(q, k, v, seq, seq, Heads, Dim);
        var attnDiff = StableAudioAttentionKernels.DotProductAttention(qDiff, kDiff, v, seq, seq, Heads, Dim);
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
        var qW = ReadWeight($"{p}.cross_attn.to_q.weight");
        var kvW = ReadWeight($"{p}.cross_attn.to_kv.weight");
        var qNormW = ReadWeight($"{p}.cross_attn.q_norm.gamma");
        var kNormW = ReadWeight($"{p}.cross_attn.k_norm.gamma");
        var outW = ReadWeight($"{p}.cross_attn.to_out.weight");

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

        StableAudioAttentionKernels.PerHeadRmsNorm(q, seq, Heads, Dim, qNormW);
        StableAudioAttentionKernels.PerHeadRmsNorm(qDiff, seq, Heads, Dim, qNormW);
        StableAudioAttentionKernels.PerHeadRmsNorm(k, nCond, Heads, Dim, kNormW);
        StableAudioAttentionKernels.PerHeadRmsNorm(kDiff, nCond, Heads, Dim, kNormW);

        var attnMain = StableAudioAttentionKernels.DotProductAttention(q, k, v, seq, nCond, Heads, Dim);
        var attnDiff = StableAudioAttentionKernels.DotProductAttention(qDiff, kDiff, v, seq, nCond, Heads, Dim);
        var attnOut = new float[attnMain.Length];
        for (int i = 0; i < attnOut.Length; i++) attnOut[i] = attnMain[i] - attnDiff[i];

        return DiffusionOps.Linear(attnOut, outW, null, seq, Dim, Dim);
    }

    private float[] FeedForward(float[] x, int seq, string p)
    {
        var w0 = ReadWeight($"{p}.ff.ff.0.proj.weight");
        var b0 = ReadWeight($"{p}.ff.ff.0.proj.bias");
        var w2 = ReadWeight($"{p}.ff.ff.2.weight");
        var b2 = ReadWeight($"{p}.ff.ff.2.bias");

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

    public StableAudioMediumGpuWeights EnsureGpuWeights(IComputeBackend backend)
    {
        return _gpuWeights ??= new StableAudioMediumGpuWeights(backend, ReadWeight, Depth, Dim, IoChannels, HeadDim, FfInner, MemoryTokens);
    }

    /// <summary>
    /// Precomputes and caches differential cross-attention K, KDiff, and V projections on GPU for all 24 layers.
    /// Since prompt and duration text conditioning is constant across all diffusion steps,
    /// this runs once before step 0, eliminating redundant GEMM and QK-norm dispatches.
    /// </summary>
    public void PrecomputeConditioningGpu(
        float[] condTokens, int nCond, bool isCond,
        StableAudioMediumGpuWorkspace ws, StableAudioMediumGpuWeights gpuWeights, IComputeBackend backend)
    {
        if (!_condEmbedCache.TryGetValue(condTokens, out var condEmbed))
        {
            condEmbed = ToCondEmbed(condTokens, nCond);
            _condEmbedCache[condTokens] = condEmbed;
        }

        var targetK = isCond ? ws.CondLayerK : ws.UncondLayerK;
        var targetKDiff = isCond ? ws.CondLayerKDiff : ws.UncondLayerKDiff;
        var targetV = isCond ? ws.CondLayerV : ws.UncondLayerV;

        for (int layer = 0; layer < Depth; layer++)
        {
            string p = $"model.model.transformer.layers.{layer}";
            var kvW = ReadWeight($"{p}.cross_attn.to_kv.weight");
            var kNormW = ReadWeight($"{p}.cross_attn.k_norm.gamma");

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

            StableAudioAttentionKernels.PerHeadRmsNorm(k, nCond, Heads, Dim, kNormW);
            StableAudioAttentionKernels.PerHeadRmsNorm(kDiff, nCond, Heads, Dim, kNormW);

            backend.WritePinned(targetK[layer], k);
            backend.WritePinned(targetKDiff[layer], kDiff);
            backend.WritePinned(targetV[layer], v);
        }
    }

    /// <summary>
    /// Executes one full 24-layer differential DiT forward pass entirely on the GPU.
    /// In integrated GPU residency mode, the input latent in <paramref name="ws"/>.Latent
    /// is transformed into <paramref name="outputVelocity"/> with zero CPU synchronizations.
    /// </summary>
    public void ForwardGpuCore(
        StableAudioMediumGpuWorkspace ws,
        int seqLen,
        float[] secondsTotalRaw,
        float timestep,
        bool isCond,
        int nCond,
        StableAudioMediumGpuWeights gpuWeights,
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
        var targetKDiff = isCond ? ws.CondLayerKDiff : ws.UncondLayerKDiff;
        var targetV = isCond ? ws.CondLayerV : ws.UncondLayerV;

        for (int layer = 0; layer < Depth; layer++)
        {
            var bw = gpuWeights.Layers[layer];
            var modTensor = ws.LayerMods[layer];

            // 1. Differential Self-attention
            visionOps.AdaLNModulate(ws.Normed, ws.XFull, modTensor, totalSeq, Dim, shiftOffset: Dim, scaleOffset: 0, isRmsNorm: true, eps: 1e-5f);
            imageOps.Sgemm(ws.Qkv, ws.Normed, bw.SelfAttnQkvW, totalSeq, Dim, 5 * Dim);
            visionOps.DiffUnpack5(ws.Qkv, ws.Q, ws.K, ws.V, ws.QDiff, ws.KDiff, totalSeq, Dim);

            visionOps.RmsNormBatched(ws.Q, ws.Q, bw.SelfAttnQNormGamma, HeadDim, totalSeq * Heads, eps: 1e-5f);
            visionOps.RmsNormBatched(ws.K, ws.K, bw.SelfAttnKNormGamma, HeadDim, totalSeq * Heads, eps: 1e-5f);
            visionOps.RmsNormBatched(ws.QDiff, ws.QDiff, bw.SelfAttnQNormGamma, HeadDim, totalSeq * Heads, eps: 1e-5f);
            visionOps.RmsNormBatched(ws.KDiff, ws.KDiff, bw.SelfAttnKNormGamma, HeadDim, totalSeq * Heads, eps: 1e-5f);

            visionOps.RoPEPartialBatched(ws.Q, basePosition: 0, headDim: HeadDim, ropeDim: RopeRotDim, ropeTheta: RopeTheta, numHeads: Heads, nTok: totalSeq, neox: true);
            visionOps.RoPEPartialBatched(ws.K, basePosition: 0, headDim: HeadDim, ropeDim: RopeRotDim, ropeTheta: RopeTheta, numHeads: Heads, nTok: totalSeq, neox: true);
            visionOps.RoPEPartialBatched(ws.QDiff, basePosition: 0, headDim: HeadDim, ropeDim: RopeRotDim, ropeTheta: RopeTheta, numHeads: Heads, nTok: totalSeq, neox: true);
            visionOps.RoPEPartialBatched(ws.KDiff, basePosition: 0, headDim: HeadDim, ropeDim: RopeRotDim, ropeTheta: RopeTheta, numHeads: Heads, nTok: totalSeq, neox: true);

            imageOps.MultiHeadAttentionTiled(ws.AttnOut, ws.Q, ws.K, ws.V, totalSeq, totalSeq, Heads, HeadDim);
            imageOps.MultiHeadAttentionTiled(ws.AttnOutDiff, ws.QDiff, ws.KDiff, ws.V, totalSeq, totalSeq, Heads, HeadDim);
            visionOps.FluxEulerStep(ws.AttnOut, ws.AttnOutDiff, -1.0f, totalSeq * Dim);

            imageOps.Sgemm(ws.AttnProj, ws.AttnOut, bw.SelfAttnToOutW, totalSeq, Dim, Dim);
            visionOps.ScaleGateAdd(ws.XFull, ws.AttnProj, modTensor, totalSeq, Dim, gateOffset: 2 * Dim);

            // 2. Differential Cross-attention
            visionOps.RmsNormBatched(ws.CrossNormed, ws.XFull, bw.CrossAttendNormGamma, Dim, totalSeq, eps: 1e-5f);
            imageOps.Sgemm(ws.CrossQBoth, ws.CrossNormed, bw.CrossAttnQW, totalSeq, Dim, 2 * Dim);
            visionOps.DiffUnpack2(ws.CrossQBoth, ws.CrossQ, ws.CrossQDiff, totalSeq, Dim);

            visionOps.RmsNormBatched(ws.CrossQ, ws.CrossQ, bw.CrossAttnQNormGamma, HeadDim, totalSeq * Heads, eps: 1e-5f);
            visionOps.RmsNormBatched(ws.CrossQDiff, ws.CrossQDiff, bw.CrossAttnQNormGamma, HeadDim, totalSeq * Heads, eps: 1e-5f);

            imageOps.MultiHeadAttentionTiled(ws.CrossAttnOut, ws.CrossQ, targetK[layer], targetV[layer], totalSeq, nCond, Heads, HeadDim);
            imageOps.MultiHeadAttentionTiled(ws.CrossAttnOutDiff, ws.CrossQDiff, targetKDiff[layer], targetV[layer], totalSeq, nCond, Heads, HeadDim);
            visionOps.FluxEulerStep(ws.CrossAttnOut, ws.CrossAttnOutDiff, -1.0f, totalSeq * Dim);

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

    public void Dispose()
    {
        _gpuWeights?.Dispose();
        if (_ownsLoader) _st.Dispose();
    }
}
