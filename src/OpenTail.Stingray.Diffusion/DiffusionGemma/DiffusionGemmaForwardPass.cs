using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Diffusion.DiffusionGemma;

/// <summary>
/// Pass execution mode for the shared Gemma-4 backbone in DiffusionGemma.
/// </summary>
public enum DiffusionGemmaPassMode
{
    EncoderPrefill,
    DecoderCanvas
}

/// <summary>
/// Forward pass, causal prompt prefill, and bidirectional canvas decode executor
/// for DiffusionGemma (google/diffusiongemma-26B-A4B-it).
/// </summary>
public sealed unsafe class DiffusionGemmaForwardPass
{
    private readonly DiffusionGemmaConfig _config;
    private readonly DiffusionGemmaTensorSet? _tensorSet;

    // Direct tensor references
    private readonly DeepSeek4TensorRef _tokEmbd;
    private readonly DeepSeek4TensorRef _outputNorm;
    private readonly DeepSeek4TensorRef _output;
    private readonly DeepSeek4TensorRef? _ropeFreqs;
    private readonly DeepSeek4TensorRef? _scPreNorm;
    private readonly DeepSeek4TensorRef? _scGate;
    private readonly DeepSeek4TensorRef? _scUp;
    private readonly DeepSeek4TensorRef? _scDown;
    private readonly IReadOnlyList<DiffusionGemmaLayerTensorsBase> _layers;

    // Persistent prompt KV cache: [layer][pos] -> float[kvDim]
    private readonly List<float[]>[] _promptKCache;
    private readonly List<float[]>[] _promptVCache;

    // Precomputed RoPE frequency tables
    private readonly float[] _ropeFreqsLocal;   // localHalf = 128
    private readonly float[] _ropeFreqsGlobal;  // globalHalf = 256

    public DiffusionGemmaForwardPass(DiffusionGemmaTensorSet tensorSet, DiffusionGemmaConfig config)
    {
        _tensorSet = tensorSet;
        _config = config;

        _tokEmbd = tensorSet.TokEmbd;
        _outputNorm = tensorSet.OutputNorm;
        _output = tensorSet.Output;
        _ropeFreqs = tensorSet.RopeFreqs;

        _scPreNorm = tensorSet.SelfCondPreNorm;
        _scGate = tensorSet.SelfCondGate;
        _scUp = tensorSet.SelfCondUp;
        _scDown = tensorSet.SelfCondDown;

        _layers = tensorSet.Layers;

        // Initialize persistent prompt KV cache
        _promptKCache = new List<float[]>[_layers.Count];
        _promptVCache = new List<float[]>[_layers.Count];
        for (int i = 0; i < _layers.Count; i++)
        {
            _promptKCache[i] = [];
            _promptVCache[i] = [];
        }

        (_ropeFreqsLocal, _ropeFreqsGlobal) = PrecomputeRopeFrequencies(_ropeFreqs);
    }

    public DiffusionGemmaForwardPass(GgufModel model, DiffusionGemmaConfig config, bool allowRealCheckpoint = false)
    {
        _config = config;

        bool isRealCheckpoint = model.FindTensor("blk.0.ffn_gate_up_exps.weight") is not null
            || model.FindTensor("blk.0.post_ffw_norm_1.weight") is not null;

        // Real checkpoints remain guarded until full verification ladder passes unless explicitly allowed
        if (isRealCheckpoint && !allowRealCheckpoint
            && Environment.GetEnvironmentVariable("STINGRAY_DIFFUSIONGEMMA_ENABLE_REAL") != "1")
        {
            throw new NotSupportedException(
                "diffusion-gemma: this checkpoint uses the Gemma-4 MoE layer layout (fused gate/up experts, router scale, " +
                "parallel dense+expert FFN norms, layer output scales, self_cond_* MLP). Architecture implementation is in-place, " +
                "but real-weight execution remains guarded until independent real-weight verification passes per CLAUDE.md Rule 14. " +
                "Set STINGRAY_DIFFUSIONGEMMA_ENABLE_REAL=1 to override.");
        }

        if (isRealCheckpoint)
        {
            _tensorSet = DiffusionGemmaTensorSet.Load(model, config);
            _tokEmbd = _tensorSet.TokEmbd;
            _outputNorm = _tensorSet.OutputNorm;
            _output = _tensorSet.Output;
            _ropeFreqs = _tensorSet.RopeFreqs;
            _scPreNorm = _tensorSet.SelfCondPreNorm;
            _scGate = _tensorSet.SelfCondGate;
            _scUp = _tensorSet.SelfCondUp;
            _scDown = _tensorSet.SelfCondDown;
            _layers = _tensorSet.Layers;
        }
        else
        {
            // Synthetic / fallback test model support
            _tokEmbd = DiffusionGemmaTensorSet.Required(model, "token_embd.weight");
            _outputNorm = DiffusionGemmaTensorSet.Required(model, "output_norm.weight");
            _output = DiffusionGemmaTensorSet.Optional(model, "output.weight") ?? _tokEmbd;
            _ropeFreqs = DiffusionGemmaTensorSet.Optional(model, "rope_freqs.weight");

            _scPreNorm = DiffusionGemmaTensorSet.Optional(model, "self_cond_pre_norm.weight");
            _scGate = DiffusionGemmaTensorSet.Optional(model, "self_cond_gate.weight");
            _scUp = DiffusionGemmaTensorSet.Optional(model, "self_cond_up.weight");
            _scDown = DiffusionGemmaTensorSet.Optional(model, "self_cond_down.weight");

            var layers = new List<DiffusionGemmaLayerTensorsBase>(config.NumLayers);
            for (int i = 0; i < config.NumLayers; i++)
            {
                bool isFull = config.IsFullAttention(i);
                var attnNorm = DiffusionGemmaTensorSet.Required(model, $"blk.{i}.attn_norm.weight");
                var attnQ = DiffusionGemmaTensorSet.Required(model, $"blk.{i}.attn_q.weight");
                var attnK = DiffusionGemmaTensorSet.Required(model, $"blk.{i}.attn_k.weight");
                var attnQNorm = DiffusionGemmaTensorSet.Optional(model, $"blk.{i}.attn_q_norm.weight") ?? attnNorm;
                var attnKNorm = DiffusionGemmaTensorSet.Optional(model, $"blk.{i}.attn_k_norm.weight") ?? attnNorm;
                var attnOutput = DiffusionGemmaTensorSet.Optional(model, $"blk.{i}.attn_output.weight")
                    ?? DiffusionGemmaTensorSet.Required(model, $"blk.{i}.attn_out.weight");
                var postAttnNorm = DiffusionGemmaTensorSet.Optional(model, $"blk.{i}.post_attention_norm.weight") ?? attnNorm;

                var ffnNorm = DiffusionGemmaTensorSet.Required(model, $"blk.{i}.ffn_norm.weight");
                var ffnGate = DiffusionGemmaTensorSet.Required(model, $"blk.{i}.ffn_gate.weight");
                var ffnUp = DiffusionGemmaTensorSet.Required(model, $"blk.{i}.ffn_up.weight");
                var ffnDown = DiffusionGemmaTensorSet.Required(model, $"blk.{i}.ffn_down.weight");
                var postFfwNorm1 = DiffusionGemmaTensorSet.Optional(model, $"blk.{i}.post_ffw_norm_1.weight");
                var preFfwNorm2 = DiffusionGemmaTensorSet.Optional(model, $"blk.{i}.pre_ffw_norm_2.weight");
                var ffnGateInp = DiffusionGemmaTensorSet.Optional(model, $"blk.{i}.ffn_gate_inp.weight");
                var ffnGateInpScale = DiffusionGemmaTensorSet.Optional(model, $"blk.{i}.ffn_gate_inp.scale");
                var ffnGateUpExps = DiffusionGemmaTensorSet.Optional(model, $"blk.{i}.ffn_gate_up_exps.weight");
                var ffnDownExps = DiffusionGemmaTensorSet.Optional(model, $"blk.{i}.ffn_down_exps.weight");
                var ffnDownExpsScale = DiffusionGemmaTensorSet.Optional(model, $"blk.{i}.ffn_down_exps.scale");
                var postFfwNorm2 = DiffusionGemmaTensorSet.Optional(model, $"blk.{i}.post_ffw_norm_2.weight");

                var postFfwNorm = DiffusionGemmaTensorSet.Optional(model, $"blk.{i}.post_ffw_norm.weight");
                var layerOutputScale = DiffusionGemmaTensorSet.Optional(model, $"blk.{i}.layer_output_scale");
                var encLayerOutputScale = DiffusionGemmaTensorSet.Optional(model, $"blk.{i}.enc_layer_output_scale");

                DiffusionGemmaLayerTensorsBase layer;
                if (isFull)
                {
                    layer = new DiffusionGemmaFullLayerTensors
                    {
                        LayerIndex = i,
                        AttnNorm = attnNorm,
                        Wq = attnQ,
                        Wk = attnK,
                        AttnQNorm = attnQNorm,
                        AttnKNorm = attnKNorm,
                        Wo = attnOutput,
                        PostAttnNorm = postAttnNorm,
                        FfnNorm = ffnNorm,
                        FfnGate = ffnGate,
                        FfnUp = ffnUp,
                        FfnDown = ffnDown,
                        PostFfwNorm1 = postFfwNorm1,
                        PreFfwNorm2 = preFfwNorm2,
                        FfnGateInp = ffnGateInp,
                        FfnGateInpScale = ffnGateInpScale,
                        FfnGateUpExps = ffnGateUpExps,
                        FfnDownExps = ffnDownExps,
                        FfnDownExpsScale = ffnDownExpsScale,
                        PostFfwNorm2 = postFfwNorm2,
                        PostFfwNorm = postFfwNorm,
                        LayerOutputScale = layerOutputScale,
                        EncLayerOutputScale = encLayerOutputScale,
                    };
                }
                else
                {
                    var attnV = DiffusionGemmaTensorSet.Required(model, $"blk.{i}.attn_v.weight");
                    layer = new DiffusionGemmaSlidingLayerTensors(attnV)
                    {
                        LayerIndex = i,
                        AttnNorm = attnNorm,
                        Wq = attnQ,
                        Wk = attnK,
                        AttnQNorm = attnQNorm,
                        AttnKNorm = attnKNorm,
                        Wo = attnOutput,
                        PostAttnNorm = postAttnNorm,
                        FfnNorm = ffnNorm,
                        FfnGate = ffnGate,
                        FfnUp = ffnUp,
                        FfnDown = ffnDown,
                        PostFfwNorm1 = postFfwNorm1,
                        PreFfwNorm2 = preFfwNorm2,
                        FfnGateInp = ffnGateInp,
                        FfnGateInpScale = ffnGateInpScale,
                        FfnGateUpExps = ffnGateUpExps,
                        FfnDownExps = ffnDownExps,
                        FfnDownExpsScale = ffnDownExpsScale,
                        PostFfwNorm2 = postFfwNorm2,
                        PostFfwNorm = postFfwNorm,
                        LayerOutputScale = layerOutputScale,
                        EncLayerOutputScale = encLayerOutputScale,
                    };
                }
                layers.Add(layer);
            }
            _layers = layers;
        }

        _promptKCache = new List<float[]>[_layers.Count];
        _promptVCache = new List<float[]>[_layers.Count];
        for (int i = 0; i < _layers.Count; i++)
        {
            _promptKCache[i] = [];
            _promptVCache[i] = [];
        }

        (_ropeFreqsLocal, _ropeFreqsGlobal) = PrecomputeRopeFrequencies(_ropeFreqs);
    }

    public int VocabSize => (int)_tokEmbd.Info.Dimensions[1];
    public int HiddenDim => _config.HiddenDim;
    public int PromptLength => _promptKCache[0].Count;

    public void ResetPrompt()
    {
        for (int i = 0; i < _layers.Count; i++)
        {
            _promptKCache[i].Clear();
            _promptVCache[i].Clear();
        }
    }

    private static (float[] local, float[] global) PrecomputeRopeFrequencies(DeepSeek4TensorRef? ropeFreqs)
    {
        const int localHalf = 128;
        const int globalHalf = 256;

        var local = new float[localHalf];
        for (int i = 0; i < localHalf; i++)
        {
            local[i] = (float)(1.0 / Math.Pow(10000.0, 2.0 * i / 256.0));
        }

        var global = new float[globalHalf];
        float[]? freqFactors = null;
        if (ropeFreqs is { } rf)
        {
            freqFactors = new float[globalHalf];
            fixed (float* dst = freqFactors)
            {
                if (rf.DType == DType.Float32)
                {
                    Buffer.MemoryCopy(rf.DataPtr, dst, globalHalf * 4, Math.Min((long)globalHalf * 4, (long)rf.Info.Dimensions[0] * 4));
                }
                else
                {
                    Dequantize.ToFloat32(new ReadOnlySpan<byte>(rf.DataPtr, (int)rf.Info.ByteSize),
                        new Span<float>(dst, globalHalf), rf.DType, globalHalf);
                }
            }
        }

        for (int i = 0; i < globalHalf; i++)
        {
            double freq = 1.0 / Math.Pow(1000000.0, 2.0 * i / 512.0);
            if (freqFactors != null && i < freqFactors.Length && freqFactors[i] != 0f)
            {
                freq /= freqFactors[i];
            }
            global[i] = (float)freq;
        }

        return (local, global);
    }

    /// <summary>
    /// Prefills prompt tokens causally into persistent prompt KV cache across all layers.
    /// Supports multi-block appends: newly prefilled tokens causally attend to all previously
    /// cached prefix KV tokens (subject to layer-specific SWA / Full geometry) and to earlier
    /// tokens in the current append.
    /// </summary>
    public void PrefillPrompt(ReadOnlySpan<int> promptTokens)
    {
        int pLen = promptTokens.Length;
        if (pLen == 0) return;

        // Embedding: Embed(token) * sqrt(hidden_dim)
        var promptH = new float[pLen, HiddenDim];
        for (int p = 0; p < pLen; p++)
        {
            var emb = new float[HiddenDim];
            EmbedToken(promptTokens[p], emb);
            for (int d = 0; d < HiddenDim; d++)
            {
                promptH[p, d] = emb[d] * _config.EmbedScale;
            }
        }

        for (int il = 0; il < _layers.Count; il++)
        {
            var layer = _layers[il];
            bool isFull = layer.IsFullAttention;
            int qHeads = isFull ? _config.FullNumQHeads : _config.SlidingNumQHeads;
            int kvHeads = isFull ? _config.FullNumKvHeads : _config.SlidingNumKvHeads;
            int headDim = isFull ? _config.FullHeadDim : _config.SlidingHeadDim;
            int qDim = qHeads * headDim;
            int kvDim = kvHeads * headDim;
            int swa = _config.SlidingWindowSize;

            int startPos = _promptKCache[il].Count;

            var qAll = new float[pLen, qDim];
            var kAll = new float[pLen, kvDim];
            var vAll = new float[pLen, kvDim];
            var curRow = new float[HiddenDim];
            var normed = new float[HiddenDim];

            for (int p = 0; p < pLen; p++)
            {
                int absPos = startPos + p;
                for (int d = 0; d < HiddenDim; d++) curRow[d] = promptH[p, d];
                fixed (float* inPtr = curRow, outPtr = normed)
                {
                    SimdKernels.RmsNorm(outPtr, inPtr, (float*)layer.AttnNorm.DataPtr, HiddenDim, _config.RmsNormEps);
                }

                fixed (float* inPtr = normed)
                fixed (float* qPtr = &qAll[p, 0], kPtr = &kAll[p, 0], vPtr = &vAll[p, 0])
                {
                    SimdKernels.MatVec(qPtr, layer.Wq.DataPtr, inPtr, qDim, HiddenDim, layer.Wq.DType);
                    SimdKernels.MatVec(kPtr, layer.Wk.DataPtr, inPtr, kvDim, HiddenDim, layer.Wk.DType);

                    if (!isFull && layer.Wv is { } wv)
                    {
                        SimdKernels.MatVec(vPtr, wv.DataPtr, inPtr, kvDim, HiddenDim, wv.DType);
                        // SWA V: weightless RMSNorm per head
                        for (int h = 0; h < kvHeads; h++)
                        {
                            SimdKernels.PureRmsNorm(vPtr + h * headDim, vPtr + h * headDim, headDim, _config.RmsNormEps);
                        }
                    }
                    else
                    {
                        // Full attention layer: V is derived from RAW K before RoPE via weightless RMSNorm!
                        for (int h = 0; h < kvHeads; h++)
                        {
                            SimdKernels.PureRmsNorm(vPtr + h * headDim, kPtr + h * headDim, headDim, _config.RmsNormEps);
                        }
                    }

                    // Learned per-head Q and K RMSNorm
                    for (int h = 0; h < qHeads; h++)
                    {
                        SimdKernels.RmsNorm(qPtr + h * headDim, qPtr + h * headDim, (float*)layer.AttnQNorm.DataPtr, headDim, _config.RmsNormEps);
                    }
                    for (int h = 0; h < kvHeads; h++)
                    {
                        SimdKernels.RmsNorm(kPtr + h * headDim, kPtr + h * headDim, (float*)layer.AttnKNorm.DataPtr, headDim, _config.RmsNormEps);
                    }

                    // Apply NeoX RoPE at absolute position (startPos + p)
                    ApplyNeoXRope(qPtr, qHeads, headDim, absPos, isFull);
                    ApplyNeoXRope(kPtr, kvHeads, headDim, absPos, isFull);
                }
            }

            // Causal prompt attention across [cached KV + new tokens]
            int gqaRatio = qHeads / kvHeads;
            var attnOut = new float[pLen, qDim];

            for (int p = 0; p < pLen; p++)
            {
                int absPos = startPos + p;
                int kGlobalStart = isFull ? 0 : Math.Max(0, absPos - swa + 1);
                int kTotalCount = absPos - kGlobalStart + 1;
                var scores = new float[kTotalCount];

                for (int h = 0; h < qHeads; h++)
                {
                    int kvH = h / gqaRatio;

                    // Attention scale is 1.0 (absorbed into Q/K norms)
                    for (int ki = 0; ki < kTotalCount; ki++)
                    {
                        int kGlobal = kGlobalStart + ki;
                        float dot = 0f;
                        if (kGlobal < startPos)
                        {
                            var cachedK = _promptKCache[il][kGlobal];
                            for (int d = 0; d < headDim; d++)
                            {
                                dot += qAll[p, h * headDim + d] * cachedK[kvH * headDim + d];
                            }
                        }
                        else
                        {
                            int newIdx = kGlobal - startPos;
                            for (int d = 0; d < headDim; d++)
                            {
                                dot += qAll[p, h * headDim + d] * kAll[newIdx, kvH * headDim + d];
                            }
                        }
                        scores[ki] = dot;
                    }

                    fixed (float* sPtr = scores)
                    {
                        SimdKernels.SoftmaxInPlace(sPtr, kTotalCount);
                    }

                    for (int d = 0; d < headDim; d++)
                    {
                        float acc = 0f;
                        for (int ki = 0; ki < kTotalCount; ki++)
                        {
                            int kGlobal = kGlobalStart + ki;
                            if (kGlobal < startPos)
                            {
                                var cachedV = _promptVCache[il][kGlobal];
                                acc += scores[ki] * cachedV[kvH * headDim + d];
                            }
                            else
                            {
                                int newIdx = kGlobal - startPos;
                                acc += scores[ki] * vAll[newIdx, kvH * headDim + d];
                            }
                        }
                        attnOut[p, h * headDim + d] = acc;
                    }
                }

                // Project attention output, post-norm, residual
                var headOut = new float[qDim];
                for (int d = 0; d < qDim; d++) headOut[d] = attnOut[p, d];
                var proj = new float[HiddenDim];
                fixed (float* inPtr = headOut, outPtr = proj)
                {
                    SimdKernels.MatVec(outPtr, layer.Wo.DataPtr, inPtr, HiddenDim, qDim, layer.Wo.DType);
                    SimdKernels.RmsNorm(outPtr, outPtr, (float*)layer.PostAttnNorm.DataPtr, HiddenDim, _config.RmsNormEps);
                }
                for (int d = 0; d < HiddenDim; d++) curRow[d] = promptH[p, d] + proj[d];

                // FFN: Dense + MoE
                ExecuteFfnAndScaling(layer, curRow, DiffusionGemmaPassMode.EncoderPrefill);
                for (int d = 0; d < HiddenDim; d++) promptH[p, d] = curRow[d];
            }

            // Append persistent prompt K and V for the newly prefilled tokens
            for (int p = 0; p < pLen; p++)
            {
                var kStored = new float[kvDim];
                var vStored = new float[kvDim];
                for (int d = 0; d < kvDim; d++)
                {
                    kStored[d] = kAll[p, d];
                    vStored[d] = vAll[p, d];
                }
                _promptKCache[il].Add(kStored);
                _promptVCache[il].Add(vStored);
            }
        }
    }

    /// <summary>
    /// Executes a denoising step over the 256-token canvas with self-conditioning,
    /// cross-attending to cached prompt KV and bidirectional self-attending across canvas.
    /// </summary>
    public float[] ForwardCanvas(ReadOnlySpan<int> canvasTokens, ReadOnlySpan<float> selfCondEmbeddings = default)
    {
        int canvasLen = canvasTokens.Length;
        int vocabSize = VocabSize;
        int promptLen = PromptLength;

        // 1. Initial canvas representations: Embed(token) * sqrt(2816) + self_cond
        var canvas = new float[canvasLen, HiddenDim];
        for (int pos = 0; pos < canvasLen; pos++)
        {
            var emb = new float[HiddenDim];
            EmbedToken(canvasTokens[pos], emb);
            for (int d = 0; d < HiddenDim; d++)
            {
                float val = emb[d] * _config.EmbedScale;
                if (!selfCondEmbeddings.IsEmpty)
                {
                    val += selfCondEmbeddings[pos * HiddenDim + d];
                }
                canvas[pos, d] = val;
            }

            // Weightless RMSNorm on combined canvas embedding
            fixed (float* cPtr = &canvas[pos, 0])
            {
                SimdKernels.PureRmsNorm(cPtr, cPtr, HiddenDim, _config.RmsNormEps);
            }
        }

        var curRow = new float[HiddenDim];
        var normed = new float[HiddenDim];

        for (int il = 0; il < _layers.Count; il++)
        {
            var layer = _layers[il];
            bool isFull = layer.IsFullAttention;
            int qHeads = isFull ? _config.FullNumQHeads : _config.SlidingNumQHeads;
            int kvHeads = isFull ? _config.FullNumKvHeads : _config.SlidingNumKvHeads;
            int headDim = isFull ? _config.FullHeadDim : _config.SlidingHeadDim;
            int qDim = qHeads * headDim;
            int kvDim = kvHeads * headDim;
            int swa = _config.SlidingWindowSize;

            var canvasQ = new float[canvasLen, qDim];
            var canvasK = new float[canvasLen, kvDim];
            var canvasV = new float[canvasLen, kvDim];

            for (int pos = 0; pos < canvasLen; pos++)
            {
                int absPos = promptLen + pos;
                for (int d = 0; d < HiddenDim; d++) curRow[d] = canvas[pos, d];

                fixed (float* inPtr = curRow, outPtr = normed)
                {
                    SimdKernels.RmsNorm(outPtr, inPtr, (float*)layer.AttnNorm.DataPtr, HiddenDim, _config.RmsNormEps);
                }

                fixed (float* inPtr = normed)
                fixed (float* qPtr = &canvasQ[pos, 0], kPtr = &canvasK[pos, 0], vPtr = &canvasV[pos, 0])
                {
                    SimdKernels.MatVec(qPtr, layer.Wq.DataPtr, inPtr, qDim, HiddenDim, layer.Wq.DType);
                    SimdKernels.MatVec(kPtr, layer.Wk.DataPtr, inPtr, kvDim, HiddenDim, layer.Wk.DType);

                    if (!isFull && layer.Wv is { } wv)
                    {
                        SimdKernels.MatVec(vPtr, wv.DataPtr, inPtr, kvDim, HiddenDim, wv.DType);
                        for (int h = 0; h < kvHeads; h++)
                        {
                            SimdKernels.PureRmsNorm(vPtr + h * headDim, vPtr + h * headDim, headDim, _config.RmsNormEps);
                        }
                    }
                    else
                    {
                        // Full attention layer: V is derived from RAW K before RoPE via weightless RMSNorm!
                        for (int h = 0; h < kvHeads; h++)
                        {
                            SimdKernels.PureRmsNorm(vPtr + h * headDim, kPtr + h * headDim, headDim, _config.RmsNormEps);
                        }
                    }

                    // Learned per-head Q and K RMSNorm
                    for (int h = 0; h < qHeads; h++)
                    {
                        SimdKernels.RmsNorm(qPtr + h * headDim, qPtr + h * headDim, (float*)layer.AttnQNorm.DataPtr, headDim, _config.RmsNormEps);
                    }
                    for (int h = 0; h < kvHeads; h++)
                    {
                        SimdKernels.RmsNorm(kPtr + h * headDim, kPtr + h * headDim, (float*)layer.AttnKNorm.DataPtr, headDim, _config.RmsNormEps);
                    }

                    // Apply NeoX RoPE at absolute position (promptLen + pos)
                    ApplyNeoXRope(qPtr, qHeads, headDim, absPos, isFull);
                    ApplyNeoXRope(kPtr, kvHeads, headDim, absPos, isFull);
                }
            }

            // Allowed prompt keys for canvas queries
            int prefixAttendable = isFull
                ? promptLen
                : Math.Min(promptLen, Math.Max(0, swa - 1));
            int promptStart = promptLen - prefixAttendable;

            int totalKeys = prefixAttendable + canvasLen;
            int gqaRatio = qHeads / kvHeads;

            var attnOut = new float[canvasLen, qDim];
            var scores = new float[totalKeys];

            for (int pos = 0; pos < canvasLen; pos++)
            {
                for (int h = 0; h < qHeads; h++)
                {
                    int kvH = h / gqaRatio;

                    // 1. Cross-attend to prefix prompt KV (scale = 1.0)
                    for (int k = 0; k < prefixAttendable; k++)
                    {
                        var promptK = _promptKCache[il][promptStart + k];
                        float dot = 0f;
                        for (int d = 0; d < headDim; d++)
                        {
                            dot += canvasQ[pos, h * headDim + d] * promptK[kvH * headDim + d];
                        }
                        scores[k] = dot;
                    }

                    // 2. Bidirectional attention across canvas (scale = 1.0)
                    for (int k = 0; k < canvasLen; k++)
                    {
                        float dot = 0f;
                        for (int d = 0; d < headDim; d++)
                        {
                            dot += canvasQ[pos, h * headDim + d] * canvasK[k, kvH * headDim + d];
                        }
                        scores[prefixAttendable + k] = dot;
                    }

                    fixed (float* sPtr = scores)
                    {
                        SimdKernels.SoftmaxInPlace(sPtr, totalKeys);
                    }

                    for (int d = 0; d < headDim; d++)
                    {
                        float acc = 0f;
                        for (int k = 0; k < prefixAttendable; k++)
                        {
                            var promptV = _promptVCache[il][promptStart + k];
                            acc += scores[k] * promptV[kvH * headDim + d];
                        }
                        for (int k = 0; k < canvasLen; k++)
                        {
                            acc += scores[prefixAttendable + k] * canvasV[k, kvH * headDim + d];
                        }
                        attnOut[pos, h * headDim + d] = acc;
                    }
                }

                // Project out and residual add
                var headOut = new float[qDim];
                for (int d = 0; d < qDim; d++) headOut[d] = attnOut[pos, d];

                var proj = new float[HiddenDim];
                fixed (float* inPtr = headOut, outPtr = proj)
                {
                    SimdKernels.MatVec(outPtr, layer.Wo.DataPtr, inPtr, HiddenDim, qDim, layer.Wo.DType);
                    SimdKernels.RmsNorm(outPtr, outPtr, (float*)layer.PostAttnNorm.DataPtr, HiddenDim, _config.RmsNormEps);
                }
                for (int d = 0; d < HiddenDim; d++) curRow[d] = canvas[pos, d] + proj[d];

                // FFN: Dense + MoE
                ExecuteFfnAndScaling(layer, curRow, DiffusionGemmaPassMode.DecoderCanvas);
                for (int d = 0; d < HiddenDim; d++) canvas[pos, d] = curRow[d];
            }
        }

        // 3. Final norm, tied embedding projection, and final logit softcapping (30.0)
        var logits = new float[canvasLen * vocabSize];
        var finalRow = new float[HiddenDim];
        var normedRow = new float[HiddenDim];
        var posLogits = new float[vocabSize];
        float softcap = _config.FinalLogitSoftcap;

        for (int pos = 0; pos < canvasLen; pos++)
        {
            for (int d = 0; d < HiddenDim; d++) finalRow[d] = canvas[pos, d];
            fixed (float* inPtr = finalRow, outPtr = normedRow)
            {
                SimdKernels.RmsNorm(outPtr, inPtr, (float*)_outputNorm.DataPtr, HiddenDim, _config.RmsNormEps);
            }

            fixed (float* inPtr = normedRow, outPtr = posLogits)
            {
                SimdKernels.MatVec(outPtr, _output.DataPtr, inPtr, vocabSize, HiddenDim, _output.DType);
            }

            // Softcap: MathF.Tanh(l / softcap) * softcap
            for (int v = 0; v < vocabSize; v++)
            {
                float l = posLogits[v];
                float capped = MathF.Tanh(l / softcap) * softcap;
                logits[pos * vocabSize + v] = capped;
            }
        }

        return logits;
    }

    private void ExecuteFfnAndScaling(DiffusionGemmaLayerTensorsBase layer, float[] curRow, DiffusionGemmaPassMode mode)
    {
        // 1. Dense FFN: RMSNorm -> gate/up (dim 2112) -> GELU(gate)*up -> down -> PostFfwNorm1
        var ffnNormed = new float[HiddenDim];
        fixed (float* inPtr = curRow, outPtr = ffnNormed)
        {
            SimdKernels.RmsNorm(outPtr, inPtr, (float*)layer.FfnNorm.DataPtr, HiddenDim, _config.RmsNormEps);
        }
        var denseOut = DenseFfn(layer, ffnNormed);

        // 2. MoE Expert branch:
        float[] combinedFfn;
        if (layer.FfnGateUpExps is { DataPtr: not null } && layer.FfnGateInp is { DataPtr: not null } && layer.FfnDownExps is { DataPtr: not null })
        {
            var moeOut = MoeFfn(layer, curRow);
            // Combine: PostFfwNorm(denseOut + moeOut)
            var sumFfn = new float[HiddenDim];
            for (int d = 0; d < HiddenDim; d++) sumFfn[d] = denseOut[d] + moeOut[d];

            if (layer.PostFfwNorm is { DataPtr: not null } pfn)
            {
                combinedFfn = new float[HiddenDim];
                fixed (float* inPtr = sumFfn, outPtr = combinedFfn)
                {
                    SimdKernels.RmsNorm(outPtr, inPtr, (float*)pfn.DataPtr, HiddenDim, _config.RmsNormEps);
                }
            }
            else
            {
                combinedFfn = sumFfn;
            }
        }
        else
        {
            combinedFfn = denseOut;
        }

        // Add residual + apply layer output scale
        float scale = mode == DiffusionGemmaPassMode.EncoderPrefill
            ? layer.EncLayerOutputScaleValue
            : layer.LayerOutputScaleValue;

        for (int d = 0; d < HiddenDim; d++)
        {
            curRow[d] = (curRow[d] + combinedFfn[d]) * scale;
        }
    }

    private float[] DenseFfn(DiffusionGemmaLayerTensorsBase layer, float[] normedInput)
    {
        int interDim = (int)layer.FfnGate.Info.Dimensions[1];
        var gate = new float[interDim];
        var up = new float[interDim];
        fixed (float* inPtr = normedInput, gPtr = gate, uPtr = up)
        {
            SimdKernels.MatVec(gPtr, layer.FfnGate.DataPtr, inPtr, interDim, HiddenDim, layer.FfnGate.DType);
            SimdKernels.MatVec(uPtr, layer.FfnUp.DataPtr, inPtr, interDim, HiddenDim, layer.FfnUp.DType);
        }
        for (int i = 0; i < interDim; i++)
        {
            float x = gate[i];
            float gelu = 0.5f * x * (1.0f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
            gate[i] = gelu * up[i];
        }
        var down = new float[HiddenDim];
        fixed (float* inPtr = gate, outPtr = down)
        {
            SimdKernels.MatVec(outPtr, layer.FfnDown.DataPtr, inPtr, HiddenDim, interDim, layer.FfnDown.DType);
        }
        if (layer.PostFfwNorm1 is { DataPtr: not null } pfn1)
        {
            var postNorm = new float[HiddenDim];
            fixed (float* inPtr = down, postPtr = postNorm)
            {
                SimdKernels.RmsNorm(postPtr, inPtr, (float*)pfn1.DataPtr, HiddenDim, _config.RmsNormEps);
            }
            return postNorm;
        }
        return down;
    }

    private float[] MoeFfn(DiffusionGemmaLayerTensorsBase layer, float[] attnRes)
    {
        int numExperts = _config.NumExperts;
        int topK = _config.NumExpertsUsed;
        int expertFfnDim = _config.ExpertIntermediateDim; // 704

        // 1. Router: pure RMSNorm(attnRes) * (1 / sqrt(D)) * routerScale
        var routerNormed = new float[HiddenDim];
        float invSqrtD = 1.0f / MathF.Sqrt(HiddenDim);
        fixed (float* inPtr = attnRes, outPtr = routerNormed)
        {
            SimdKernels.PureRmsNorm(outPtr, inPtr, HiddenDim, _config.RmsNormEps);
        }

        // Apply router scale if available
        if (layer.FfnGateInpScale is { } gateScale && gateScale.DataPtr != null)
        {
            var scalePtr = (float*)gateScale.DataPtr;
            for (int d = 0; d < HiddenDim; d++)
            {
                routerNormed[d] = routerNormed[d] * invSqrtD * scalePtr[d];
            }
        }
        else
        {
            for (int d = 0; d < HiddenDim; d++)
            {
                routerNormed[d] *= invSqrtD;
            }
        }

        // Router logits
        var routerLogits = new float[numExperts];
        var gateInp = layer.FfnGateInp!.Value;
        fixed (float* inPtr = routerNormed, outPtr = routerLogits)
        {
            SimdKernels.MatVec(outPtr, gateInp.DataPtr, inPtr, numExperts, HiddenDim, gateInp.DType);
        }

        // Softmax over 128 experts + top-K selection
        float maxVal = float.NegativeInfinity;
        for (int e = 0; e < numExperts; e++)
        {
            if (routerLogits[e] > maxVal) maxVal = routerLogits[e];
        }
        float sumExp = 0f;
        var expLogits = new float[numExperts];
        for (int e = 0; e < numExperts; e++)
        {
            float ex = MathF.Exp(routerLogits[e] - maxVal);
            expLogits[e] = ex;
            sumExp += ex;
        }
        float invSum = sumExp > 0f ? 1.0f / sumExp : 0f;
        for (int e = 0; e < numExperts; e++) expLogits[e] *= invSum;

        // Select top-K
        var topIndices = new int[topK];
        var topWeights = new float[topK];
        float selSum = 0f;
        for (int k = 0; k < topK; k++)
        {
            int bestIdx = -1;
            float bestVal = float.NegativeInfinity;
            for (int e = 0; e < numExperts; e++)
            {
                if (expLogits[e] > bestVal)
                {
                    bestVal = expLogits[e];
                    bestIdx = e;
                }
            }
            topIndices[k] = bestIdx;
            topWeights[k] = bestVal;
            selSum += bestVal;
            expLogits[bestIdx] = float.NegativeInfinity;
        }

        // Renormalize top-K
        if (selSum > 0f)
        {
            float invSel = 1.0f / selSum;
            for (int k = 0; k < topK; k++) topWeights[k] *= invSel;
        }

        // 2. Expert execution: moeInput = RMSNorm(attnRes, pre_ffw_norm_2)
        var moeInput = new float[HiddenDim];
        if (layer.PreFfwNorm2 is { DataPtr: not null } pfn2)
        {
            fixed (float* inPtr = attnRes, outPtr = moeInput)
            {
                SimdKernels.RmsNorm(outPtr, inPtr, (float*)pfn2.DataPtr, HiddenDim, _config.RmsNormEps);
            }
        }
        else
        {
            Array.Copy(attnRes, moeInput, HiddenDim);
        }

        int fusedCols = expertFfnDim * 2; // 1408
        var gateUp = new float[fusedCols];
        var act = new float[expertFfnDim];
        var expertDown = new float[HiddenDim];
        var moeAccum = new float[HiddenDim];

        for (int k = 0; k < topK; k++)
        {
            int expert = topIndices[k];
            float w = topWeights[k];

            // Fused gate/up: [2816, 1408]
            PerExpertMatVec(layer.FfnGateUpExps!.Value, expert, moeInput, gateUp, fusedCols);

            // GEGLU: first 704 values are gate, second 704 are up
            for (int i = 0; i < expertFfnDim; i++)
            {
                float x = gateUp[i];
                float up = gateUp[expertFfnDim + i];
                float gelu = 0.5f * x * (1.0f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
                act[i] = gelu * up;
            }

            // Down projection: [704, 2816]
            PerExpertMatVecDown(layer.FfnDownExps!.Value, expert, act, expertDown, expertFfnDim);

            // Per-expert down scale
            float expScale = layer.GetExpertDownScale(expert);
            for (int d = 0; d < HiddenDim; d++)
            {
                moeAccum[d] += w * expertDown[d] * expScale;
            }
        }

        // Post-FFW norm 2 on MoE output
        var moeOut = new float[HiddenDim];
        if (layer.PostFfwNorm2 is { DataPtr: not null } pfn2Out)
        {
            fixed (float* inPtr = moeAccum, outPtr = moeOut)
            {
                SimdKernels.RmsNorm(outPtr, inPtr, (float*)pfn2Out.DataPtr, HiddenDim, _config.RmsNormEps);
            }
        }
        else
        {
            Array.Copy(moeAccum, moeOut, HiddenDim);
        }
        return moeOut;
    }

    private void PerExpertMatVec(DeepSeek4TensorRef tensor, int expert, float[] input, float[] output, int outDim)
    {
        int inDim = (int)tensor.Info.Dimensions[0];
        long bytesPerRow = ((long)inDim / DTypeInfo.BlockSize(tensor.DType)) * DTypeInfo.BytesPerBlock(tensor.DType);
        byte* expertPtr = tensor.DataPtr + (long)expert * outDim * bytesPerRow;
        fixed (float* inPtr = input, outPtr = output)
        {
            SimdKernels.MatVec(outPtr, expertPtr, inPtr, outDim, inDim, tensor.DType);
        }
    }

    private void PerExpertMatVecDown(DeepSeek4TensorRef tensor, int expert, float[] input, float[] output, int inDim)
    {
        int outDim = (int)tensor.Info.Dimensions[1];
        long bytesPerRow = ((long)inDim / DTypeInfo.BlockSize(tensor.DType)) * DTypeInfo.BytesPerBlock(tensor.DType);
        byte* expertPtr = tensor.DataPtr + (long)expert * outDim * bytesPerRow;
        fixed (float* inPtr = input, outPtr = output)
        {
            SimdKernels.MatVec(outPtr, expertPtr, inPtr, outDim, inDim, tensor.DType);
        }
    }

    private void ApplyNeoXRope(float* data, int numHeads, int headDim, int pos, bool isFull)
    {
        int half = headDim / 2;
        float[] freqs = isFull ? _ropeFreqsGlobal : _ropeFreqsLocal;

        for (int h = 0; h < numHeads; h++)
        {
            float* head = data + (long)h * headDim;
            for (int j = 0; j < half; j++)
            {
                float angle = pos * freqs[j];
                float c = MathF.Cos(angle);
                float s = MathF.Sin(angle);

                float x0 = head[j];
                float x1 = head[j + half];

                head[j] = x0 * c - x1 * s;
                head[j + half] = x0 * s + x1 * c;
            }
        }
    }

    /// <summary>
    /// Exact full-vocabulary soft-embedding sum E[pos] = EmbedScale * sum_v softmax(logits[pos] / temperature)[v] * Embed[v].
    /// </summary>
    public float[] ComputeSoftEmbeddings(ReadOnlySpan<float> canvasLogits, float temperature, int canvasLen)
    {
        int vocabSize = VocabSize;
        var probs = new float[(long)canvasLen * vocabSize];
        for (int pos = 0; pos < canvasLen; pos++)
        {
            DiffusionGemmaSelfConditioning.ComputeSoftProbabilities(
                canvasLogits.Slice(pos * vocabSize, vocabSize), temperature, probs.AsSpan(pos * vocabSize, vocabSize));
        }

        var soft = new float[canvasLen * HiddenDim];
        var row = new float[HiddenDim];
        for (int v = 0; v < vocabSize; v++)
        {
            EmbedToken(v, row);
            for (int pos = 0; pos < canvasLen; pos++)
            {
                float p = probs[(long)pos * vocabSize + v];
                if (p == 0f) continue;
                var dst = soft.AsSpan(pos * HiddenDim, HiddenDim);
                System.Numerics.Tensors.TensorPrimitives.MultiplyAdd(row, p, dst, dst);
            }
        }

        for (int i = 0; i < soft.Length; i++) soft[i] *= _config.EmbedScale;
        return soft;
    }

    /// <summary>
    /// Applies the learned self-conditioning MLP to soft embeddings:
    /// signal = down(GELU(gate(pre_norm(soft))) * up(pre_norm(soft))).
    /// </summary>
    public void ApplySelfCondMlp(ReadOnlySpan<float> softEmbed, Span<float> scOut, int canvasLen)
    {
        if (_scGate is null || _scUp is null || _scDown is null || _scPreNorm is null)
        {
            softEmbed.Slice(0, canvasLen * HiddenDim).CopyTo(scOut);
            return;
        }

        int interDim = _config.SelfCondIntermediateDim;
        var normed = new float[HiddenDim];
        var gate = new float[interDim];
        var up = new float[interDim];

        for (int pos = 0; pos < canvasLen; pos++)
        {
            fixed (float* inPtr = softEmbed.Slice(pos * HiddenDim, HiddenDim), nPtr = normed, gPtr = gate, uPtr = up, outPtr = scOut.Slice(pos * HiddenDim, HiddenDim))
            {
                // Pre-norm
                SimdKernels.RmsNorm(nPtr, inPtr, (float*)_scPreNorm.Value.DataPtr, HiddenDim, _config.RmsNormEps);

                // Gate & Up
                SimdKernels.MatVec(gPtr, _scGate.Value.DataPtr, nPtr, interDim, HiddenDim, _scGate.Value.DType);
                SimdKernels.MatVec(uPtr, _scUp.Value.DataPtr, nPtr, interDim, HiddenDim, _scUp.Value.DType);

                // GELU(gate) * up
                for (int i = 0; i < interDim; i++)
                {
                    float x = gate[i];
                    float gelu = 0.5f * x * (1.0f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
                    gate[i] = gelu * up[i];
                }

                // Down
                SimdKernels.MatVec(outPtr, _scDown.Value.DataPtr, gPtr, HiddenDim, interDim, _scDown.Value.DType);
            }
        }
    }

    private void EmbedToken(int token, float[] destination)
    {
        var tensor = _tokEmbd;
        int cols = HiddenDim;
        long bytesPerRow = ((long)cols / DTypeInfo.BlockSize(tensor.DType)) * DTypeInfo.BytesPerBlock(tensor.DType);
        byte* src = tensor.DataPtr + (long)token * bytesPerRow;
        fixed (float* dst = destination)
        {
            SimdKernels.DequantRow(src, dst, cols, tensor.DType);
        }
    }

    public void Reset()
    {
        for (int i = 0; i < _layers.Count; i++)
        {
            _promptKCache[i].Clear();
            _promptVCache[i].Clear();
        }
    }
}
