using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Diffusion.DiffusionGemma;

/// <summary>
/// Resolved layer tensors for DiffusionGemma.
/// </summary>
public sealed unsafe class DiffusionGemmaLayerTensors
{
    public DeepSeek4TensorRef? AttnNorm;
    public DeepSeek4TensorRef? AttnPostNorm;
    public DeepSeek4TensorRef? Wq;
    public DeepSeek4TensorRef? Wk;
    public DeepSeek4TensorRef? Wv;
    public DeepSeek4TensorRef? Wo;
    public DeepSeek4TensorRef? QNorm;
    public DeepSeek4TensorRef? KNorm;

    public DeepSeek4TensorRef? FfnNorm;
    public DeepSeek4TensorRef? FfnPostNorm;

    // Dense FFN
    public DeepSeek4TensorRef? FfnGate;
    public DeepSeek4TensorRef? FfnUp;
    public DeepSeek4TensorRef? FfnDown;

    // MoE
    public DeepSeek4TensorRef? FfnGateInp;
    public DeepSeek4TensorRef? FfnGateExps;
    public DeepSeek4TensorRef? FfnUpExps;
    public DeepSeek4TensorRef? FfnDownExps;
}

/// <summary>
/// Forward pass and prompt prefill executor for DiffusionGemma.
/// </summary>
public sealed unsafe class DiffusionGemmaForwardPass
{
    private readonly GgufModel _model;
    private readonly DiffusionGemmaConfig _config;
    private readonly DeepSeek4TensorRef _tokEmbd;
    private readonly DeepSeek4TensorRef _outputNorm;
    private readonly DeepSeek4TensorRef _output;
    private readonly DeepSeek4TensorRef? _scGate;
    private readonly DeepSeek4TensorRef? _scUp;
    private readonly DeepSeek4TensorRef? _scDown;
    private readonly IReadOnlyList<DiffusionGemmaLayerTensors> _layers;

    // Persistent prompt KV cache: [layer][pos][kvDim]
    private readonly List<float[]>[] _promptKCache;
    private readonly List<float[]>[] _promptVCache;

    public DiffusionGemmaForwardPass(GgufModel model, DiffusionGemmaConfig config)
    {
        _model = model;
        _config = config;

        DeepSeek4TensorRef Required(string name)
        {
            var info = model.FindTensor(name)
                ?? throw new InvalidOperationException($"Missing required DiffusionGemma tensor: {name}");
            return new DeepSeek4TensorRef(name, info, model.GetTensorDataPtr(info));
        }

        DeepSeek4TensorRef? Optional(string name)
        {
            var info = model.FindTensor(name);
            return info is null ? null : new DeepSeek4TensorRef(name, info.Value, model.GetTensorDataPtr(info.Value));
        }

        _tokEmbd = Required("token_embd.weight");
        _outputNorm = Required("output_norm.weight");
        _output = Optional("output.weight") ?? _tokEmbd;

        _scGate = Optional("self_cond.gate.weight");
        _scUp = Optional("self_cond.up.weight");
        _scDown = Optional("self_cond.down.weight");

        var layers = new List<DiffusionGemmaLayerTensors>(config.NumLayers);
        for (int i = 0; i < config.NumLayers; i++)
        {
            var lt = new DiffusionGemmaLayerTensors
            {
                AttnNorm = Required($"blk.{i}.attn_norm.weight"),
                AttnPostNorm = Optional($"blk.{i}.attn_post_norm.weight"),
                Wq = Required($"blk.{i}.attn_q.weight"),
                Wk = Required($"blk.{i}.attn_k.weight"),
                Wv = Required($"blk.{i}.attn_v.weight"),
                Wo = Optional($"blk.{i}.attn_output.weight") ?? Required($"blk.{i}.attn_out.weight"),
                QNorm = Optional($"blk.{i}.attn_q_norm.weight"),
                KNorm = Optional($"blk.{i}.attn_k_norm.weight"),
                FfnNorm = Required($"blk.{i}.ffn_norm.weight"),
                FfnPostNorm = Optional($"blk.{i}.ffn_post_norm.weight"),
                FfnGate = Required($"blk.{i}.ffn_gate.weight"),
                FfnUp = Required($"blk.{i}.ffn_up.weight"),
                FfnDown = Required($"blk.{i}.ffn_down.weight"),
                FfnGateInp = Optional($"blk.{i}.ffn_gate_inp.weight"),
                FfnGateExps = Optional($"blk.{i}.ffn_gate_exps.weight"),
                FfnUpExps = Optional($"blk.{i}.ffn_up_exps.weight"),
                FfnDownExps = Optional($"blk.{i}.ffn_down_exps.weight"),
            };
            layers.Add(lt);
        }
        _layers = layers;

        _promptKCache = new List<float[]>[config.NumLayers];
        _promptVCache = new List<float[]>[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            _promptKCache[i] = [];
            _promptVCache[i] = [];
        }
    }

    public int VocabSize => (int)_tokEmbd.Info.Dimensions[1];
    public int HiddenDim => _config.HiddenDim;
    public int PromptLength => _promptKCache[0].Count;

    /// <summary>
    /// Prefills prompt tokens causally into persistent prompt KV cache.
    /// </summary>
    public void PrefillPrompt(ReadOnlySpan<int> promptTokens)
    {
        for (int p = 0; p < promptTokens.Length; p++)
        {
            int token = promptTokens[p];
            var cur = new float[HiddenDim];
            EmbedToken(token, cur);

            for (int il = 0; il < _config.NumLayers; il++)
            {
                var layer = _layers[il];
                bool isFull = _config.IsFullAttention(il);
                int kvHeads = isFull ? _config.FullNumKvHeads : _config.SlidingNumKvHeads;
                int headDim = isFull ? _config.FullHeadDim : _config.SlidingHeadDim;
                int kvDim = kvHeads * headDim;

                var normed = new float[HiddenDim];
                fixed (float* inPtr = cur, outPtr = normed)
                {
                    float* weightPtr = (float*)layer.AttnNorm!.Value.DataPtr;
                    SimdKernels.RmsNorm(outPtr, inPtr, weightPtr, HiddenDim, _config.RmsNormEps);
                }

                // Compute K and V and store in prompt KV cache
                var k = new float[kvDim];
                var v = new float[kvDim];
                fixed (float* inPtr = normed, kPtr = k, vPtr = v)
                {
                    SimdKernels.MatVec(kPtr, layer.Wk!.Value.DataPtr, inPtr, kvDim, HiddenDim, layer.Wk.Value.DType);
                    SimdKernels.MatVec(vPtr, layer.Wv!.Value.DataPtr, inPtr, kvDim, HiddenDim, layer.Wv.Value.DType);
                }

                if (layer.KNorm is { } kn)
                {
                    for (int h = 0; h < kvHeads; h++)
                    {
                        fixed (float* kHeadPtr = &k[h * headDim])
                        {
                            SimdKernels.RmsNorm(kHeadPtr, kHeadPtr, (float*)kn.DataPtr, headDim, _config.RmsNormEps);
                        }
                    }
                }

                _promptKCache[il].Add(k);
                _promptVCache[il].Add(v);

                // Residual forward step
                var ffnNormed = new float[HiddenDim];
                fixed (float* inPtr = cur, outPtr = ffnNormed)
                {
                    float* weightPtr = (float*)layer.FfnNorm!.Value.DataPtr;
                    SimdKernels.RmsNorm(outPtr, inPtr, weightPtr, HiddenDim, _config.RmsNormEps);
                }
                var ffnOut = DenseFfn(layer, ffnNormed);
                for (int d = 0; d < HiddenDim; d++) cur[d] += ffnOut[d];
            }
        }
    }

    /// <summary>
    /// Executes a denoising step over the 256-token canvas with self-conditioning.
    /// </summary>
    public float[] ForwardCanvas(ReadOnlySpan<int> canvasTokens, ReadOnlySpan<float> selfCondEmbeddings)
    {
        int canvasLen = canvasTokens.Length;
        int vocabSize = VocabSize;

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
        }

        // 2. Transformer layers
        var curRow = new float[HiddenDim];
        var nextCanvas = new float[canvasLen, HiddenDim];

        for (int il = 0; il < _config.NumLayers; il++)
        {
            var layer = _layers[il];
            bool isFull = _config.IsFullAttention(il);
            int qHeads = isFull ? _config.FullNumQHeads : _config.SlidingNumQHeads;
            int kvHeads = isFull ? _config.FullNumKvHeads : _config.SlidingNumKvHeads;
            int headDim = isFull ? _config.FullHeadDim : _config.SlidingHeadDim;
            int qDim = qHeads * headDim;
            int kvDim = kvHeads * headDim;

            // Compute Q, K, V for all canvas tokens
            var canvasQ = new float[canvasLen, qDim];
            var canvasK = new float[canvasLen, kvDim];
            var canvasV = new float[canvasLen, kvDim];

            for (int pos = 0; pos < canvasLen; pos++)
            {
                for (int d = 0; d < HiddenDim; d++) curRow[d] = canvas[pos, d];

                var normed = new float[HiddenDim];
                fixed (float* inPtr = curRow, outPtr = normed)
                {
                    float* weightPtr = (float*)layer.AttnNorm!.Value.DataPtr;
                    SimdKernels.RmsNorm(outPtr, inPtr, weightPtr, HiddenDim, _config.RmsNormEps);
                }

                fixed (float* inPtr = normed)
                {
                    fixed (float* qPtr = &canvasQ[pos, 0], kPtr = &canvasK[pos, 0], vPtr = &canvasV[pos, 0])
                    {
                        SimdKernels.MatVec(qPtr, layer.Wq!.Value.DataPtr, inPtr, qDim, HiddenDim, layer.Wq.Value.DType);
                        SimdKernels.MatVec(kPtr, layer.Wk!.Value.DataPtr, inPtr, kvDim, HiddenDim, layer.Wk.Value.DType);
                        SimdKernels.MatVec(vPtr, layer.Wv!.Value.DataPtr, inPtr, kvDim, HiddenDim, layer.Wv.Value.DType);
                    }
                }

                if (layer.QNorm is { } qn)
                {
                    for (int h = 0; h < qHeads; h++)
                    {
                        fixed (float* qHeadPtr = &canvasQ[pos, h * headDim])
                        {
                            SimdKernels.RmsNorm(qHeadPtr, qHeadPtr, (float*)qn.DataPtr, headDim, _config.RmsNormEps);
                        }
                    }
                }
                if (layer.KNorm is { } kn)
                {
                    for (int h = 0; h < kvHeads; h++)
                    {
                        fixed (float* kHeadPtr = &canvasK[pos, h * headDim])
                        {
                            SimdKernels.RmsNorm(kHeadPtr, kHeadPtr, (float*)kn.DataPtr, headDim, _config.RmsNormEps);
                        }
                    }
                }
            }

            // Bidirectional canvas attention + prefix prompt attention
            int promptLen = PromptLength;
            int prefixAttendable = isFull
                ? promptLen
                : Math.Min(promptLen, Math.Max(0, _config.SlidingWindowSize - 1));

            int totalKeys = prefixAttendable + canvasLen;
            float scale = 1.0f / MathF.Sqrt(headDim);
            int gqaRatio = qHeads / kvHeads;

            var attnOut = new float[canvasLen, qDim];
            var scores = new float[totalKeys];

            for (int pos = 0; pos < canvasLen; pos++)
            {
                for (int h = 0; h < qHeads; h++)
                {
                    int kvH = h / gqaRatio;

                    // 1. Cross-attend to prefix prompt KV
                    int promptStart = promptLen - prefixAttendable;
                    for (int k = 0; k < prefixAttendable; k++)
                    {
                        var promptK = _promptKCache[il][promptStart + k];
                        float dot = 0f;
                        for (int d = 0; d < headDim; d++)
                        {
                            dot += canvasQ[pos, h * headDim + d] * promptK[kvH * headDim + d];
                        }
                        scores[k] = dot * scale;
                    }

                    // 2. Bidirectional attention across canvas
                    for (int k = 0; k < canvasLen; k++)
                    {
                        float dot = 0f;
                        for (int d = 0; d < headDim; d++)
                        {
                            dot += canvasQ[pos, h * headDim + d] * canvasK[k, kvH * headDim + d];
                        }
                        scores[prefixAttendable + k] = dot * scale;
                    }

                    fixed (float* sPtr = scores)
                    {
                        SimdKernels.SoftmaxInPlace(sPtr, totalKeys);
                    }

                    // Weighted sum
                    for (int d = 0; d < headDim; d++)
                    {
                        float acc = 0f;
                        // Prompt V
                        for (int k = 0; k < prefixAttendable; k++)
                        {
                            var promptV = _promptVCache[il][promptStart + k];
                            acc += scores[k] * promptV[kvH * headDim + d];
                        }

                        // Canvas V
                        for (int k = 0; k < canvasLen; k++)
                        {
                            acc += scores[prefixAttendable + k] * canvasV[k, kvH * headDim + d];
                        }
                        attnOut[pos, h * headDim + d] = acc;
                    }
                }
            }

            // Project out and residual add
            for (int pos = 0; pos < canvasLen; pos++)
            {
                var headOut = new float[qDim];
                for (int d = 0; d < qDim; d++) headOut[d] = attnOut[pos, d];

                var proj = new float[HiddenDim];
                fixed (float* inPtr = headOut, outPtr = proj)
                {
                    SimdKernels.MatVec(outPtr, layer.Wo!.Value.DataPtr, inPtr, HiddenDim, qDim, layer.Wo.Value.DType);
                }

                // Attn post-norm + residual
                if (layer.AttnPostNorm is { } apn)
                {
                    fixed (float* ptr = proj)
                    {
                        SimdKernels.RmsNorm(ptr, ptr, (float*)apn.DataPtr, HiddenDim, _config.RmsNormEps);
                    }
                }
                for (int d = 0; d < HiddenDim; d++) canvas[pos, d] += proj[d];

                // FFN
                for (int d = 0; d < HiddenDim; d++) curRow[d] = canvas[pos, d];
                var ffnNormed = new float[HiddenDim];
                fixed (float* inPtr = curRow, outPtr = ffnNormed)
                {
                    float* weightPtr = (float*)layer.FfnNorm!.Value.DataPtr;
                    SimdKernels.RmsNorm(outPtr, inPtr, weightPtr, HiddenDim, _config.RmsNormEps);
                }

                var ffnOut = DenseFfn(layer, ffnNormed);
                if (layer.FfnGateExps is not null)
                {
                    var moeOut = MoeFfn(layer, ffnNormed);
                    for (int d = 0; d < HiddenDim; d++) ffnOut[d] += moeOut[d];
                }

                if (layer.FfnPostNorm is { } fpn)
                {
                    fixed (float* ptr = ffnOut)
                    {
                        SimdKernels.RmsNorm(ptr, ptr, (float*)fpn.DataPtr, HiddenDim, _config.RmsNormEps);
                    }
                }

                for (int d = 0; d < HiddenDim; d++) canvas[pos, d] += ffnOut[d];
            }
        }

        // 3. Final norm, tied embedding projection, and softcapping
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
                float* weightPtr = (float*)_outputNorm.DataPtr;
                SimdKernels.RmsNorm(outPtr, inPtr, weightPtr, HiddenDim, _config.RmsNormEps);
            }

            fixed (float* inPtr = normedRow, outPtr = posLogits)
            {
                SimdKernels.MatVec(outPtr, _output.DataPtr, inPtr, vocabSize, HiddenDim, _output.DType);
            }

            // Softcap: tanh(l / softcap) * softcap
            for (int v = 0; v < vocabSize; v++)
            {
                float l = posLogits[v];
                float capped = MathF.Tanh(l / softcap) * softcap;
                logits[pos * vocabSize + v] = capped;
            }
        }

        return logits;
    }

    /// <summary>
    /// Exact soft-embedding sum E[pos] = EmbedScale * sum_v softmax(logits[pos] / temperature)[v] * Embed[v], over
    /// the whole vocabulary (no tail pruning, so step traces stay comparable with a reference). Vocabulary-major
    /// so each (possibly quantised) embedding row is dequantised once per step, not once per canvas position.
    /// The learned self-conditioning MLP that follows in the reference is NOT applied here (see the plan's open items).
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
                var dst = soft.AsSpan(pos * HiddenDim, HiddenDim);
                System.Numerics.Tensors.TensorPrimitives.MultiplyAdd(row, p, dst, dst);
            }
        }

        for (int i = 0; i < soft.Length; i++) soft[i] *= _config.EmbedScale;
        return soft;
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

    private float[] DenseFfn(DiffusionGemmaLayerTensors layer, float[] normedInput)
    {
        int interDim = (int)layer.FfnGate!.Value.Info.Dimensions[1];
        var gate = new float[interDim];
        var up = new float[interDim];
        fixed (float* inPtr = normedInput, gPtr = gate, uPtr = up)
        {
            SimdKernels.MatVec(gPtr, layer.FfnGate.Value.DataPtr, inPtr, interDim, HiddenDim, layer.FfnGate.Value.DType);
            SimdKernels.MatVec(uPtr, layer.FfnUp!.Value.DataPtr, inPtr, interDim, HiddenDim, layer.FfnUp.Value.DType);
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
            SimdKernels.MatVec(outPtr, layer.FfnDown!.Value.DataPtr, inPtr, HiddenDim, interDim, layer.FfnDown.Value.DType);
        }
        return down;
    }

    private float[] MoeFfn(DiffusionGemmaLayerTensors layer, float[] normedInput)
    {
        int numExperts = _config.NumExperts;
        int topK = _config.NumExpertsUsed;
        int interDim = (int)layer.FfnGateExps!.Value.Info.Dimensions[1];

        var logits = new float[numExperts];
        fixed (float* inPtr = normedInput, outPtr = logits)
        {
            SimdKernels.MatVec(outPtr, layer.FfnGateInp!.Value.DataPtr, inPtr, numExperts, HiddenDim, layer.FfnGateInp.Value.DType);
        }

        int[] topIndices = DeepSeek4Graph.SelectTopKIndices(logits, topK);
        var weights = new float[topK];
        float sumWeight = 0f;
        for (int k = 0; k < topK; k++)
        {
            float e = MathF.Exp(logits[topIndices[k]]);
            weights[k] = e;
            sumWeight += e;
        }
        float invSum = 1.0f / MathF.Max(1e-6f, sumWeight);
        for (int k = 0; k < topK; k++) weights[k] *= invSum;

        var accumulated = new float[HiddenDim];
        var expertGate = new float[interDim];
        var expertUp = new float[interDim];
        var expertDown = new float[HiddenDim];

        for (int k = 0; k < topK; k++)
        {
            int expert = topIndices[k];
            float w = weights[k];

            PerExpertMatVec(layer.FfnGateExps!.Value, expert, normedInput, expertGate, interDim);
            PerExpertMatVec(layer.FfnUpExps!.Value, expert, normedInput, expertUp, interDim);

            for (int i = 0; i < interDim; i++)
            {
                float x = expertGate[i];
                float gelu = 0.5f * x * (1.0f + MathF.Tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
                expertGate[i] = gelu * expertUp[i];
            }

            PerExpertMatVecDown(layer.FfnDownExps!.Value, expert, expertGate, expertDown, interDim);
            for (int i = 0; i < HiddenDim; i++) accumulated[i] += expertDown[i] * w;
        }

        return accumulated;
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

    public void Reset()
    {
        for (int i = 0; i < _config.NumLayers; i++)
        {
            _promptKCache[i].Clear();
            _promptVCache[i].Clear();
        }
    }
}
