using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine;

// ============================================================================================
// ALPHA / UNTESTED -- see GptOssAlpha.cs's file header for overall status/scope; everything
// there applies here. Forward-pass dispatch for gpt-oss: standard GQA attention with biased
// QKVO, attention sinks, alternating sliding/full-window masking, and biased MoE with the OAI
// SwiGLU + select-then-softmax gating. NEVER RUN -- no gpt-oss GGUF was loaded while writing
// this (a download was started in parallel, per explicit user direction not to wait for it).
//
// KNOWN, DELIBERATE SIMPLIFICATIONS/GAPS:
//  1. RoPE is plain (non-YaRN) here even though the external plan the user shared claims
//     gpt-oss-20b uses YaRN (factor 32, orig-ctx 4096, beta_fast 32, beta_slow 1, theta 150000).
//     Those specific numbers were NOT independently confirmed against a real GGUF this session
//     (see docs/060-gpt-oss-implementation-plan.md's own caveat on this) -- if true, this is a
//     real, known gap parallel to deepseek32's now-CLOSED YaRN gap, and porting the SAME
//     already-verified YaRN formula chain (this codebase's `ApplyYarnRope` pattern in
//     DeepSeek32ForwardPass.cs) is the natural fix once metadata confirms it's needed. Left
//     plain here deliberately rather than guessing YaRN parameters into the constructor from an
//     unverified external source.
//  2. Per-layer RoPE frequency base (global freq_base vs. SWA-layer freq_base_swa,
//     GptOssHyperparams.RopeFreqBase/RopeFreqBaseSwa) IS implemented (the one piece of the
//     reference's per-layer RoPE handling that's simple and unambiguous from the reference
//     alone) -- this is independent of gap 1 above.
//  3. No batched/packed prefill -- Prefill loops Forward per token, same as every other alpha
//     forward pass in this codebase so far.
// ============================================================================================

/// <summary>
/// ALPHA/UNTESTED. Implements <see cref="IForwardPass"/> for gpt-oss: GQA attention with biased
/// QKVO projections, attention sinks, alternating sliding-window/full masking, and MoE with
/// per-expert biases, the OAI SwiGLU activation, and select-then-softmax gating. See this file's
/// header for gaps.
/// </summary>
public sealed unsafe class GptOssForwardPass : IForwardPass
{
    private readonly GgufModel _model;
    private readonly GptOssHyperparams _hp;
    private readonly GptOssTensorSet _tensors;
    private readonly int _embedDim, _numHeads, _numHeadsKv, _headDim, _numLayer;

    // Raw per-layer KV cache: one [numHeadsKv * headDim]-wide K/V pair per token. Global layers
    // attend over the full cache; SWA layers attend only over the trailing SlidingWindow tokens
    // (implemented as a masking rule at attention time, not a physically-truncated cache -- the
    // reference's own real paged-cache semantics for SWA are more involved than this alpha
    // attempts; see docs/060-...md Phase 6 for the KV-cache-policy question left open there).
    private readonly List<float[]>[] _kCache, _vCache;

    public GptOssForwardPass(GgufModel model, GptOssHyperparams hp)
    {
        _model = model;
        _hp = hp;
        _tensors = GptOssTensorSet.Load(model, hp);
        _embedDim = hp.EmbedDim;
        _numHeads = hp.NumHeads;
        _numHeadsKv = hp.NumHeadsKv;
        _headDim = hp.HeadDim;
        _numLayer = hp.NumLayer;

        _kCache = new List<float[]>[_numLayer];
        _vCache = new List<float[]>[_numLayer];
        for (int il = 0; il < _numLayer; il++)
        {
            _kCache[il] = [];
            _vCache[il] = [];
        }
    }

    public int VocabSize => _hp.VocabSize;
    public int MaxSeqLen => 1 << 20;

    public ReadOnlySpan<float> Forward(int token, int position)
    {
        var cur = new float[_embedDim];
        EmbedTokenInto(token, cur);

        for (int il = 0; il < _numLayer; il++)
        {
            var layer = _tensors.Layers[il];
            var residual = (float[])cur.Clone();

            var attnNormed = new float[_embedDim];
            fixed (float* inPtr = cur, outPtr = attnNormed)
            {
                float* weightPtr = (float*)layer.AttnNorm!.Value.DataPtr;
                SimdKernels.RmsNorm(outPtr, inPtr, weightPtr, _embedDim, _hp.RmsNormEps);
            }

            var attnOut = Attention(il, layer, attnNormed, position);
            for (int i = 0; i < _embedDim; i++) cur[i] = residual[i] + attnOut[i];

            residual = (float[])cur.Clone();
            var ffnNormed = new float[_embedDim];
            fixed (float* inPtr = cur, outPtr = ffnNormed)
            {
                float* weightPtr = (float*)layer.AttnPostNorm!.Value.DataPtr;
                SimdKernels.RmsNorm(outPtr, inPtr, weightPtr, _embedDim, _hp.RmsNormEps);
            }

            var ffnOut = MoeFfn(layer, ffnNormed);
            for (int i = 0; i < _embedDim; i++) cur[i] = residual[i] + ffnOut[i];
        }

        var normed = new float[_embedDim];
        fixed (float* inPtr = cur, outPtr = normed)
        {
            float* weightPtr = (float*)_tensors.OutputNorm.DataPtr;
            SimdKernels.RmsNorm(outPtr, inPtr, weightPtr, _embedDim, _hp.RmsNormEps);
        }

        var logits = new float[VocabSize];
        fixed (float* inPtr = normed, outPtr = logits)
        {
            SimdKernels.MatVec(outPtr, _tensors.Output.DataPtr, inPtr, VocabSize, _embedDim, _tensors.Output.DType);
        }
        return logits;
    }

    private float[] Attention(int il, GptOssLayerTensors layer, float[] normedInput, int position)
    {
        bool isSwa = _hp.IsSwaLayer(il);
        float freqBase = isSwa ? _hp.RopeFreqBaseSwa : _hp.RopeFreqBase;

        var q = new float[_numHeads * _headDim];
        var k = new float[_numHeadsKv * _headDim];
        var v = new float[_numHeadsKv * _headDim];
        fixed (float* inPtr = normedInput, qPtr = q, kPtr = k, vPtr = v)
        {
            SimdKernels.MatVec(qPtr, layer.Wq!.Value.DataPtr, inPtr, _numHeads * _headDim, _embedDim, layer.Wq.Value.DType);
            SimdKernels.MatVec(kPtr, layer.Wk!.Value.DataPtr, inPtr, _numHeadsKv * _headDim, _embedDim, layer.Wk.Value.DType);
            SimdKernels.MatVec(vPtr, layer.Wv!.Value.DataPtr, inPtr, _numHeadsKv * _headDim, _embedDim, layer.Wv.Value.DType);
        }
        AddBiasIfPresent(q, layer.WqB);
        AddBiasIfPresent(k, layer.WkB);
        AddBiasIfPresent(v, layer.WvB);

        for (int h = 0; h < _numHeads; h++)
        {
            ApplyRopeNeox(q.AsSpan(h * _headDim, _headDim), position, freqBase);
        }
        for (int h = 0; h < _numHeadsKv; h++)
        {
            ApplyRopeNeox(k.AsSpan(h * _headDim, _headDim), position, freqBase);
        }

        _kCache[il].Add(k);
        _vCache[il].Add(v);

        int cacheLen = _kCache[il].Count;
        int windowStart = isSwa && _hp.SlidingWindow > 0 ? Math.Max(0, cacheLen - _hp.SlidingWindow) : 0;
        int numKeys = cacheLen - windowStart;

        int groupSize = _numHeads / _numHeadsKv; // GQA: groupSize Q heads share one KV head.
        var attnOut = new float[_numHeads * _headDim];
        var scores = new float[numKeys];
        float scale = 1f / MathF.Sqrt(_headDim);
        float? sink = null;

        for (int h = 0; h < _numHeads; h++)
        {
            int kvHead = h / groupSize;
            if (layer.AttnSinks is { } sinksTensor)
            {
                sink = ((float*)sinksTensor.DataPtr)[h];
            }

            var qHead = q.AsSpan(h * _headDim, _headDim);
            for (int t = 0; t < numKeys; t++)
            {
                var kt = _kCache[il][windowStart + t].AsSpan(kvHead * _headDim, _headDim);
                float dot = 0f;
                for (int d = 0; d < _headDim; d++) dot += qHead[d] * kt[d];
                scores[t] = dot * scale;
            }
            GptOssGraph.SoftmaxWithSink(scores, sink);

            var outHead = attnOut.AsSpan(h * _headDim, _headDim);
            outHead.Clear();
            for (int t = 0; t < numKeys; t++)
            {
                var vt = _vCache[il][windowStart + t].AsSpan(kvHead * _headDim, _headDim);
                float w = scores[t];
                for (int d = 0; d < _headDim; d++) outHead[d] += vt[d] * w;
            }
        }

        var result = new float[_embedDim];
        fixed (float* inPtr = attnOut, outPtr = result)
        {
            SimdKernels.MatVec(outPtr, layer.Wo!.Value.DataPtr, inPtr, _embedDim, _numHeads * _headDim, layer.Wo.Value.DType);
        }
        AddBiasIfPresent(result, layer.WoB);
        return result;
    }

    private float[] MoeFfn(GptOssLayerTensors layer, float[] normedInput)
    {
        int numExperts = _hp.NumExperts;
        int topK = _hp.NumExpertsUsed;

        var logits = new float[numExperts];
        fixed (float* inPtr = normedInput, outPtr = logits)
        {
            SimdKernels.MatVec(outPtr, layer.FfnGateInp!.Value.DataPtr, inPtr, numExperts, _embedDim, layer.FfnGateInp.Value.DType);
        }
        AddBiasIfPresent(logits, layer.FfnGateInpB);

        var expertIndices = new int[topK];
        var expertWeights = new float[topK];
        GptOssGraph.SelectThenSoftmaxGate(logits, topK, expertIndices, expertWeights);

        var result = new float[_embedDim];
        int ffnDim = _hp.ExpertFeedForwardLength;
        var gate = new float[ffnDim];
        var up = new float[ffnDim];
        var activated = new float[ffnDim];
        var down = new float[_embedDim];
        for (int k = 0; k < topK; k++)
        {
            int e = expertIndices[k];
            float w = expertWeights[k];

            PerExpertMatVec(layer.FfnGateExps!.Value, e, normedInput, gate, ffnDim);
            AddPerExpertBiasIfPresent(gate, layer.FfnGateExpsB, e, ffnDim);
            PerExpertMatVec(layer.FfnUpExps!.Value, e, normedInput, up, ffnDim);
            AddPerExpertBiasIfPresent(up, layer.FfnUpExpsB, e, ffnDim);

            GptOssGraph.SwigluOai(gate, up, activated);

            PerExpertMatVecDown(layer.FfnDownExps!.Value, e, activated, down, ffnDim);
            AddPerExpertBiasIfPresent(down, layer.FfnDownExpsB, e, _embedDim);

            for (int i = 0; i < _embedDim; i++) result[i] += down[i] * w;
        }
        return result;
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

    /// <summary>
    /// Adds one expert's slice of a per-expert bias tensor (shape [outDim, numExperts], row-major
    /// by expert -- same slicing convention as the per-expert weight tensors). Per-expert MoE
    /// biases are a tensor shape not seen elsewhere in this codebase's MoE path (see
    /// docs/060-...md's architecture-mapping table) -- this slicing was NOT independently
    /// re-verified against a real GGUF's actual bias tensor layout.
    /// </summary>
    private static void AddPerExpertBiasIfPresent(float[] data, DeepSeek4TensorRef? bias, int expert, int dim)
    {
        if (bias is not { } b) return;
        float* biasPtr = (float*)b.DataPtr + (long)expert * dim;
        for (int i = 0; i < dim; i++) data[i] += biasPtr[i];
    }

    private static void AddBiasIfPresent(float[] data, DeepSeek4TensorRef? bias)
    {
        if (bias is not { } b) return;
        float* biasPtr = (float*)b.DataPtr;
        for (int i = 0; i < data.Length; i++) data[i] += biasPtr[i];
    }

    /// <summary>
    /// NEOX-style RoPE (rotates pairs (i, i+half)) -- confirmed as gpt-oss's convention via
    /// llama-model.cpp's rope-type switch (LLM_ARCH_OPENAI_MOE groups with
    /// LLAMA_ROPE_TYPE_NEOX, alongside qwen3next/mimo2/mellum/etc, NOT with the
    /// LLAMA_ROPE_TYPE_NORM/interleaved group llama/deepseek/etc fall into). Originally written
    /// as the interleaved variant by mistake (defaulted without checking) and caught/fixed
    /// 2026-09-02 by actually reading llama-model.cpp's rope-type table before trusting the
    /// assumption -- see docs/060-gpt-oss-implementation-plan.md's progress log.
    /// </summary>
    private static void ApplyRopeNeox(Span<float> x, int position, float freqBase)
    {
        int dim = x.Length;
        int half = dim / 2;
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Pow(freqBase, -2f * i / dim);
            float theta = position * freq;
            float cos = MathF.Cos(theta), sin = MathF.Sin(theta);
            float x0 = x[i], x1 = x[i + half];
            x[i] = x0 * cos - x1 * sin;
            x[i + half] = x0 * sin + x1 * cos;
        }
    }

    private void EmbedTokenInto(int token, Span<float> dest)
    {
        var info = _tensors.TokEmbd.Info;
        int bytesPerRow = (_embedDim / DTypeInfo.BlockSize(info.DType)) * DTypeInfo.BytesPerBlock(info.DType);
        byte* rowPtr = _tensors.TokEmbd.DataPtr + (long)token * bytesPerRow;
        fixed (float* destPtr = dest)
        {
            SimdKernels.DequantRow(rowPtr, destPtr, _embedDim, info.DType);
        }
    }

    // ── IForwardPass minimal surface ────────────────────────────────────────────────────────

    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
    {
        ReadOnlySpan<float> last = default;
        for (int i = 0; i < tokens.Count; i++) last = Forward(tokens[i], startPos + i).ToArray();
        return last;
    }

    public void TruncateTo(int length)
    {
        if (length == 0) { ResetCache(); return; }
        int current = _kCache.Length > 0 ? _kCache[0].Count : 0;
        if (length != current)
        {
            throw new NotSupportedException("GptOssForwardPass (alpha): only full reset (TruncateTo(0)) or a no-op is supported.");
        }
    }

    public void ResetCache()
    {
        for (int il = 0; il < _numLayer; il++)
        {
            _kCache[il].Clear();
            _vCache[il].Clear();
        }
    }

    public void Dispose() { }
}
