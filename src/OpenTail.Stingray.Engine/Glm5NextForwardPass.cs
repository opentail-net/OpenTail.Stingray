using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine;

// ============================================================================================
// ALPHA / UNTESTED -- Forward-pass implementation for GLM5-Next (GLM-5.3-Flash). Ported 2026-10-03
// from examples/llama.cpp/llama.cpp/src/models/glm5-next.cpp.
//
// Key components:
//  1. 4-stream multi-head hyper-connections (mHC) with Sinkhorn balancing on the 4x4 combine matrix.
//  2. 34 KDA layers (Gated DeltaNet linear attention with 1D causal convolution and delta rule update).
//  3. 11 MLA layers (compressed latent MLA with 4-token K-pool DSA indexer).
//  4. 288-expert MoE (top-8 routed, sigmoid gating, normalized weights, scale 2.5) with SwiGLU
//     intermediate values clamped to 10.0.
// ============================================================================================

/// <summary>
/// Forward pass for GLM5-Next ("glm5next" GGUF architecture).
/// </summary>
public sealed unsafe class Glm5NextForwardPass : IForwardPass
{
    private readonly GgufModel _model;
    private readonly Glm5NextHyperparams _hp;
    private readonly Glm5NextTensorSet _tensors;
    private readonly int _embedDim, _numHeads, _numLayer;

    // 4-stream mHC residual states
    private readonly float[][] _streams;

    // KDA recurrent states: [layer][head, headDim, headDim]
    private readonly float[][,,] _kdaState;
    // KDA conv1d buffers: [layer][3 (q,k,v), dInner, 3 (history)]
    private readonly float[][,,] _kdaConvState;

    // MLA K/V cache: [layer][token][kvLoraRank]
    private readonly List<float[]>[] _mlaKvCache;

    // K-pool indexer caches:
    // Staging tokens for current pool (up to 4): [layer][token]
    private readonly List<(float[] Key, float[] Gate)>[] _kpoolStaging;
    // Completed pooled keys: [layer][poolIdx] -> 128-dim pooled key
    private readonly List<float[]>[] _kpoolCache;

    public Glm5NextForwardPass(GgufModel model, Glm5NextHyperparams hp)
    {
        _model = model;
        _hp = hp;
        _tensors = Glm5NextTensorSet.Load(model, hp);
        _embedDim = hp.EmbedDim;
        _numHeads = hp.NumHeads;
        _numLayer = hp.NumLayer;

        _streams = new float[hp.HcMult][];
        for (int s = 0; s < hp.HcMult; s++) _streams[s] = new float[_embedDim];

        _kdaState = new float[_numLayer][,,];
        _kdaConvState = new float[_numLayer][,,];
        _mlaKvCache = new List<float[]>[_numLayer];
        _kpoolStaging = new List<(float[], float[])>[_numLayer];
        _kpoolCache = new List<float[]>[_numLayer];

        for (int il = 0; il < _numLayer; il++)
        {
            if (hp.IsRecurrent(il))
            {
                _kdaState[il] = new float[hp.NumHeads, hp.HeadDimKda, hp.HeadDimKda];
                _kdaConvState[il] = new float[3, hp.DInner, hp.SsmDConv - 1];
            }
            else
            {
                _mlaKvCache[il] = [];
                _kpoolStaging[il] = [];
                _kpoolCache[il] = [];
            }
        }

        VocabSize = (int)_tensors.TokEmbd.Info.Dimensions[1];
    }

    public int VocabSize { get; private set; }
    public int MaxSeqLen => 1 << 20;

    public ReadOnlySpan<float> Forward(int token, int position)
    {
        var embedded = new float[_embedDim];
        EmbedTokenInto(token, embedded);

        // The mHC streams are the current token's residual state, not sequence history (that lives in
        // the KDA / MLA / K-pool caches): every token starts from its own embedding replicated x hc.
        for (int s = 0; s < _hp.HcMult; s++)
        {
            embedded.CopyTo(_streams[s], 0);
        }

        for (int il = 0; il < _numLayer; il++)
        {
            var layer = _tensors.Layers[il];

            // 1. Attention sublayer with mHC
            var (preInAttn, postAttn, combAttn) = ComputeHcPre(_streams, layer.HcAttnFn!.Value, layer.HcAttnScale!.Value, layer.HcAttnBase!.Value);

            var attnNormed = new float[_embedDim];
            fixed (float* inPtr = preInAttn, outPtr = attnNormed)
            {
                float* weightPtr = (float*)layer.AttnNorm!.Value.DataPtr;
                SimdKernels.RmsNorm(outPtr, inPtr, weightPtr, _embedDim, _hp.RmsNormEps);
            }

            var sublayerOutAttn = _hp.IsRecurrent(il)
                ? KdaLayer(il, layer, attnNormed)
                : MlaLayer(il, layer, attnNormed, position);

            ApplyHcPost(_streams, sublayerOutAttn, postAttn, combAttn);

            // 2. FFN sublayer with mHC
            var (preInFfn, postFfn, combFfn) = ComputeHcPre(_streams, layer.HcFfnFn!.Value, layer.HcFfnScale!.Value, layer.HcFfnBase!.Value);

            var ffnNormed = new float[_embedDim];
            fixed (float* inPtr = preInFfn, outPtr = ffnNormed)
            {
                float* weightPtr = (float*)layer.FfnNorm!.Value.DataPtr;
                SimdKernels.RmsNorm(outPtr, inPtr, weightPtr, _embedDim, _hp.RmsNormEps);
            }

            var sublayerOutFfn = il < _hp.LeadingDenseBlockCount
                ? DenseFfn(layer, ffnNormed)
                : MoeFfn(layer, ffnNormed);

            ApplyHcPost(_streams, sublayerOutFfn, postFfn, combFfn);
        }

        // Final mean collapse of 4 residual streams
        var meanStream = new float[_embedDim];
        float invHc = 1.0f / _hp.HcMult;
        for (int s = 0; s < _hp.HcMult; s++)
        {
            for (int i = 0; i < _embedDim; i++) meanStream[i] += _streams[s][i] * invHc;
        }

        var outputNormed = new float[_embedDim];
        fixed (float* inPtr = meanStream, outPtr = outputNormed)
        {
            float* weightPtr = (float*)_tensors.OutputNorm.DataPtr;
            SimdKernels.RmsNorm(outPtr, inPtr, weightPtr, _embedDim, _hp.RmsNormEps);
        }

        var logits = new float[VocabSize];
        fixed (float* inPtr = outputNormed, outPtr = logits)
        {
            SimdKernels.MatVec(outPtr, _tensors.Output.DataPtr, inPtr, VocabSize, _embedDim, _tensors.Output.DType);
        }
        return logits;
    }

    private void EmbedTokenInto(int token, float[] destination)
    {
        var tensor = _tensors.TokEmbd;
        int cols = _embedDim;
        long bytesPerRow = ((long)cols / DTypeInfo.BlockSize(tensor.DType)) * DTypeInfo.BytesPerBlock(tensor.DType);
        byte* src = tensor.DataPtr + (long)token * bytesPerRow;
        fixed (float* dst = destination)
        {
            SimdKernels.DequantRow(src, dst, cols, tensor.DType);
        }
    }

    private (float[] MixedInput, float[] Post, float[,] Comb) ComputeHcPre(
        float[][] streams, DeepSeek4TensorRef hcFn, DeepSeek4TensorRef hcScale, DeepSeek4TensorRef hcBase)
    {
        int hc = _hp.HcMult; // 4
        int hcDim = hc * _embedDim;

        var flat = new float[hcDim];
        for (int s = 0; s < hc; s++)
        {
            streams[s].CopyTo(flat.AsSpan(s * _embedDim, _embedDim));
        }

        var flatNorm = new float[hcDim];
        float ss = 0f;
        for (int i = 0; i < hcDim; i++) ss += flat[i] * flat[i];
        float invRms = 1f / MathF.Sqrt(ss / hcDim + _hp.RmsNormEps);
        for (int i = 0; i < hcDim; i++) flatNorm[i] = flat[i] * invRms;

        int mixDim = (2 + hc) * hc; // 24
        var mixes = new float[mixDim];
        fixed (float* inPtr = flatNorm, outPtr = mixes)
        {
            SimdKernels.MatVec(outPtr, hcFn.DataPtr, inPtr, mixDim, hcDim, hcFn.DType);
        }

        float* scalePtr = (float*)hcScale.DataPtr;
        float* basePtr = (float*)hcBase.DataPtr;

        float scalePre = scalePtr[0];
        float scalePost = scalePtr[1];
        float scaleComb = scalePtr[2];

        // 1. Pre weights (mixes[0..3])
        var pre = new float[hc];
        for (int s = 0; s < hc; s++)
        {
            float val = mixes[s] * scalePre + basePtr[s];
            pre[s] = (1f / (1f + MathF.Exp(-val))) + _hp.HcEps;
        }

        // 2. Post weights (mixes[4..7])
        var post = new float[hc];
        for (int d = 0; d < hc; d++)
        {
            float val = mixes[hc + d] * scalePost + basePtr[hc + d];
            post[d] = 2.0f / (1f + MathF.Exp(-val));
        }

        // 3. Comb matrix (mixes[8..23], 4x4)
        var comb = new float[hc, hc];
        for (int s = 0; s < hc; s++)
        {
            for (int d = 0; d < hc; d++)
            {
                int idx = 2 * hc + s * hc + d;
                comb[s, d] = mixes[idx] * scaleComb + basePtr[idx];
            }
        }

        // Sinkhorn balancing on comb matrix
        SinkhornBalancing(comb, hc, _hp.HcSinkhornIters, _hp.HcEps);

        // Mix input streams down to 1 vector
        var mixedInput = new float[_embedDim];
        for (int s = 0; s < hc; s++)
        {
            float w = pre[s];
            for (int i = 0; i < _embedDim; i++) mixedInput[i] += streams[s][i] * w;
        }

        return (mixedInput, post, comb);
    }

    private static void SinkhornBalancing(float[,] comb, int hc, int iterations, float eps)
    {
        // Row softmax
        for (int s = 0; s < hc; s++)
        {
            float max = float.NegativeInfinity;
            for (int d = 0; d < hc; d++) if (comb[s, d] > max) max = comb[s, d];
            float sum = 0f;
            for (int d = 0; d < hc; d++)
            {
                float e = MathF.Exp(comb[s, d] - max);
                comb[s, d] = e;
                sum += e;
            }
            float inv = 1f / sum;
            for (int d = 0; d < hc; d++) comb[s, d] = comb[s, d] * inv + eps;
        }

        // Alternating col/row normalization
        for (int iter = 0; iter < iterations; iter++)
        {
            // Normalize columns
            for (int d = 0; d < hc; d++)
            {
                float colSum = eps;
                for (int s = 0; s < hc; s++) colSum += comb[s, d];
                float inv = 1f / colSum;
                for (int s = 0; s < hc; s++) comb[s, d] *= inv;
            }
            // Normalize rows
            for (int s = 0; s < hc; s++)
            {
                float rowSum = eps;
                for (int d = 0; d < hc; d++) rowSum += comb[s, d];
                float inv = 1f / rowSum;
                for (int d = 0; d < hc; d++) comb[s, d] *= inv;
            }
        }
    }

    private void ApplyHcPost(float[][] streams, float[] sublayerOut, float[] post, float[,] comb)
    {
        int hc = _hp.HcMult;
        var oldStreams = new float[hc][];
        for (int s = 0; s < hc; s++) oldStreams[s] = (float[])streams[s].Clone();

        for (int d = 0; d < hc; d++)
        {
            float p = post[d];
            for (int i = 0; i < _embedDim; i++)
            {
                float sum = sublayerOut[i] * p;
                for (int s = 0; s < hc; s++)
                {
                    sum += oldStreams[s][i] * comb[s, d];
                }
                streams[d][i] = sum;
            }
        }
    }

    private float[] KdaLayer(int il, Glm5NextLayerTensors layer, float[] normedInput)
    {
        int dInner = _hp.DInner;
        int headDim = _hp.HeadDimKda;
        int numHeads = _hp.NumHeads;

        // 1. Projections
        var qProj = new float[dInner];
        var kProj = new float[dInner];
        var vProj = new float[dInner];

        fixed (float* inPtr = normedInput, qPtr = qProj, kPtr = kProj, vPtr = vProj)
        {
            SimdKernels.MatVec(qPtr, layer.Wq!.Value.DataPtr, inPtr, dInner, _embedDim, layer.Wq.Value.DType);
            SimdKernels.MatVec(kPtr, layer.Wk!.Value.DataPtr, inPtr, dInner, _embedDim, layer.Wk.Value.DType);
            SimdKernels.MatVec(vPtr, layer.Wv!.Value.DataPtr, inPtr, dInner, _embedDim, layer.Wv.Value.DType);
        }

        // 2. 1D Causal convolution on Q, K, V
        var qConv = ApplyConv1d(il, 0, layer.SsmQConv!.Value, qProj);
        var kConv = ApplyConv1d(il, 1, layer.SsmKConv!.Value, kProj);
        var vConv = ApplyConv1d(il, 2, layer.SsmVConv!.Value, vProj);

        // 3. Decay gate g1
        var fA = new float[headDim];
        fixed (float* inPtr = normedInput, outPtr = fA)
        {
            SimdKernels.MatVec(outPtr, layer.SsmFA!.Value.DataPtr, inPtr, headDim, _embedDim, layer.SsmFA.Value.DType);
        }
        var g1 = new float[dInner];
        fixed (float* inPtr = fA, outPtr = g1)
        {
            SimdKernels.MatVec(outPtr, layer.SsmFB!.Value.DataPtr, inPtr, dInner, headDim, layer.SsmFB.Value.DType);
        }
        float* dtBPtr = (float*)layer.SsmDtB!.Value.DataPtr;
        float* aPtr = (float*)layer.SsmA!.Value.DataPtr;

        for (int h = 0; h < numHeads; h++)
        {
            float a = aPtr[h];
            for (int d = 0; d < headDim; d++)
            {
                int idx = h * headDim + d;
                float val = g1[idx] + dtBPtr[idx];
                float softplus = MathF.Max(val, 0f) + MathF.Log(1f + MathF.Exp(-MathF.Abs(val)));
                g1[idx] = softplus * a;
            }
        }

        // 4. Beta
        var beta = new float[numHeads];
        fixed (float* inPtr = normedInput, outPtr = beta)
        {
            SimdKernels.MatVec(outPtr, layer.SsmBeta!.Value.DataPtr, inPtr, numHeads, _embedDim, layer.SsmBeta.Value.DType);
        }
        for (int h = 0; h < numHeads; h++) beta[h] = 1f / (1f + MathF.Exp(-beta[h]));

        // 5. L2 norm on Q and K per head
        for (int h = 0; h < numHeads; h++)
        {
            float qSum = 1e-6f, kSum = 1e-6f;
            for (int d = 0; d < headDim; d++)
            {
                int idx = h * headDim + d;
                qSum += qConv[idx] * qConv[idx];
                kSum += kConv[idx] * kConv[idx];
            }
            float invQ = 1f / MathF.Sqrt(qSum);
            float invK = 1f / MathF.Sqrt(kSum);
            for (int d = 0; d < headDim; d++)
            {
                int idx = h * headDim + d;
                qConv[idx] *= invQ;
                kConv[idx] *= invK;
            }
        }

        // 6. Recurrent delta-rule update and output calculation
        var scanOut = new float[dInner];
        for (int h = 0; h < numHeads; h++)
        {
            float b = beta[h];

            // Decay state and compute k^T * S
            var kDotS = new float[headDim];
            for (int i = 0; i < headDim; i++)
            {
                float decay = MathF.Exp(g1[h * headDim + i]);
                for (int j = 0; j < headDim; j++)
                {
                    _kdaState[il][h, i, j] *= decay;
                    kDotS[j] += kConv[h * headDim + i] * _kdaState[il][h, i, j];
                }
            }

            // Delta = beta * (v - k^T * S)
            var delta = new float[headDim];
            for (int j = 0; j < headDim; j++)
            {
                delta[j] = b * (vConv[h * headDim + j] - kDotS[j]);
            }

            // S += k (x) delta
            for (int i = 0; i < headDim; i++)
            {
                float ki = kConv[h * headDim + i];
                for (int j = 0; j < headDim; j++)
                {
                    _kdaState[il][h, i, j] += ki * delta[j];
                }
            }

            // O = q * S
            for (int j = 0; j < headDim; j++)
            {
                float sum = 0f;
                for (int i = 0; i < headDim; i++)
                {
                    sum += qConv[h * headDim + i] * _kdaState[il][h, i, j];
                }
                scanOut[h * headDim + j] = sum;
            }
        }

        // 7. Output gate g2 & RMSNorm
        var gA = new float[headDim];
        fixed (float* inPtr = normedInput, outPtr = gA)
        {
            SimdKernels.MatVec(outPtr, layer.SsmGA!.Value.DataPtr, inPtr, headDim, _embedDim, layer.SsmGA.Value.DType);
        }
        var g2 = new float[dInner];
        fixed (float* inPtr = gA, outPtr = g2)
        {
            SimdKernels.MatVec(outPtr, layer.SsmGB!.Value.DataPtr, inPtr, dInner, headDim, layer.SsmGB.Value.DType);
        }

        var normedScan = new float[dInner];
        fixed (float* inPtr = scanOut, outPtr = normedScan)
        {
            float* weightPtr = (float*)layer.SsmONorm!.Value.DataPtr;
            SimdKernels.RmsNorm(outPtr, inPtr, weightPtr, dInner, _hp.RmsNormEps);
        }

        for (int i = 0; i < dInner; i++)
        {
            float sig = 1f / (1f + MathF.Exp(-g2[i]));
            normedScan[i] *= sig;
        }

        var kdaOut = new float[_embedDim];
        fixed (float* inPtr = normedScan, outPtr = kdaOut)
        {
            SimdKernels.MatVec(outPtr, layer.Wo!.Value.DataPtr, inPtr, _embedDim, dInner, layer.Wo.Value.DType);
        }
        return kdaOut;
    }

    private float[] ApplyConv1d(int il, int qkvIdx, DeepSeek4TensorRef convWeight, float[] x)
    {
        int dInner = _hp.DInner;
        int dConv = _hp.SsmDConv; // 4
        int hist = dConv - 1; // 3

        var result = new float[dInner];
        float* wPtr = (float*)convWeight.DataPtr;

        for (int c = 0; c < dInner; c++)
        {
            float sum = x[c] * wPtr[c * dConv + hist];
            for (int k = 0; k < hist; k++)
            {
                sum += _kdaConvState[il][qkvIdx, c, k] * wPtr[c * dConv + k];
            }

            // SiLU activation
            float silu = sum / (1f + MathF.Exp(-sum));
            result[c] = silu;

            // Shift history
            for (int k = 0; k < hist - 1; k++)
            {
                _kdaConvState[il][qkvIdx, c, k] = _kdaConvState[il][qkvIdx, c, k + 1];
            }
            _kdaConvState[il][qkvIdx, c, hist - 1] = x[c];
        }
        return result;
    }

    private float[] MlaLayer(int il, Glm5NextLayerTensors layer, float[] normedInput, int position)
    {
        int qLoraRank = _hp.QLoraRank;
        int kvLoraRank = _hp.KvLoraRank;
        int numHeads = _hp.NumHeads;
        int headDimK = _hp.EmbedHeadKMla;
        int headDimV = _hp.EmbedHeadVMla;

        // Q projection
        var qr = new float[qLoraRank];
        fixed (float* inPtr = normedInput, outPtr = qr)
        {
            SimdKernels.MatVec(outPtr, layer.WqA!.Value.DataPtr, inPtr, qLoraRank, _embedDim, layer.WqA.Value.DType);
        }
        var qrNormed = new float[qLoraRank];
        fixed (float* inPtr = qr, outPtr = qrNormed)
        {
            float* weightPtr = (float*)layer.AttnQANorm!.Value.DataPtr;
            SimdKernels.RmsNorm(outPtr, inPtr, weightPtr, qLoraRank, _hp.RmsNormEps);
        }
        var q = new float[numHeads * headDimK];
        fixed (float* inPtr = qrNormed, outPtr = q)
        {
            SimdKernels.MatVec(outPtr, layer.WqB!.Value.DataPtr, inPtr, numHeads * headDimK, qLoraRank, layer.WqB.Value.DType);
        }

        // KV compression (NoPE MLA - no RoPE channels)
        var kvCmpr = new float[kvLoraRank];
        fixed (float* inPtr = normedInput, outPtr = kvCmpr)
        {
            SimdKernels.MatVec(outPtr, layer.WkvAMqa!.Value.DataPtr, inPtr, kvLoraRank, _embedDim, layer.WkvAMqa.Value.DType);
        }
        var kvCmprNormed = new float[kvLoraRank];
        fixed (float* inPtr = kvCmpr, outPtr = kvCmprNormed)
        {
            float* weightPtr = (float*)layer.AttnKvANorm!.Value.DataPtr;
            SimdKernels.RmsNorm(outPtr, inPtr, weightPtr, kvLoraRank, _hp.RmsNormEps);
        }
        _mlaKvCache[il].Add(kvCmprNormed);

        // K-pool DSA Indexer
        int kpool = _hp.IndexerKPool; // 4
        int indexerHeadDim = _hp.IndexerHeadSize; // 128
        int numIndexerHeads = _hp.IndexerNumHeads; // 32

        var ik = new float[indexerHeadDim];
        fixed (float* inPtr = normedInput, outPtr = ik)
        {
            SimdKernels.MatVec(outPtr, layer.IndexerAttnK!.Value.DataPtr, inPtr, indexerHeadDim, _embedDim, layer.IndexerAttnK.Value.DType);
        }
        LayerNormInPlace(ik, layer.IndexerKNorm!.Value, layer.IndexerKNormBias);

        var ig = new float[indexerHeadDim];
        fixed (float* inPtr = normedInput, outPtr = ig)
        {
            SimdKernels.MatVec(outPtr, layer.IndexerKPoolGate!.Value.DataPtr, inPtr, indexerHeadDim, _embedDim, layer.IndexerKPoolGate.Value.DType);
        }

        _kpoolStaging[il].Add((ik, ig));
        if (_kpoolStaging[il].Count == kpool)
        {
            // Completed a pool of 4 tokens -> pool them
            var pooledK = PoolTokens(_kpoolStaging[il], layer.IndexerKPoolApe!.Value, indexerHeadDim);
            _kpoolCache[il].Add(pooledK);
            _kpoolStaging[il].Clear();
        }

        // Absorb wk_b per head
        var qEff = new float[numHeads * kvLoraRank];
        for (int h = 0; h < numHeads; h++)
        {
            var qHead = q.AsSpan(h * headDimK, headDimK);
            var absorbed = new float[kvLoraRank];
            PerHeadMatVec(layer.WkB!.Value, h, kvLoraRank, headDimK, qHead, absorbed);
            absorbed.CopyTo(qEff.AsSpan(h * kvLoraRank, kvLoraRank));
        }

        // Full attention over the cached KV tokens. The K-pool DSA indexer selects at most IndexerTopK keys, so for
        // numKeys <= IndexerTopK selection keeps every key and full attention is exact. Beyond that the pooled-key
        // top-k selection (kpool blocks + tail) is not implemented: fail loudly rather than attend to everything.
        int numKeys = _mlaKvCache[il].Count;
        if (_hp.IndexerTopK > 0 && numKeys > _hp.IndexerTopK)
        {
            throw new NotSupportedException(
                $"glm5next: context of {numKeys} keys exceeds indexer.top_k={_hp.IndexerTopK}; K-pool sparse key " +
                "selection is not implemented, so results past this length would be wrong.");
        }
        float kqScale = 1.0f / MathF.Sqrt(headDimK);
        var attnOut = new float[numHeads * headDimV];
        var scores = new float[numKeys];

        for (int h = 0; h < numHeads; h++)
        {
            var qHead = qEff.AsSpan(h * kvLoraRank, kvLoraRank);
            for (int t = 0; t < numKeys; t++)
            {
                var kt = _mlaKvCache[il][t];
                float dot = 0f;
                for (int d = 0; d < kvLoraRank; d++) dot += qHead[d] * kt[d];
                scores[t] = dot * kqScale;
            }
            fixed (float* scoresPtr = scores)
            {
                SimdKernels.SoftmaxInPlace(scoresPtr, numKeys);
            }
            var weighted = new float[kvLoraRank];
            for (int t = 0; t < numKeys; t++)
            {
                var vt = _mlaKvCache[il][t];
                float w = scores[t];
                for (int d = 0; d < kvLoraRank; d++) weighted[d] += vt[d] * w;
            }
            var outHead = attnOut.AsSpan(h * headDimV, headDimV);
            PerHeadMatVec(layer.WvB!.Value, h, headDimV, kvLoraRank, weighted, outHead);
        }

        var result = new float[_embedDim];
        fixed (float* inPtr = attnOut, outPtr = result)
        {
            SimdKernels.MatVec(outPtr, layer.Wo!.Value.DataPtr, inPtr, _embedDim, numHeads * headDimV, layer.Wo.Value.DType);
        }
        return result;
    }

    private static float[] PoolTokens(List<(float[] Key, float[] Gate)> staging, DeepSeek4TensorRef apeTensor, int headDim)
    {
        int kpool = staging.Count; // 4
        float* apePtr = (float*)apeTensor.DataPtr;
        var pooled = new float[headDim];

        for (int d = 0; d < headDim; d++)
        {
            float max = float.NegativeInfinity;
            for (int r = 0; r < kpool; r++)
            {
                float logit = staging[r].Gate[d] + apePtr[r * headDim + d];
                if (logit > max) max = logit;
            }
            float sum = 0f;
            for (int r = 0; r < kpool; r++)
            {
                float e = MathF.Exp((staging[r].Gate[d] + apePtr[r * headDim + d]) - max);
                sum += e;
            }
            float inv = 1f / sum;
            for (int r = 0; r < kpool; r++)
            {
                float prob = MathF.Exp((staging[r].Gate[d] + apePtr[r * headDim + d]) - max) * inv;
                pooled[d] += staging[r].Key[d] * prob;
            }
        }
        return pooled;
    }

    private float[] DenseFfn(Glm5NextLayerTensors layer, float[] normedInput)
    {
        int interDim = (int)layer.FfnGate!.Value.Info.Dimensions[1];
        var gate = new float[interDim];
        var up = new float[interDim];
        fixed (float* inPtr = normedInput, gPtr = gate, uPtr = up)
        {
            SimdKernels.MatVec(gPtr, layer.FfnGate.Value.DataPtr, inPtr, interDim, _embedDim, layer.FfnGate.Value.DType);
            SimdKernels.MatVec(uPtr, layer.FfnUp!.Value.DataPtr, inPtr, interDim, _embedDim, layer.FfnUp.Value.DType);
        }

        float clamp = _hp.SwiGluClampExp;
        for (int i = 0; i < interDim; i++)
        {
            float g = gate[i];
            float silu = g / (1f + MathF.Exp(-g));
            float val = silu * up[i];
            if (val > clamp) val = clamp;
            else if (val < -clamp) val = -clamp;
            gate[i] = val;
        }

        var down = new float[_embedDim];
        fixed (float* inPtr = gate, outPtr = down)
        {
            SimdKernels.MatVec(outPtr, layer.FfnDown!.Value.DataPtr, inPtr, _embedDim, interDim, layer.FfnDown.Value.DType);
        }
        return down;
    }

    private float[] MoeFfn(Glm5NextLayerTensors layer, float[] normedInput)
    {
        int numExperts = _hp.NumExperts;
        int topK = _hp.NumExpertsUsed;
        int interDim = (int)layer.FfnGateExps!.Value.Info.Dimensions[1];

        var logits = new float[numExperts];
        fixed (float* inPtr = normedInput, outPtr = logits)
        {
            SimdKernels.MatVec(outPtr, layer.FfnGateInp!.Value.DataPtr, inPtr, numExperts, _embedDim, layer.FfnGateInp.Value.DType);
        }
        if (layer.FfnExpProbsB is { } biasRef)
        {
            float* bias = (float*)biasRef.DataPtr;
            for (int e = 0; e < numExperts; e++) logits[e] += bias[e];
        }

        var scores = new float[numExperts];
        for (int e = 0; e < numExperts; e++)
        {
            scores[e] = 1f / (1f + MathF.Exp(-logits[e]));
        }

        int[] topIndices = DeepSeek4Graph.SelectTopKIndices(scores, topK);
        var weights = new float[topK];
        float sumWeight = 0f;
        for (int k = 0; k < topK; k++)
        {
            weights[k] = scores[topIndices[k]];
            sumWeight += weights[k];
        }
        if (_hp.ExpertWeightsNorm && sumWeight > 0f)
        {
            float inv = 1f / sumWeight;
            for (int k = 0; k < topK; k++) weights[k] *= inv;
        }
        for (int k = 0; k < topK; k++) weights[k] *= _hp.ExpertWeightsScale;

        var accumulated = new float[_embedDim];
        var expertGate = new float[interDim];
        var expertUp = new float[interDim];
        var expertDown = new float[_embedDim];
        float clamp = _hp.SwiGluClampExp;

        for (int k = 0; k < topK; k++)
        {
            int expert = topIndices[k];
            float w = weights[k];

            PerExpertMatVec(layer.FfnGateExps!.Value, expert, normedInput, expertGate, interDim);
            PerExpertMatVec(layer.FfnUpExps!.Value, expert, normedInput, expertUp, interDim);

            for (int i = 0; i < interDim; i++)
            {
                float g = expertGate[i];
                float silu = g / (1f + MathF.Exp(-g));
                float val = silu * expertUp[i];
                if (val > clamp) val = clamp;
                else if (val < -clamp) val = -clamp;
                expertGate[i] = val;
            }

            PerExpertMatVecDown(layer.FfnDownExps!.Value, expert, expertGate, expertDown, interDim);
            for (int i = 0; i < _embedDim; i++) accumulated[i] += expertDown[i] * w;
        }

        // Shared expert
        int sharedInterDim = (int)layer.FfnGateShexp!.Value.Info.Dimensions[1];
        var shGate = new float[sharedInterDim];
        var shUp = new float[sharedInterDim];
        fixed (float* inPtr = normedInput, gPtr = shGate, uPtr = shUp)
        {
            SimdKernels.MatVec(gPtr, layer.FfnGateShexp.Value.DataPtr, inPtr, sharedInterDim, _embedDim, layer.FfnGateShexp.Value.DType);
            SimdKernels.MatVec(uPtr, layer.FfnUpShexp!.Value.DataPtr, inPtr, sharedInterDim, _embedDim, layer.FfnUpShexp.Value.DType);
        }
        for (int i = 0; i < sharedInterDim; i++)
        {
            float g = shGate[i];
            float silu = g / (1f + MathF.Exp(-g));
            float val = silu * shUp[i];
            if (val > clamp) val = clamp;
            else if (val < -clamp) val = -clamp;
            shGate[i] = val;
        }
        var shDown = new float[_embedDim];
        fixed (float* inPtr = shGate, outPtr = shDown)
        {
            SimdKernels.MatVec(outPtr, layer.FfnDownShexp!.Value.DataPtr, inPtr, _embedDim, sharedInterDim, layer.FfnDownShexp.Value.DType);
        }

        for (int i = 0; i < _embedDim; i++) accumulated[i] += shDown[i];
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

    private void PerHeadMatVec(DeepSeek4TensorRef tensor, int head, int outDim, int inDim, ReadOnlySpan<float> input, Span<float> output)
    {
        long bytesPerRow = ((long)inDim / DTypeInfo.BlockSize(tensor.DType)) * DTypeInfo.BytesPerBlock(tensor.DType);
        byte* headPtr = tensor.DataPtr + (long)head * outDim * bytesPerRow;
        fixed (float* inPtr = input, outPtr = output)
        {
            SimdKernels.MatVec(outPtr, headPtr, inPtr, outDim, inDim, tensor.DType);
        }
    }

    private static void LayerNormInPlace(float[] x, DeepSeek4TensorRef weight, DeepSeek4TensorRef? bias)
    {
        int n = x.Length;
        float mean = 0f;
        for (int i = 0; i < n; i++) mean += x[i];
        mean /= n;
        float var = 0f;
        for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
        var /= n;
        float inv = 1f / MathF.Sqrt(var + 1e-5f);
        float* w = (float*)weight.DataPtr;
        float* b = bias is { } bv ? (float*)bv.DataPtr : null;
        for (int i = 0; i < n; i++)
        {
            x[i] = (x[i] - mean) * inv * w[i] + (b is not null ? b[i] : 0f);
        }
    }

    public ReadOnlySpan<float> Prefill(IReadOnlyList<int> tokens, int startPos = 0)
    {
        ReadOnlySpan<float> last = default;
        for (int i = 0; i < tokens.Count; i++) last = Forward(tokens[i], startPos + i).ToArray();
        return last;
    }

    public void TruncateTo(int length)
    {
        if (length == 0) { ResetCache(); return; }
        throw new NotSupportedException("Glm5NextForwardPass (alpha): only full reset (TruncateTo(0)) is supported.");
    }

    public void ResetCache()
    {
        for (int s = 0; s < _hp.HcMult; s++) Array.Clear(_streams[s]);

        for (int il = 0; il < _numLayer; il++)
        {
            if (_hp.IsRecurrent(il))
            {
                Array.Clear(_kdaState[il]);
                Array.Clear(_kdaConvState[il]);
            }
            else
            {
                _mlaKvCache[il].Clear();
                _kpoolStaging[il].Clear();
                _kpoolCache[il].Clear();
            }
        }
    }

    public void Dispose() { }
}
