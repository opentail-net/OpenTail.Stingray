using System.Buffers;
using System.Numerics.Tensors;
using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.SD3;

/// <summary>
/// Multimodal Diffusion Transformer (MMDiT) for Stable Diffusion 3 / 3.5.
/// Supports dual-stream joint transformer blocks and single-stream self-attention blocks.
/// Supports CPU SIMD and Vulkan GPU SGEMM.
/// </summary>
public sealed class MMDiTModel : IDisposable
{
    private readonly IWeightLoader _weights;
    private readonly IComputeBackend? _backend;
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);
    private readonly Dictionary<string, CoreTensor>? _gpuWeights;
    private readonly string _prefix;

    public int HiddenSize { get; }
    public int NumHeads { get; }
    public int HeadDim { get; }
    public int Depth { get; }
    public int InChannels { get; }
    public int OutChannels { get; }
    public int PatchSize { get; }
    public int ContextSize { get; }
    public int AdmInChannels { get; }

    public MMDiTModel(
        IWeightLoader weights,
        string prefix = "model.diffusion_model.",
        int hiddenSize = 1536,
        int numHeads = 24,
        int depth = 24,
        int inChannels = 16,
        int outChannels = 16,
        int patchSize = 2,
        int contextSize = 4096,
        int admInChannels = 2048,
        IComputeBackend? backend = null)
    {
        _weights = weights;
        _prefix = prefix;
        HiddenSize = hiddenSize;
        NumHeads = numHeads;
        HeadDim = hiddenSize / numHeads;
        Depth = depth;
        InChannels = inChannels;
        OutChannels = outChannels;
        PatchSize = patchSize;
        ContextSize = contextSize;
        AdmInChannels = admInChannels;
        _backend = backend;
        if (_backend is not null)
            _gpuWeights = new Dictionary<string, CoreTensor>(StringComparer.Ordinal);
    }

    private float[] GetWeight(string name)
    {
        string fullName = _prefix + name;
        if (!_weightCache.TryGetValue(fullName, out var w))
        {
            w = _weights.ReadF32(fullName);
            _weightCache[fullName] = w;
        }
        return w;
    }

    private float[]? TryGetWeight(string name)
    {
        string fullName = _prefix + name;
        if (_weightCache.TryGetValue(fullName, out var w)) return w;
        if (_weights.Contains(fullName))
        {
            w = _weights.ReadF32(fullName);
            _weightCache[fullName] = w;
            return w;
        }
        return null;
    }

    private CoreTensor GetGpuWeight(string name, float[] cpuWeight)
    {
        string fullName = _prefix + name;
        if (_gpuWeights!.TryGetValue(fullName, out var wGpu)) return wGpu;

        wGpu = _backend!.Upload(cpuWeight.AsSpan(), TensorShape.D1(cpuWeight.Length));
        _gpuWeights[fullName] = wGpu;
        return wGpu;
    }

    public float[] Lin(string name, float[] x, int n, int inDim, int outDim)
    {
        var wF = GetWeight($"{name}.weight");
        var bF = TryGetWeight($"{name}.bias");

        if (_backend is null)
        {
            return DiffusionOps.Linear(x, wF, bF, n, inDim, outDim);
        }

        var wGpu = GetGpuWeight($"{name}.weight", wF);
        var xGpu = _backend.Upload(x.AsSpan(0, n * inDim), TensorShape.D1(n * inDim));
        var cGpu = _backend.Allocate(TensorShape.D1(n * outDim));
        var result = new float[n * outDim];

        try
        {
            _backend.Sgemm(cGpu, xGpu, wGpu, n, inDim, outDim);
            _backend.Synchronize();
            _backend.Download(cGpu, result);
        }
        finally
        {
            _backend.Free(xGpu);
            _backend.Free(cGpu);
        }

        if (bF is not null)
        {
            Parallel.For(0, n, i =>
            {
                int off = i * outDim;
                for (int o = 0; o < outDim; o++)
                    result[off + o] += bF[o];
            });
        }

        return result;
    }

    public float[] ComputeTimeAndPooledEmbedding(float timestep, float[] pooledY)
    {
        // 1. Timestep Fourier embedding: [256] -> [hiddenSize]
        int dim = 256;
        var sinEmb = new float[dim];
        int half = dim / 2;
        float logMaxPeriod = MathF.Log(10000.0f);

        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-logMaxPeriod * i / half);
            float arg = timestep * freq;
            sinEmb[i]        = MathF.Cos(arg);
            sinEmb[half + i] = MathF.Sin(arg);
        }

        var tEmb = Lin("t_embedder.mlp.0", sinEmb, 1, dim, HiddenSize);
        DiffusionOps.SiluInPlace(tEmb);
        tEmb = Lin("t_embedder.mlp.2", tEmb, 1, HiddenSize, HiddenSize);

        // 2. Pooled Y embedding: [2048] -> [hiddenSize]
        var yEmb = Lin("y_embedder.mlp.0", pooledY, 1, AdmInChannels, HiddenSize);
        DiffusionOps.SiluInPlace(yEmb);
        yEmb = Lin("y_embedder.mlp.2", yEmb, 1, HiddenSize, HiddenSize);

        for (int i = 0; i < HiddenSize; i++)
            tEmb[i] += yEmb[i];

        return tEmb;
    }

    public float[] Forward(float[] latents, float timestep, float[] textContext, float[] pooledY, int latH, int latW, int numTextTokens)
    {
        int p = PatchSize;
        int imgH = latH / p;
        int imgW = latW / p;
        int numImgTokens = imgH * imgW;
        int inPatchDim = InChannels * p * p;

        // 1. Patchify latents: [1, 16, latH, latW] -> [numImgTokens, inPatchDim]
        var imgTokens = new float[numImgTokens * inPatchDim];
        for (int py = 0; py < imgH; py++)
        for (int px = 0; px < imgW; px++)
        {
            int tokenIdx = py * imgW + px;
            int dstBase = tokenIdx * inPatchDim;
            int idx = 0;

            for (int ic = 0; ic < InChannels; ic++)
            for (int dy = 0; dy < p; dy++)
            for (int dx = 0; dx < p; dx++)
            {
                int iy = py * p + dy;
                int ix = px * p + dx;
                imgTokens[dstBase + idx++] = latents[ic * latH * latW + iy * latW + ix];
            }
        }

        // Linear x_embedder projection: inPatchDim -> HiddenSize
        var x = Lin("x_embedder.proj", imgTokens, numImgTokens, inPatchDim, HiddenSize);

        // 2. Project text context: ContextSize (4096) -> HiddenSize
        var c = Lin("context_embedder", textContext, numTextTokens, ContextSize, HiddenSize);

        // 3. Time + Pooled Embedding
        var tVec = ComputeTimeAndPooledEmbedding(timestep, pooledY);

        // 4. Joint MMDiT Transformer Blocks
        for (int b = 0; b < Depth; b++)
        {
            string blk = $"joint_blocks.{b}";

            // Modulations: 6 parameters for image, 6 parameters for text
            // [6 * HiddenSize]
            var imgMod = Lin($"{blk}.x_block.adaLN_modulation.1", tVec, 1, HiddenSize, 6 * HiddenSize);
            var txtMod = Lin($"{blk}.context_block.adaLN_modulation.1", tVec, 1, HiddenSize, 6 * HiddenSize);

            // ── Self/Joint Attention ────────────────────────────────────────
            var xNorm1 = ModulateNorm(x, imgMod, 0, numImgTokens, HiddenSize);
            var cNorm1 = ModulateNorm(c, txtMod, 0, numTextTokens, HiddenSize);

            var xQ = Lin($"{blk}.x_block.attn.qkv.0", xNorm1, numImgTokens, HiddenSize, HiddenSize);
            var xK = Lin($"{blk}.x_block.attn.qkv.1", xNorm1, numImgTokens, HiddenSize, HiddenSize);
            var xV = Lin($"{blk}.x_block.attn.qkv.2", xNorm1, numImgTokens, HiddenSize, HiddenSize);

            var cQ = Lin($"{blk}.context_block.attn.qkv.0", cNorm1, numTextTokens, HiddenSize, HiddenSize);
            var cK = Lin($"{blk}.context_block.attn.qkv.1", cNorm1, numTextTokens, HiddenSize, HiddenSize);
            var cV = Lin($"{blk}.context_block.attn.qkv.2", cNorm1, numTextTokens, HiddenSize, HiddenSize);

            // Concatenate image + text tokens
            int totalTokens = numImgTokens + numTextTokens;
            var jointQ = ConcatSeq(xQ, cQ, numImgTokens, numTextTokens, HiddenSize);
            var jointK = ConcatSeq(xK, cK, numImgTokens, numTextTokens, HiddenSize);
            var jointV = ConcatSeq(xV, cV, numImgTokens, numTextTokens, HiddenSize);

            var jointAttn = JointMultiHeadAttention(jointQ, jointK, jointV, totalTokens, HiddenSize, NumHeads, HeadDim);

            var xAttn = jointAttn.AsSpan(0, numImgTokens * HiddenSize).ToArray();
            var cAttn = jointAttn.AsSpan(numImgTokens * HiddenSize, numTextTokens * HiddenSize).ToArray();

            var xProj = Lin($"{blk}.x_block.attn.proj", xAttn, numImgTokens, HiddenSize, HiddenSize);
            var cProj = Lin($"{blk}.context_block.attn.proj", cAttn, numTextTokens, HiddenSize, HiddenSize);

            ApplyGateAndResidual(x, xProj, imgMod, 2, numImgTokens, HiddenSize);
            ApplyGateAndResidual(c, cProj, txtMod, 2, numTextTokens, HiddenSize);

            // ── FeedForward (MLP) ───────────────────────────────────────────
            var xNorm2 = ModulateNorm(x, imgMod, 3, numImgTokens, HiddenSize);
            var cNorm2 = ModulateNorm(c, txtMod, 3, numTextTokens, HiddenSize);

            int mlpHidden = HiddenSize * 4;
            var xMlp1 = Lin($"{blk}.x_block.mlp.fc1", xNorm2, numImgTokens, HiddenSize, mlpHidden);
            DiffusionOps.GeluInPlace(xMlp1);
            var xMlp2 = Lin($"{blk}.x_block.mlp.fc2", xMlp1, numImgTokens, mlpHidden, HiddenSize);

            var cMlp1 = Lin($"{blk}.context_block.mlp.fc1", cNorm2, numTextTokens, HiddenSize, mlpHidden);
            DiffusionOps.GeluInPlace(cMlp1);
            var cMlp2 = Lin($"{blk}.context_block.mlp.fc2", cMlp1, numTextTokens, mlpHidden, HiddenSize);

            ApplyGateAndResidual(x, xMlp2, imgMod, 5, numImgTokens, HiddenSize);
            ApplyGateAndResidual(c, cMlp2, txtMod, 5, numTextTokens, HiddenSize);
        }

        // 5. Final Layer: modulation + linear projection back to patch channels
        var finalMod = Lin("final_layer.adaLN_modulation.1", tVec, 1, HiddenSize, 2 * HiddenSize);
        var finalNorm = ModulateNorm(x, finalMod, 0, numImgTokens, HiddenSize);

        int outPatchDim = OutChannels * p * p;
        var unpatchified = Lin("final_layer.linear", finalNorm, numImgTokens, HiddenSize, outPatchDim);

        // 6. Unpatchify back to [1, 16, latH, latW]
        var outLatent = new float[OutChannels * latH * latW];
        for (int py = 0; py < imgH; py++)
        for (int px = 0; px < imgW; px++)
        {
            int tokenIdx = py * imgW + px;
            int srcBase = tokenIdx * outPatchDim;
            int idx = 0;

            for (int ch = 0; ch < OutChannels; ch++)
            for (int dy = 0; dy < p; dy++)
            for (int dx = 0; dx < p; dx++)
            {
                int y = py * p + dy;
                int xCoord = px * p + dx;
                outLatent[ch * latH * latW + y * latW + xCoord] = unpatchified[srcBase + idx++];
            }
        }

        return outLatent;
    }

    private static float[] ModulateNorm(float[] tokens, float[] modVec, int modIdxOffset, int numTokens, int dim)
    {
        // modVec contains shift [dim] and scale [dim] starting at modIdxOffset * dim
        int shiftOff = modIdxOffset * dim;
        int scaleOff = (modIdxOffset + 1) * dim;

        var normed = (float[])tokens.Clone();
        for (int i = 0; i < numTokens; i++)
        {
            int off = i * dim;
            // LayerNorm over token
            float mean = 0f;
            for (int d = 0; d < dim; d++) mean += normed[off + d];
            mean /= dim;

            float var = 0f;
            for (int d = 0; d < dim; d++)
            {
                float diff = normed[off + d] - mean;
                var += diff * diff;
            }
            float invStd = 1f / MathF.Sqrt(var / dim + 1e-6f);

            for (int d = 0; d < dim; d++)
            {
                float n = (normed[off + d] - mean) * invStd;
                normed[off + d] = n * (1f + modVec[scaleOff + d]) + modVec[shiftOff + d];
            }
        }
        return normed;
    }

    private static void ApplyGateAndResidual(float[] target, float[] branch, float[] modVec, int gateIdx, int numTokens, int dim)
    {
        int gateOff = gateIdx * dim;
        Parallel.For(0, numTokens, i =>
        {
            int off = i * dim;
            for (int d = 0; d < dim; d++)
                target[off + d] += branch[off + d] * modVec[gateOff + d];
        });
    }

    private static float[] ConcatSeq(float[] seq1, float[] seq2, int n1, int n2, int dim)
    {
        var cat = new float[(n1 + n2) * dim];
        Array.Copy(seq1, 0, cat, 0, n1 * dim);
        Array.Copy(seq2, 0, cat, n1 * dim, n2 * dim);
        return cat;
    }

    private static float[] JointMultiHeadAttention(float[] q, float[] k, float[] v, int totalTokens, int dim, int nHeads, int headDim)
    {
        float scale = 1f / MathF.Sqrt(headDim);
        var output = new float[totalTokens * dim];

        Parallel.For(0, nHeads, h =>
        {
            int headOffset = h * headDim;
            var scores = new float[totalTokens];

            for (int qi = 0; qi < totalTokens; qi++)
            {
                int qBase = qi * dim + headOffset;

                for (int kj = 0; kj < totalTokens; kj++)
                {
                    int kBase = kj * dim + headOffset;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                        dot += q[qBase + d] * k[kBase + d];
                    scores[kj] = dot * scale;
                }

                DiffusionOps.Softmax(scores, 0, totalTokens);

                int outBase = qi * dim + headOffset;
                for (int d = 0; d < headDim; d++)
                {
                    float sum = 0f;
                    for (int kj = 0; kj < totalTokens; kj++)
                        sum += scores[kj] * v[kj * dim + headOffset + d];
                    output[outBase + d] = sum;
                }
            }
        });

        return output;
    }

    public void Dispose()
    {
        if (_gpuWeights is not null)
        {
            foreach (var t in _gpuWeights.Values) _backend!.Free(t);
            _gpuWeights.Clear();
        }
        _weightCache.Clear();
    }
}

