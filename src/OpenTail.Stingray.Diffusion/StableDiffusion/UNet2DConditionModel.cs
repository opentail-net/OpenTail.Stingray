using System.Buffers;
using System.Numerics.Tensors;
using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.StableDiffusion;

/// <summary>
/// Stable Diffusion 1.5 UNet (2D Condition Model).
/// Supports both CPU (SIMD AVX2/AVX-512) and GPU (Vulkan/CUDA SGEMM via IComputeBackend).
/// </summary>
public sealed class UNet2DConditionModel : IDisposable
{
    private readonly IWeightLoader _weights;
    private readonly IComputeBackend? _backend;
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);
    private readonly Dictionary<string, CoreTensor>? _gpuWeights;
    private readonly string _prefix;

    private const int ModelChannels = 320;
    private const int TimeEmbedDim = 1280;
    private const int ContextDim = 768;
    private const int NumHeads = 8;
    private const int MaxColChunkFloats = 8 * 1024 * 1024; // 32MB im2col buffer chunk

    public UNet2DConditionModel(IWeightLoader weights, string prefix = "model.diffusion_model.", IComputeBackend? backend = null)
    {
        _weights = weights;
        _prefix = prefix;
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

    public float[] Conv(string name, float[] x, int inC, int h, int w, int outC, int ksize, int stride = 1, int padding = -1)
    {
        var wF = GetWeight($"{name}.weight");
        var bF = TryGetWeight($"{name}.bias");

        if (_backend is null)
        {
            return DiffusionOps.Conv2D(x, wF, bF, 1, inC, h, w, outC, ksize, ksize, stride, padding);
        }

        if (padding < 0) padding = (ksize - 1) / 2;
        int outH = (h + 2 * padding - ksize) / stride + 1;
        int outW = (w + 2 * padding - ksize) / stride + 1;
        int hw = outH * outW;
        int kPts = inC * ksize * ksize;

        var wGpu = GetGpuWeight($"{name}.weight", wF);

        int chunkRows = kPts > 0 ? Math.Max(1, Math.Min(outH, MaxColChunkFloats / (outW * kPts))) : outH;
        var output = new float[outC * hw];

        var colBuf = ArrayPool<float>.Shared.Rent(chunkRows * outW * kPts);
        var resBuf = ArrayPool<float>.Shared.Rent(chunkRows * outW * outC);

        try
        {
            for (int rowStart = 0; rowStart < outH; rowStart += chunkRows)
            {
                int rowEnd = Math.Min(rowStart + chunkRows, outH);
                int chunkH = rowEnd - rowStart;
                int chunkHW = chunkH * outW;
                int colSize = chunkHW * kPts;

                // Build im2col for this chunk
                int idx = 0;
                for (int oh = rowStart; oh < rowEnd; oh++)
                {
                    int ih0 = oh * stride - padding;
                    for (int ow = 0; ow < outW; ow++)
                    {
                        int iw0 = ow * stride - padding;
                        for (int ic = 0; ic < inC; ic++)
                        {
                            int inChannelBase = ic * h * w;
                            for (int kh = 0; kh < ksize; kh++)
                            {
                                int ih = ih0 + kh;
                                if ((uint)ih < (uint)h)
                                {
                                    int inRow = inChannelBase + ih * w;
                                    for (int kw = 0; kw < ksize; kw++)
                                    {
                                        int iw = iw0 + kw;
                                        colBuf[idx++] = ((uint)iw < (uint)w) ? x[inRow + iw] : 0f;
                                    }
                                }
                                else
                                {
                                    for (int kw = 0; kw < ksize; kw++)
                                        colBuf[idx++] = 0f;
                                }
                            }
                        }
                    }
                }

                // GPU SGEMM
                var colGpu = _backend.Upload(colBuf.AsSpan(0, colSize), TensorShape.D1(colSize));
                var cGpu = _backend.Allocate(TensorShape.D1(chunkHW * outC));
                try
                {
                    _backend.Sgemm(cGpu, colGpu, wGpu, chunkHW, kPts, outC);
                    _backend.Synchronize();
                    _backend.Download(cGpu, resBuf.AsSpan(0, chunkHW * outC));
                }
                finally
                {
                    _backend.Free(colGpu);
                    _backend.Free(cGpu);
                }

                int basePos = rowStart * outW;
                for (int pos = 0; pos < chunkHW; pos++)
                {
                    int absPos = basePos + pos;
                    for (int oc = 0; oc < outC; oc++)
                        output[oc * hw + absPos] = resBuf[pos * outC + oc] + (bF is not null ? bF[oc] : 0f);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(colBuf);
            ArrayPool<float>.Shared.Return(resBuf);
        }

        return output;
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

    /// <summary>
    /// Computes sinusoidal timestep embedding and passes through 2-layer MLP.
    /// </summary>
    public float[] ComputeTimeEmbedding(float timestep)
    {
        int dim = ModelChannels; // 320
        var sinEmb = new float[dim];
        int half = dim / 2;
        float maxPeriod = 10000.0f;
        float logMaxPeriod = MathF.Log(maxPeriod);

        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-logMaxPeriod * i / half);
            float arg = timestep * freq;
            sinEmb[i]        = MathF.Cos(arg);
            sinEmb[half + i] = MathF.Sin(arg);
        }

        var emb = Lin("time_embed.0", sinEmb, 1, dim, TimeEmbedDim);
        DiffusionOps.SiluInPlace(emb);
        return Lin("time_embed.2", emb, 1, TimeEmbedDim, TimeEmbedDim);
    }

    /// <summary>
    /// ResBlock with GroupNorm, SiLU, Conv2D, timestep embedding projection, and residual.
    /// </summary>
    public float[] ResBlock(string prefix, float[] x, float[] tEmb, int inC, int outC, int h, int w)
    {
        // 1. in_layers: GroupNorm(32, inC) + SiLU + Conv2D(inC -> outC, 3x3)
        var gn1W = GetWeight($"{prefix}.in_layers.0.weight");
        var gn1B = GetWeight($"{prefix}.in_layers.0.bias");
        var hNorm = (float[])x.Clone();
        DiffusionOps.GroupNorm(hNorm, gn1W, gn1B, 1, inC, h, w, groups: 32);
        DiffusionOps.SiluInPlace(hNorm);

        var hOut = Conv($"{prefix}.in_layers.2", hNorm, inC, h, w, outC, 3);

        // 2. emb_layers: SiLU(tEmb) -> Linear(1280 -> outC) added spatially
        var tEmbAct = (float[])tEmb.Clone();
        DiffusionOps.SiluInPlace(tEmbAct);
        var tProj = Lin($"{prefix}.emb_layers.1", tEmbAct, 1, TimeEmbedDim, outC);

        int spatial = h * w;
        for (int c = 0; c < outC; c++)
        {
            float bias = tProj[c];
            int cOff = c * spatial;
            for (int s = 0; s < spatial; s++)
                hOut[cOff + s] += bias;
        }

        // 3. out_layers: GroupNorm(32, outC) + SiLU + Conv2D(outC -> outC, 3x3)
        var gn2W = GetWeight($"{prefix}.out_layers.0.weight");
        var gn2B = GetWeight($"{prefix}.out_layers.0.bias");
        DiffusionOps.GroupNorm(hOut, gn2W, gn2B, 1, outC, h, w, groups: 32);
        DiffusionOps.SiluInPlace(hOut);

        hOut = Conv($"{prefix}.out_layers.3", hOut, outC, h, w, outC, 3);

        // 4. Skip connection (nin_shortcut if inC != outC)
        float[] xRes;
        if (TryGetWeight($"{prefix}.skip_connection.weight") is not null)
        {
            xRes = Conv($"{prefix}.skip_connection", x, inC, h, w, outC, 1, stride: 1, padding: 0);
        }
        else if (inC != outC)
        {
            throw new InvalidOperationException($"ResBlock {prefix} has inChannels ({inC}) != outChannels ({outC}) but no skip_connection weight.");
        }
        else
        {
            xRes = x;
        }

        // Residual add
        for (int i = 0; i < hOut.Length; i++)
            hOut[i] += xRes[i];

        return hOut;
    }

    /// <summary>
    /// SpatialTransformer block (Self-Attention + Cross-Attention + GEGLU FeedForward).
    /// </summary>
    public float[] SpatialTransformer(string prefix, float[] x, float[] context, int c, int h, int w)
    {
        int hw = h * w;

        // 1. norm + proj_in (Conv2D 1x1)
        var normW = GetWeight($"{prefix}.norm.weight");
        var normB = GetWeight($"{prefix}.norm.bias");
        var xNorm = (float[])x.Clone();
        DiffusionOps.GroupNorm(xNorm, normW, normB, 1, c, h, w, groups: 32);

        var xProj = Conv($"{prefix}.proj_in", xNorm, c, h, w, c, 1, stride: 1, padding: 0);

        // Permute [1, C, H, W] -> [H*W, C] sequence
        var xSeq = new float[hw * c];
        for (int ch = 0; ch < c; ch++)
        {
            int chOff = ch * hw;
            for (int s = 0; s < hw; s++)
                xSeq[s * c + ch] = xProj[chOff + s];
        }

        // 2. Transformer Block (depth = 1 in SD 1.5)
        string tb = $"{prefix}.transformer_blocks.0";

        // A. Self-Attention:
        var saNormW = GetWeight($"{tb}.norm1.weight");
        var saNormB = GetWeight($"{tb}.norm1.bias");
        var saNorm = (float[])xSeq.Clone();
        DiffusionOps.LayerNorm(saNorm, saNormW, saNormB, c);

        var saQ = Lin($"{tb}.attn1.to_q", saNorm, hw, c, c);
        var saK = Lin($"{tb}.attn1.to_k", saNorm, hw, c, c);
        var saV = Lin($"{tb}.attn1.to_v", saNorm, hw, c, c);

        var saAttnOut = MultiHeadAttention(saQ, saK, saV, hw, hw, c, NumHeads);
        var saProjOut = Lin($"{tb}.attn1.to_out.0", saAttnOut, hw, c, c);

        for (int i = 0; i < xSeq.Length; i++)
            xSeq[i] += saProjOut[i];

        // B. Cross-Attention (to CLIP text context: 77 tokens, 768 dim):
        var caNormW = GetWeight($"{tb}.norm2.weight");
        var caNormB = GetWeight($"{tb}.norm2.bias");
        var caNorm = (float[])xSeq.Clone();
        DiffusionOps.LayerNorm(caNorm, caNormW, caNormB, c);

        var caQ = Lin($"{tb}.attn2.to_q", caNorm, hw, c, c);
        var caK = Lin($"{tb}.attn2.to_k", context, 77, ContextDim, c);
        var caV = Lin($"{tb}.attn2.to_v", context, 77, ContextDim, c);

        var caAttnOut = MultiHeadAttention(caQ, caK, caV, hw, 77, c, NumHeads);
        var caProjOut = Lin($"{tb}.attn2.to_out.0", caAttnOut, hw, c, c);

        for (int i = 0; i < xSeq.Length; i++)
            xSeq[i] += caProjOut[i];

        // C. Feed-Forward with GEGLU:
        var ffNormW = GetWeight($"{tb}.norm3.weight");
        var ffNormB = GetWeight($"{tb}.norm3.bias");
        var ffNorm = (float[])xSeq.Clone();
        DiffusionOps.LayerNorm(ffNorm, ffNormW, ffNormB, c);

        int mlpDim = c * 4;
        var ffH = Lin($"{tb}.ff.net.0.proj", ffNorm, hw, c, mlpDim * 2);
        var ffGated = new float[hw * mlpDim];
        Parallel.For(0, hw, s =>
        {
            int srcOff = s * mlpDim * 2;
            int dstOff = s * mlpDim;
            for (int d = 0; d < mlpDim; d++)
            {
                float val = ffH[srcOff + d];
                float gate = ffH[srcOff + mlpDim + d];
                float geluGate = 0.5f * gate * (1.0f + MathF.Tanh(0.79788456f * (gate + 0.044715f * gate * gate * gate)));
                ffGated[dstOff + d] = val * geluGate;
            }
        });

        var ffOut = Lin($"{tb}.ff.net.2", ffGated, hw, mlpDim, c);

        for (int i = 0; i < xSeq.Length; i++)
            xSeq[i] += ffOut[i];

        // Permute [H*W, C] back to [1, C, H, W]
        var xSpatial = new float[hw * c];
        for (int ch = 0; ch < c; ch++)
        {
            int chOff = ch * hw;
            for (int s = 0; s < hw; s++)
                xSpatial[chOff + s] = xSeq[s * c + ch];
        }

        // proj_out (Conv2D 1x1) + residual with input x
        var projOut = Conv($"{prefix}.proj_out", xSpatial, c, h, w, c, 1, stride: 1, padding: 0);

        for (int i = 0; i < x.Length; i++)
            projOut[i] += x[i];

        return projOut;
    }

    private static float[] MultiHeadAttention(float[] q, float[] k, float[] v, int qLen, int kvLen, int c, int nHeads)
    {
        int headDim = c / nHeads;
        float scale = 1f / MathF.Sqrt(headDim);
        var output = new float[qLen * c];

        Parallel.For(0, nHeads, h =>
        {
            int headOffset = h * headDim;
            var scores = new float[kvLen];

            for (int qi = 0; qi < qLen; qi++)
            {
                int qBase = qi * c + headOffset;

                for (int kj = 0; kj < kvLen; kj++)
                {
                    int kBase = kj * c + headOffset;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                        dot += q[qBase + d] * k[kBase + d];
                    scores[kj] = dot * scale;
                }

                DiffusionOps.Softmax(scores, 0, kvLen);

                int outBase = qi * c + headOffset;
                for (int d = 0; d < headDim; d++)
                {
                    float val = 0f;
                    for (int kj = 0; kj < kvLen; kj++)
                        val += scores[kj] * v[kj * c + headOffset + d];
                    output[outBase + d] = val;
                }
            }
        });

        return output;
    }

    public float[] Forward(float[] x, float timestep, float[] context, int latH, int latW, IReadOnlyList<float[]>? controlDownResiduals = null, float[]? controlMidResidual = null)
    {
        var tEmb = ComputeTimeEmbedding(timestep);
        var savedInputs = new List<float[]>(12);

        // ── Input Blocks ────────────────────────────────────────────────────────
        int h = latH, w = latW;
        var cur = Conv("input_blocks.0.0", x, 4, h, w, 320, 3);
        savedInputs.Add(cur);

        cur = ResBlock("input_blocks.1.0", cur, tEmb, 320, 320, h, w);
        cur = SpatialTransformer("input_blocks.1.1", cur, context, 320, h, w);
        savedInputs.Add(cur);

        cur = ResBlock("input_blocks.2.0", cur, tEmb, 320, 320, h, w);
        cur = SpatialTransformer("input_blocks.2.1", cur, context, 320, h, w);
        savedInputs.Add(cur);

        // Block 3: Downsample (Conv2D 320 -> 320, stride 2)
        cur = Conv("input_blocks.3.0.op", cur, 320, h, w, 320, 3, stride: 2);
        h /= 2; w /= 2;
        savedInputs.Add(cur);

        cur = ResBlock("input_blocks.4.0", cur, tEmb, 320, 640, h, w);
        cur = SpatialTransformer("input_blocks.4.1", cur, context, 640, h, w);
        savedInputs.Add(cur);

        cur = ResBlock("input_blocks.5.0", cur, tEmb, 640, 640, h, w);
        cur = SpatialTransformer("input_blocks.5.1", cur, context, 640, h, w);
        savedInputs.Add(cur);

        // Block 6: Downsample (Conv2D 640 -> 640, stride 2)
        cur = Conv("input_blocks.6.0.op", cur, 640, h, w, 640, 3, stride: 2);
        h /= 2; w /= 2;
        savedInputs.Add(cur);

        cur = ResBlock("input_blocks.7.0", cur, tEmb, 640, 1280, h, w);
        cur = SpatialTransformer("input_blocks.7.1", cur, context, 1280, h, w);
        savedInputs.Add(cur);

        cur = ResBlock("input_blocks.8.0", cur, tEmb, 1280, 1280, h, w);
        cur = SpatialTransformer("input_blocks.8.1", cur, context, 1280, h, w);
        savedInputs.Add(cur);

        // Block 9: Downsample (Conv2D 1280 -> 1280, stride 2)
        cur = Conv("input_blocks.9.0.op", cur, 1280, h, w, 1280, 3, stride: 2);
        h /= 2; w /= 2;
        savedInputs.Add(cur);

        cur = ResBlock("input_blocks.10.0", cur, tEmb, 1280, 1280, h, w);
        savedInputs.Add(cur);

        cur = ResBlock("input_blocks.11.0", cur, tEmb, 1280, 1280, h, w);
        savedInputs.Add(cur);

        // Apply ControlNet down residuals to skip connections
        if (controlDownResiduals is not null)
        {
            for (int i = 0; i < Math.Min(savedInputs.Count, controlDownResiduals.Count); i++)
            {
                var res = controlDownResiduals[i];
                var inp = savedInputs[i];
                for (int j = 0; j < Math.Min(inp.Length, res.Length); j++)
                    inp[j] += res[j];
            }
        }

        // ── Middle Block ────────────────────────────────────────────────────────
        cur = ResBlock("middle_block.0", cur, tEmb, 1280, 1280, h, w);
        cur = SpatialTransformer("middle_block.1", cur, context, 1280, h, w);
        cur = ResBlock("middle_block.2", cur, tEmb, 1280, 1280, h, w);

        if (controlMidResidual is not null)
        {
            for (int j = 0; j < Math.Min(cur.Length, controlMidResidual.Length); j++)
                cur[j] += controlMidResidual[j];
        }

        // ── Output Blocks ───────────────────────────────────────────────────────
        static float[] CatSkip(float[] current, float[] skip, int curC, int skipC, int curH, int curW)
        {
            int hw = curH * curW;
            var cat = new float[(curC + skipC) * hw];
            Array.Copy(current, 0, cat, 0, curC * hw);
            Array.Copy(skip, 0, cat, curC * hw, skipC * hw);
            return cat;
        }

        // Block 0: ResBlock(1280 + 1280 -> 1280)
        cur = CatSkip(cur, savedInputs[11], 1280, 1280, h, w);
        cur = ResBlock("output_blocks.0.0", cur, tEmb, 2560, 1280, h, w);

        // Block 1: ResBlock(1280 + 1280 -> 1280)
        cur = CatSkip(cur, savedInputs[10], 1280, 1280, h, w);
        cur = ResBlock("output_blocks.1.0", cur, tEmb, 2560, 1280, h, w);

        // Block 2: ResBlock(1280 + 1280 -> 1280) + Upsample(1280 -> 1280)
        cur = CatSkip(cur, savedInputs[9], 1280, 1280, h, w);
        cur = ResBlock("output_blocks.2.0", cur, tEmb, 2560, 1280, h, w);
        cur = DiffusionOps.Upsample2x(cur, 1, 1280, h, w);
        h *= 2; w *= 2;
        cur = Conv("output_blocks.2.1.conv", cur, 1280, h, w, 1280, 3);

        // Block 3: ResBlock(1280 + 1280 -> 1280) + SpatialTransformer(1280)
        cur = CatSkip(cur, savedInputs[8], 1280, 1280, h, w);
        cur = ResBlock("output_blocks.3.0", cur, tEmb, 2560, 1280, h, w);
        cur = SpatialTransformer("output_blocks.3.1", cur, context, 1280, h, w);

        // Block 4: ResBlock(1280 + 1280 -> 1280) + SpatialTransformer(1280)
        cur = CatSkip(cur, savedInputs[7], 1280, 1280, h, w);
        cur = ResBlock("output_blocks.4.0", cur, tEmb, 2560, 1280, h, w);
        cur = SpatialTransformer("output_blocks.4.1", cur, context, 1280, h, w);

        // Block 5: ResBlock(1280 + 640 -> 1280) + SpatialTransformer(1280) + Upsample(1280 -> 1280)
        cur = CatSkip(cur, savedInputs[6], 1280, 640, h, w);
        cur = ResBlock("output_blocks.5.0", cur, tEmb, 1920, 1280, h, w);
        cur = SpatialTransformer("output_blocks.5.1", cur, context, 1280, h, w);
        cur = DiffusionOps.Upsample2x(cur, 1, 1280, h, w);
        h *= 2; w *= 2;
        cur = Conv("output_blocks.5.2.conv", cur, 1280, h, w, 1280, 3);

        // Block 6: ResBlock(1280 + 640 -> 640) + SpatialTransformer(640)
        cur = CatSkip(cur, savedInputs[5], 1280, 640, h, w);
        cur = ResBlock("output_blocks.6.0", cur, tEmb, 1920, 640, h, w);
        cur = SpatialTransformer("output_blocks.6.1", cur, context, 640, h, w);

        // Block 7: ResBlock(640 + 640 -> 640) + SpatialTransformer(640)
        cur = CatSkip(cur, savedInputs[4], 640, 640, h, w);
        cur = ResBlock("output_blocks.7.0", cur, tEmb, 1280, 640, h, w);
        cur = SpatialTransformer("output_blocks.7.1", cur, context, 640, h, w);

        // Block 8: ResBlock(640 + 320 -> 640) + SpatialTransformer(640) + Upsample(640 -> 640)
        cur = CatSkip(cur, savedInputs[3], 640, 320, h, w);
        cur = ResBlock("output_blocks.8.0", cur, tEmb, 960, 640, h, w);
        cur = SpatialTransformer("output_blocks.8.1", cur, context, 640, h, w);
        cur = DiffusionOps.Upsample2x(cur, 1, 640, h, w);
        h *= 2; w *= 2;
        cur = Conv("output_blocks.8.2.conv", cur, 640, h, w, 640, 3);

        // Block 9: ResBlock(640 + 320 -> 320) + SpatialTransformer(320)
        cur = CatSkip(cur, savedInputs[2], 640, 320, h, w);
        cur = ResBlock("output_blocks.9.0", cur, tEmb, 960, 320, h, w);
        cur = SpatialTransformer("output_blocks.9.1", cur, context, 320, h, w);

        // Block 10: ResBlock(320 + 320 -> 320) + SpatialTransformer(320)
        cur = CatSkip(cur, savedInputs[1], 320, 320, h, w);
        cur = ResBlock("output_blocks.10.0", cur, tEmb, 640, 320, h, w);
        cur = SpatialTransformer("output_blocks.10.1", cur, context, 320, h, w);

        // Block 11: ResBlock(320 + 320 -> 320) + SpatialTransformer(320)
        cur = CatSkip(cur, savedInputs[0], 320, 320, h, w);
        cur = ResBlock("output_blocks.11.0", cur, tEmb, 640, 320, h, w);
        cur = SpatialTransformer("output_blocks.11.1", cur, context, 320, h, w);

        // ── Final Output ────────────────────────────────────────────────────────
        var outGnW = GetWeight("out.0.weight");
        var outGnB = GetWeight("out.0.bias");
        DiffusionOps.GroupNorm(cur, outGnW, outGnB, 1, 320, h, w, groups: 32);
        DiffusionOps.SiluInPlace(cur);

        return Conv("out.2", cur, 320, h, w, 4, 3);
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


