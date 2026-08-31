using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.Wan;

/// <summary>
/// Native C# Wan 2.1 / 2.2 Video Diffusion Transformer (DiT).
/// Reference: stable-diffusion.cpp:src/model/diffusion/wan.hpp:WanModel
/// </summary>
public sealed class WanModel : IDisposable
{
    private readonly IWeightLoader _weights;
    private readonly string _prefix;
    private readonly IComputeBackend? _backend;
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);
    private readonly Dictionary<string, CoreTensor>? _gpuWeights;
    private readonly int _numLayers;
    private readonly int _dim;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _ffnDim;
    private bool _disposed;

    public const int InChannels = 64;   // 16 * 2 * 2
    public const int OutChannels = 16;
    public const int TextDim = 4096;    // UMT5 / T5-XXL text dimension

    public int NumLayers => _numLayers;
    public int Dim => _dim;
    public int NumHeads => _numHeads;

    public WanModel(IWeightLoader weights, string prefix = "", int numLayers = 30, int dim = 1536, int numHeads = 12, IComputeBackend? backend = null)
    {
        _weights = weights;
        _prefix = prefix;
        _backend = backend;
        (_numLayers, _dim, _numHeads) = DetectConfig(weights, prefix, numLayers, dim, numHeads);
        _headDim = _dim / _numHeads;
        _ffnDim = _dim == 1536 ? 8960 : (_dim == 5120 ? 13824 : _dim * 4);
        if (backend is not null)
            _gpuWeights = new Dictionary<string, CoreTensor>(StringComparer.Ordinal);
    }

    private static (int numLayers, int dim, int numHeads) DetectConfig(IWeightLoader weights, string prefix, int defLayers, int defDim, int defHeads)
    {
        int detectedLayers = defLayers;
        for (int i = 60; i >= 0; i--)
        {
            string key = $"{prefix}blocks.{i}.self_attn.q.weight";
            if (weights.Contains(key) || weights.Contains("model.diffusion_model." + key))
            {
                detectedLayers = i + 1;
                break;
            }
        }

        int detectedDim = defDim;
        string patchKey = $"{prefix}patch_embedding.weight";
        if (!weights.Contains(patchKey)) patchKey = "model.diffusion_model." + patchKey;
        if (weights.Contains(patchKey))
        {
            var w = weights.ReadF32(patchKey);
            detectedDim = w.Length / InChannels;
        }

        int detectedHeads = detectedDim == 1536 ? 12 : (detectedDim == 5120 ? 40 : defHeads);
        return (detectedLayers, detectedDim, detectedHeads);
    }

    private string Resolve(string name)
    {
        string direct = _prefix + name;
        if (_weights.Contains(direct)) return direct;
        if (_weights.Contains("model.diffusion_model." + direct)) return "model.diffusion_model." + direct;
        if (_weights.Contains("diffusion_model." + direct)) return "diffusion_model." + direct;
        return direct;
    }

    private float[] GetWeight(string name)
    {
        string fullName = Resolve(name);
        if (_weightCache.TryGetValue(fullName, out var cached)) return cached;
        var data = _weights.ReadF32(fullName);
        _weightCache[fullName] = data;
        return data;
    }

    private float[]? TryGetWeight(string name)
    {
        string fullName = Resolve(name);
        if (_weightCache.TryGetValue(fullName, out var cached)) return cached;
        if (_weights.Contains(fullName))
        {
            var data = _weights.ReadF32(fullName);
            _weightCache[fullName] = data;
            return data;
        }
        return null;
    }

    /// <summary>
    /// Executes the forward pass of the Wan DiT model.
    /// </summary>
    public float[] Forward(
        float[] latent,
        float timestep,
        float[] textContext,
        int numFrames,
        int latH,
        int latW)
    {
        int patchH = latH / 2;
        int patchW = latW / 2;
        int numTokens = numFrames * patchH * patchW;
        int numTxtTokens = textContext.Length / TextDim;

        // 1. Pack 16-channel video latent into 64-channel patches [numTokens, 64]
        var packed = PackLatents(latent, numFrames, latH, latW);

        // 2. Patch & Text input projections
        var x = Linear("patch_embedding", packed, InChannels, _dim);
        var txtProj = ComputeTextEmbedding(textContext);

        // 3. Timestep embedding (sinusoidal 256 -> linear dim -> silu -> linear dim)
        var tEmb = ComputeTimestepEmbedding(timestep);

        // Real Wan DiT: a SHARED "time_projection" linear (dim -> 6*dim, applied once, not
        // per-block) whose output is then ADDITIVELY combined with each block's own per-block
        // "modulation" parameter (`scale_shift_table`, a learned constant -- NOT a Linear weight;
        // confirmed against `WanTimeTextImageEmbedding`/`WanTransformerBlock.forward` in
        // transformer_wan.py, and against the real GGUF's `time_projection.1.weight`
        // [dim,6*dim] + per-block `blocks.{i}.modulation` [dim,6,1] tensor shapes).
        var timeProjSilu = (float[])tEmb.Clone();
        DiffusionOps.SiluInPlace(timeProjSilu);
        var timestepProj = Linear("time_projection.1", timeProjSilu, _dim, _dim * 6);

        // 4. 3D-RoPE positional frequencies
        var (cos, sin) = WanRoPE.Compute3DRoPE(numFrames, patchH, patchW, _headDim);

        // 5. Transformer Blocks
        for (int b = 0; b < _numLayers; b++)
        {
            string p = $"blocks.{b}";
            x = TransformerBlock(p, x, timestepProj, cos, sin, txtProj, numTokens, numTxtTokens);
        }

        // 6. Final Layer (AdaLN + Linear dim -> 64)
        // Real: `shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)` --
        // additive broadcast of the RAW (unprojected) temb against the head's own 2*dim constant,
        // not a Linear (see WanTransformer3DModel.forward's final-layer block).
        var headModParam = GetWeight("head.modulation");
        var headShift = new float[_dim];
        var headScale = new float[_dim];
        for (int d = 0; d < _dim; d++)
        {
            headShift[d] = headModParam[d] + tEmb[d];
            headScale[d] = headModParam[_dim + d] + tEmb[d];
        }

        var normed = (float[])x.Clone();
        LayerNormNoAffine(normed, _dim);
        normed = Modulate(normed, numTokens, headShift, headScale);

        var outPacked = Linear("head.head", normed, _dim, InChannels);

        // 7. Unpack patches [numTokens, 64] -> [16, numFrames, latH, latW]
        return UnpackLatents(outPacked, numFrames, latH, latW);
    }

    private float[] TransformerBlock(
        string prefix,
        float[] x,
        float[] timestepProj,
        float[] cos,
        float[] sin,
        float[] txtContext,
        int numTokens,
        int numTxt)
    {
        // Real: per-block additive AdaLN modulation = this block's own learned constant
        // (`scale_shift_table`, GGUF `{prefix}.modulation`) + the SHARED `timestepProj` computed
        // once in Forward -- NOT a per-block Linear (confirmed against `WanTransformerBlock.
        // forward`'s `self.scale_shift_table + temb` in transformer_wan.py, and the real GGUF's
        // bare (bias-less, non-`.weight`-suffixed) `blocks.{i}.modulation` [dim,6,1] tensor).
        var modParam = GetWeight($"{prefix}.modulation");
        var mod = new float[_dim * 6];
        for (int i = 0; i < mod.Length; i++) mod[i] = modParam[i] + timestepProj[i];
        var s1 = mod.AsSpan(0 * _dim, _dim);
        var sc1 = mod.AsSpan(1 * _dim, _dim);
        var g1 = mod.AsSpan(2 * _dim, _dim);
        var s2 = mod.AsSpan(3 * _dim, _dim);
        var sc2 = mod.AsSpan(4 * _dim, _dim);
        var g2 = mod.AsSpan(5 * _dim, _dim);

        // 1. Self-attention: affine-free LayerNorm (real `norm1`, `elementwise_affine=False`,
        // no learned weight/bias tensor in the checkpoint) -> AdaLN modulate -> self-attn (3D-RoPE)
        // -> gated residual.
        var norm1 = (float[])x.Clone();
        LayerNormNoAffine(norm1, _dim);
        var normed1 = Modulate(norm1, numTokens, s1, sc1);
        var selfAttn = SelfAttention($"{prefix}.self_attn", normed1, cos, sin, numTokens);
        ApplyGatedResidual(x, selfAttn, numTokens, g1);

        // 2. Cross-Attention with T5/UMT5 text tokens: real per-block AFFINE LayerNorm (the
        // checkpoint's own `{prefix}.norm3.weight`/`.bias` -- the only per-block LayerNorm with
        // real learned params; `cross_attn_norm=True` in the real config) -> cross-attn -> plain
        // (UNGATED) residual add. Previously this ran cross-attention on raw, un-normalized `x`.
        var norm3W = GetWeight($"{prefix}.norm3.weight");
        var norm3B = GetWeight($"{prefix}.norm3.bias");
        var normedCross = (float[])x.Clone();
        DiffusionOps.LayerNorm(normedCross, norm3W, norm3B, _dim);
        var crossAttn = CrossAttention($"{prefix}.cross_attn", normedCross, txtContext, numTokens, numTxt);
        for (int i = 0; i < x.Length; i++)
            x[i] += crossAttn[i];

        // 3. Modulated FeedForward (GELU approx tanh): affine-free LayerNorm (real `norm3` in
        // diffusers' own naming, i.e. the FFN pre-norm; also `elementwise_affine=False`) -> AdaLN
        // modulate -> FFN -> gated residual.
        var norm2 = (float[])x.Clone();
        LayerNormNoAffine(norm2, _dim);
        var normed2 = Modulate(norm2, numTokens, s2, sc2);
        var ffn = FeedForward($"{prefix}.ffn", normed2, numTokens);
        ApplyGatedResidual(x, ffn, numTokens, g2);

        return x;
    }

    /// <summary>Mean/variance LayerNorm with no learned affine (real `FP32LayerNorm(...,
    /// elementwise_affine=False)`, used for the two per-block AdaLN-modulated pre-norms).</summary>
    private static void LayerNormNoAffine(Span<float> x, int dim, float eps = 1e-6f)
    {
        int n = x.Length / dim;
        for (int row = 0; row < n; row++)
        {
            var row_ = x.Slice(row * dim, dim);
            float mean = 0f;
            for (int d = 0; d < dim; d++) mean += row_[d];
            mean /= dim;
            for (int d = 0; d < dim; d++) row_[d] -= mean;
            float var = 0f;
            for (int d = 0; d < dim; d++) var += row_[d] * row_[d];
            var /= dim;
            float scale = 1f / MathF.Sqrt(var + eps);
            for (int d = 0; d < dim; d++) row_[d] *= scale;
        }
    }

    private float[] SelfAttention(string prefix, float[] x, float[] cos, float[] sin, int seqLen)
    {
        var q = Linear($"{prefix}.q", x, _dim, _dim);
        var k = Linear($"{prefix}.k", x, _dim, _dim);
        var v = Linear($"{prefix}.v", x, _dim, _dim);

        // RMSNorm on Q and K per head
        var normQ = TryGetWeight($"{prefix}.norm_q.weight");
        if (normQ is not null) RmsNormHeads(q, seqLen, _numHeads, _headDim, normQ);
        var normK = TryGetWeight($"{prefix}.norm_k.weight");
        if (normK is not null) RmsNormHeads(k, seqLen, _numHeads, _headDim, normK);

        // Apply 3D-RoPE
        WanRoPE.ApplyRoPE(q, cos, sin, seqLen, _numHeads, _headDim);
        WanRoPE.ApplyRoPE(k, cos, sin, seqLen, _numHeads, _headDim);

        var attn = MultiHeadAttention(q, k, v, seqLen, seqLen, _numHeads, _headDim);
        return Linear($"{prefix}.o", attn, _dim, _dim);
    }

    private float[] CrossAttention(string prefix, float[] x, float[] context, int seqLen, int ctxLen)
    {
        var q = Linear($"{prefix}.q", x, _dim, _dim);
        var k = Linear($"{prefix}.k", context, _dim, _dim);
        var v = Linear($"{prefix}.v", context, _dim, _dim);

        var normQ = TryGetWeight($"{prefix}.norm_q.weight");
        if (normQ is not null) RmsNormHeads(q, seqLen, _numHeads, _headDim, normQ);
        var normK = TryGetWeight($"{prefix}.norm_k.weight");
        if (normK is not null) RmsNormHeads(k, ctxLen, _numHeads, _headDim, normK);

        var attn = MultiHeadAttention(q, k, v, seqLen, ctxLen, _numHeads, _headDim);
        return Linear($"{prefix}.o", attn, _dim, _dim);
    }

    private static float[] MultiHeadAttention(float[] q, float[] k, float[] v, int qSeq, int kvSeq, int numHeads, int headDim)
    {
        float scale = 1.0f / MathF.Sqrt(headDim);
        var output = new float[qSeq * numHeads * headDim];

        for (int h = 0; h < numHeads; h++)
        {
            for (int i = 0; i < qSeq; i++)
            {
                int qRow = (i * numHeads + h) * headDim;
                var scores = new float[kvSeq];
                float maxScore = float.NegativeInfinity;

                for (int j = 0; j < kvSeq; j++)
                {
                    int kRow = (j * numHeads + h) * headDim;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                        dot += q[qRow + d] * k[kRow + d];
                    dot *= scale;
                    scores[j] = dot;
                    if (dot > maxScore) maxScore = dot;
                }

                float sumExp = 0f;
                for (int j = 0; j < kvSeq; j++)
                {
                    scores[j] = MathF.Exp(scores[j] - maxScore);
                    sumExp += scores[j];
                }
                float invSum = 1f / sumExp;
                for (int j = 0; j < kvSeq; j++) scores[j] *= invSum;

                int outRow = (i * numHeads + h) * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    float sum = 0f;
                    for (int j = 0; j < kvSeq; j++)
                        sum += scores[j] * v[(j * numHeads + h) * headDim + d];
                    output[outRow + d] = sum;
                }
            }
        }

        return output;
    }

    private static void RmsNormHeads(float[] qk, int seqLen, int numHeads, int headDim, float[] gamma)
    {
        for (int s = 0; s < seqLen; s++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int off = (s * numHeads + h) * headDim;
                float sumSq = 0f;
                for (int d = 0; d < headDim; d++)
                {
                    float val = qk[off + d];
                    sumSq += val * val;
                }
                float invRms = 1.0f / MathF.Sqrt(sumSq / headDim + 1e-6f);
                for (int d = 0; d < headDim; d++)
                    qk[off + d] = qk[off + d] * invRms * gamma[d];
            }
        }
    }

    private float[] FeedForward(string prefix, float[] x, int seqLen)
    {
        var h1 = Linear($"{prefix}.0", x, _dim, _ffnDim);
        DiffusionOps.GeluInPlace(h1);
        return Linear($"{prefix}.2", h1, _ffnDim, _dim);
    }

    private float[] Modulate(float[] x, int seqLen, ReadOnlySpan<float> shift, ReadOnlySpan<float> scale)
    {
        var outF = new float[x.Length];
        for (int i = 0; i < seqLen; i++)
        {
            int off = i * _dim;
            for (int d = 0; d < _dim; d++)
            {
                float val = x[off + d];
                outF[off + d] = val * (1.0f + scale[d]) + shift[d];
            }
        }
        return outF;
    }

    private void ApplyGatedResidual(float[] x, float[] branch, int seqLen, ReadOnlySpan<float> gate)
    {
        for (int i = 0; i < seqLen; i++)
        {
            int off = i * _dim;
            for (int d = 0; d < _dim; d++)
                x[off + d] += branch[off + d] * gate[d];
        }
    }

    private float[] ComputeTimestepEmbedding(float timestep)
    {
        var emb = new float[256];
        int half = 128;
        float factor = 10000.0f;
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-MathF.Log(factor) * i / half);
            emb[i] = MathF.Cos(timestep * freq);
            emb[half + i] = MathF.Sin(timestep * freq);
        }

        var t0 = Linear("time_embedding.0", emb, 256, _dim);
        DiffusionOps.SiluInPlace(t0);
        return Linear("time_embedding.2", t0, _dim, _dim);
    }

    // Real Wan DiT text conditioning projection: `PixArtAlphaTextProjection(text_embed_dim, dim,
    // act_fn="gelu_tanh")` in transformer_wan.py's `WanTimeTextImageEmbedding` -- linear_1 -> GELU
    // (tanh-approx) -> linear_2, NOT a single linear (confirmed real GGUF tensor names are
    // `text_embedding.0`/`text_embedding.2`, matching `time_embedding`'s own two-layer shape).
    private float[] ComputeTextEmbedding(float[] textContext)
    {
        var t0 = Linear("text_embedding.0", textContext, TextDim, _dim);
        for (int i = 0; i < t0.Length; i++) t0[i] = DiffusionOps.Gelu(t0[i]);
        return Linear("text_embedding.2", t0, _dim, _dim);
    }

    private static float[] DiffusionOpsSilu(float[] x)
    {
        var res = (float[])x.Clone();
        DiffusionOps.SiluInPlace(res);
        return res;
    }

    private float[] Linear(string name, float[] x, int inDim, int outDim)
    {
        var w = GetWeight($"{name}.weight");
        var b = TryGetWeight($"{name}.bias");
        int rows = x.Length / inDim;
        var outF = new float[rows * outDim];

        for (int r = 0; r < rows; r++)
        {
            int inOff = r * inDim;
            int outOff = r * outDim;
            for (int o = 0; o < outDim; o++)
            {
                float sum = b is not null ? b[o] : 0f;
                int wOff = o * inDim;
                for (int i = 0; i < inDim; i++)
                    sum += x[inOff + i] * w[wOff + i];
                outF[outOff + o] = sum;
            }
        }
        return outF;
    }

    public static float[] PackLatents(float[] latents, int numFrames, int latH, int latW)
    {
        int patchH = latH / 2;
        int patchW = latW / 2;
        int numTokens = numFrames * patchH * patchW;
        var packed = new float[numTokens * InChannels];

        for (int f = 0; f < numFrames; f++)
        {
            for (int ph = 0; ph < patchH; ph++)
            {
                for (int pw = 0; pw < patchW; pw++)
                {
                    int tokenIdx = (f * patchH + ph) * patchW + pw;
                    int tokenOff = tokenIdx * InChannels;
                    int chanOffset = 0;

                    for (int c = 0; c < OutChannels; c++)
                    {
                        for (int dy = 0; dy < 2; dy++)
                        {
                            for (int dx = 0; dx < 2; dx++)
                            {
                                int y = ph * 2 + dy;
                                int x = pw * 2 + dx;
                                int srcIdx = ((c * numFrames + f) * latH + y) * latW + x;
                                packed[tokenOff + chanOffset++] = latents[srcIdx];
                            }
                        }
                    }
                }
            }
        }
        return packed;
    }

    public static float[] UnpackLatents(float[] packed, int numFrames, int latH, int latW)
    {
        int patchH = latH / 2;
        int patchW = latW / 2;
        var unpacked = new float[OutChannels * numFrames * latH * latW];

        for (int f = 0; f < numFrames; f++)
        {
            for (int ph = 0; ph < patchH; ph++)
            {
                for (int pw = 0; pw < patchW; pw++)
                {
                    int tokenIdx = (f * patchH + ph) * patchW + pw;
                    int tokenOff = tokenIdx * InChannels;
                    int chanOffset = 0;

                    for (int c = 0; c < OutChannels; c++)
                    {
                        for (int dy = 0; dy < 2; dy++)
                        {
                            for (int dx = 0; dx < 2; dx++)
                            {
                                int y = ph * 2 + dy;
                                int x = pw * 2 + dx;
                                int dstIdx = ((c * numFrames + f) * latH + y) * latW + x;
                                unpacked[dstIdx] = packed[tokenOff + chanOffset++];
                            }
                        }
                    }
                }
            }
        }
        return unpacked;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _weights.Dispose();
        }
    }
}
