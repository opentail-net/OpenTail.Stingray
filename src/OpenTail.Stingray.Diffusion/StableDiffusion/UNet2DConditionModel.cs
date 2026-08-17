using System.Numerics.Tensors;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Diffusion.StableDiffusion;

/// <summary>
/// Stable Diffusion 1.5 UNet (2D Condition Model).
/// Input: Noisy latent [4, H, W], timestep t ∈ [0, 999], CLIP text context [77, 768].
/// Output: Predicted noise [4, H, W].
///
/// Architecture:
/// - 4 Resolution Levels (Channels: 320, 640, 1280, 1280)
/// - 12 Input Blocks with Skip Connections
/// - Middle Block (ResBlock + SpatialTransformer + ResBlock)
/// - 12 Output Blocks with Skip Connection Concatenations
/// - SpatialTransformer with Asymmetric Cross-Attention (Query: H×W, Key/Value: 77)
/// - GEGLU FeedForward activation
/// </summary>
public sealed class UNet2DConditionModel
{
    private readonly IWeightLoader _weights;
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);
    private readonly string _prefix;

    private const int ModelChannels = 320;
    private const int TimeEmbedDim = 1280;
    private const int ContextDim = 768;
    private const int NumHeads = 8;

    public UNet2DConditionModel(IWeightLoader weights, string prefix = "model.diffusion_model.")
    {
        _weights = weights;
        _prefix = prefix;
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

    /// <summary>
    /// Computes sinusoidal timestep embedding and passes through 2-layer MLP.
    /// Uses flip_sin_to_cos = true (CompVis / SD1.5 standard).
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
            sinEmb[i]        = MathF.Cos(arg); // flip_sin_to_cos: first cos, then sin
            sinEmb[half + i] = MathF.Sin(arg);
        }

        // Linear: 320 -> 1280
        var w0 = GetWeight("time_embed.0.weight");
        var b0 = GetWeight("time_embed.0.bias");
        var emb = DiffusionOps.Linear(sinEmb, w0, b0, 1, dim, TimeEmbedDim);

        // SiLU
        DiffusionOps.SiluInPlace(emb);

        // Linear: 1280 -> 1280
        var w2 = GetWeight("time_embed.2.weight");
        var b2 = GetWeight("time_embed.2.bias");
        return DiffusionOps.Linear(emb, w2, b2, 1, TimeEmbedDim, TimeEmbedDim);
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

        var conv1W = GetWeight($"{prefix}.in_layers.2.weight");
        var conv1B = GetWeight($"{prefix}.in_layers.2.bias");
        var hOut = DiffusionOps.Conv2D(hNorm, conv1W, conv1B, 1, inC, h, w, outC, 3, 3);

        // 2. emb_layers: SiLU(tEmb) -> Linear(1280 -> outC) added spatially
        var tEmbAct = (float[])tEmb.Clone();
        DiffusionOps.SiluInPlace(tEmbAct);
        var embW = GetWeight($"{prefix}.emb_layers.1.weight");
        var embB = GetWeight($"{prefix}.emb_layers.1.bias");
        var tProj = DiffusionOps.Linear(tEmbAct, embW, embB, 1, TimeEmbedDim, outC);

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

        var conv2W = GetWeight($"{prefix}.out_layers.3.weight");
        var conv2B = GetWeight($"{prefix}.out_layers.3.bias");
        hOut = DiffusionOps.Conv2D(hOut, conv2W, conv2B, 1, outC, h, w, outC, 3, 3);

        // 4. Skip connection (nin_shortcut if inC != outC)
        float[] xRes;
        var skipW = TryGetWeight($"{prefix}.skip_connection.weight");
        if (skipW is not null)
        {
            var skipB = TryGetWeight($"{prefix}.skip_connection.bias");
            xRes = DiffusionOps.Conv2D(x, skipW, skipB, 1, inC, h, w, outC, 1, 1, stride: 1, padding: 0);
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

        var projInW = GetWeight($"{prefix}.proj_in.weight");
        var projInB = GetWeight($"{prefix}.proj_in.bias");
        var xProj = DiffusionOps.Conv2D(xNorm, projInW, projInB, 1, c, h, w, c, 1, 1, stride: 1, padding: 0);

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

        var saQW = GetWeight($"{tb}.attn1.to_q.weight");
        var saKW = GetWeight($"{tb}.attn1.to_k.weight");
        var saVW = GetWeight($"{tb}.attn1.to_v.weight");
        var saOutW = GetWeight($"{tb}.attn1.to_out.0.weight");
        var saOutB = GetWeight($"{tb}.attn1.to_out.0.bias");

        var saQ = DiffusionOps.Linear(saNorm, saQW, null, hw, c, c);
        var saK = DiffusionOps.Linear(saNorm, saKW, null, hw, c, c);
        var saV = DiffusionOps.Linear(saNorm, saVW, null, hw, c, c);

        var saAttnOut = MultiHeadAttention(saQ, saK, saV, hw, hw, c, NumHeads);
        var saProjOut = DiffusionOps.Linear(saAttnOut, saOutW, saOutB, hw, c, c);

        for (int i = 0; i < xSeq.Length; i++)
            xSeq[i] += saProjOut[i];

        // B. Cross-Attention (to CLIP text context: 77 tokens, 768 dim):
        var caNormW = GetWeight($"{tb}.norm2.weight");
        var caNormB = GetWeight($"{tb}.norm2.bias");
        var caNorm = (float[])xSeq.Clone();
        DiffusionOps.LayerNorm(caNorm, caNormW, caNormB, c);

        var caQW = GetWeight($"{tb}.attn2.to_q.weight");
        var caKW = GetWeight($"{tb}.attn2.to_k.weight");
        var caVW = GetWeight($"{tb}.attn2.to_v.weight");
        var caOutW = GetWeight($"{tb}.attn2.to_out.0.weight");
        var caOutB = GetWeight($"{tb}.attn2.to_out.0.bias");

        var caQ = DiffusionOps.Linear(caNorm, caQW, null, hw, c, c);
        var caK = DiffusionOps.Linear(context, caKW, null, 77, ContextDim, c);
        var caV = DiffusionOps.Linear(context, caVW, null, 77, ContextDim, c);

        var caAttnOut = MultiHeadAttention(caQ, caK, caV, hw, 77, c, NumHeads);
        var caProjOut = DiffusionOps.Linear(caAttnOut, caOutW, caOutB, hw, c, c);

        for (int i = 0; i < xSeq.Length; i++)
            xSeq[i] += caProjOut[i];

        // C. Feed-Forward with GEGLU:
        var ffNormW = GetWeight($"{tb}.norm3.weight");
        var ffNormB = GetWeight($"{tb}.norm3.bias");
        var ffNorm = (float[])xSeq.Clone();
        DiffusionOps.LayerNorm(ffNorm, ffNormW, ffNormB, c);

        var ff1W = GetWeight($"{tb}.ff.net.0.proj.weight");
        var ff1B = GetWeight($"{tb}.ff.net.0.proj.bias");
        var ff2W = GetWeight($"{tb}.ff.net.2.weight");
        var ff2B = GetWeight($"{tb}.ff.net.2.bias");

        int mlpDim = c * 4;
        // net.0.proj projects c -> mlpDim * 2
        var ffH = DiffusionOps.Linear(ffNorm, ff1W, ff1B, hw, c, mlpDim * 2);
        var ffGated = new float[hw * mlpDim];
        Parallel.For(0, hw, s =>
        {
            int srcOff = s * mlpDim * 2;
            int dstOff = s * mlpDim;
            for (int d = 0; d < mlpDim; d++)
            {
                float val = ffH[srcOff + d];
                float gate = ffH[srcOff + mlpDim + d];
                // GELU(gate): approximate tanh version
                float geluGate = 0.5f * gate * (1.0f + MathF.Tanh(0.79788456f * (gate + 0.044715f * gate * gate * gate)));
                ffGated[dstOff + d] = val * geluGate;
            }
        });

        var ffOut = DiffusionOps.Linear(ffGated, ff2W, ff2B, hw, mlpDim, c);

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
        var projOutW = GetWeight($"{prefix}.proj_out.weight");
        var projOutB = GetWeight($"{prefix}.proj_out.bias");
        var projOut = DiffusionOps.Conv2D(xSpatial, projOutW, projOutB, 1, c, h, w, c, 1, 1, stride: 1, padding: 0);

        for (int i = 0; i < x.Length; i++)
            projOut[i] += x[i];

        return projOut;
    }

    /// <summary>
    /// Multi-head scaled dot-product attention for Q [qLen, C], K [kvLen, C], V [kvLen, C].
    /// </summary>
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

                // Compute scores: Q_h[qi] . K_h[kj] * scale
                for (int kj = 0; kj < kvLen; kj++)
                {
                    int kBase = kj * c + headOffset;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                        dot += q[qBase + d] * k[kBase + d];
                    scores[kj] = dot * scale;
                }

                // Softmax
                DiffusionOps.Softmax(scores, 0, kvLen);

                // Weighted sum: sum_kj (scores[kj] * V_h[kj])
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

    /// <summary>
    /// UNet 2D Condition Model Forward Pass:
    /// Takes noisy latent x [4, H, W], scalar timestep t, and CLIP text context [77, 768].
    /// Returns predicted noise [4, H, W].
    /// </summary>
    public float[] Forward(float[] x, float timestep, float[] context, int latH, int latW)
    {
        // 1. Timestep embedding: [1280]
        var tEmb = ComputeTimeEmbedding(timestep);

        var savedInputs = new List<float[]>(12);

        // ── Input Blocks ────────────────────────────────────────────────────────
        // Block 0: Conv2D(4 -> 320, 3x3)
        int h = latH, w = latW;
        var in0W = GetWeight("input_blocks.0.0.weight");
        var in0B = GetWeight("input_blocks.0.0.bias");
        var cur = DiffusionOps.Conv2D(x, in0W, in0B, 1, 4, h, w, 320, 3, 3);
        savedInputs.Add(cur);

        // Block 1: ResBlock(320 -> 320) + SpatialTransformer(320)
        cur = ResBlock("input_blocks.1.0", cur, tEmb, 320, 320, h, w);
        cur = SpatialTransformer("input_blocks.1.1", cur, context, 320, h, w);
        savedInputs.Add(cur);

        // Block 2: ResBlock(320 -> 320) + SpatialTransformer(320)
        cur = ResBlock("input_blocks.2.0", cur, tEmb, 320, 320, h, w);
        cur = SpatialTransformer("input_blocks.2.1", cur, context, 320, h, w);
        savedInputs.Add(cur);

        // Block 3: Downsample (Conv2D 320 -> 320, stride 2)
        var ds3W = GetWeight("input_blocks.3.0.op.weight");
        var ds3B = GetWeight("input_blocks.3.0.op.bias");
        cur = DiffusionOps.Conv2D(cur, ds3W, ds3B, 1, 320, h, w, 320, 3, 3, stride: 2);
        h /= 2; w /= 2;
        savedInputs.Add(cur);

        // Block 4: ResBlock(320 -> 640) + SpatialTransformer(640)
        cur = ResBlock("input_blocks.4.0", cur, tEmb, 320, 640, h, w);
        cur = SpatialTransformer("input_blocks.4.1", cur, context, 640, h, w);
        savedInputs.Add(cur);

        // Block 5: ResBlock(640 -> 640) + SpatialTransformer(640)
        cur = ResBlock("input_blocks.5.0", cur, tEmb, 640, 640, h, w);
        cur = SpatialTransformer("input_blocks.5.1", cur, context, 640, h, w);
        savedInputs.Add(cur);

        // Block 6: Downsample (Conv2D 640 -> 640, stride 2)
        var ds6W = GetWeight("input_blocks.6.0.op.weight");
        var ds6B = GetWeight("input_blocks.6.0.op.bias");
        cur = DiffusionOps.Conv2D(cur, ds6W, ds6B, 1, 640, h, w, 640, 3, 3, stride: 2);
        h /= 2; w /= 2;
        savedInputs.Add(cur);

        // Block 7: ResBlock(640 -> 1280) + SpatialTransformer(1280)
        cur = ResBlock("input_blocks.7.0", cur, tEmb, 640, 1280, h, w);
        cur = SpatialTransformer("input_blocks.7.1", cur, context, 1280, h, w);
        savedInputs.Add(cur);

        // Block 8: ResBlock(1280 -> 1280) + SpatialTransformer(1280)
        cur = ResBlock("input_blocks.8.0", cur, tEmb, 1280, 1280, h, w);
        cur = SpatialTransformer("input_blocks.8.1", cur, context, 1280, h, w);
        savedInputs.Add(cur);

        // Block 9: Downsample (Conv2D 1280 -> 1280, stride 2)
        var ds9W = GetWeight("input_blocks.9.0.op.weight");
        var ds9B = GetWeight("input_blocks.9.0.op.bias");
        cur = DiffusionOps.Conv2D(cur, ds9W, ds9B, 1, 1280, h, w, 1280, 3, 3, stride: 2);
        h /= 2; w /= 2;
        savedInputs.Add(cur);

        // Block 10: ResBlock(1280 -> 1280)
        cur = ResBlock("input_blocks.10.0", cur, tEmb, 1280, 1280, h, w);
        savedInputs.Add(cur);

        // Block 11: ResBlock(1280 -> 1280)
        cur = ResBlock("input_blocks.11.0", cur, tEmb, 1280, 1280, h, w);
        savedInputs.Add(cur);

        // ── Middle Block ────────────────────────────────────────────────────────
        // ResBlock(1280, 1280) + SpatialTransformer(1280) + ResBlock(1280, 1280)
        cur = ResBlock("middle_block.0", cur, tEmb, 1280, 1280, h, w);
        cur = SpatialTransformer("middle_block.1", cur, context, 1280, h, w);
        cur = ResBlock("middle_block.2", cur, tEmb, 1280, 1280, h, w);

        // ── Output Blocks ───────────────────────────────────────────────────────
        // Helper to concatenate skip connection along channel dimension C
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
        var us2W = GetWeight("output_blocks.2.1.conv.weight");
        var us2B = GetWeight("output_blocks.2.1.conv.bias");
        h *= 2; w *= 2;
        cur = DiffusionOps.Conv2D(cur, us2W, us2B, 1, 1280, h, w, 1280, 3, 3);

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
        var us5W = GetWeight("output_blocks.5.2.conv.weight");
        var us5B = GetWeight("output_blocks.5.2.conv.bias");
        h *= 2; w *= 2;
        cur = DiffusionOps.Conv2D(cur, us5W, us5B, 1, 1280, h, w, 1280, 3, 3);

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
        var us8W = GetWeight("output_blocks.8.2.conv.weight");
        var us8B = GetWeight("output_blocks.8.2.conv.bias");
        h *= 2; w *= 2;
        cur = DiffusionOps.Conv2D(cur, us8W, us8B, 1, 640, h, w, 640, 3, 3);

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
        // GroupNorm(32, 320) + SiLU + Conv2D(320 -> 4, 3x3)
        var outGnW = GetWeight("out.0.weight");
        var outGnB = GetWeight("out.0.bias");
        DiffusionOps.GroupNorm(cur, outGnW, outGnB, 1, 320, h, w, groups: 32);
        DiffusionOps.SiluInPlace(cur);

        var outConvW = GetWeight("out.2.weight");
        var outConvB = GetWeight("out.2.bias");
        return DiffusionOps.Conv2D(cur, outConvW, outConvB, 1, 320, h, w, 4, 3, 3);
    }
}
