using System.Numerics.Tensors;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>Weights of one audiocraft LM decoder layer (MusicGen, AudioGen): pre-norm self-attention, cross-attention
/// over the text encoder, and a bias-free GELU FFN.</summary>
internal sealed class AudiocraftLmLayer
{
    public required CfmLinearWeight SelfQ { get; init; }
    public required CfmLinearWeight SelfK { get; init; }
    public required CfmLinearWeight SelfV { get; init; }
    public required CfmLinearWeight SelfO { get; init; }
    public required float[] SelfNormWeight { get; init; }
    public required float[] SelfNormBias { get; init; }
    public required CfmLinearWeight CrossQ { get; init; }
    public required CfmLinearWeight CrossK { get; init; }
    public required CfmLinearWeight CrossV { get; init; }
    public required CfmLinearWeight CrossO { get; init; }
    public required float[] CrossNormWeight { get; init; }
    public required float[] CrossNormBias { get; init; }
    public required CfmLinearWeight Fc1 { get; init; }
    public required CfmLinearWeight Fc2 { get; init; }
    public required float[] FfnNormWeight { get; init; }
    public required float[] FfnNormBias { get; init; }
}

/// <summary>An audiocraft LM decoder: dimensions, layers, token tables, output norm and heads, the text-to-decoder
/// projection, and how the model adds its position term (MusicGen: a learned table; AudioGen: cos/sin).</summary>
internal sealed class AudiocraftLmModel
{
    public required int Hidden { get; init; }
    public required int NumHeads { get; init; }
    public required int HeadDim { get; init; }
    public required int FfnDim { get; init; }
    public required int TextDim { get; init; }
    public required int CodebookSize { get; init; }
    public required AudiocraftLmLayer[] Layers { get; init; }
    public required float[][] EmbedTokens { get; init; }
    public required float[] OutNormWeight { get; init; }
    public required float[] OutNormBias { get; init; }
    public required CfmLinearWeight[] LmHeads { get; init; }
    public required CfmLinearWeight TextProj { get; init; }
    public required float[] TextProjBias { get; init; }
    /// <summary>Adds the position term for <c>position</c> to a summed token embedding in place.</summary>
    public required Action<float[], int> AddPosition { get; init; }
}

/// <summary>Per-sequence decoder state: self-attention K/V per layer (grown each step), the cross-attention K/V computed
/// once from the text encoder, and the next position.</summary>
public class AudiocraftLmKvCache
{
    public List<float[]>[] SelfK { get; }
    public List<float[]>[] SelfV { get; }
    public float[][]? CrossK { get; set; } // [layer][crossLen * hidden]
    public float[][]? CrossV { get; set; }
    public int CrossLen { get; set; }
    public int Position { get; set; } // next absolute position to embed (== number of self-attn steps so far)

    public AudiocraftLmKvCache(int numLayers)
    {
        SelfK = new List<float[]>[numLayers];
        SelfV = new List<float[]>[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            SelfK[i] = [];
            SelfV[i] = [];
        }
    }
}

/// <summary>
/// The audiocraft LM decoder step shared by <see cref="MusicGen.MusicGenTransformer"/> and
/// <see cref="AudioGen.AudioGenTransformer"/>, whose layer math is identical. <see cref="StepBatch"/> runs several
/// sequences sharing the token column and position (CFG's conditional and unconditional branches) through every
/// projection as one B-row matmul, so the weights stream once per step; each sequence attends over its own cache. A
/// B = 1 call is the single-sequence step, with the same per-row arithmetic.
/// </summary>
internal static class AudiocraftLmKernels
{
    /// <summary>Projects the text encoder output to the decoder width, then through each layer's cross K/V -- once per
    /// generation, reused every step.</summary>
    public static unsafe void PrepareCrossAttention(AudiocraftLmModel m, float[][] encoderHiddenStates, AudiocraftLmKvCache cache)
    {
        int crossLen = encoderHiddenStates.Length;
        int hidden = m.Hidden;
        var rawFlat = new float[crossLen * m.TextDim];
        for (int i = 0; i < crossLen; i++) Array.Copy(encoderHiddenStates[i], 0, rawFlat, i * m.TextDim, m.TextDim);

        var flat = new float[crossLen * hidden];
        fixed (float* rp = rawFlat, fp = flat, bp = m.TextProjBias)
            m.TextProj.MatMul(rp, crossLen, fp, bp);

        cache.CrossK = new float[m.Layers.Length][];
        cache.CrossV = new float[m.Layers.Length][];
        cache.CrossLen = crossLen;
        fixed (float* fp = flat)
        {
            for (int l = 0; l < m.Layers.Length; l++)
            {
                var k = new float[crossLen * hidden];
                var v = new float[crossLen * hidden];
                fixed (float* kp = k, vp = v)
                {
                    m.Layers[l].CrossK.MatMul(fp, crossLen, kp);
                    m.Layers[l].CrossV.MatMul(fp, crossLen, vp);
                }
                cache.CrossK[l] = k;
                cache.CrossV[l] = v;
            }
        }
    }

    /// <summary>One decode step for <paramref name="caches"/>.Length sequences sharing <paramref name="tokenColumn"/> and
    /// position. Returns <c>[sequence][codebook][CodebookSize]</c>.</summary>
    public static unsafe float[][][] StepBatch(AudiocraftLmModel m, int[] tokenColumn, AudiocraftLmKvCache[] caches)
    {
        int hidden = m.Hidden;
        int b = caches.Length;
        int codebooks = m.EmbedTokens.Length;
        var x0 = new float[hidden];
        for (int q = 0; q < codebooks; q++)
        {
            int tok = tokenColumn[q];
            var table = m.EmbedTokens[q];
            for (int d = 0; d < hidden; d++) x0[d] += table[tok * hidden + d];
        }
        m.AddPosition(x0, caches[0].Position);
        var x = new float[b * hidden];
        for (int i = 0; i < b; i++) Array.Copy(x0, 0, x, i * hidden, hidden);

        for (int li = 0; li < m.Layers.Length; li++)
            x = DecoderLayer(m, x, b, m.Layers[li], li, caches);

        var normed = LayerNormRows(x, b, hidden, m.OutNormWeight, m.OutNormBias);
        var logits = new float[b][][];
        for (int i = 0; i < b; i++) logits[i] = new float[codebooks][];
        var l = new float[b * m.CodebookSize];
        fixed (float* np = normed, lp = l)
        {
            for (int q = 0; q < codebooks; q++)
            {
                m.LmHeads[q].MatMul(np, b, lp);
                for (int i = 0; i < b; i++)
                    logits[i][q] = l.AsSpan(i * m.CodebookSize, m.CodebookSize).ToArray();
            }
        }

        foreach (var c in caches) c.Position++;
        return logits;
    }

    private static float[] LayerNormRows(float[] x, int b, int hidden, float[] weight, float[] bias)
    {
        var output = new float[b * hidden];
        for (int i = 0; i < b; i++)
            LayerNorm(x.AsSpan(i * hidden, hidden), weight, bias, output.AsSpan(i * hidden, hidden));
        return output;
    }

    private static float[] DecoderLayer(AudiocraftLmModel m, float[] x, int b, AudiocraftLmLayer lw, int layerIndex, AudiocraftLmKvCache[] caches)
    {
        var selfOut = SelfAttention(m, LayerNormRows(x, b, m.Hidden, lw.SelfNormWeight, lw.SelfNormBias), b, lw, layerIndex, caches);
        var afterSelf = new float[x.Length];
        TensorPrimitives.Add(x, selfOut, afterSelf);

        var crossOut = CrossAttention(m, LayerNormRows(afterSelf, b, m.Hidden, lw.CrossNormWeight, lw.CrossNormBias), b, lw, layerIndex, caches);
        var afterCross = new float[x.Length];
        TensorPrimitives.Add(afterSelf, crossOut, afterCross);

        var ffnOut = Ffn(m, LayerNormRows(afterCross, b, m.Hidden, lw.FfnNormWeight, lw.FfnNormBias), b, lw);
        var output = new float[x.Length];
        TensorPrimitives.Add(afterCross, ffnOut, output);
        return output;
    }

    private static unsafe float[] SelfAttention(AudiocraftLmModel m, float[] x, int b, AudiocraftLmLayer lw, int layerIndex, AudiocraftLmKvCache[] caches)
    {
        int hidden = m.Hidden, nHeads = m.NumHeads, headDim = m.HeadDim;
        float scale = 1f / MathF.Sqrt(headDim);

        var q = new float[b * hidden];
        var k = new float[b * hidden];
        var v = new float[b * hidden];
        fixed (float* xp = x, qp = q, kp = k, vp = v)
        {
            lw.SelfQ.MatMul(xp, b, qp);
            lw.SelfK.MatMul(xp, b, kp);
            lw.SelfV.MatMul(xp, b, vp);
        }
        for (int i = 0; i < b; i++)
        {
            caches[i].SelfK[layerIndex].Add(k.AsSpan(i * hidden, hidden).ToArray());
            caches[i].SelfV[layerIndex].Add(v.AsSpan(i * hidden, hidden).ToArray());
        }

        var context = new float[b * hidden];
        Parallel.For(0, b * nHeads, job =>
        {
            int i = job / nHeads, h = job % nHeads;
            var cache = caches[i];
            int histLen = cache.SelfK[layerIndex].Count;
            int qOff = i * hidden + h * headDim, off = h * headDim;
            var scores = new float[histLen];
            for (int j = 0; j < histLen; j++)
            {
                var kj = cache.SelfK[layerIndex][j];
                float dot = 0f;
                for (int d = 0; d < headDim; d++) dot += q[qOff + d] * kj[off + d];
                scores[j] = dot * scale;
            }
            DenseKernels.SoftmaxInPlace(scores); // max-subtracted (TensorPrimitives.SoftMax can overflow to NaN)

            var ctxSpan = context.AsSpan(qOff, headDim);
            for (int j = 0; j < histLen; j++)
            {
                float s = scores[j];
                var vj = cache.SelfV[layerIndex][j];
                for (int d = 0; d < headDim; d++) ctxSpan[d] += s * vj[off + d];
            }
        });

        var output = new float[b * hidden];
        fixed (float* cp = context, op = output)
            lw.SelfO.MatMul(cp, b, op);
        return output;
    }

    private static unsafe float[] CrossAttention(AudiocraftLmModel m, float[] x, int b, AudiocraftLmLayer lw, int layerIndex, AudiocraftLmKvCache[] caches)
    {
        int hidden = m.Hidden, nHeads = m.NumHeads, headDim = m.HeadDim;
        float scale = 1f / MathF.Sqrt(headDim);

        var q = new float[b * hidden];
        fixed (float* xp = x, qp = q)
            lw.CrossQ.MatMul(xp, b, qp);

        var context = new float[b * hidden];
        Parallel.For(0, b * nHeads, job =>
        {
            int i = job / nHeads, h = job % nHeads;
            var cache = caches[i];
            var crossK = cache.CrossK![layerIndex];
            var crossV = cache.CrossV![layerIndex];
            int crossLen = cache.CrossLen;
            int qOff = i * hidden + h * headDim, off = h * headDim;
            var scores = new float[crossLen];
            for (int j = 0; j < crossLen; j++)
            {
                float dot = 0f;
                int kBase = j * hidden + off;
                for (int d = 0; d < headDim; d++) dot += q[qOff + d] * crossK[kBase + d];
                scores[j] = dot * scale;
            }
            DenseKernels.SoftmaxInPlace(scores); // max-subtracted (TensorPrimitives.SoftMax can overflow to NaN)

            var ctxSpan = context.AsSpan(qOff, headDim);
            for (int j = 0; j < crossLen; j++)
            {
                float s = scores[j];
                int vBase = j * hidden + off;
                for (int d = 0; d < headDim; d++) ctxSpan[d] += s * crossV[vBase + d];
            }
        });

        var output = new float[b * hidden];
        fixed (float* cp = context, op = output)
            lw.CrossO.MatMul(cp, b, op);
        return output;
    }

    /// <summary>`fc1 -> GELU -> fc2`, no biases, exact erf GELU (both models' `activation: gelu`).</summary>
    private static unsafe float[] Ffn(AudiocraftLmModel m, float[] x, int b, AudiocraftLmLayer lw)
    {
        var mid = new float[b * m.FfnDim];
        fixed (float* xp = x, mp = mid)
            lw.Fc1.MatMul(xp, b, mp);
        for (int i = 0; i < mid.Length; i++) mid[i] = Gelu(mid[i]);

        var output = new float[b * m.Hidden];
        fixed (float* mp = mid, op = output)
            lw.Fc2.MatMul(mp, b, op);
        return output;
    }

    private static float Gelu(float x) => 0.5f * x * (1f + Erf(x / 1.4142135f));

    // Abramowitz-Stegun 7.1.26 approximation, max error ~1.5e-7.
    private static float Erf(float x)
    {
        float sign = MathF.Sign(x);
        x = MathF.Abs(x);
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float t = 1f / (1f + p * x);
        float y = 1f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * MathF.Exp(-x * x);
        return sign * y;
    }

    private static unsafe void LayerNorm(ReadOnlySpan<float> x, float[] weight, float[] bias, Span<float> output, float eps = 1e-5f)
    {
        fixed (float* op = output, xp = x, wp = weight, bp = bias)
            SimdKernels.LayerNorm(op, xp, wp, bp, x.Length, eps);
    }
}
