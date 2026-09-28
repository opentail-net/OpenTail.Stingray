
namespace OpenTail.Stingray.Audio.AudioGen;

/// <summary>
/// Real AudioGen decoder-only LM forward pass, transcribed from the real `audiocraft.modules
/// .transformer` (`StreamingTransformerLayer`/`StreamingMultiheadAttention`) and
/// `audiocraft.models.lm` (`LMModel.forward`) source (pip-installed and read directly,
/// 2026-09-02 -- see docs/done/063-audiogen-implementation-plan.md).
///
/// <para><b>Real differences from MusicGen's HF-format decoder</b> (do not copy MusicGen's
/// assumptions here): (1) self- AND cross-attention use a single FUSED `in_proj_weight`
/// (`[3*hidden,hidden]`, Q/K/V concatenated) rather than separate Q/K/V matrices -- for
/// cross-attention, the query third is applied to the hidden state, the key/value thirds to the
/// conditioning tensor (real `nn.MultiheadAttention`-style behavior when `query != key`).
/// (2) Positional embedding is real sinusoidal, COMPUTED not loaded (`positional_embedding:
/// sin`): `phase = pos / max_period^(i/(halfDim-1))`, embedding = `concat([cos(phase),
/// sin(phase)])` -- note COS FIRST, sin second (the opposite half-order from MusicGen HF's
/// stored `[sin,cos]` buffer; confirmed from the real `create_sin_embedding` source, do not
/// assume the two conventions match). (3) Self-attention Q IS scaled by `1/sqrt(headDim)` (same
/// as MusicGen). (4) Real layer order (confirmed `norm_first=true`, `StreamingTransformerLayer
/// .forward`): `x += self_attn(norm1(x)); x += cross_attn(norm_cross(x), condSrc); x +=
/// ffn(norm2(x))` -- same shape as MusicGen's pre-norm layer, just different norm names/no
/// linear-layer bias anywhere (`bias_ff`/`bias_attn`/`bias_proj` all false; LayerNorms still
/// carry bias). (5) Final `out_norm` after the last layer, then per-codebook `linears.{q}` heads
/// (no bias).</para>
/// </summary>
public static class AudioGenTransformer
{
    public sealed class KvCache
    {
        public List<float[]>[] SelfK { get; }
        public List<float[]>[] SelfV { get; }
        public float[][]? CrossK { get; set; } // [layer][crossLen * hidden]
        public float[][]? CrossV { get; set; }
        public int CrossLen { get; set; }
        public int Position { get; set; }

        public KvCache()
        {
            SelfK = new List<float[]>[AudioGenConfig.NumLayers];
            SelfV = new List<float[]>[AudioGenConfig.NumLayers];
            for (int i = 0; i < AudioGenConfig.NumLayers; i++)
            {
                SelfK[i] = [];
                SelfV[i] = [];
            }
        }
    }

    /// <summary>Precomputes cross-attention K/V from the T5 text encoder's output, projected once through `output_proj` (T5's 1024-dim -&gt; 1536-dim) then through each layer's fused cross in_proj_weight's K/V thirds -- done once per generation, reused every decode step.</summary>
    public static unsafe void PrepareCrossAttention(AudioGenTransformerWeights w, float[][] encoderHiddenStates, KvCache cache)
    {
        int crossLen = encoderHiddenStates.Length;
        int hidden = AudioGenConfig.HiddenSize;
        int textDim = AudioGenConfig.TextDModel;

        var rawFlat = new float[crossLen * textDim];
        for (int i = 0; i < crossLen; i++) Array.Copy(encoderHiddenStates[i], 0, rawFlat, i * textDim, textDim);

        var flat = new float[crossLen * hidden];
        fixed (float* rp = rawFlat, fp = flat, bp = w.OutputProjBias)
            w.OutputProjWeight.MatMul(rp, crossLen, fp, bp);

        cache.CrossK = new float[AudioGenConfig.NumLayers][];
        cache.CrossV = new float[AudioGenConfig.NumLayers][];
        cache.CrossLen = crossLen;

        fixed (float* fp = flat)
        {
            for (int l = 0; l < AudioGenConfig.NumLayers; l++)
            {
                var k = new float[crossLen * hidden];
                var v = new float[crossLen * hidden];
                fixed (float* kp = k, vp = v)
                {
                    w.Layers[l].CrossAttnKWeight.MatMul(fp, crossLen, kp);
                    w.Layers[l].CrossAttnVWeight.MatMul(fp, crossLen, vp);
                }
                cache.CrossK[l] = k;
                cache.CrossV[l] = v;
            }
        }
    }

    /// <summary>Runs one decode step: `tokenColumn[codebook]` -&gt; summed embedding + sinusoidal position -&gt; N decoder layers (growing <paramref name="cache"/>) -&gt; final norm -&gt; per-codebook logits. Returns `[codebook][CodebookSize]`.</summary>
    public static float[][] Step(AudioGenTransformerWeights w, int[] tokenColumn, KvCache cache) =>
        StepBatch(w, tokenColumn, [cache])[0];

    /// <summary>
    /// <see cref="Step"/> for several sequences that share the token column and position (classifier-free guidance's
    /// conditional and unconditional branches): every projection runs as one B-row matmul, so each weight is streamed
    /// once per step instead of once per branch, while each sequence attends over its own <paramref name="caches"/>
    /// entry. Per-row arithmetic is the same as running <see cref="Step"/> on each cache. Returns `[b][codebook][CodebookSize]`.
    /// </summary>
    public static unsafe float[][][] StepBatch(AudioGenTransformerWeights w, int[] tokenColumn, KvCache[] caches)
    {
        int hidden = AudioGenConfig.HiddenSize;
        int b = caches.Length;
        var x0 = new float[hidden];
        for (int q = 0; q < AudioGenConfig.NumCodebooks; q++)
        {
            int tok = tokenColumn[q];
            var table = w.EmbedTokens[q];
            for (int d = 0; d < hidden; d++) x0[d] += table[tok * hidden + d];
        }

        int pos = caches[0].Position;
        AddSinusoidalPositionEmbedding(x0, pos, hidden);
        var x = new float[b * hidden];
        for (int i = 0; i < b; i++) Array.Copy(x0, 0, x, i * hidden, hidden);

        for (int li = 0; li < w.Layers.Length; li++)
            x = DecoderLayer(x, b, w.Layers[li], li, caches);

        var normed = LayerNormRows(x, b, w.OutNormWeight, w.OutNormBias);
        var logits = new float[b][][];
        for (int i = 0; i < b; i++) logits[i] = new float[AudioGenConfig.NumCodebooks][];
        var l = new float[b * AudioGenConfig.CodebookSize];
        fixed (float* np = normed, lp = l)
        {
            for (int q = 0; q < AudioGenConfig.NumCodebooks; q++)
            {
                w.LmHeads[q].MatMul(np, b, lp);
                for (int i = 0; i < b; i++)
                    logits[i][q] = l.AsSpan(i * AudioGenConfig.CodebookSize, AudioGenConfig.CodebookSize).ToArray();
            }
        }

        foreach (var c in caches) c.Position++;
        return logits;
    }

    /// <summary>Real `create_sin_embedding`: `phase = pos / maxPeriod^(i/(halfDim-1))` for `i` in `[0,halfDim)`, embedding = `concat([cos(phase), sin(phase)])` -- cos in the FIRST half, sin in the second (confirmed from the real `audiocraft.modules.transformer` source; do not assume MusicGen HF's `[sin,cos]` order applies here).</summary>
    private static void AddSinusoidalPositionEmbedding(float[] x, int position, int dim)
    {
        int halfDim = dim / 2;
        for (int i = 0; i < halfDim; i++)
        {
            float exponent = i / (float)(halfDim - 1);
            float divisor = MathF.Pow(AudioGenConfig.SinusoidalMaxPeriod, exponent);
            float phase = position / divisor;
            x[i] += MathF.Cos(phase);
            x[halfDim + i] += MathF.Sin(phase);
        }
    }

    private static float[] LayerNormRows(float[] x, int b, float[] weight, float[] bias)
    {
        int hidden = AudioGenConfig.HiddenSize;
        var output = new float[b * hidden];
        for (int i = 0; i < b; i++)
            LayerNorm(x.AsSpan(i * hidden, hidden), weight, bias, output.AsSpan(i * hidden, hidden));
        return output;
    }

    private static float[] DecoderLayer(float[] x, int b, AudioGenDecoderLayerWeights lw, int layerIndex, KvCache[] caches)
    {
        var selfOut = SelfAttention(LayerNormRows(x, b, lw.Norm1Weight, lw.Norm1Bias), b, lw, layerIndex, caches);
        var afterSelf = new float[x.Length];
        TensorPrimitives.Add(x, selfOut, afterSelf);

        var crossOut = CrossAttention(LayerNormRows(afterSelf, b, lw.NormCrossWeight, lw.NormCrossBias), b, lw, layerIndex, caches);
        var afterCross = new float[x.Length];
        TensorPrimitives.Add(afterSelf, crossOut, afterCross);

        var ffnOut = Ffn(LayerNormRows(afterCross, b, lw.Norm2Weight, lw.Norm2Bias), b, lw);
        var output = new float[x.Length];
        TensorPrimitives.Add(afterCross, ffnOut, output);
        return output;
    }

    private static unsafe float[] SelfAttention(float[] x, int b, AudioGenDecoderLayerWeights lw, int layerIndex, KvCache[] caches)
    {
        int hidden = AudioGenConfig.HiddenSize;
        int nHeads = AudioGenConfig.NumHeads;
        int headDim = AudioGenConfig.HeadDim;
        float scale = 1f / MathF.Sqrt(headDim);

        var q = new float[b * hidden];
        var k = new float[b * hidden];
        var v = new float[b * hidden];
        fixed (float* xp = x, qp = q, kp = k, vp = v)
        {
            lw.SelfAttnQWeight.MatMul(xp, b, qp);
            lw.SelfAttnKWeight.MatMul(xp, b, kp);
            lw.SelfAttnVWeight.MatMul(xp, b, vp);
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
            SoftmaxInPlace(scores);

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
            lw.SelfAttnOutProjWeight.MatMul(cp, b, op);
        return output;
    }

    private static unsafe float[] CrossAttention(float[] x, int b, AudioGenDecoderLayerWeights lw, int layerIndex, KvCache[] caches)
    {
        int hidden = AudioGenConfig.HiddenSize;
        int nHeads = AudioGenConfig.NumHeads;
        int headDim = AudioGenConfig.HeadDim;
        float scale = 1f / MathF.Sqrt(headDim);

        var q = new float[b * hidden];
        fixed (float* xp = x, qp = q)
            lw.CrossAttnQWeight.MatMul(xp, b, qp);

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
            SoftmaxInPlace(scores);

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
            lw.CrossAttnOutProjWeight.MatMul(cp, b, op);
        return output;
    }

    /// <summary>Real `fc1 -> GELU -> fc2` (`linear1`/`linear2`), no bias (`bias_ff: false`).</summary>
    private static unsafe float[] Ffn(float[] x, int b, AudioGenDecoderLayerWeights lw)
    {
        int hidden = AudioGenConfig.HiddenSize;
        int ffn = AudioGenConfig.FfnDim;
        var mid = new float[b * ffn];
        fixed (float* xp = x, mp = mid)
            lw.Linear1Weight.MatMul(xp, b, mp);

        for (int i = 0; i < mid.Length; i++) mid[i] = Gelu(mid[i]);

        var output = new float[b * hidden];
        fixed (float* mp = mid, op = output)
            lw.Linear2Weight.MatMul(mp, b, op);
        return output;
    }

    /// <summary>Real (erf-based) GELU -- config `activation: gelu` is PyTorch's default `F.gelu` (exact erf form), same convention as MusicGen's decoder.</summary>
    private static float Gelu(float x) => 0.5f * x * (1f + Erf(x / 1.4142135f));

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
        {
            SimdKernels.LayerNorm(op, xp, wp, bp, x.Length, eps);
        }
    }

    private static void SoftmaxInPlace(float[] scores)
    {
        TensorPrimitives.SoftMax(scores, scores);
    }
}
