
namespace OpenTail.Stingray.Audio.AudioGen;

/// <summary>
/// Real AudioGen decoder-only LM forward pass, transcribed from the real `audiocraft.modules
/// .transformer` (`StreamingTransformerLayer`/`StreamingMultiheadAttention`) and
/// `audiocraft.models.lm` (`LMModel.forward`) source (pip-installed and read directly,
/// 2026-09-02 -- see docs/063-audiogen-implementation-plan.md).
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
    public static unsafe float[][] Step(AudioGenTransformerWeights w, int[] tokenColumn, KvCache cache)
    {
        int hidden = AudioGenConfig.HiddenSize;
        var x = new float[hidden];
        for (int q = 0; q < AudioGenConfig.NumCodebooks; q++)
        {
            int tok = tokenColumn[q];
            var table = w.EmbedTokens[q];
            for (int d = 0; d < hidden; d++) x[d] += table[tok * hidden + d];
        }

        int pos = cache.Position;
        AddSinusoidalPositionEmbedding(x, pos, hidden);

        foreach (var (layer, li) in w.Layers.Select((l, i) => (l, i)))
            x = DecoderLayer(x, layer, li, cache);

        var normed = new float[hidden];
        LayerNorm(x, w.OutNormWeight, w.OutNormBias, normed);

        var logits = new float[AudioGenConfig.NumCodebooks][];
        fixed (float* np = normed)
        {
            for (int q = 0; q < AudioGenConfig.NumCodebooks; q++)
            {
                var l = new float[AudioGenConfig.CodebookSize];
                fixed (float* lp = l)
                    w.LmHeads[q].MatMul(np, 1, lp);
                logits[q] = l;
            }
        }

        cache.Position++;
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

    private static unsafe float[] DecoderLayer(float[] x, AudioGenDecoderLayerWeights lw, int layerIndex, KvCache cache)
    {
        int hidden = AudioGenConfig.HiddenSize;

        var normed1 = new float[hidden];
        LayerNorm(x, lw.Norm1Weight, lw.Norm1Bias, normed1);
        var selfOut = SelfAttention(normed1, lw, layerIndex, cache);
        var afterSelf = new float[hidden];
        TensorPrimitives.Add(x, selfOut, afterSelf);

        var normedCross = new float[hidden];
        LayerNorm(afterSelf, lw.NormCrossWeight, lw.NormCrossBias, normedCross);
        var crossOut = CrossAttention(normedCross, lw, layerIndex, cache);
        var afterCross = new float[hidden];
        TensorPrimitives.Add(afterSelf, crossOut, afterCross);

        var normed2 = new float[hidden];
        LayerNorm(afterCross, lw.Norm2Weight, lw.Norm2Bias, normed2);
        var ffnOut = Ffn(normed2, lw);
        var output = new float[hidden];
        TensorPrimitives.Add(afterCross, ffnOut, output);
        return output;
    }

    private static unsafe float[] SelfAttention(float[] x, AudioGenDecoderLayerWeights lw, int layerIndex, KvCache cache)
    {
        int hidden = AudioGenConfig.HiddenSize;
        int nHeads = AudioGenConfig.NumHeads;
        int headDim = AudioGenConfig.HeadDim;
        float scale = 1f / MathF.Sqrt(headDim);

        var q = new float[hidden];
        var k = new float[hidden];
        var v = new float[hidden];
        fixed (float* xp = x, qp = q, kp = k, vp = v)
        {
            lw.SelfAttnQWeight.MatMul(xp, 1, qp);
            lw.SelfAttnKWeight.MatMul(xp, 1, kp);
            lw.SelfAttnVWeight.MatMul(xp, 1, vp);
        }

        cache.SelfK[layerIndex].Add(k);
        cache.SelfV[layerIndex].Add(v);
        int histLen = cache.SelfK[layerIndex].Count;

        var context = new float[hidden];
        Parallel.For(0, nHeads, h =>
        {
            int off = h * headDim;
            var scores = new float[histLen];
            for (int j = 0; j < histLen; j++)
            {
                var kj = cache.SelfK[layerIndex][j];
                float dot = 0f;
                for (int d = 0; d < headDim; d++) dot += q[off + d] * kj[off + d];
                scores[j] = dot * scale;
            }
            SoftmaxInPlace(scores);

            var ctxSpan = context.AsSpan(off, headDim);
            for (int j = 0; j < histLen; j++)
            {
                float s = scores[j];
                var vj = cache.SelfV[layerIndex][j];
                for (int d = 0; d < headDim; d++) ctxSpan[d] += s * vj[off + d];
            }
        });

        var output = new float[hidden];
        fixed (float* cp = context, op = output)
            lw.SelfAttnOutProjWeight.MatMul(cp, 1, op);
        return output;
    }

    private static unsafe float[] CrossAttention(float[] x, AudioGenDecoderLayerWeights lw, int layerIndex, KvCache cache)
    {
        int hidden = AudioGenConfig.HiddenSize;
        int nHeads = AudioGenConfig.NumHeads;
        int headDim = AudioGenConfig.HeadDim;
        float scale = 1f / MathF.Sqrt(headDim);

        var q = new float[hidden];
        fixed (float* xp = x, qp = q)
            lw.CrossAttnQWeight.MatMul(xp, 1, qp);

        var crossK = cache.CrossK![layerIndex];
        var crossV = cache.CrossV![layerIndex];
        int crossLen = cache.CrossLen;

        var context = new float[hidden];
        Parallel.For(0, nHeads, h =>
        {
            int off = h * headDim;
            var scores = new float[crossLen];
            for (int j = 0; j < crossLen; j++)
            {
                float dot = 0f;
                int kBase = j * hidden + off;
                for (int d = 0; d < headDim; d++) dot += q[off + d] * crossK[kBase + d];
                scores[j] = dot * scale;
            }
            SoftmaxInPlace(scores);

            var ctxSpan = context.AsSpan(off, headDim);
            for (int j = 0; j < crossLen; j++)
            {
                float s = scores[j];
                int vBase = j * hidden + off;
                for (int d = 0; d < headDim; d++) ctxSpan[d] += s * crossV[vBase + d];
            }
        });

        var output = new float[hidden];
        fixed (float* cp = context, op = output)
            lw.CrossAttnOutProjWeight.MatMul(cp, 1, op);
        return output;
    }

    /// <summary>Real `fc1 -> GELU -> fc2` (`linear1`/`linear2`), no bias (`bias_ff: false`).</summary>
    private static unsafe float[] Ffn(float[] x, AudioGenDecoderLayerWeights lw)
    {
        int hidden = AudioGenConfig.HiddenSize;
        int ffn = AudioGenConfig.FfnDim;
        var mid = new float[ffn];
        fixed (float* xp = x, mp = mid)
            lw.Linear1Weight.MatMul(xp, 1, mp);

        for (int i = 0; i < mid.Length; i++) mid[i] = Gelu(mid[i]);

        var output = new float[hidden];
        fixed (float* mp = mid, op = output)
            lw.Linear2Weight.MatMul(mp, 1, op);
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

    private static void LayerNorm(ReadOnlySpan<float> x, float[] weight, float[] bias, Span<float> output, float eps = 1e-5f)
    {
        int n = x.Length;
        float mean = 0f;
        for (int i = 0; i < n; i++) mean += x[i];
        mean /= n;

        float variance = 0f;
        for (int i = 0; i < n; i++) { float d = x[i] - mean; variance += d * d; }
        variance /= n;

        float invStd = 1f / MathF.Sqrt(variance + eps);
        for (int i = 0; i < n; i++) output[i] = (x[i] - mean) * invStd * weight[i] + bias[i];
    }

    private static void SoftmaxInPlace(float[] scores)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < scores.Length; i++) if (scores[i] > max) max = scores[i];
        float sum = 0f;
        for (int i = 0; i < scores.Length; i++)
        {
            float e = MathF.Exp(scores[i] - max);
            scores[i] = e;
            sum += e;
        }
        float invSum = 1f / sum;
        for (int i = 0; i < scores.Length; i++) scores[i] *= invSum;
    }
}
