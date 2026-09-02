
namespace OpenTail.Stingray.Audio.MusicGen;

/// <summary>
/// Real MusicGen decoder-only LM forward pass, transcribed from the real `transformers`
/// `modeling_musicgen.py` (`MusicgenDecoder`/`MusicgenDecoderLayer`/`MusicgenAttention`) --
/// standard OPT/Bart-style PRE-norm decoder: `LayerNorm -> SelfAttn -> +residual`, `LayerNorm ->
/// CrossAttn(encoder_hidden_states) -> +residual`, `LayerNorm -> fc1 -> GELU -> fc2 ->
/// +residual`, then one final `LayerNorm` after the last layer. Self-attention IS scaled by
/// `1/sqrt(headDim)` (unlike T5) -- a real, easy-to-get-backwards difference between the two
/// attention flavors this single checkpoint contains.
///
/// <para>Embedding: <see cref="MusicGenConfig.NumCodebooks"/> separate embedding tables, SUMMED
/// (not concatenated) per step, plus the checkpoint's own precomputed sinusoidal position
/// embedding buffer added on top (`embed_scale` is 1.0 here -- real config `scale_embedding:
/// false`). `KvCache` carries self-attention K/V incrementally across autoregressive steps;
/// cross-attention K/V is computed ONCE from the text encoder's output and reused every step
/// (the real encoder_hidden_states never change during decoding).</para>
/// </summary>
public static class MusicGenTransformer
{
    /// <summary>Per-layer self-attention KV cache plus the once-computed cross-attention K/V, both grown/set as decoding proceeds.</summary>
    public sealed class KvCache
    {
        public List<float[]>[] SelfK { get; }
        public List<float[]>[] SelfV { get; }
        public float[][]? CrossK { get; set; } // [layer][crossLen * hidden], set once from EncodeCrossAttention
        public float[][]? CrossV { get; set; }
        public int CrossLen { get; set; }
        public int Position { get; set; } // next absolute position to embed (== number of self-attn steps so far)

        public KvCache()
        {
            SelfK = new List<float[]>[MusicGenConfig.DecoderNumLayers];
            SelfV = new List<float[]>[MusicGenConfig.DecoderNumLayers];
            for (int i = 0; i < MusicGenConfig.DecoderNumLayers; i++)
            {
                SelfK[i] = [];
                SelfV[i] = [];
            }
        }
    }

    /// <summary>Precomputes cross-attention K/V projections from the text encoder's output hidden states -- done once per generation, reused every decode step. Real MusicGen projects the T5 encoder's 768-dim output up to the decoder's 1024-dim hidden size via `enc_to_dec_proj` FIRST (see <see cref="MusicGenTransformerWeights.EncToDecProjWeight"/>'s doc comment) -- do not feed raw T5 output straight into the cross-attention K/V weights.</summary>
    public static unsafe void PrepareCrossAttention(MusicGenTransformerWeights w, float[][] encoderHiddenStates, KvCache cache)
    {
        int crossLen = encoderHiddenStates.Length;
        int hidden = MusicGenConfig.DecoderHiddenSize;
        int textDim = MusicGenConfig.TextDModel;

        var rawFlat = new float[crossLen * textDim];
        for (int i = 0; i < crossLen; i++) Array.Copy(encoderHiddenStates[i], 0, rawFlat, i * textDim, textDim);

        var flat = new float[crossLen * hidden];
        fixed (float* rp = rawFlat, fp2 = flat, bp = w.EncToDecProjBias)
            w.EncToDecProjWeight.MatMul(rp, crossLen, fp2, bp);

        cache.CrossK = new float[MusicGenConfig.DecoderNumLayers][];
        cache.CrossV = new float[MusicGenConfig.DecoderNumLayers][];
        cache.CrossLen = crossLen;

        fixed (float* fp = flat)
        {
            for (int l = 0; l < MusicGenConfig.DecoderNumLayers; l++)
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

    /// <summary>
    /// Runs one decode step: `tokenColumn[codebook]` (one input token per codebook, from
    /// <see cref="DelayPattern.NextInputColumn"/>) -&gt; summed embedding -&gt; N decoder layers
    /// (growing <paramref name="cache"/>) -&gt; final LayerNorm -&gt; per-codebook logits.
    /// Returns `[codebook][CodebookSize]`.
    /// </summary>
    public static unsafe float[][] Step(MusicGenTransformerWeights w, int[] tokenColumn, KvCache cache)
    {
        int hidden = MusicGenConfig.DecoderHiddenSize;
        var x = new float[hidden];
        for (int q = 0; q < MusicGenConfig.NumCodebooks; q++)
        {
            int tok = tokenColumn[q];
            var table = w.EmbedTokens[q];
            for (int d = 0; d < hidden; d++) x[d] += table[tok * hidden + d];
        }

        int pos = cache.Position;
        for (int d = 0; d < hidden; d++) x[d] += w.EmbedPositions[pos * hidden + d];

        foreach (var (layer, li) in w.Layers.Select((l, i) => (l, i)))
            x = DecoderLayer(x, layer, li, cache);

        var normed = new float[hidden];
        LayerNorm(x, w.FinalLayerNormWeight, w.FinalLayerNormBias, normed);

        var logits = new float[MusicGenConfig.NumCodebooks][];
        fixed (float* np = normed)
        {
            for (int q = 0; q < MusicGenConfig.NumCodebooks; q++)
            {
                var l = new float[MusicGenConfig.CodebookSize];
                fixed (float* lp = l)
                    w.LmHeads[q].MatMul(np, 1, lp);
                logits[q] = l;
            }
        }

        cache.Position++;
        return logits;
    }

    private static unsafe float[] DecoderLayer(float[] x, MusicGenDecoderLayerWeights lw, int layerIndex, KvCache cache)
    {
        int hidden = MusicGenConfig.DecoderHiddenSize;

        var normed1 = new float[hidden];
        LayerNorm(x, lw.SelfAttnLayerNormWeight, lw.SelfAttnLayerNormBias, normed1);
        var selfOut = SelfAttention(normed1, lw, layerIndex, cache);
        var afterSelf = new float[hidden];
        TensorPrimitives.Add(x, selfOut, afterSelf);

        var normed2 = new float[hidden];
        LayerNorm(afterSelf, lw.CrossAttnLayerNormWeight, lw.CrossAttnLayerNormBias, normed2);
        var crossOut = CrossAttention(normed2, lw, layerIndex, cache);
        var afterCross = new float[hidden];
        TensorPrimitives.Add(afterSelf, crossOut, afterCross);

        var normed3 = new float[hidden];
        LayerNorm(afterCross, lw.FinalLayerNormWeight, lw.FinalLayerNormBias, normed3);
        var ffnOut = Ffn(normed3, lw);
        var output = new float[hidden];
        TensorPrimitives.Add(afterCross, ffnOut, output);
        return output;
    }

    private static unsafe float[] SelfAttention(float[] x, MusicGenDecoderLayerWeights lw, int layerIndex, KvCache cache)
    {
        int hidden = MusicGenConfig.DecoderHiddenSize;
        int nHeads = MusicGenConfig.DecoderNumHeads;
        int headDim = MusicGenConfig.DecoderHeadDim;
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
            lw.SelfAttnOWeight.MatMul(cp, 1, op);
        return output;
    }

    private static unsafe float[] CrossAttention(float[] x, MusicGenDecoderLayerWeights lw, int layerIndex, KvCache cache)
    {
        int hidden = MusicGenConfig.DecoderHiddenSize;
        int nHeads = MusicGenConfig.DecoderNumHeads;
        int headDim = MusicGenConfig.DecoderHeadDim;
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
            lw.CrossAttnOWeight.MatMul(cp, 1, op);
        return output;
    }

    /// <summary>Real `fc1 -> GELU -> fc2`, no biases (confirmed: neither tensor has a `.bias` entry in the real checkpoint).</summary>
    private static unsafe float[] Ffn(float[] x, MusicGenDecoderLayerWeights lw)
    {
        int hidden = MusicGenConfig.DecoderHiddenSize;
        int ffn = MusicGenConfig.DecoderFfnDim;
        var mid = new float[ffn];
        fixed (float* xp = x, mp = mid)
            lw.Fc1Weight.MatMul(xp, 1, mp);

        for (int i = 0; i < mid.Length; i++) mid[i] = Gelu(mid[i]);

        var output = new float[hidden];
        fixed (float* mp = mid, op = output)
            lw.Fc2Weight.MatMul(mp, 1, op);
        return output;
    }

    /// <summary>Real (erf-based) GELU -- MusicGen's config declares `activation_function: "gelu"`, the exact erf form, not the tanh approximation ("gelu_new"/"gelu_pytorch_tanh") used elsewhere in this codebase.</summary>
    private static float Gelu(float x) => 0.5f * x * (1f + Erf(x / 1.4142135f));

    private static float Erf(float x)
    {
        // Abramowitz-Stegun 7.1.26 approximation, max error ~1.5e-7 -- matches System.Math precision needs here.
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
