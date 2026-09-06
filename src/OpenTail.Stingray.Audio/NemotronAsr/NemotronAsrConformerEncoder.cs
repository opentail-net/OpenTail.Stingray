
namespace OpenTail.Stingray.Audio.NemotronAsr;

/// <summary>
/// Nemotron 3.5 ASR's 24-layer FastConformer encoder stack (offline/full-utterance mode),
/// transcribed directly from `examples/audio.cpp/src/models/nemotron_asr/encoder.cpp`'s
/// `build_encoder_layer`/`NemotronEncoderRuntime::encode` (the real offline entry point, NOT the
/// streaming-chunk path) and `framework/modules/asr_helpers.cpp`'s
/// `fill_asr_chunked_attention_bias` (not guessed). Each block: macaron FFN1 (half-step residual,
/// SiLU) -&gt; Transformer-XL relative-position self-attention with untied u/v biases, under a
/// real CHUNKED attention mask (NOT plain causal): frames are grouped into chunks of size
/// `lookahead_tokens+1` (default lookahead=3 -&gt; chunk=4), and query chunk `qc` may attend to key
/// chunk `kc` iff `0 &lt;= qc-kc &lt;= floor((sliding_window-1)/chunk)` (`sliding_window=57` -&gt;
/// `left_chunks=14`) -&gt; depthwise-conv+GLU conv module (real LayerNorm, kernel=9, causal-padded)
/// -&gt; macaron FFN2 (half-step residual) -&gt; final LayerNorm. A per-frame one-hot `prompt` vector
/// (`num_prompts=128`, default `prompt_id=101` = the en-US language/task slot) is concatenated
/// after the Conformer stack and run through a 2-layer ReLU MLP (`prompt_kernel.0/2` in the real
/// checkpoint) before a final projection to `decoder_hidden_size` -- this last projection is
/// assumed (not yet independently confirmed) to be the SAME real tensor as `joint.enc` (the RNNT
/// joint network's own encoder-side projection, `Linear(1024-&gt;640)`), since no separate
/// `encoder_projector` tensor exists in this checkpoint and the dimensions match exactly; flag
/// this assumption if RNNT decode later doesn't golden-verify.
/// </summary>
public static class NemotronAsrConformerEncoder
{
    /// <summary>
    /// hiddenIn is per-frame [T][HiddenDim] from <see cref="NemotronAsrSubsampling"/>. Returns the
    /// final `decoder_hidden_size`-wide per-frame projection (matching `joint.enc`'s output dim)
    /// after the full Conformer stack + prompt conditioning.
    /// </summary>
    public static float[][] Forward(NemotronAsrWeights w, float[][] hiddenIn, int promptId = 101, int lookaheadTokens = 3)
    {
        int t = hiddenIn.Length;
        int hidden = w.HiddenDim;

        var posEmb = RelativePositionalEncoding(t, hidden);
        var mask = ChunkedAttentionMask(t, w.SubsampleFactor > 0 ? t : t, leftContext: ComputeSlidingWindow(w) - 1, rightContext: lookaheadTokens);

        var x = hiddenIn;
        foreach (var layer in w.Layers)
            x = ConformerLayer(w, layer, x, posEmb, mask, t);

        // Prompt conditioning: concat one-hot(num_prompts) -> Linear(hidden+128->2048) -> ReLU -> Linear(2048->hidden)
        var promptOneHot = new float[128];
        promptOneHot[promptId] = 1f;
        var afterPrompt = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var concat = new float[hidden + 128];
            Array.Copy(x[i], concat, hidden);
            Array.Copy(promptOneHot, 0, concat, hidden, 128);
            var h1 = Linear(concat, w.PromptKernel0Weight, w.PromptKernel0Bias, hidden + 128, w.PromptIntermediateSize);
            ReluInPlace(h1);
            afterPrompt[i] = Linear(h1, w.PromptKernel2Weight, w.PromptKernel2Bias, w.PromptIntermediateSize, hidden);
        }

        // Final projection to decoder_hidden_size -- see class doc comment re: joint.enc assumption.
        var output = new float[t][];
        for (int i = 0; i < t; i++)
            output[i] = Linear(afterPrompt[i], w.JointEncWeight, w.JointEncBias, hidden, w.JointDim);
        return output;
    }

    private static int ComputeSlidingWindow(NemotronAsrWeights w) => 57; // asr.encoder.sliding_window, real checkpoint metadata constant

    private static float[][] ConformerLayer(NemotronAsrWeights w, NemotronAsrConformerLayer l, float[][] x, float[][] posEmbProjectedInput, float[,] mask, int t)
    {
        int hidden = w.HiddenDim;

        // FF1 (macaron, half-step)
        var ff1Norm = LayerNormRows(x, hidden, l.NormFf1Weight, l.NormFf1Bias);
        var ff1 = FeedForward(ff1Norm, l.Ff1Linear1Weight, l.Ff1Linear1Bias, l.Ff1Linear2Weight, l.Ff1Linear2Bias, hidden, w.FfDim);
        var afterFf1 = AddScaled(x, ff1, 0.5f);

        // Self-attention (Transformer-XL rel-pos, chunked mask)
        var attnNorm = LayerNormRows(afterFf1, hidden, l.NormAttnWeight, l.NormAttnBias);
        var attn = RelPosSelfAttention(w, l, attnNorm, posEmbProjectedInput, mask, t);
        var afterAttn = Add(afterFf1, attn);

        // Conv module
        var convNorm = LayerNormRows(afterAttn, hidden, l.NormConvWeight, l.NormConvBias);
        var conv = ConvModule(w, l, convNorm, t);
        var afterConv = Add(afterAttn, conv);

        // FF2 (macaron, half-step)
        var ff2Norm = LayerNormRows(afterConv, hidden, l.NormFf2Weight, l.NormFf2Bias);
        var ff2 = FeedForward(ff2Norm, l.Ff2Linear1Weight, l.Ff2Linear1Bias, l.Ff2Linear2Weight, l.Ff2Linear2Bias, hidden, w.FfDim);
        var afterFf2 = AddScaled(afterConv, ff2, 0.5f);

        return LayerNormRows(afterFf2, hidden, l.NormOutWeight, l.NormOutBias);
    }

    private static float[][] FeedForward(float[][] xRows, float[] w1, float[]? b1, float[] w2, float[]? b2, int hidden, int ffDim)
    {
        int t = xRows.Length;
        var output = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var h = Linear(xRows[i], w1, b1, hidden, ffDim);
            SiluInPlace(h);
            output[i] = Linear(h, w2, b2, ffDim, hidden);
        }
        return output;
    }

    private static float[][] RelPosSelfAttention(NemotronAsrWeights w, NemotronAsrConformerLayer l, float[][] xRows, float[][] posEmbRaw, float[,] mask, int t)
    {
        int hidden = w.HiddenDim;
        int heads = w.NumHeads;
        int headDim = w.HeadDim;
        float scale = 1f / MathF.Sqrt(headDim);

        var q = new float[t][];
        var k = new float[t][];
        var v = new float[t][];
        for (int i = 0; i < t; i++)
        {
            q[i] = Linear(xRows[i], l.AttnQWeight, null, hidden, hidden);
            k[i] = Linear(xRows[i], l.AttnKWeight, null, hidden, hidden);
            v[i] = Linear(xRows[i], l.AttnVWeight, null, hidden, hidden);
        }

        int posLen = posEmbRaw.Length; // 2t-1
        var pProj = new float[posLen][];
        for (int p = 0; p < posLen; p++)
            pProj[p] = Linear(posEmbRaw[p], l.AttnPosWeight, null, hidden, hidden);

        var context = new float[t][];
        for (int i = 0; i < t; i++) context[i] = new float[hidden];

        var scoresRow = new float[t];
        for (int h = 0; h < heads; h++)
        {
            int hBase = h * headDim;
            for (int i = 0; i < t; i++)
            {
                float maxScore = float.NegativeInfinity;
                for (int j = 0; j < t; j++)
                {
                    if (mask[i, j] < 0f) { scoresRow[j] = float.NegativeInfinity; continue; }
                    float ac = 0f;
                    for (int d = 0; d < headDim; d++)
                        ac += (q[i][hBase + d] + l.AttnPosBiasU[d + h * headDim]) * k[j][hBase + d];

                    int pIdx = (t - 1) - i + j; // relative_shift, direct index form
                    float bd = 0f;
                    for (int d = 0; d < headDim; d++)
                        bd += (q[i][hBase + d] + l.AttnPosBiasV[d + h * headDim]) * pProj[pIdx][hBase + d];

                    float s = (ac + bd) * scale;
                    scoresRow[j] = s;
                    if (s > maxScore) maxScore = s;
                }
                float sum = 0f;
                for (int j = 0; j < t; j++)
                {
                    if (float.IsNegativeInfinity(scoresRow[j])) { scoresRow[j] = 0f; continue; }
                    scoresRow[j] = MathF.Exp(scoresRow[j] - maxScore);
                    sum += scoresRow[j];
                }
                float invSum = sum > 0f ? 1f / sum : 0f;
                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0f;
                    for (int j = 0; j < t; j++) acc += scoresRow[j] * invSum * v[j][hBase + d];
                    context[i][hBase + d] = acc;
                }
            }
        }

        var output = new float[t][];
        for (int i = 0; i < t; i++)
            output[i] = Linear(context[i], l.AttnOutWeight, null, hidden, hidden);
        return output;
    }

    private static float[][] ConvModule(NemotronAsrWeights w, NemotronAsrConformerLayer l, float[][] xRows, int t)
    {
        int hidden = w.HiddenDim;
        var glu = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var pw1 = Linear(xRows[i], l.ConvPw1Weight, null, hidden, 2 * hidden);
            var g = new float[hidden];
            for (int c = 0; c < hidden; c++)
                g[c] = pw1[c] * Sigmoid(pw1[hidden + c]);
            glu[i] = g;
        }

        // Causal depthwise conv1d over time, kernel=w.ConvKernel, left-pad only (kernel-1).
        int k = w.ConvKernel;
        var conv = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[hidden];
            for (int c = 0; c < hidden; c++)
            {
                float sum = l.ConvDwBias?[c] ?? 0f;
                for (int kk = 0; kk < k; kk++)
                {
                    int srcT = i - (k - 1) + kk;
                    if (srcT < 0) continue;
                    sum += l.ConvDwWeight[c * k + kk] * glu[srcT][c];
                }
                row[c] = sum;
            }
            conv[i] = row;
        }

        var normed = LayerNormRows(conv, hidden, l.ConvNormWeight, l.ConvNormBias);
        for (int i = 0; i < t; i++) SiluInPlace(normed[i]);

        var output = new float[t][];
        for (int i = 0; i < t; i++)
            output[i] = Linear(normed[i], l.ConvPw2Weight, null, hidden, hidden);
        return output;
    }

    /// <summary>Real chunked attention mask, transcribed directly from
    /// `asr_helpers.cpp`'s `fill_asr_chunked_attention_bias`: chunks of size
    /// `rightContext+1`; query chunk may attend to key chunk iff
    /// `0 &lt;= qChunk-kChunk &lt;= leftContext/chunk`. Returns 0f (allowed) or -1 (masked) per
    /// [query,key] pair (sentinel, not the reference's literal -1e9 -- consumed as a hard mask
    /// here rather than an additive softmax bias).</summary>
    private static float[,] ChunkedAttentionMask(int frames, int validFrames, int leftContext, int rightContext)
    {
        var mask = new float[frames, frames];
        int chunk = rightContext + 1;
        int leftChunks = leftContext >= 0 ? leftContext / chunk : int.MaxValue;
        for (int q = 0; q < frames; q++)
        {
            for (int kk = 0; kk < frames; kk++) mask[q, kk] = -1f;
            int qChunk = q / chunk;
            for (int kk = 0; kk < validFrames; kk++)
            {
                int kChunk = kk / chunk;
                int diff = qChunk - kChunk;
                if (diff >= 0 && diff <= leftChunks) mask[q, kk] = 0f;
            }
        }
        return mask;
    }

    /// <summary>Real relative positional encoding table, matching `make_relative_positional_encoding`
    /// (direct sin/cos form, mathematically equivalent to the reference's angle-recurrence form):
    /// row p (p=0..2T-2) holds `sin/cos(offset * inv_freq[i])` for `offset = (T-1)-p`.</summary>
    private static float[][] RelativePositionalEncoding(int frames, int hidden)
    {
        int posLen = 2 * frames - 1;
        int half = hidden / 2;
        var table = new float[posLen][];
        for (int p = 0; p < posLen; p++)
        {
            var row = new float[hidden];
            double offset = (frames - 1) - p;
            for (int i = 0; i < half; i++)
            {
                double invFreq = 1.0 / Math.Pow(10000.0, 2.0 * i / hidden);
                row[2 * i] = (float)Math.Sin(offset * invFreq);
                row[2 * i + 1] = (float)Math.Cos(offset * invFreq);
            }
            table[p] = row;
        }
        return table;
    }

    private static float[][] LayerNormRows(float[][] xRows, int dim, float[] weight, float[] bias)
    {
        var output = new float[xRows.Length][];
        for (int i = 0; i < xRows.Length; i++)
        {
            var row = xRows[i];
            double mean = 0;
            for (int c = 0; c < dim; c++) mean += row[c];
            mean /= dim;
            double variance = 0;
            for (int c = 0; c < dim; c++) { double d = row[c] - mean; variance += d * d; }
            variance /= dim;
            float invStd = (float)(1.0 / Math.Sqrt(variance + NemotronAsrWeights.LayerNormEps));
            var outRow = new float[dim];
            for (int c = 0; c < dim; c++) outRow[c] = (float)((row[c] - mean) * invStd) * weight[c] + bias[c];
            output[i] = outRow;
        }
        return output;
    }

    private static float[][] Add(float[][] a, float[][] b)
    {
        var output = new float[a.Length][];
        for (int i = 0; i < a.Length; i++)
        {
            var row = new float[a[i].Length];
            for (int c = 0; c < row.Length; c++) row[c] = a[i][c] + b[i][c];
            output[i] = row;
        }
        return output;
    }

    private static float[][] AddScaled(float[][] a, float[][] b, float scale)
    {
        var output = new float[a.Length][];
        for (int i = 0; i < a.Length; i++)
        {
            var row = new float[a[i].Length];
            for (int c = 0; c < row.Length; c++) row[c] = a[i][c] + b[i][c] * scale;
            output[i] = row;
        }
        return output;
    }

    private static void SiluInPlace(float[] x)
    {
        for (int i = 0; i < x.Length; i++) x[i] = x[i] / (1f + MathF.Exp(-x[i]));
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private static void ReluInPlace(float[] x)
    {
        for (int i = 0; i < x.Length; i++) if (x[i] < 0f) x[i] = 0f;
    }

    private static float[] Linear(float[] input, float[] weight, float[]? bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float sum = bias?[o] ?? 0f;
            int wBase = o * inDim;
            for (int i = 0; i < inDim; i++) sum += input[i] * weight[wBase + i];
            output[o] = sum;
        }
        return output;
    }
}
