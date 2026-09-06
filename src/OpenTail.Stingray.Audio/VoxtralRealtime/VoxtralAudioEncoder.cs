
namespace OpenTail.Stingray.Audio.VoxtralRealtime;

/// <summary>
/// Real Voxtral Realtime audio-tower forward pass, transcribed directly from
/// `examples/audio.cpp/src/models/voxtral_realtime/audio_encoder.cpp`'s
/// `build_audio_attention`/`build_audio_mlp`/`AudioGraph` (non-streaming/offline path, not the
/// KV-cache streaming path -- this port processes a full clip in one shot) -- see
/// `docs/audio-review-progress.md`'s Voxtral section for the architecture derivation. NOT a
/// Whisper-shaped encoder: a real Llama/Mistral-style transformer (RoPE-NEOX + full MHA + SwiGLU +
/// RMSNorm) applied to conv-downsampled mel features, with a local causal sliding-window mask.
/// </summary>
public static class VoxtralAudioEncoder
{
    /// <summary>Returns [tokens][TextHiddenSize] real audio embeddings ready to splice into the
    /// text decoder's token stream, matching the reference's
    /// `VoxtralRealtimeAudioEmbeddings`.</summary>
    public static float[][] Forward(VoxtralAudioEncoderWeights w, float[] melChannelMajor, int melFrames)
    {
        int melBins = VoxtralAudioEncoderWeights.NumMelBins;
        int hidden = VoxtralAudioEncoderWeights.HiddenSize;

        // Conv1d(melBins->hidden, k=3, s=1, pad=1) + GELU(erf).
        var stage1 = Conv1dSamePad(melChannelMajor, melBins, melFrames, w.Conv1Weight, w.Conv1Bias, hidden, kernel: 3, stride: 1);
        GeluErfInPlace(stage1);
        // Conv1d(hidden->hidden, k=3, s=2, pad=1) + GELU(erf) -- 2x downsample.
        int steps = (melFrames + 2 * 1 - 3) / 2 + 1;
        var stage2 = Conv1dSamePad(stage1, hidden, melFrames, w.Conv2Weight, w.Conv2Bias, hidden, kernel: 3, stride: 2);
        GeluErfInPlace(stage2);

        // Transpose channel-major [hidden, steps] -> frame-major [steps][hidden].
        var x = new float[steps][];
        for (int t = 0; t < steps; t++)
        {
            var row = new float[hidden];
            for (int c = 0; c < hidden; c++) row[c] = stage2[c * steps + t];
            x[t] = row;
        }

        for (int layer = 0; layer < VoxtralAudioEncoderWeights.NumLayers; layer++)
        {
            var lw = w.Layers[layer];
            var attnIn = RmsNormRows(x, hidden, lw.AttnNorm);
            var attn = SelfAttention(attnIn, hidden, lw);
            AddRows(x, attn);
            var mlpIn = RmsNormRows(x, hidden, lw.FinalNorm);
            var mlp = Mlp(mlpIn, hidden, lw);
            AddRows(x, mlp);
        }

        x = RmsNormRows(x, hidden, w.NormWeight);

        // Downsample by DownsampleFactor: reshape 4 adjacent hidden-dim steps into one vector.
        int factor = VoxtralAudioEncoderWeights.DownsampleFactor;
        int tokens = steps / factor;
        int textHidden = VoxtralAudioEncoderWeights.TextHiddenSize;
        var downsampled = new float[tokens][];
        for (int t = 0; t < tokens; t++)
        {
            var row = new float[hidden * factor];
            for (int f = 0; f < factor; f++)
                Array.Copy(x[t * factor + f], 0, row, f * hidden, hidden);
            downsampled[t] = row;
        }

        var output = new float[tokens][];
        for (int t = 0; t < tokens; t++)
        {
            var proj1 = LinearNoBias(downsampled[t], w.Projector1Weight, hidden * factor, textHidden);
            GeluErfInPlace(proj1);
            output[t] = LinearNoBias(proj1, w.Projector2Weight, textHidden, textHidden);
        }
        return output;
    }

    private static float[] SelfAttention(float[][] xRows, int hidden, VoxtralAudioLayerWeights w)
    {
        int frames = xRows.Length;
        int heads = VoxtralAudioEncoderWeights.NumHeads;
        int headDim = VoxtralAudioEncoderWeights.HeadDim;
        int window = VoxtralAudioEncoderWeights.SlidingWindow;
        float invSqrtD = 1f / MathF.Sqrt(headDim);

        var q = new float[frames][];
        var k = new float[frames][];
        var v = new float[frames][];
        for (int t = 0; t < frames; t++)
        {
            q[t] = LinearBias(xRows[t], w.QWeight, w.QBias, hidden, heads * headDim);
            k[t] = LinearNoBias(xRows[t], w.KWeight, hidden, heads * headDim);
            v[t] = LinearBias(xRows[t], w.VWeight, w.VBias, hidden, heads * headDim);
            RopeNeoxInPlace(q[t], heads, headDim, t);
            RopeNeoxInPlace(k[t], heads, headDim, t);
        }

        var contextFlat = new float[frames][];
        for (int t = 0; t < frames; t++) contextFlat[t] = new float[heads * headDim];

        var scores = new float[frames];
        for (int h = 0; h < heads; h++)
        {
            int chBase = h * headDim;
            for (int i = 0; i < frames; i++)
            {
                int begin = Math.Max(0, i - window + 1);
                float maxScore = float.NegativeInfinity;
                for (int j = begin; j <= i; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += q[i][chBase + d] * k[j][chBase + d];
                    dot *= invSqrtD;
                    scores[j] = dot;
                    if (dot > maxScore) maxScore = dot;
                }
                float sum = 0f;
                for (int j = begin; j <= i; j++) { scores[j] = MathF.Exp(scores[j] - maxScore); sum += scores[j]; }
                float invSum = 1f / sum;
                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0f;
                    for (int j = begin; j <= i; j++) acc += scores[j] * invSum * v[j][chBase + d];
                    contextFlat[i][chBase + d] = acc;
                }
            }
        }

        var output = new float[frames][];
        for (int t = 0; t < frames; t++) output[t] = LinearBias(contextFlat[t], w.OWeight, w.OBias, heads * headDim, hidden);

        var flatOut = new float[frames * hidden];
        for (int t = 0; t < frames; t++) Array.Copy(output[t], 0, flatOut, t * hidden, hidden);
        return flatOut;
    }

    /// <summary>Real GGML `GGML_ROPE_TYPE_NEOX` convention: rotates pairs `(i, i+headDim/2)`
    /// within each head (half-split, NOT the interleaved GPT-J `(2i,2i+1)` pairing) -- applied
    /// across the full head_dim (no partial-rotation `n_dims` reduction here, unlike CosyVoice3's
    /// RoPE usage elsewhere in this codebase).</summary>
    private static void RopeNeoxInPlace(float[] qkv, int heads, int headDim, int position)
    {
        int half = headDim / 2;
        for (int h = 0; h < heads; h++)
        {
            int baseIdx = h * headDim;
            for (int i = 0; i < half; i++)
            {
                float freq = MathF.Pow(VoxtralAudioEncoderWeights.RopeTheta, -2f * i / headDim);
                float angle = position * freq;
                float cos = MathF.Cos(angle), sin = MathF.Sin(angle);
                float a = qkv[baseIdx + i];
                float b = qkv[baseIdx + half + i];
                qkv[baseIdx + i] = a * cos - b * sin;
                qkv[baseIdx + half + i] = a * sin + b * cos;
            }
        }
    }

    private static float[] Mlp(float[][] xRows, int hidden, VoxtralAudioLayerWeights w)
    {
        int frames = xRows.Length;
        int inter = VoxtralAudioEncoderWeights.IntermediateSize;
        var flatOut = new float[frames * hidden];
        for (int t = 0; t < frames; t++)
        {
            var gate = LinearNoBias(xRows[t], w.GateWeight, hidden, inter);
            var up = LinearNoBias(xRows[t], w.UpWeight, hidden, inter);
            for (int i = 0; i < inter; i++) gate[i] = Silu(gate[i]) * up[i];
            var down = LinearBias(gate, w.DownWeight, w.DownBias, inter, hidden);
            Array.Copy(down, 0, flatOut, t * hidden, hidden);
        }
        return flatOut;
    }

    private static void AddRows(float[][] x, float[] flatDelta)
    {
        int hidden = x[0].Length;
        for (int t = 0; t < x.Length; t++)
            for (int c = 0; c < hidden; c++)
                x[t][c] += flatDelta[t * hidden + c];
    }

    private static float[][] RmsNormRows(float[][] xRows, int hidden, float[] weight)
    {
        var output = new float[xRows.Length][];
        for (int t = 0; t < xRows.Length; t++)
        {
            var row = xRows[t];
            double sumSq = 0;
            for (int c = 0; c < hidden; c++) sumSq += (double)row[c] * row[c];
            float invRms = (float)(1.0 / Math.Sqrt(sumSq / hidden + VoxtralAudioEncoderWeights.RmsNormEps));
            var outRow = new float[hidden];
            for (int c = 0; c < hidden; c++) outRow[c] = row[c] * invRms * weight[c];
            output[t] = outRow;
        }
        return output;
    }

    private static float Silu(float x) => x / (1f + MathF.Exp(-x));

    private static void GeluErfInPlace(float[] x)
    {
        for (int i = 0; i < x.Length; i++)
        {
            float v = x[i];
            x[i] = 0.5f * v * (1f + Erf(v * 0.70710678f));
        }
    }

    /// <summary>Abramowitz-Stegun erf approximation (same one used elsewhere in this codebase for
    /// exact-erf GELU, e.g. RvcHubertEncoder).</summary>
    private static float Erf(float x)
    {
        float sign = x < 0 ? -1f : 1f;
        x = MathF.Abs(x);
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float t = 1f / (1f + p * x);
        float y = 1f - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * MathF.Exp(-x * x);
        return sign * y;
    }

    private static float[] LinearNoBias(float[] input, float[] weight, int inDim, int outDim)
    {
        var output = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float sum = 0f;
            int wBase = o * inDim;
            for (int i = 0; i < inDim; i++) sum += input[i] * weight[wBase + i];
            output[o] = sum;
        }
        return output;
    }

    private static float[] LinearBias(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float sum = bias[o];
            int wBase = o * inDim;
            for (int i = 0; i < inDim; i++) sum += input[i] * weight[wBase + i];
            output[o] = sum;
        }
        return output;
    }

    /// <summary>Real "same"-padding Conv1d over a [C][T] channel-major tensor with support for
    /// stride (used for the stem's stride-2 second conv). Weight real PyTorch layout
    /// [outCh, inCh, kernel].</summary>
    private static float[] Conv1dSamePad(float[] x, int inCh, int frames, float[] weight, float[] bias, int outCh, int kernel, int stride)
    {
        int pad = (kernel - 1) / 2;
        int outFrames = (frames + 2 * pad - kernel) / stride + 1;
        var output = new float[outCh * outFrames];
        for (int oc = 0; oc < outCh; oc++)
        {
            float b = bias[oc];
            int wBase = oc * inCh * kernel;
            for (int ot = 0; ot < outFrames; ot++)
            {
                float sum = b;
                int start = ot * stride - pad;
                for (int ic = 0; ic < inCh; ic++)
                {
                    int wIcBase = wBase + ic * kernel;
                    for (int kk = 0; kk < kernel; kk++)
                    {
                        int it = start + kk;
                        if ((uint)it >= (uint)frames) continue;
                        sum += weight[wIcBase + kk] * x[ic * frames + it];
                    }
                }
                output[oc * outFrames + ot] = sum;
            }
        }
        return output;
    }
}
