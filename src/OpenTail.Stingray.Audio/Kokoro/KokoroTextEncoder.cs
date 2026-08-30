
namespace OpenTail.Stingray.Audio.Kokoro;

/// <summary>
/// StyleTTS2 TextEncoder (examples/kokoro-py/modules.py TextEncoder): token embedding ->
/// 3x (Conv1d k=5,pad=2 + custom channels-LayerNorm + LeakyReLU(0.2)) -> BiLSTM. This is the
/// `t_en` fed into KModel.forward's `asr = t_en @ pred_aln_trg` -- a DIFFERENT module from
/// ProsodyPredictor's internal "text_encoder" (which is actually DurationEncoder, see
/// ProsodyPredictorWeights). Assumes a single utterance with no padding (text_mask all-false),
/// so all masked_fill_ calls in the reference are no-ops here.
/// </summary>
public static class KokoroTextEncoder
{
    private const float LayerNormEps = 1e-5f;
    private const int KernelSize = 5;
    private const int Padding = 2;
    private const float LeakyReluSlope = 0.2f;

    /// <summary>Returns t_en, channel-first [channels, T] (channels = KokoroWeights.HiddenDim, 512).</summary>
    public static float[] Forward(KokoroWeights weights, ReadOnlySpan<int> inputIds)
    {
        var w = weights.TextEncoder;
        int t = inputIds.Length;
        int channels = weights.HiddenDim; // 512

        // 1. Embedding lookup -> [T, channels] row-major.
        var x = new float[t * channels];
        for (int i = 0; i < t; i++)
        {
            int tok = inputIds[i];
            Array.Copy(w.EmbeddingWeight, tok * channels, x, i * channels, channels);
        }

        // Work channel-first [channels, T] for the conv stack (matches x.transpose(1,2) in torch).
        var cf = Transpose(x, t, channels);

        for (int layer = 0; layer < 3; layer++)
        {
            var conv = Conv1d(cf, w.ConvWeight[layer], w.ConvBias[layer], channels, channels, t, KernelSize, Padding);
            var normed = ChannelLayerNormPerStep(conv, w.ConvLnGamma[layer], w.ConvLnBeta[layer], channels, t);
            for (int i = 0; i < normed.Length; i++)
                normed[i] = normed[i] >= 0f ? normed[i] : LeakyReluSlope * normed[i];
            cf = normed;
        }

        // 2. Transpose back to [T, channels] for the BiLSTM, then run it (hidden = channels/2 per direction).
        var tf = Transpose(cf, channels, t);
        int hiddenPerDir = channels / 2;
        var lstmWeights = new BiLstmWeights
        {
            WeightIhL0 = w.LstmWeightIhL0,
            WeightHhL0 = w.LstmWeightHhL0,
            BiasIhL0 = w.LstmBiasIhL0,
            BiasHhL0 = w.LstmBiasHhL0,
            WeightIhL0Rev = w.LstmWeightIhL0Rev,
            WeightHhL0Rev = w.LstmWeightHhL0Rev,
            BiasIhL0Rev = w.LstmBiasIhL0Rev,
            BiasHhL0Rev = w.LstmBiasHhL0Rev,
        };
        var lstmOut = KokoroLstm.Bidirectional(lstmWeights, tf, t, channels, hiddenPerDir);

        // 3. Transpose LSTM output [T, channels] back to channel-first [channels, T] (matches x.transpose(-1,-2)).
        return Transpose(lstmOut, t, channels);
    }

    /// <summary>[rows, cols] row-major -> [cols, rows] row-major.</summary>
    private static float[] Transpose(float[] src, int rows, int cols)
    {
        var dst = new float[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                dst[c * rows + r] = src[r * cols + c];
        return dst;
    }

    /// <summary>Conv1d over channel-first [inCh, T] input with zero padding, weight [outCh, inCh, kernel] row-major.</summary>
    private static float[] Conv1d(float[] input, float[] weight, float[] bias, int inCh, int outCh, int t, int kernel, int padding)
    {
        var output = new float[outCh * t];
        System.Threading.Tasks.Parallel.For(0, outCh, oc =>
        {
            var outRow = new float[t];
            Array.Fill(outRow, bias[oc]);
            int wOcBase = oc * inCh * kernel;
            for (int ic = 0; ic < inCh; ic++)
            {
                var inRow = input.AsSpan(ic * t, t);
                int wBase = wOcBase + ic * kernel;
                for (int k = 0; k < kernel; k++)
                {
                    int shift = k - padding;
                    int start = Math.Max(0, -shift);
                    int end = Math.Min(t, t - shift);
                    int len = end - start;
                    if (len <= 0) continue;
                    var inSlice = inRow.Slice(start + shift, len);
                    var outSlice = outRow.AsSpan(start, len);
                    System.Numerics.Tensors.TensorPrimitives.MultiplyAdd(inSlice, weight[wBase + k], outSlice, outSlice);
                }
            }
            Array.Copy(outRow, 0, output, oc * t, t);
        });
        return output;
    }

    /// <summary>
    /// modules.py's custom LayerNorm: normalizes over the channel dim independently at each
    /// time step (x.transpose(1,-1) -> layer_norm over channels -> transpose back), gamma/beta
    /// naming (not weight/bias), eps=1e-5 (LayerNorm class default, distinct from ALBERT's 1e-12).
    /// Input/output are channel-first [channels, T].
    /// </summary>
    private static float[] ChannelLayerNormPerStep(float[] input, float[] gamma, float[] beta, int channels, int t)
    {
        var output = new float[channels * t];
        for (int ti = 0; ti < t; ti++)
        {
            double mean = 0;
            for (int c = 0; c < channels; c++) mean += input[c * t + ti];
            mean /= channels;

            double variance = 0;
            for (int c = 0; c < channels; c++)
            {
                double d = input[c * t + ti] - mean;
                variance += d * d;
            }
            variance /= channels;
            float invStd = (float)(1.0 / Math.Sqrt(variance + LayerNormEps));

            for (int c = 0; c < channels; c++)
            {
                float normed = (float)((input[c * t + ti] - mean) * invStd);
                output[c * t + ti] = normed * gamma[c] + beta[c];
            }
        }
        return output;
    }
}
