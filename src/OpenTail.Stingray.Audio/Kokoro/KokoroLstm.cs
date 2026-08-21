using System;

namespace OpenTail.Stingray.Audio.Kokoro;

/// <summary>
/// Standard single-layer bidirectional LSTM (PyTorch gate order i,f,g,o), used by
/// TextEncoder, DurationEncoder (x<see cref="KokoroWeights.NStyleLayers"/>), pred.lstm and
/// pred.shared -- all four use the exact same cell math, just different weight sets and
/// input/hidden sizes, so this is written once as shared infrastructure.
///
/// Weight layout: weight_ih [4*hidden, input], weight_hh [4*hidden, hidden], both already
/// row-major "rows=4*hidden" per the GGUF reversed-dims convention (see docs/audio-review-progress.md),
/// directly indexable without transpose. No batching/masking: this codebase always drives
/// Kokoro with a single utterance and no padding.
/// </summary>
public static class KokoroLstm
{
    /// <summary>
    /// Runs both directions and concatenates per PyTorch's bidirectional convention
    /// (output[:, :hidden] = forward, output[:, hidden:] = backward). input is [T, inputSize]
    /// row-major; returns [T, 2*hiddenSize] row-major.
    /// </summary>
    public static float[] Bidirectional(BiLstmWeights w, float[] input, int t, int inputSize, int hiddenSize)
    {
        float[] fwd = RunDirection(w.WeightIhL0, w.WeightHhL0, w.BiasIhL0, w.BiasHhL0, input, t, inputSize, hiddenSize, reverse: false);
        float[] bwd = RunDirection(w.WeightIhL0Rev, w.WeightHhL0Rev, w.BiasIhL0Rev, w.BiasHhL0Rev, input, t, inputSize, hiddenSize, reverse: true);

        var output = new float[t * 2 * hiddenSize];
        for (int i = 0; i < t; i++)
        {
            Array.Copy(fwd, i * hiddenSize, output, i * 2 * hiddenSize, hiddenSize);
            Array.Copy(bwd, i * hiddenSize, output, i * 2 * hiddenSize + hiddenSize, hiddenSize);
        }
        return output;
    }

    private static float[] RunDirection(
        float[] weightIh, float[] weightHh, float[] biasIh, float[] biasHh,
        float[] input, int t, int inputSize, int hiddenSize, bool reverse)
    {
        var h = new float[hiddenSize];
        var c = new float[hiddenSize];
        var gates = new float[4 * hiddenSize];
        var output = new float[t * hiddenSize];

        for (int step = 0; step < t; step++)
        {
            int ti = reverse ? (t - 1 - step) : step;

            for (int g = 0; g < 4 * hiddenSize; g++)
            {
                float sum = biasIh[g] + biasHh[g];
                int ihRow = g * inputSize;
                for (int d = 0; d < inputSize; d++)
                    sum += weightIh[ihRow + d] * input[ti * inputSize + d];
                int hhRow = g * hiddenSize;
                for (int d = 0; d < hiddenSize; d++)
                    sum += weightHh[hhRow + d] * h[d];
                gates[g] = sum;
            }

            for (int j = 0; j < hiddenSize; j++)
            {
                float i_g = Sigmoid(gates[j]);
                float f_g = Sigmoid(gates[hiddenSize + j]);
                float g_g = MathF.Tanh(gates[2 * hiddenSize + j]);
                float o_g = Sigmoid(gates[3 * hiddenSize + j]);
                float cNew = f_g * c[j] + i_g * g_g;
                c[j] = cNew;
                h[j] = o_g * MathF.Tanh(cNew);
            }

            Array.Copy(h, 0, output, ti * hiddenSize, hiddenSize);
        }
        return output;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
}
