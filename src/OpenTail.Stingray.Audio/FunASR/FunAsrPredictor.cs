using System;
using System.Threading.Tasks;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.FunASR;

/// <summary>
/// Real CIF (Continuous Integrate-and-Fire) predictor for Paraformer, transcribed from the real
/// `funasr` Python package's `CifPredictorV2` (`funasr/models/paraformer/cif_predictor.py`),
/// NOT the base `CifPredictor` class -- confirmed from this checkpoint's real
/// `predictor.cif_conv1d.weight` tensor shape (a FULL, non-grouped Conv1d), see docs/audio-
/// review-progress.md's FunASR section for the full derivation. Do not re-derive.
///
/// Real forward: `ReLU(Conv1d(pad(hidden), kernel=3, groups=1))` -&gt; `Sigmoid(Linear(_, 1))`
/// -&gt; alphas -&gt; append a synthetic final "tail" alpha (this checkpoint's real
/// `pf.predictor.tail_threshold` metadata) + a zero hidden frame -&gt; the classic sequential
/// CIF integrate-and-fire algorithm (real vectorized source is a cumsum/floor trick
/// mathematically equivalent to this sequential form for `threshold=1.0`).
/// </summary>
public static class FunAsrPredictor
{
    /// <summary>Runs the real CIF predictor. Returns (acousticEmbeds, tokenCount) -- acousticEmbeds has exactly tokenCount rows of the encoder's hidden dim.</summary>
    public static (float[][] AcousticEmbeds, int TokenCount) Predict(FunAsrWeights w, float[][] hidden)
    {
        int t = hidden.Length;
        int c = w.EncoderDim;

        // ReLU(Conv1d(pad(hidden.T, left=1, right=1), kernel=3, groups=1)) -- a FULL conv, not
        // depthwise (confirmed from the real tensor shape, see class doc comment).
        var convOut = new float[t][];
        for (int ti = 0; ti < t; ti++) convOut[ti] = new float[c];
        Parallel.For(0, c, oc =>
        {
            int wOcBase = oc * c * 3;
            float bias = w.PredictorCifConv1dBias[oc];
            for (int ti = 0; ti < t; ti++)
            {
                float sum = bias;
                for (int ic = 0; ic < c; ic++)
                {
                    int wBase = wOcBase + ic * 3;
                    for (int k = 0; k < 3; k++)
                    {
                        int srcT = ti - 1 + k;
                        if ((uint)srcT < (uint)t) sum += hidden[srcT][ic] * w.PredictorCifConv1dWeight[wBase + k];
                    }
                }
                convOut[ti][oc] = MathF.Max(0f, sum);
            }
        });

        // alphas = sigmoid(Linear(convOut, 512->1))
        var alphas = new float[t + 1]; // +1 for the synthetic tail frame appended below
        for (int ti = 0; ti < t; ti++)
        {
            float logit = w.PredictorCifOutputBias[0];
            for (int d = 0; d < c; d++) logit += convOut[ti][d] * w.PredictorCifOutputWeight[d];
            alphas[ti] = Sigmoid(logit);
        }
        alphas[t] = w.CifTailThreshold;

        var hiddenExt = new float[t + 1][];
        Array.Copy(hidden, hiddenExt, t);
        hiddenExt[t] = new float[c];

        float tokenNumFloat = 0f;
        for (int i = 0; i < alphas.Length; i++) tokenNumFloat += alphas[i];
        int tokenNum = (int)MathF.Floor(tokenNumFloat);

        // Classic sequential CIF integrate-and-fire, threshold = w.CifThreshold (1.0 for this checkpoint).
        var tokens = new System.Collections.Generic.List<float[]>();
        float accumulatedWeight = 0f;
        var accumulatedState = new float[c];
        for (int ti = 0; ti < alphas.Length; ti++)
        {
            float a = alphas[ti];
            if (accumulatedWeight + a >= w.CifThreshold)
            {
                float remaining = w.CifThreshold - accumulatedWeight;
                float carry = a - remaining;
                var emitted = new float[c];
                for (int d = 0; d < c; d++) emitted[d] = accumulatedState[d] + remaining * hiddenExt[ti][d];
                tokens.Add(emitted);

                accumulatedWeight = carry;
                for (int d = 0; d < c; d++) accumulatedState[d] = carry * hiddenExt[ti][d];
            }
            else
            {
                accumulatedWeight += a;
                for (int d = 0; d < c; d++) accumulatedState[d] += a * hiddenExt[ti][d];
            }
        }

        int finalCount = Math.Min(tokenNum, tokens.Count);
        var result = new float[finalCount][];
        for (int i = 0; i < finalCount; i++) result[i] = tokens[i];
        return (result, finalCount);
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
}
