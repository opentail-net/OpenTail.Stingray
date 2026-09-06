
namespace OpenTail.Stingray.Audio.NemotronAsr;

/// <summary>
/// Nemotron 3.5 ASR's RNN-Transducer prediction network + joint network + frame-synchronous greedy
/// decode loop, transcribed directly from `examples/audio.cpp/src/models/nemotron_asr/decoder.cpp`
/// (`ensure_graph`/`run_step`/`decode`, not guessed). Real predictor: `embed(token) -> 2-layer
/// LSTM -> Linear(640-&gt;640, bias)` (`joint.pred` -- NOT `decoder_projector` as a separately-named
/// tensor; confirmed by shape match: 640-&gt;640 with bias). Real joint: `relu(encoder_frame[640] +
/// decoder_cache[640]) -&gt; Linear(640-&gt;vocab, bias)` (`joint.joint_net.2`). Critically, the
/// predictor LSTM state and `decoder_cache` are ONLY updated when a NON-BLANK token was just
/// emitted (or on the very first step, which force-updates regardless of the initial blank
/// input) -- repeated blank predictions reuse the frozen `decoder_cache` against successive
/// encoder frames without re-running the LSTM. Getting this skip-on-blank behavior right matters:
/// naively re-running the predictor every step would silently diverge from the reference's real
/// greedy-decode semantics.
/// </summary>
public static class NemotronAsrRnntDecoder
{
    public static List<int> Greedy(NemotronAsrWeights w, float[][] encoderFrames, int? maxTokens = null)
    {
        int hidden = w.PredHidden;
        int layers = w.PredNumLayers;
        var h = new float[layers][];
        var c = new float[layers][];
        for (int l = 0; l < layers; l++) { h[l] = new float[hidden]; c[l] = new float[hidden]; }
        var decoderCache = new float[hidden];

        int limit = maxTokens ?? (encoderFrames.Length * w.MaxSymbolsPerStep + 1);
        var tokens = new List<int> { w.BlankTokenId };

        int frameIndex = 0;
        int symbolsAtFrame = 0;
        int inputToken = w.BlankTokenId;
        bool cacheInitialized = false;

        while (frameIndex < encoderFrames.Length && tokens.Count - 1 < limit)
        {
            bool updatePredictor = !cacheInitialized || inputToken != w.BlankTokenId;
            if (updatePredictor)
                RunPredictor(w, inputToken, h, c, decoderCache);
            cacheInitialized = true;

            var logits = Joint(w, encoderFrames[frameIndex], decoderCache);
            int token = ArgMax(logits);
            tokens.Add(token);

            bool blank = token == w.BlankTokenId;
            if (!blank) symbolsAtFrame++;
            bool forceAdvance = symbolsAtFrame >= w.MaxSymbolsPerStep;
            if (blank || forceAdvance)
            {
                frameIndex++;
                symbolsAtFrame = 0;
            }
            inputToken = token;
        }

        return tokens;
    }

    private static void RunPredictor(NemotronAsrWeights w, int token, float[][] h, float[][] c, float[] decoderCacheOut)
    {
        int hidden = w.PredHidden;
        var x = new float[hidden];
        int embedBase = token * hidden;
        for (int d = 0; d < hidden; d++) x[d] = w.PredEmbedWeight[embedBase + d];

        var input = x;
        var lstms = new[] { w.PredLstm0, w.PredLstm1 };
        for (int l = 0; l < w.PredNumLayers; l++)
        {
            var (newH, newC) = LstmCell(lstms[l], input, h[l], c[l], hidden);
            h[l] = newH;
            c[l] = newC;
            input = newH;
        }

        var projected = Linear(input, w.JointPredWeight, w.JointPredBias, hidden, hidden);
        Array.Copy(projected, decoderCacheOut, hidden);
    }

    /// <summary>Real PyTorch LSTMCell math, gates packed in order i,f,g,o (real PyTorch
    /// convention).</summary>
    private static (float[] H, float[] C) LstmCell(NemotronAsrLstmLayer l, float[] x, float[] hPrev, float[] cPrev, int hidden)
    {
        var gates = new float[4 * hidden];
        for (int g = 0; g < 4 * hidden; g++)
        {
            float sum = l.BiasIh[g] + l.BiasHh[g];
            int wIhBase = g * x.Length;
            for (int i = 0; i < x.Length; i++) sum += l.WeightIh[wIhBase + i] * x[i];
            int wHhBase = g * hidden;
            for (int i = 0; i < hidden; i++) sum += l.WeightHh[wHhBase + i] * hPrev[i];
            gates[g] = sum;
        }

        var hNew = new float[hidden];
        var cNew = new float[hidden];
        for (int d = 0; d < hidden; d++)
        {
            float i = Sigmoid(gates[d]);
            float f = Sigmoid(gates[hidden + d]);
            float g = MathF.Tanh(gates[2 * hidden + d]);
            float o = Sigmoid(gates[3 * hidden + d]);
            float cn = f * cPrev[d] + i * g;
            cNew[d] = cn;
            hNew[d] = o * MathF.Tanh(cn);
        }
        return (hNew, cNew);
    }

    private static float[] Joint(NemotronAsrWeights w, float[] encoderFrame, float[] decoderCache)
    {
        int hidden = w.JointDim;
        var sum = new float[hidden];
        for (int d = 0; d < hidden; d++) sum[d] = MathF.Max(0f, encoderFrame[d] + decoderCache[d]);
        return Linear(sum, w.JointNet2Weight, w.JointNet2Bias, hidden, w.VocabSize);
    }

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private static float[] Linear(float[] input, float[] weight, float[] bias, int inDim, int outDim)
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
}
