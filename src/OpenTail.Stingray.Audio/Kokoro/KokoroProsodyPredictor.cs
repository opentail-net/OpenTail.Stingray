
namespace OpenTail.Stingray.Audio.Kokoro;

/// <summary>
/// StyleTTS2 ProsodyPredictor's duration path (modules.py ProsodyPredictor + DurationEncoder,
/// model.py forward lines 93-97): DurationEncoder alternates BiLSTM blocks (input
/// d_model+style_dim=640, hidden 256/direction) with AdaLayerNorm blocks (each re-concatenating
/// the style vector onto the channel dim afterward, so every LSTM after the first also sees
/// 640 input channels), then predictor.lstm (same 640->512 shape) feeds duration_proj (Linear
/// 512->max_dur) whose sigmoid-summed output is the (pre-round/clamp) per-token duration.
/// F0Ntrain (predictor.shared/F0/N stacks) is a separate, not-yet-implemented stage.
///
/// NOTE: ProsodyPredictor.text_encoder in the reference is DurationEncoder -- a different
/// module from the top-level KModel.text_encoder (see KokoroTextEncoder.cs). Do not conflate.
/// </summary>
public static class KokoroProsodyPredictor
{
    /// <summary>
    /// DurationEncoder.forward. dEn is channel-first [HiddenDim, T] (KokoroBertEncoder's
    /// ProjectToWorkingDim output). style is the StyleDim-length style-conditioning vector
    /// (ref_s[:, StyleDim:] in model.py). Returns `d`, batch_first [T, HiddenDim+StyleDim].
    /// </summary>
    public static float[] EncodeDuration(KokoroWeights weights, float[] dEn, float[] style, int t)
    {
        var pred = weights.Predictor;
        int channels = weights.HiddenDim;
        int styleDim = weights.StyleDim;
        int catDim = channels + styleDim;
        int hiddenPerDir = channels / 2;

        var xCat = ConcatChannelFirst(dEn, style, channels, styleDim, t);

        int n = weights.NStyleLayers;
        for (int i = 0; i < n; i++)
        {
            var lstmInput = Transpose(xCat, catDim, t);
            var lstmOut = KokoroLstm.Bidirectional(pred.DurEncLstm[i], lstmInput, t, catDim, hiddenPerDir);
            var lstmChannelFirst = Transpose(lstmOut, t, channels);
            var normed = KokoroAdaLayerNorm.Forward(lstmChannelFirst, style, pred.DurEncAdaLnWeight[i], pred.DurEncAdaLnBias[i], channels, styleDim, t);
            xCat = ConcatChannelFirst(normed, style, channels, styleDim, t);
        }

        return Transpose(xCat, catDim, t);
    }

    /// <summary>
    /// predictor.lstm + duration_proj + sigmoid-sum (model.py lines 94-96, pre-round/clamp).
    /// d is `EncodeDuration`'s output, [T, HiddenDim+StyleDim]. Returns per-token float
    /// durations (caller applies /speed, round, clamp(min=1) as needed).
    /// </summary>
    public static float[] PredictDurations(KokoroWeights weights, float[] d, int t)
    {
        var pred = weights.Predictor;
        int channels = weights.HiddenDim;
        int styleDim = weights.StyleDim;
        int catDim = channels + styleDim;
        int hiddenPerDir = channels / 2;
        int maxDur = weights.MaxDur;

        var lstmOut = KokoroLstm.Bidirectional(pred.SharedLstm, d, t, catDim, hiddenPerDir);

        var durations = new float[t];
        var logits = new float[maxDur];
        var sigmoids = new float[maxDur];
        unsafe
        {
            fixed (float* w = pred.DurProjWeight, b = pred.DurProjBias, l = logits, s = sigmoids, inp = lstmOut)
            {
                for (int ti = 0; ti < t; ti++)
                {
                    SimdKernels.MatVecF32(l, w, inp + ti * channels, maxDur, channels);
                    var lSpan = logits.AsSpan();
                    var bSpan = pred.DurProjBias.AsSpan();
                    var sSpan = sigmoids.AsSpan();
                    System.Numerics.Tensors.TensorPrimitives.Add(lSpan, bSpan, lSpan);
                    System.Numerics.Tensors.TensorPrimitives.Sigmoid(lSpan, sSpan);
                    durations[ti] = System.Numerics.Tensors.TensorPrimitives.Sum(sSpan);
                }
            }
        }
        return durations;
    }

    /// <summary>
    /// ProsodyPredictor.F0Ntrain (modules.py, model.py line 104: `F0_pred, N_pred =
    /// self.predictor.F0Ntrain(en, s)`). `en` is channel-first `[HiddenDim+StyleDim, frames]`
    /// (`d.transpose(-1,-2) @ pred_aln_trg`, frame-rate after length regulation -- see
    /// KokoroAlignment). Returns (F0Curve, NCurve), each length `frames*2` (the F0/N
    /// AdainResBlk1d stacks upsample x2 partway through), matching `F0_curve`/`N` in
    /// istftnet.py's Decoder.forward.
    /// </summary>
    public static (float[] F0Curve, float[] NCurve) F0Ntrain(KokoroWeights weights, float[] en, int frames, float[] style)
    {
        var pred = weights.Predictor;
        int hidden = weights.HiddenDim;    // 512
        int half = hidden / 2;             // 256
        int catDim = hidden + weights.StyleDim; // 640
        int hiddenPerDir = half;
        int styleDim = weights.StyleDim;

        var enBatchFirst = Transpose(en, catDim, frames);
        var sharedOut = KokoroLstm.Bidirectional(pred.SharedFeatureLstm, enBatchFirst, frames, catDim, hiddenPerDir); // [frames, hidden]
        var xChannelFirst = Transpose(sharedOut, frames, hidden); // [hidden, frames]

        int framesUp = frames * 2;

        var f0 = KokoroAdainResBlk1d.Forward(pred.F0[0], xChannelFirst, hidden, hidden, frames, style, styleDim);
        f0 = KokoroAdainResBlk1d.Forward(pred.F0[1], f0, hidden, half, frames, style, styleDim); // upsamples frames -> framesUp
        f0 = KokoroAdainResBlk1d.Forward(pred.F0[2], f0, half, half, framesUp, style, styleDim);
        var f0Curve = Conv1x1(f0, pred.F0ProjWeight, pred.F0ProjBias, half, 1, framesUp);

        var n = KokoroAdainResBlk1d.Forward(pred.N[0], xChannelFirst, hidden, hidden, frames, style, styleDim);
        n = KokoroAdainResBlk1d.Forward(pred.N[1], n, hidden, half, frames, style, styleDim);
        n = KokoroAdainResBlk1d.Forward(pred.N[2], n, half, half, framesUp, style, styleDim);
        var nCurve = Conv1x1(n, pred.NProjWeight, pred.NProjBias, half, 1, framesUp);

        return (f0Curve, nCurve);
    }

    private static float[] Conv1x1(float[] input, float[] weight, float[] bias, int inCh, int outCh, int t)
    {
        var output = new float[outCh * t];
        for (int oc = 0; oc < outCh; oc++)
        {
            int wBase = oc * inCh;
            for (int ti = 0; ti < t; ti++)
            {
                float sum = bias[oc];
                for (int ic = 0; ic < inCh; ic++)
                    sum += weight[wBase + ic] * input[ic * t + ti];
                output[oc * t + ti] = sum;
            }
        }
        return output;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private static float[] ConcatChannelFirst(float[] main, float[] style, int mainChannels, int styleDim, int t)
    {
        int total = mainChannels + styleDim;
        var output = new float[total * t];
        for (int c = 0; c < mainChannels; c++)
            for (int ti = 0; ti < t; ti++)
                output[c * t + ti] = main[c * t + ti];
        for (int c = 0; c < styleDim; c++)
            for (int ti = 0; ti < t; ti++)
                output[(mainChannels + c) * t + ti] = style[c];
        return output;
    }

    private static float[] Transpose(float[] src, int rows, int cols)
    {
        var dst = new float[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                dst[c * rows + r] = src[r * cols + c];
        return dst;
    }
}
