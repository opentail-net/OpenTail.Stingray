
namespace OpenTail.Stingray.Audio.FunASR;

/// <summary>Real forward pass for Fun-ASR-Nano-2512's audio-tower SAN-M encoder -- see
/// <see cref="FunAsrNanoEncoderWeights"/>'s doc comment for the architecture derivation.
/// Transcribed directly from `encoder.cpp`'s `ensure_graph`: `hidden = input*sqrt(d_model) +
/// sinusoidal_positions(over input_size channels) -> stem (projection block, input_size-&gt;d_model)
/// -> 49 main residual blocks -> LayerNorm -> 20 timestamp-prediction residual blocks ->
/// LayerNorm`.</summary>
public static class FunAsrNanoEncoder
{
    /// <summary>Real LFR (Low Frame Rate) splice, transcribed directly from
    /// `examples/audio.cpp/src/framework/audio/kaldi_fbank.cpp` (NOT the funasr Python
    /// `apply_lfr`-derived <see cref="FunAsrRealMelExtractor.ApplyLfr"/>, which uses a different,
    /// more complex right-padding formula for the OLD CIF-Paraformer pipeline -- this real C++
    /// reference instead just CLAMPS the source frame index to <c>[0, melFrames-1]</c> on both
    /// ends, a much simpler and unambiguous algorithm): `frames = 1 + (melFrames-1)/lfrN`;
    /// `leftPad = (lfrM-1)/2`; for each output frame/stacked-frame pair,
    /// `sourceFrame = clamp(outputFrame*lfrN + stackedFrame - leftPad, 0, melFrames-1)`.</summary>
    public static float[][] ApplyRealLfr(float[][] logMel, int lfrM, int lfrN)
    {
        int melFrames = logMel.Length;
        int featDim = logMel[0].Length;
        int leftPad = (lfrM - 1) / 2;
        int frames = 1 + (melFrames - 1) / lfrN;
        var output = new float[frames][];
        for (int o = 0; o < frames; o++)
        {
            var row = new float[lfrM * featDim];
            for (int s = 0; s < lfrM; s++)
            {
                int sourceFrame = Math.Clamp(o * lfrN + s - leftPad, 0, melFrames - 1);
                Array.Copy(logMel[sourceFrame], 0, row, s * featDim, featDim);
            }
            output[o] = row;
        }
        return output;
    }

    /// <summary>mel is frame-major [T][InputSize] (560-dim, post mel+LFR via
    /// <see cref="ApplyRealLfr"/> -- no CMVN, this architecture's real frontend has none).
    /// Returns per-frame encoder hidden states [T][DModel].</summary>
    public static float[][] Forward(FunAsrNanoEncoderWeights w, float[][] mel)
    {
        int t = mel.Length;
        float scale = MathF.Sqrt(FunAsrNanoEncoderWeights.DModel);
        var positions = SinusoidalPositions(t, FunAsrNanoEncoderWeights.InputSize);

        var hidden = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[FunAsrNanoEncoderWeights.InputSize];
            for (int c = 0; c < row.Length; c++) row[c] = mel[i][c] * scale + positions[i][c];
            hidden[i] = row;
        }

        var x = FunAsrNanoSanmBlock.ProjectionBlock(
            hidden, w.Stem, FunAsrNanoEncoderWeights.InputSize, FunAsrNanoEncoderWeights.DModel,
            FunAsrNanoEncoderWeights.AttentionHeads, FunAsrNanoEncoderWeights.FfnDim, FunAsrNanoEncoderWeights.KernelSize, FunAsrNanoEncoderWeights.LayerNormEps);

        foreach (var layer in w.MainLayers)
            x = FunAsrNanoSanmBlock.ResidualBlock(
                x, layer, FunAsrNanoEncoderWeights.DModel, FunAsrNanoEncoderWeights.AttentionHeads,
                FunAsrNanoEncoderWeights.FfnDim, FunAsrNanoEncoderWeights.KernelSize, FunAsrNanoEncoderWeights.LayerNormEps);

        x = LayerNormRows(x, FunAsrNanoEncoderWeights.DModel, w.MainNorm.Weight, w.MainNorm.Bias, FunAsrNanoEncoderWeights.LayerNormEps);

        foreach (var layer in w.TimestampLayers)
            x = FunAsrNanoSanmBlock.ResidualBlock(
                x, layer, FunAsrNanoEncoderWeights.DModel, FunAsrNanoEncoderWeights.AttentionHeads,
                FunAsrNanoEncoderWeights.FfnDim, FunAsrNanoEncoderWeights.KernelSize, FunAsrNanoEncoderWeights.LayerNormEps);

        return LayerNormRows(x, FunAsrNanoEncoderWeights.DModel, w.TimestampNorm.Weight, w.TimestampNorm.Bias, FunAsrNanoEncoderWeights.LayerNormEps);
    }

    /// <summary>Real sinusoidal position table, matching `make_sinusoidal_positions` exactly:
    /// position = frame+1 (1-indexed), `increment = ln(10000)/(half-1)`,
    /// `inv_timescale[i] = exp(-increment*i)`, `sin`/`cos` halves.</summary>
    private static float[][] SinusoidalPositions(int frames, int channels)
    {
        int half = channels / 2;
        float increment = MathF.Log(10000f) / (half - 1);
        var table = new float[frames][];
        for (int f = 0; f < frames; f++)
        {
            var row = new float[channels];
            float position = f + 1;
            for (int i = 0; i < half; i++)
            {
                float invTimescale = MathF.Exp(-increment * i);
                float phase = position * invTimescale;
                row[i] = MathF.Sin(phase);
                row[half + i] = MathF.Cos(phase);
            }
            table[f] = row;
        }
        return table;
    }

    private static float[][] LayerNormRows(float[][] xRows, int dim, float[] weight, float[] bias, float eps)
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
            float invStd = (float)(1.0 / Math.Sqrt(variance + eps));
            var outRow = new float[dim];
            for (int c = 0; c < dim; c++) outRow[c] = (float)((row[c] - mean) * invStd) * weight[c] + bias[c];
            output[i] = outRow;
        }
        return output;
    }
}
