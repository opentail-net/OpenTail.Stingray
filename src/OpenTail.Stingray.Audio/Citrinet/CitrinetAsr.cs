namespace OpenTail.Stingray.Audio.Citrinet;

/// <summary>
/// Real Citrinet ASR forward pass (NVIDIA NeMo's larger "Jasper"-family CTC CNN, with real
/// Squeeze-and-Excitation blocks), ported from `citrinet_asr/runtime.cpp`'s
/// `build_citrinet_graph`/`jasper_block`/`squeeze_excite`/`greedy_ctc_ids` (not guessed).
/// Channel-major `[channels][frames]` throughout. Real per-block structure: N separable-conv
/// repeats (stride placed on the LAST repeat only, unlike MarbleNet where stride is on the first)
/// -&gt; optional Squeeze-Excite gate -&gt; optional 1x1-conv+BN residual (itself strided when
/// `residual_mode=="stride_add"`) -&gt; unconditional ReLU -&gt; a real 1x1-conv decoder to
/// `NumClasses` logits/frame -&gt; greedy CTC decode (argmax, collapse repeats, drop blank).
/// </summary>
public static class CitrinetAsr
{
    public static float[][] Forward(CitrinetAsrWeights w, float[][] melChannelMajor)
    {
        var x = melChannelMajor;
        foreach (var block in w.Blocks)
            x = JasperBlock(x, block);

        int frames = x[0].Length, inChannels = x.Length;
        var logits = new float[frames][];
        for (int t = 0; t < frames; t++)
        {
            var row = new float[w.NumClasses];
            for (int c = 0; c < w.NumClasses; c++)
            {
                float sum = w.DecoderBias[c];
                int wBase = c * inChannels;
                for (int i = 0; i < inChannels; i++) sum += w.DecoderWeight[wBase + i] * x[i][t];
                row[c] = sum;
            }
            logits[t] = row;
        }
        return logits;
    }

    private static float[][] JasperBlock(float[][] input, CitrinetJasperBlock block)
    {
        var residualInput = input;
        var x = input;
        if (block.Separable)
        {
            for (int r = 0; r < block.SeparableRepeats.Length; r++)
            {
                x = Conv1d(x, block.SeparableRepeats[r].Depthwise);
                x = Conv1d(x, block.SeparableRepeats[r].Pointwise);
                if (r + 1 != block.SeparableRepeats.Length) Relu(x);
            }
        }
        else
        {
            for (int r = 0; r < block.ConvRepeats.Length; r++)
            {
                x = Conv1d(x, block.ConvRepeats[r]);
                if (r + 1 != block.ConvRepeats.Length) Relu(x);
            }
        }
        if (block.SqueezeExcite is { } se)
            x = SqueezeExcite(x, se);
        if (block.ResidualConvBn is { } res)
        {
            var projected = Conv1d(residualInput, res);
            for (int c = 0; c < x.Length; c++)
                for (int t = 0; t < x[c].Length; t++)
                    x[c][t] += projected[c][t];
        }
        Relu(x);
        return x;
    }

    /// <summary>Real SE gate, ported from `conditioning_modules.cpp`'s `SqueezeExcite1dModule`:
    /// global average pool over time -&gt; `fc1` (channels-&gt;hidden, no bias) -&gt; ReLU -&gt;
    /// `fc2` (hidden-&gt;channels, no bias) -&gt; Sigmoid -&gt; broadcast-multiply back onto the
    /// original per-frame input.</summary>
    private static float[][] SqueezeExcite(float[][] x, CitrinetSqueezeExcite se)
    {
        int channels = x.Length, frames = x[0].Length;
        var pooled = new float[channels];
        for (int c = 0; c < channels; c++)
        {
            float sum = 0f;
            for (int t = 0; t < frames; t++) sum += x[c][t];
            pooled[c] = sum / frames;
        }

        int hidden = se.Fc1.OutChannels;
        var gate1 = new float[hidden];
        for (int h = 0; h < hidden; h++)
        {
            float sum = 0f;
            int wBase = h * channels;
            for (int c = 0; c < channels; c++) sum += se.Fc1.Weight[wBase + c] * pooled[c];
            gate1[h] = MathF.Max(0f, sum);
        }

        var gate2 = new float[channels];
        for (int c = 0; c < channels; c++)
        {
            float sum = 0f;
            int wBase = c * hidden;
            for (int h = 0; h < hidden; h++) sum += se.Fc2.Weight[wBase + h] * gate1[h];
            gate2[c] = 1f / (1f + MathF.Exp(-sum));
        }

        var output = new float[channels][];
        for (int c = 0; c < channels; c++)
        {
            var row = new float[frames];
            float gate = gate2[c];
            var inRow = x[c];
            for (int t = 0; t < frames; t++) row[t] = inRow[t] * gate;
            output[c] = row;
        }
        return output;
    }

    private static void Relu(float[][] x)
    {
        foreach (var row in x)
            for (int t = 0; t < row.Length; t++)
                if (row[t] < 0f) row[t] = 0f;
    }

    private static float[][] Conv1d(float[][] input, CitrinetConvBn conv)
    {
        int inFrames = input[0].Length;
        int outFrames = (inFrames + 2 * conv.Padding - conv.Dilation * (conv.Kernel - 1) - 1) / conv.Stride + 1;
        var output = new float[conv.OutChannels][];
        for (int c = 0; c < conv.OutChannels; c++) output[c] = new float[outFrames];

        if (conv.Depthwise)
        {
            for (int c = 0; c < conv.OutChannels; c++)
            {
                var inRow = input[c];
                var outRow = output[c];
                int wBase = c * conv.Kernel;
                float bias = conv.Bias[c];
                for (int t = 0; t < outFrames; t++)
                {
                    float sum = bias;
                    int inStart = t * conv.Stride - conv.Padding;
                    for (int k = 0; k < conv.Kernel; k++)
                    {
                        int inIdx = inStart + k * conv.Dilation;
                        if (inIdx < 0 || inIdx >= inFrames) continue;
                        sum += conv.Weight[wBase + k] * inRow[inIdx];
                    }
                    outRow[t] = sum;
                }
            }
        }
        else
        {
            for (int oc = 0; oc < conv.OutChannels; oc++)
            {
                var outRow = output[oc];
                float bias = conv.Bias[oc];
                for (int t = 0; t < outFrames; t++)
                {
                    float sum = bias;
                    int inStart = t * conv.Stride - conv.Padding;
                    for (int ic = 0; ic < conv.InChannels; ic++)
                    {
                        var inRow = input[ic];
                        int wBase = (oc * conv.InChannels + ic) * conv.Kernel;
                        for (int k = 0; k < conv.Kernel; k++)
                        {
                            int inIdx = inStart + k * conv.Dilation;
                            if (inIdx < 0 || inIdx >= inFrames) continue;
                            sum += conv.Weight[wBase + k] * inRow[inIdx];
                        }
                    }
                    outRow[t] = sum;
                }
            }
        }
        return output;
    }

    /// <summary>Real greedy CTC decode, ported from `runtime.cpp`'s `greedy_ctc_ids`: per-frame
    /// argmax, collapse consecutive repeats, drop `blankId`.</summary>
    public static int[] GreedyCtcIds(float[][] logits, int blankId)
    {
        var ids = new List<int>(logits.Length);
        int prev = -1;
        foreach (var row in logits)
        {
            int best = 0;
            float bestValue = row[0];
            for (int c = 1; c < row.Length; c++)
            {
                if (row[c] > bestValue) { bestValue = row[c]; best = c; }
            }
            if (best == prev) continue;
            prev = best;
            if (best == blankId) continue;
            ids.Add(best);
        }
        return [.. ids];
    }
}
