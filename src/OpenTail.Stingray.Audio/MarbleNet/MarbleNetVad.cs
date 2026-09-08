namespace OpenTail.Stingray.Audio.MarbleNet;

public readonly struct MarbleNetSpeechSegment(int startSample, int endSample, float confidence)
{
    public int StartSample { get; } = startSample;
    public int EndSample { get; } = endSample;
    public float Confidence { get; } = confidence;
}

/// <summary>
/// Real MarbleNet VAD forward pass (a small NeMo "Jasper"-family depthwise-separable-conv CNN),
/// ported from `marblenet_vad/runtime.cpp`'s `build_marblenet_graph`/`jasper_block`/`conv1d` (not
/// guessed). Channel-major `[channels][frames]` throughout, matching this codebase's existing
/// convolutional-model convention. Six Jasper blocks (5 separable, 1 plain kernel-1) each end in
/// an unconditional ReLU (even blocks with no residual), then a real per-frame Linear decoder
/// (`decoder.layer0`) projects to 2 logits/frame (silence/speech).
/// </summary>
public static class MarbleNetVad
{
    /// <summary>Runs the full Jasper encoder + decoder on `[NMels][frames]` log-mel features,
    /// returning `[frames][NumClasses]` logits (row-major, one row per output frame).</summary>
    public static float[][] Forward(MarbleNetVadWeights w, float[][] melChannelMajor)
    {
        var x = melChannelMajor;
        foreach (var block in w.Blocks)
            x = JasperBlock(x, block);

        // Decoder: per-frame Linear (kernel-1 conv with real bias), [NumClasses][channels] weight.
        int frames = x[0].Length, inChannels = x.Length;
        var logits = new float[frames][];
        for (int t = 0; t < frames; t++)
        {
            var row = new float[MarbleNetVadWeights.NumClasses];
            for (int c = 0; c < MarbleNetVadWeights.NumClasses; c++)
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

    private static float[][] JasperBlock(float[][] input, MarbleNetJasperBlock block)
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
        if (block.ResidualConvBn is { } res)
        {
            var projected = Conv1d(residualInput, res);
            for (int c = 0; c < x.Length; c++)
                for (int t = 0; t < x[c].Length; t++)
                    x[c][t] += projected[c][t];
        }
        Relu(x); // Real: unconditional, even for blocks with no residual.
        return x;
    }

    private static void Relu(float[][] x)
    {
        foreach (var row in x)
            for (int t = 0; t < row.Length; t++)
                if (row[t] < 0f) row[t] = 0f;
    }

    /// <summary>Real, generic 1D conv (regular or depthwise, per `conv.Depthwise`), symmetric
    /// zero-padding (`conv.Padding` each side), dilation, stride.</summary>
    private static float[][] Conv1d(float[][] input, MarbleNetConvBn conv)
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

    /// <summary>Real per-frame speech probability, ported from `runtime.cpp`'s
    /// `speech_probability` (softmax over 2 classes; a real `classes==1` sigmoid branch exists in
    /// the reference for single-logit checkpoints but MarbleNet's real `num_classes=2`, so only
    /// the softmax path is reachable here).</summary>
    public static float SpeechProbability(float[] logits)
    {
        float maxLogit = MathF.Max(logits[0], logits[1]);
        float silence = MathF.Exp(logits[0] - maxLogit);
        float speech = MathF.Exp(logits[1] - maxLogit);
        return speech / (silence + speech);
    }

    /// <summary>Real threshold-based segment decode, ported from `runtime.cpp`'s
    /// `decode_segments`: contiguous runs of `probability &gt;= threshold` frames become one
    /// segment, `frame -&gt; sample` via `hopLength * outputStride`, then rescaled to
    /// `targetSampleRate` if different from the model's real 16kHz.</summary>
    public static List<MarbleNetSpeechSegment> DecodeSegments(MarbleNetVadWeights w, float[][] logits, int targetSampleRate, float threshold = 0.5f)
    {
        var segments = new List<MarbleNetSpeechSegment>();
        bool active = false;
        int startFrame = 0;
        double confidenceSum = 0;
        int confidenceCount = 0;
        int frameToSamples = MarbleNetVadWeights.HopLength * w.OutputStride;

        void Emit(int endFrame)
        {
            int startSample = startFrame * frameToSamples * targetSampleRate / MarbleNetVadWeights.SampleRate;
            int endSample = endFrame * frameToSamples * targetSampleRate / MarbleNetVadWeights.SampleRate;
            float confidence = (float)(confidenceSum / Math.Max(confidenceCount, 1));
            segments.Add(new MarbleNetSpeechSegment(startSample, endSample, confidence));
        }

        for (int f = 0; f < logits.Length; f++)
        {
            float probability = SpeechProbability(logits[f]);
            if (probability >= threshold)
            {
                if (!active) { active = true; startFrame = f; confidenceSum = 0; confidenceCount = 0; }
                confidenceSum += probability;
                confidenceCount++;
                continue;
            }
            if (!active) continue;
            Emit(f);
            active = false;
        }
        if (active) Emit(logits.Length);
        return segments;
    }
}
