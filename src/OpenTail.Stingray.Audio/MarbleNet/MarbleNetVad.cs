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

        return JasperKernels.LinearDecoder(x, w.DecoderWeight, w.DecoderBias, MarbleNetVadWeights.NumClasses);
    }

    private static float[][] JasperBlock(float[][] input, MarbleNetJasperBlock block)
    {
        var residualInput = input;
        var x = input;
        if (block.Separable)
        {
            for (int r = 0; r < block.SeparableRepeats.Length; r++)
            {
                x = JasperKernels.Conv1d(x, block.SeparableRepeats[r].Depthwise);
                x = JasperKernels.Conv1d(x, block.SeparableRepeats[r].Pointwise);
                if (r + 1 != block.SeparableRepeats.Length) JasperKernels.Relu(x);
            }
        }
        else
        {
            for (int r = 0; r < block.ConvRepeats.Length; r++)
            {
                x = JasperKernels.Conv1d(x, block.ConvRepeats[r]);
                if (r + 1 != block.ConvRepeats.Length) JasperKernels.Relu(x);
            }
        }
        if (block.ResidualConvBn is { } res)
        {
            var projected = JasperKernels.Conv1d(residualInput, res);
            JasperKernels.AddResidualInPlace(x, projected);
        }
        JasperKernels.Relu(x); // Real: unconditional, even for blocks with no residual.
        return x;
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
