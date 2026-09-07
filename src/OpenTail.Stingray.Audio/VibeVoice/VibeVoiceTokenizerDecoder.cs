namespace OpenTail.Stingray.Audio.VibeVoice;

/// <summary>
/// Real forward pass for VibeVoice's acoustic-latent-to-waveform tokenizer DECODER, ported from
/// `tokenizer_audio.cpp`'s `build_decoder` (not guessed): for stage 0, a stride-1 causal Conv1d
/// stem; for every later stage, a causal `ConvTranspose1d` upsample (real ratio for that stage)
/// -- each followed by that stage's real ConvNeXt-1D blocks (<see cref="VibeVoiceConvNeXtBlock"/>,
/// shared with the encoder) -- then an optional final channel RMSNorm and a stride-1 causal
/// Conv1d head projecting to the output waveform's channel count.
/// </summary>
public static class VibeVoiceTokenizerDecoder
{
    /// <summary>Decodes `[VaeDim][latentFrames]` channel-major latents into
    /// `[channels][waveformSamples]` (channels=1 for mono).</summary>
    public static float[][] Decode(VibeVoiceTokenizerDecoderWeights w, float[][] latentChannelMajor, float eps)
    {
        var hidden = VibeVoiceConvNeXtBlock.CausalConv1d(latentChannelMajor, w.StemWeight, w.StemBias, stride: 1).Output;
        hidden = RunStage(hidden, w.Stages[0], eps);

        for (int i = 0; i < w.UpsampleRatios.Length; i++)
        {
            hidden = CausalConvTranspose1d(hidden, w.UpsampleWeights[i], w.UpsampleBiases[i], w.UpsampleRatios[i]);
            hidden = RunStage(hidden, w.Stages[i + 1], eps);
        }

        if (w.FinalNormWeight is { } finalNorm)
            hidden = VibeVoiceConvNeXtBlock.ChannelRmsNorm(hidden, finalNorm, eps);

        return VibeVoiceConvNeXtBlock.CausalConv1d(hidden, w.HeadWeight, w.HeadBias, stride: 1).Output;
    }

    /// <summary>Real STREAMING decode, ported from `build_decoder_streaming` (not guessed): same
    /// per-stage structure as <see cref="Decode"/>, but every conv uses the real per-call CACHE
    /// convention (<see cref="VibeVoiceConvNeXtBlock.SConv1dStreaming"/>/
    /// <see cref="VibeVoiceConvNeXtBlock.SConvTranspose1dStreaming"/>/
    /// <see cref="VibeVoiceConvNeXtBlock.ForwardStreaming"/>). Call once per real
    /// generated-latent chunk with the SAME <paramref name="state"/> (create via
    /// <see cref="VibeVoiceTokenizerStreamingState.ForDecoder"/> once per stream).</summary>
    public static float[][] DecodeStreaming(VibeVoiceTokenizerDecoderWeights w, VibeVoiceTokenizerStreamingState state, float[][] latentChannelMajorChunk, float eps)
    {
        state.BeginChunk();
        var hidden = VibeVoiceConvNeXtBlock.SConv1dStreaming(latentChannelMajorChunk, w.StemWeight, w.StemBias, stride: 1, ref state.Next());
        hidden = RunStageStreaming(hidden, w.Stages[0], eps, state);

        for (int i = 0; i < w.UpsampleRatios.Length; i++)
        {
            hidden = VibeVoiceConvNeXtBlock.SConvTranspose1dStreaming(hidden, w.UpsampleWeights[i], w.UpsampleBiases[i], w.UpsampleRatios[i], ref state.Next());
            hidden = RunStageStreaming(hidden, w.Stages[i + 1], eps, state);
        }

        if (w.FinalNormWeight is { } finalNorm)
            hidden = VibeVoiceConvNeXtBlock.ChannelRmsNorm(hidden, finalNorm, eps);

        return VibeVoiceConvNeXtBlock.SConv1dStreaming(hidden, w.HeadWeight, w.HeadBias, stride: 1, ref state.Next());
    }

    private static float[][] RunStage(float[][] hidden, VibeVoiceConvNeXtBlockWeights[] blocks, float eps)
    {
        foreach (var block in blocks) hidden = VibeVoiceConvNeXtBlock.Forward(hidden, block, eps);
        return hidden;
    }

    private static float[][] RunStageStreaming(float[][] hidden, VibeVoiceConvNeXtBlockWeights[] blocks, float eps, VibeVoiceTokenizerStreamingState state)
    {
        foreach (var block in blocks) hidden = VibeVoiceConvNeXtBlock.ForwardStreaming(hidden, block, eps, ref state.Next());
        return hidden;
    }

    /// <summary>Real causal `ConvTranspose1d` (`sconv_transpose1d`, not guessed): a standard
    /// unpadded transpose-conv (`padding=0, output_padding=0`) followed by trimming ALL of
    /// `kernel-stride` from the RIGHT end only (real `kTokenizerConvTransposeTrimRightRatio=1.0`
    /// -&gt; `padding_left=0`) -- a genuinely different crop than a symmetric PyTorch
    /// `ConvTranspose1d(padding,output_padding)` call would produce, and different again from
    /// Higgs Audio TTS's codec decoder (which trims from BOTH ends per its own real ratio).
    /// Weight layout `[inChannels][outChannels][kernel]`.</summary>
    private static float[][] CausalConvTranspose1d(float[][] input, float[][][] weight, float[] bias, int stride)
    {
        int inChannels = input.Length;
        int outChannels = weight[0].Length;
        int kernel = weight[0][0].Length;
        int inLen = input[0].Length;
        int fullLen = (inLen - 1) * stride + kernel;

        var full = new float[outChannels][];
        for (int oc = 0; oc < outChannels; oc++)
        {
            full[oc] = new float[fullLen];
            float b = bias[oc];
            for (int o = 0; o < fullLen; o++) full[oc][o] = b;
        }

        for (int ic = 0; ic < inChannels; ic++)
        {
            var inRow = input[ic];
            for (int i = 0; i < inLen; i++)
            {
                float v = inRow[i];
                if (v == 0f) continue;
                int baseOut = i * stride;
                for (int oc = 0; oc < outChannels; oc++)
                {
                    var wRow = weight[ic][oc];
                    var outRow = full[oc];
                    for (int k = 0; k < kernel; k++) outRow[baseOut + k] += wRow[k] * v;
                }
            }
        }

        int paddingTotal = kernel - stride;
        if (paddingTotal < 0) throw new InvalidOperationException("VibeVoice tokenizer ConvTranspose1d padding_total is negative.");
        int outLen = fullLen - paddingTotal;
        if (outLen <= 0) throw new InvalidOperationException("VibeVoice tokenizer ConvTranspose1d unpad removed all frames.");

        var output = new float[outChannels][];
        for (int oc = 0; oc < outChannels; oc++)
        {
            output[oc] = new float[outLen];
            Array.Copy(full[oc], 0, output[oc], 0, outLen);
        }
        return output;
    }
}
