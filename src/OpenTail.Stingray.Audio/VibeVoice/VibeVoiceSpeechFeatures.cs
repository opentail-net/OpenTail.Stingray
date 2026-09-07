namespace OpenTail.Stingray.Audio.VibeVoice;

/// <summary>
/// Real full speech-feature extraction for VibeVoice ASR, ported from `speech_encoder.cpp`'s
/// non-streaming branch (not guessed): `encode_acoustic(audio) -&gt; sample_gaussian -&gt;
/// project_acoustic` PLUS `encode_semantic(audio) -&gt; project_semantic`, frame-aligned
/// (truncated to `min(acousticFrames, semanticFrames)`), then summed element-wise -- a plain
/// addition of the two connectors' projected embeddings, not a concatenation or gated mix.
/// </summary>
public static class VibeVoiceSpeechFeatures
{
    /// <summary>Extracts the combined `[hiddenSize][frames]` channel-major speech embedding
    /// stream to splice into the text decoder via
    /// <see cref="VibeVoiceLlmTensorSource.EnableSpeechConditioning"/>.</summary>
    public static float[][] Extract(
        VibeVoiceTokenizerEncoderWeights acousticEncoder,
        VibeVoiceTokenizerConfig acousticConfig,
        VibeVoiceConnectorWeights acousticConnector,
        VibeVoiceTokenizerEncoderWeights semanticEncoder,
        VibeVoiceTokenizerConfig semanticConfig,
        VibeVoiceConnectorWeights semanticConnector,
        float[] monoWaveform,
        Random gaussianRng)
    {
        var acousticMean = VibeVoiceTokenizerEncoder.Encode(acousticEncoder, monoWaveform, acousticConfig.LayerNormEps);
        var acousticSampled = VibeVoiceAcousticLatentSampler.Sample(acousticMean, acousticConfig.FixStd, gaussianRng);
        var acousticProjected = VibeVoiceConnector.Project(acousticConnector, acousticSampled);

        var semanticLatent = VibeVoiceTokenizerEncoder.Encode(semanticEncoder, monoWaveform, semanticConfig.LayerNormEps);
        var semanticProjected = VibeVoiceConnector.Project(semanticConnector, semanticLatent);

        int frames = Math.Min(acousticProjected[0].Length, semanticProjected[0].Length);
        int hiddenSize = acousticProjected.Length;
        if (hiddenSize != semanticProjected.Length)
            throw new InvalidDataException("VibeVoice-ASR acoustic and semantic connector hidden sizes mismatch.");

        var combined = new float[hiddenSize][];
        for (int c = 0; c < hiddenSize; c++)
        {
            var row = new float[frames];
            for (int t = 0; t < frames; t++) row[t] = acousticProjected[c][t] + semanticProjected[c][t];
            combined[c] = row;
        }
        return combined;
    }
}
