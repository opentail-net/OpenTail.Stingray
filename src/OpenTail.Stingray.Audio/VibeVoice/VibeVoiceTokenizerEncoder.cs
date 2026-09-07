namespace OpenTail.Stingray.Audio.VibeVoice;

/// <summary>
/// Real forward pass for VibeVoice's tokenizer encoder (used for BOTH the acoustic and semantic
/// tokenizers -- same architecture, different config/weights), ported from
/// `examples/audio.cpp/src/models/vibevoice_asr/speech_tokenizer.cpp`'s `build_encoder` (not
/// guessed). Real per-stage math: causal downsample Conv1d (stride 1 for stage 0, then
/// `EncoderRatios` reversed for later stages) -&gt; that stage's real depth-count of ConvNeXt-1D
/// blocks (<see cref="VibeVoiceConvNeXtBlock"/>) -&gt; (after all stages) an optional final channel
/// RMSNorm -&gt; a stride-1 causal Conv1d head projecting to `VaeDim`. Deterministic -- ASR
/// encoding never samples the acoustic tokenizer's Gaussian VAE (that's a real, separate,
/// TTS-generation-only path in the reference, `sample_vibevoice_acoustic_latents_gaussian`, not
/// used by `encode_acoustic`/`encode_semantic` and therefore not ported here).
/// </summary>
public static class VibeVoiceTokenizerEncoder
{
    /// <summary>Encodes a mono waveform (real 24kHz input, `[1][samples]` channels-major with one
    /// channel) into `[VaeDim][frames]` latent features.</summary>
    public static float[][] Encode(VibeVoiceTokenizerEncoderWeights w, float[] monoWaveform, float layerNormEps)
    {
        float[][] hidden = [monoWaveform];

        for (int stage = 0; stage < w.Stages.Length; stage++)
        {
            (hidden, _) = VibeVoiceConvNeXtBlock.CausalConv1d(
                hidden, w.DownsampleWeights[stage], w.DownsampleBiases[stage], w.DownsampleStrides[stage]);

            foreach (var block in w.Stages[stage])
                hidden = VibeVoiceConvNeXtBlock.Forward(hidden, block, layerNormEps);
        }

        if (w.FinalNormWeight is { } finalNorm)
            hidden = VibeVoiceConvNeXtBlock.ChannelRmsNorm(hidden, finalNorm, layerNormEps);

        (hidden, _) = VibeVoiceConvNeXtBlock.CausalConv1d(hidden, w.HeadWeight, w.HeadBias, stride: 1);
        return hidden;
    }

    /// <summary>Real STREAMING encode, ported from `build_encoder_streaming` (not guessed): same
    /// per-stage structure as <see cref="Encode"/>, but every conv uses the real per-call CACHE
    /// convention (<see cref="VibeVoiceConvNeXtBlock.SConv1dStreaming"/>/
    /// <see cref="VibeVoiceConvNeXtBlock.ForwardStreaming"/>) instead of whole-clip zero-padding.
    /// Call once per real audio chunk with the SAME <paramref name="state"/> (create via
    /// <see cref="VibeVoiceTokenizerStreamingState.ForEncoder"/> once per stream) for correct
    /// causal continuity across chunk boundaries -- a fresh state per call is equivalent to the
    /// non-streaming <see cref="Encode"/>.</summary>
    public static float[][] EncodeStreaming(VibeVoiceTokenizerEncoderWeights w, VibeVoiceTokenizerStreamingState state, float[] monoWaveformChunk, float layerNormEps)
    {
        float[][] hidden = [monoWaveformChunk];
        state.BeginChunk();

        for (int stage = 0; stage < w.Stages.Length; stage++)
        {
            hidden = VibeVoiceConvNeXtBlock.SConv1dStreaming(hidden, w.DownsampleWeights[stage], w.DownsampleBiases[stage], w.DownsampleStrides[stage], ref state.Next());
            foreach (var block in w.Stages[stage])
                hidden = VibeVoiceConvNeXtBlock.ForwardStreaming(hidden, block, layerNormEps, ref state.Next());
        }

        if (w.FinalNormWeight is { } finalNorm)
            hidden = VibeVoiceConvNeXtBlock.ChannelRmsNorm(hidden, finalNorm, layerNormEps);

        return VibeVoiceConvNeXtBlock.SConv1dStreaming(hidden, w.HeadWeight, w.HeadBias, stride: 1, ref state.Next());
    }
}
