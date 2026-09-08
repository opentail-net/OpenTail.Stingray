namespace OpenTail.Stingray.Audio.MossTts;

/// <summary>Real stereo waveform to encode (real per-channel float samples, 48kHz).</summary>
public readonly struct MossTtsAudioInput(float[] left, float[] right)
{
    public float[] Left { get; } = left;
    public float[] Right { get; } = right;
}

/// <summary>
/// Native C# port of MOSS-Audio-Tokenizer-Nano's ENCODER forward pass (voice-cloning reference-
/// audio conditioning), from `examples/audio.cpp/src/framework/codecs/
/// moss_audio_tokenizer_codec_runtime.cpp`'s `MossAudioTokenizerEncoder::encode`/`Impl::
/// prepare_graph` (not guessed). Real, documented "structural mirror" of
/// <see cref="MossTtsAudioCodecDecoder"/>: interleave the stereo channels into one stream, then
/// four stages of `[PatchDownsample -&gt; input_proj -&gt; N causal-RoPE-attention/LayerScale/
/// exact-erf-GELU-MLP transformer layers -&gt; optional output_proj]` (patch happens BEFORE the
/// transformer for the encoder, the exact reverse order of the decoder's patch-AFTER), a final
/// `PatchDownsample` by `EncoderFinalPatch=4`, then real residual-RVQ nearest-code quantization
/// (<see cref="MossTtsAudioCodecQuantizerWeights.Encode"/>) into `[NumQuantizers][frames]` codes.
/// </summary>
public static class MossTtsAudioCodecEncoder
{
    /// <summary>Encodes a real stereo waveform into `[NumQuantizers][validFrames]` RVQ codes, one
    /// frame per <see cref="MossTtsAudioCodecDecoderWeights.SamplesPerFrame"/> input samples (per
    /// channel). Real, deliberate padding behavior matching the reference: the input is
    /// zero-padded up to a whole number of frames for the transformer graph, but the returned code
    /// length is `floor(validSamples / SamplesPerFrame)` -- any partial trailing frame contributes
    /// no code (matches `MossAudioTokenizerEncoder::encode`'s own `valid_frames` truncation).</summary>
    public static int[][] Encode(
        MossTtsAudioCodecEncoderWeights encoderWeights,
        MossTtsAudioCodecQuantizerWeights quantizerWeights,
        MossTtsAudioInput audio)
    {
        int rawPerChannel = audio.Left.Length;
        if (audio.Right.Length != rawPerChannel)
            throw new ArgumentException("MOSS codec encoder channels must have equal length.");
        if (rawPerChannel <= 0)
            throw new ArgumentException("MOSS codec encoder requires a non-empty waveform.");

        const int samplesPerFrame = MossTtsAudioCodecDecoderWeights.SamplesPerFrame;
        const int channels = MossTtsAudioCodecDecoderWeights.Channels;
        int frames = (rawPerChannel + samplesPerFrame - 1) / samplesPerFrame;
        int validFrames = rawPerChannel / samplesPerFrame;
        if (validFrames <= 0)
            throw new ArgumentException("MOSS codec encoder input is shorter than one codec frame.");
        int perChannel = frames * samplesPerFrame;
        int interleaved = perChannel * channels;

        // Real stereo interleave (channel-fastest), zero-padded to a whole number of frames --
        // exact inverse of the decoder's de-interleave.
        var hidden = new float[interleaved][];
        for (int i = 0; i < perChannel; i++)
        {
            hidden[channels * i] = [i < rawPerChannel ? audio.Left[i] : 0f];
            hidden[channels * i + 1] = [i < rawPerChannel ? audio.Right[i] : 0f];
        }

        foreach (var stage in encoderWeights.Stages)
        {
            var downsampled = MossTtsAudioCodecDecoder.PatchDownsample(hidden, stage.Patch);
            hidden = MossTtsAudioCodecDecoder.RunTransformerStage(downsampled, stage);
        }
        hidden = MossTtsAudioCodecDecoder.PatchDownsample(hidden, MossTtsAudioCodecEncoderWeights.EncoderFinalPatch);

        // hidden is now [outputSteps][CodeDim] feature-last -- flatten to frame-major for the
        // quantizer, truncated to the real valid frame count (matches the reference's own
        // `latent.resize(valid_frames * kCodeDim)`).
        int codeDim = MossTtsAudioCodecQuantizerWeights.CodeDim;
        var latentFrameMajor = new float[validFrames * codeDim];
        for (int f = 0; f < validFrames; f++)
            Array.Copy(hidden[f], 0, latentFrameMajor, f * codeDim, codeDim);

        return quantizerWeights.Encode(latentFrameMajor, validFrames);
    }
}
