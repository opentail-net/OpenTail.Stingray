using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Parakeet;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class ParakeetTests
{
    [Fact]
    public void ParakeetMelExtractor_ExtractMel_Produces80ChannelLogMelSpectrogram()
    {
        var extractor = new ParakeetMelExtractor();
        int sampleRate = 16000;
        int durationSamples = sampleRate * 1; // 1 second of audio
        var pcm = new float[durationSamples];

        // Synthesize test audio (sine wave 440 Hz)
        for (int i = 0; i < durationSamples; i++)
        {
            pcm[i] = 0.5f * MathF.Sin(2.0f * MathF.PI * 440.0f * i / sampleRate);
        }

        float[] mel = extractor.ExtractMel(pcm);

        Assert.NotNull(mel);
        Assert.NotEmpty(mel);
        Assert.Equal(0, mel.Length % ParakeetMelExtractor.NumMels);

        int numFrames = mel.Length / ParakeetMelExtractor.NumMels;
        Assert.True(numFrames > 90); // ~100 frames for 1 second with 10ms hop

        for (int i = 0; i < mel.Length; i++)
        {
            Assert.False(float.IsNaN(mel[i]), $"NaN at mel index {i}");
            Assert.False(float.IsInfinity(mel[i]), $"Infinity at mel index {i}");
        }
    }

    [Fact]
    public void ParakeetTokenizer_EncodeAndDecode_PreservesTextAndHandlesSpecialTokens()
    {
        var tokenizer = new ParakeetTokenizer();
        string text = "the speech recognition model is fast and accurate";

        int[] tokens = tokenizer.Encode(text);

        Assert.NotNull(tokens);
        Assert.NotEmpty(tokens);

        foreach (int t in tokens)
        {
            Assert.InRange(t, 0, tokenizer.VocabSize - 1);
            Assert.NotEqual(ParakeetTokenizer.BlankTokenId, t);
        }

        string decoded = tokenizer.Decode(tokens);
        Assert.NotNull(decoded);
        Assert.NotEmpty(decoded);
        Assert.Equal(text, decoded, ignoreCase: true);
    }

    // ParakeetConformerEncoder is now a real, weight-driven port (no procedural fast-path
    // constructor exists anymore -- see docs/audio-review-progress.md's Parakeet section).
    // Encoder/decoder/pipeline coverage against real GGUF weights lives in
    // OpenTail.Stingray.Tests.Audio/ParakeetConformerEncoderTests.cs (HeavyTestBase).
}
