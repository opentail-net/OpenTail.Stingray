
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Structural (synthetic-weight) test for VoxtralAudioEncoder -- confirms shapes and
/// finiteness without needing the real ~9GB checkpoint (that load test is a separate, follow-on
/// step once the download in models/_models/voxtral-mini-realtime/ completes).</summary>
public sealed class VoxtralAudioEncoderStructuralTests
{
    [Fact]
    public void Forward_SyntheticWeights_ProducesExpectedTokenCountAndFiniteOutput()
    {
        var rng = new Random(42);
        float[] RandArray(int n)
        {
            var a = new float[n];
            for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
            return a;
        }

        int melBins = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoderWeights.NumMelBins;
        int textHidden = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoderWeights.TextHiddenSize;
        int factor = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoderWeights.DownsampleFactor;

        var w = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoderWeights.CreateSynthetic(RandArray);

        int melFrames = 64; // small enough for a fast structural (not perf) check
        var mel = RandArray(melBins * melFrames);

        var tokens = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralAudioEncoder.Forward(w, mel, melFrames);

        // Stem: conv1(stride1) preserves length, conv2(stride2) halves it -> melFrames/2 steps,
        // then downsample by 4 -> melFrames/8 tokens (matching audio_length_per_tok=8 real config).
        int expectedSteps = melFrames / 2;
        int expectedTokens = expectedSteps / factor;
        Assert.Equal(expectedTokens, tokens.Length);
        foreach (var row in tokens)
        {
            Assert.Equal(textHidden, row.Length);
            foreach (var v in row) Assert.True(float.IsFinite(v));
        }
    }
}
