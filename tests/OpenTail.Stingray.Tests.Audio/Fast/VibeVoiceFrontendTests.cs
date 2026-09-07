using OpenTail.Stingray.Audio.VibeVoice;

namespace OpenTail.Stingray.Tests.Audio.Fast;

/// <summary>Structural tests for <see cref="VibeVoiceFrontend"/>.</summary>
public sealed class VibeVoiceFrontendTests
{
    private static VibeVoiceFrontendConfig MakeConfig(bool normalize) => new()
    {
        SampleRate = 16000,
        NormalizeAudio = normalize,
        TargetDbFs = -20f,
        Eps = 1e-8f,
    };

    [Fact]
    public void Normalize_Disabled_ReturnsUnchangedCopy()
    {
        float[] input = [0.1f, -0.2f, 0.3f];
        var output = VibeVoiceFrontend.Normalize(input, MakeConfig(normalize: false));
        Assert.Equal(input, output);
        Assert.NotSame(input, output);
    }

    [Fact]
    public void Normalize_QuietAudio_ScalesTowardTargetRms_NeverClips()
    {
        var rng = new Random(1);
        var input = new float[8000];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 0.02 - 0.01); // very quiet

        var output = VibeVoiceFrontend.Normalize(input, MakeConfig(normalize: true));

        Assert.All(output, s => Assert.True(float.IsFinite(s)));
        Assert.All(output, s => Assert.True(MathF.Abs(s) <= 1.0f));

        double sumSq = 0;
        foreach (float s in output) sumSq += (double)s * s;
        float rms = (float)Math.Sqrt(sumSq / output.Length);
        Assert.True(rms > 0.02f); // gained up from the quiet input
    }

    [Fact]
    public void Normalize_LoudAudio_PeakLimitsToOne()
    {
        var input = new float[1000];
        for (int i = 0; i < input.Length; i++) input[i] = MathF.Sin(i * 0.1f) * 0.9f;

        var output = VibeVoiceFrontend.Normalize(input, MakeConfig(normalize: true));

        Assert.All(output, s => Assert.True(MathF.Abs(s) <= 1.0f));
    }
}
