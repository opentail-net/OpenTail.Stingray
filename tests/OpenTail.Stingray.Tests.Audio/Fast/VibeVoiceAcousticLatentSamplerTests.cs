using OpenTail.Stingray.Audio.VibeVoice;

namespace OpenTail.Stingray.Tests.Audio.Fast;

/// <summary>Structural tests for <see cref="VibeVoiceAcousticLatentSampler"/>.</summary>
public sealed class VibeVoiceAcousticLatentSamplerTests
{
    [Fact]
    public void Sample_ProducesFiniteOutput_WithSameShapeAsMean()
    {
        var mean = new float[][]
        {
            [1f, 2f, 3f, 4f],
            [-1f, -2f, -3f, -4f],
        };

        var sampled = VibeVoiceAcousticLatentSampler.Sample(mean, fixStd: 0.5f, new Random(1));

        Assert.Equal(mean.Length, sampled.Length);
        Assert.Equal(mean[0].Length, sampled[0].Length);
        foreach (var row in sampled)
            Assert.All(row, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void Sample_IsDeterministic_ForTheSameSeed()
    {
        var mean = new float[][] { [1f, 2f, 3f] };

        var a = VibeVoiceAcousticLatentSampler.Sample(mean, 0.5f, new Random(42));
        var b = VibeVoiceAcousticLatentSampler.Sample(mean, 0.5f, new Random(42));

        Assert.Equal(a[0], b[0]);
    }

    [Fact]
    public void Sample_DiffersFromMean_WhenStdIsNonzero()
    {
        var mean = new float[][] { [0f, 0f, 0f, 0f, 0f] };
        var sampled = VibeVoiceAcousticLatentSampler.Sample(mean, fixStd: 1.0f, new Random(3));
        Assert.NotEqual(mean[0], sampled[0]);
    }
}
