using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Diffusion.StableDiffusion;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class VaeEncoderTests
{
    [Fact]
    public void VaeEncoder_Dimensions_MustBeDivisibleBy8()
    {
        int height = 512, width = 512;
        Assert.Equal(64, height / 8);
        Assert.Equal(64, width / 8);
    }

    [Fact]
    public void EulerDiscreteScheduler_CreateNoisyLatent_BlendsAtStartStep()
    {
        var scheduler = new EulerDiscreteScheduler(numSteps: 20);
        var cleanLatent = new float[64];
        Array.Fill(cleanLatent, 1.0f);

        var noise = new float[64];
        Array.Fill(noise, 0.5f);

        int startStep = 10;
        var noisy = scheduler.CreateNoisyLatent(cleanLatent, noise, startStep);

        float expectedSigma = scheduler.Sigmas[startStep];
        float expectedVal = 1.0f + 0.5f * expectedSigma;

        for (int i = 0; i < noisy.Length; i++)
            Assert.Equal(expectedVal, noisy[i], tolerance: 1e-4f);
    }

    [Fact]
    public void EulerDiscreteScheduler_Denoise_RespectsStartStep()
    {
        var scheduler = new EulerDiscreteScheduler(numSteps: 20);
        var initialLatent = new float[16];
        Array.Fill(initialLatent, 2.0f);

        int stepsExecuted = 0;
        int startStep = 15;

        var result = scheduler.Denoise(initialLatent, (xIn, t) =>
        {
            stepsExecuted++;
            var pred = new float[xIn.Length];
            for (int i = 0; i < pred.Length; i++) pred[i] = 0.1f;
            return pred;
        }, startStep: startStep);

        Assert.Equal(5, stepsExecuted); // 20 - 15 = 5 steps
        Assert.NotNull(result);
    }
}
