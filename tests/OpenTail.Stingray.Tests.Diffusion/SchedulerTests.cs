using OpenTail.Stingray.Diffusion.StableDiffusion;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class SchedulerTests
{
    [Fact]
    public void EulerDiscreteScheduler_BuildsCorrectSchedule()
    {
        var scheduler = new EulerDiscreteScheduler(numInferenceSteps: 20);

        Assert.Equal(20, scheduler.NumSteps);
        Assert.Equal(21, scheduler.Sigmas.Length);
        Assert.Equal(20, scheduler.Timesteps.Length);

        // First timestep is 999, last is 0
        Assert.Equal(999f, scheduler.Timesteps[0]);
        Assert.Equal(0f, scheduler.Timesteps[^1]);

        // Sigmas are descending and end with 0
        Assert.True(scheduler.Sigmas[0] > 10.0f); // ~14.61 for SD1.5
        Assert.Equal(0f, scheduler.Sigmas[^1]);

        for (int i = 0; i < scheduler.Sigmas.Length - 1; i++)
        {
            Assert.True(scheduler.Sigmas[i] > scheduler.Sigmas[i + 1], $"Sigma at {i} ({scheduler.Sigmas[i]}) not > sigma at {i+1} ({scheduler.Sigmas[i+1]})");
        }
    }

    [Fact]
    public void EulerDiscreteScheduler_CombineGuidance_ComputesCFG()
    {
        var scheduler = new EulerDiscreteScheduler(numInferenceSteps: 10);
        var cond = new float[] { 2.0f, -1.0f, 0.5f };
        var uncond = new float[] { 1.0f, 0.0f, 0.0f };

        // Guidance scale = 7.5: uncond + 7.5 * (cond - uncond)
        // 0: 1.0 + 7.5 * (2.0 - 1.0) = 8.5
        // 1: 0.0 + 7.5 * (-1.0 - 0.0) = -7.5
        // 2: 0.0 + 7.5 * (0.5 - 0.0) = 3.75
        var combined = scheduler.CombineGuidance(cond, uncond, 7.5f);

        Assert.Equal(8.5f, combined[0], precision: 4);
        Assert.Equal(-7.5f, combined[1], precision: 4);
        Assert.Equal(3.75f, combined[2], precision: 4);
    }
}
