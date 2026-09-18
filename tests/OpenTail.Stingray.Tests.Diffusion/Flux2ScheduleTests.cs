namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Real numeric checks of the FLUX.2 timestep schedule fix (docs/087) -- confirms the real
/// `get_schedule`/`generalized_time_snr_shift` boundary and monotonicity properties described in
/// `examples/flux2/src/flux2/sampling.py`.
/// </summary>
public sealed class Flux2ScheduleTests
{
    [Fact]
    public void GetSchedule_IsBoundaryPreservingAndMonotonicallyDecreasing()
    {
        var schedule = OpenTail.Stingray.Diffusion.Flux2.Flux2Schedule.GetSchedule(numSteps: 20, imageSeqLen: 1024);

        Assert.Equal(21, schedule.Length);
        Assert.Equal(1.0f, schedule[0], precision: 5);
        Assert.Equal(0.0f, schedule[^1], precision: 5);

        for (int i = 1; i < schedule.Length; i++)
            Assert.True(schedule[i] < schedule[i - 1], $"schedule[{i}]={schedule[i]} must be < schedule[{i - 1}]={schedule[i - 1]}");
    }

    [Fact]
    public void GetSchedule_IsNonLinear_DiffersFromPlainLinearRamp()
    {
        // The real bug (docs/087): a previous version silently used a plain linear ramp instead
        // of this real shifted schedule. Confirm the two are NOT the same (i.e. the shift is real
        // and has a measurable effect), for a realistic image_seq_len.
        var schedule = OpenTail.Stingray.Diffusion.Flux2.Flux2Schedule.GetSchedule(numSteps: 20, imageSeqLen: 1024);

        bool anyDifferentFromLinear = false;
        for (int i = 0; i <= 20; i++)
        {
            float linear = 1.0f - i / 20.0f;
            if (MathF.Abs(schedule[i] - linear) > 1e-3f) { anyDifferentFromLinear = true; break; }
        }
        Assert.True(anyDifferentFromLinear, "Real FLUX.2 schedule should differ from a plain linear ramp for a realistic image size.");
    }

    [Fact]
    public void ComputeEmpiricalMu_LargerImages_ProduceDifferentMu()
    {
        // Real formula is resolution-dependent -- confirm two different image_seq_len values
        // (below the 4300 threshold) produce different mu, i.e. the schedule genuinely adapts.
        float muSmall = OpenTail.Stingray.Diffusion.Flux2.Flux2Schedule.ComputeEmpiricalMu(imageSeqLen: 256, numSteps: 20);
        float muLarge = OpenTail.Stingray.Diffusion.Flux2.Flux2Schedule.ComputeEmpiricalMu(imageSeqLen: 4096, numSteps: 20);
        Assert.NotEqual(muSmall, muLarge);
    }
}
