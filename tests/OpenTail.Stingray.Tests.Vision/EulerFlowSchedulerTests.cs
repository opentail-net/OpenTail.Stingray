
namespace OpenTail.Stingray.Tests.Vision;

public sealed class EulerFlowSchedulerTests
{
    [Fact]
    public void Linear_NumSteps_MatchesRequestedStepCount()
    {
        var sched = OpenTail.Stingray.Diffusion.EulerFlowScheduler.Linear(numSteps: 4);
        Assert.Equal(4, sched.NumSteps);
    }

    [Fact]
    public void Denoise_ConstantZeroVelocity_LeavesNoiseUnchanged()
    {
        var sched = OpenTail.Stingray.Diffusion.EulerFlowScheduler.Linear(numSteps: 4);
        float[] noise = [1f, 2f, 3f];
        float[] result = sched.Denoise(noise, (x, t) => new float[x.Length]);
        Assert.Equal(noise, result);
    }

    [Fact]
    public void Denoise_ConstantVelocity_SubtractsVelocityScaledByDt()
    {
        // 2 steps (t=1.0 -> 0.5 -> 0.0): dt=0.5 on each step
        var sched = OpenTail.Stingray.Diffusion.EulerFlowScheduler.Linear(numSteps: 2);
        float[] noise = [1.0f];
        // v = 1.0 -> x = 1.0 - 0.5*1.0 - 0.5*1.0 = 0.0
        float[] result = sched.Denoise(noise, (x, t) => [1.0f]);
        Assert.Equal(0.0f, result[0], precision: 5);
    }

    [Fact]
    public void Denoise_DoesNotMutateInputArray()
    {
        var sched = OpenTail.Stingray.Diffusion.EulerFlowScheduler.Linear(numSteps: 2);
        float[] noise = [5f, 5f];
        sched.Denoise(noise, (x, t) => [1f, 1f]);
        Assert.Equal([5f, 5f], noise);
    }

    [Fact]
    public void Denoise_InvokesProgressCallbackForEveryStep()
    {
        var sched = OpenTail.Stingray.Diffusion.EulerFlowScheduler.Linear(numSteps: 3);
        var seen = new System.Collections.Generic.List<(int step, int total)>();
        sched.Denoise([0f], (x, t) => [0f], (step, total) => seen.Add((step, total)));
        Assert.Equal([(1, 3), (2, 3), (3, 3)], seen);
    }

    [Fact]
    public void Denoise_ReceivesTimestepsDescendingFromOne()
    {
        var sched = OpenTail.Stingray.Diffusion.EulerFlowScheduler.Linear(numSteps: 4);
        var seenT = new System.Collections.Generic.List<float>();
        sched.Denoise([0f], (x, t) => { seenT.Add(t); return [0f]; });

        Assert.Equal(4, seenT.Count);
        Assert.Equal(1.0f, seenT[0], precision: 5);
        for (int i = 1; i < seenT.Count; i++)
            Assert.True(seenT[i] < seenT[i - 1], "timesteps must strictly decrease toward 0");
    }

    [Fact]
    public void Linear_Shift_ChangesTimestepSpacingVsNoShift()
    {
        var sched1x = OpenTail.Stingray.Diffusion.EulerFlowScheduler.Linear(numSteps: 4, shift: 1f);
        var sched3x = OpenTail.Stingray.Diffusion.EulerFlowScheduler.Linear(numSteps: 4, shift: 3f);
        var t1 = new System.Collections.Generic.List<float>();
        var t3 = new System.Collections.Generic.List<float>();
        sched1x.Denoise([0f], (x, t) => { t1.Add(t); return [0f]; });
        sched3x.Denoise([0f], (x, t) => { t3.Add(t); return [0f]; });
        // Same step 0 (t=1 exactly, invariant under the shift formula), later steps diverge.
        Assert.Equal(t1[0], t3[0], precision: 5);
        Assert.NotEqual(t1[1], t3[1]);
    }

    [Fact]
    public void PackLatent_ThenUnpack_RoundTripsExactly()
    {
        const int c = 2, h = 4, w = 4;
        float[] latent = new float[c * h * w];
        for (int i = 0; i < latent.Length; i++) latent[i] = i;

        float[] packed = OpenTail.Stingray.Diffusion.EulerFlowScheduler.PackLatent(latent, c, h, w, patchSize: 2);
        float[] unpacked = OpenTail.Stingray.Diffusion.EulerFlowScheduler.UnpackLatent(packed, c, h, w, patchSize: 2);

        Assert.Equal(latent, unpacked);
    }

    [Fact]
    public void PackLatent_ProducesExpectedShape()
    {
        const int c = 16, h = 4, w = 4, patchSize = 2;
        float[] latent = new float[c * h * w];
        float[] packed = OpenTail.Stingray.Diffusion.EulerFlowScheduler.PackLatent(latent, c, h, w, patchSize);
        // nPatches = (h/p)*(w/p) = 2*2 = 4; patchDim = p*p*c = 64
        Assert.Equal(4 * 64, packed.Length);
    }

    [Fact]
    public void PackLatentSpatialFirst_ThenUnpack_RoundTripsExactly()
    {
        const int c = 3, h = 4, w = 4;
        float[] latent = new float[c * h * w];
        for (int i = 0; i < latent.Length; i++) latent[i] = i * 0.5f;

        float[] packed = OpenTail.Stingray.Diffusion.EulerFlowScheduler.PackLatentSpatialFirst(latent, c, h, w, patchSize: 2);
        float[] unpacked = OpenTail.Stingray.Diffusion.EulerFlowScheduler.UnpackLatentSpatialFirst(packed, c, h, w, patchSize: 2);

        Assert.Equal(latent, unpacked);
    }

    [Fact]
    public void PackLatent_And_PackLatentSpatialFirst_ProduceDifferentOrderings()
    {
        // Same input, two different patchify conventions (channel-first vs spatial-first) —
        // the two should disagree on ordering whenever there's more than one channel.
        const int c = 2, h = 2, w = 2;
        float[] latent = [1, 2, 3, 4, 5, 6, 7, 8];

        float[] channelFirst = OpenTail.Stingray.Diffusion.EulerFlowScheduler.PackLatent(latent, c, h, w, patchSize: 2);
        float[] spatialFirst = OpenTail.Stingray.Diffusion.EulerFlowScheduler.PackLatentSpatialFirst(latent, c, h, w, patchSize: 2);

        Assert.NotEqual(channelFirst, spatialFirst);
        // Both are permutations of the same underlying values.
        Assert.Equal(channelFirst.OrderBy(x => x), spatialFirst.OrderBy(x => x));
    }

    [Fact]
    public void SampleNoise_SameSeed_IsDeterministic()
    {
        float[] a = OpenTail.Stingray.Diffusion.EulerFlowScheduler.SampleNoise(16, seed: 42);
        float[] b = OpenTail.Stingray.Diffusion.EulerFlowScheduler.SampleNoise(16, seed: 42);
        Assert.Equal(a, b);
    }

    [Fact]
    public void SampleNoise_DifferentSeeds_ProduceDifferentValues()
    {
        float[] a = OpenTail.Stingray.Diffusion.EulerFlowScheduler.SampleNoise(16, seed: 1);
        float[] b = OpenTail.Stingray.Diffusion.EulerFlowScheduler.SampleNoise(16, seed: 2);
        Assert.NotEqual(a, b);
    }

    [Fact]
    public void SampleNoise_ReturnsRequestedLength()
    {
        Assert.Equal(7, OpenTail.Stingray.Diffusion.EulerFlowScheduler.SampleNoise(7, seed: 0).Length);
        Assert.Equal(8, OpenTail.Stingray.Diffusion.EulerFlowScheduler.SampleNoise(8, seed: 0).Length);
    }

    [Fact]
    public void SampleNoise_ValuesAreFiniteAndRoughlyStandardNormal()
    {
        float[] noise = OpenTail.Stingray.Diffusion.EulerFlowScheduler.SampleNoise(4096, seed: 7);
        Assert.All(noise, v => Assert.True(float.IsFinite(v)));
        double mean = noise.Average(v => (double)v);
        Assert.InRange(mean, -0.15, 0.15);
    }
}
