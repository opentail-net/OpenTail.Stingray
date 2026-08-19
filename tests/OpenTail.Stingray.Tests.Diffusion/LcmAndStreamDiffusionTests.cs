using OpenTail.Stingray.Diffusion;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class LcmAndStreamDiffusionTests
{
    [Fact]
    public void LcmScheduler_BuildTimesteps_ConstructsEvenlySpacedSchedule()
    {
        var scheduler = new LcmScheduler(numTrainTimesteps: 1000);
        int[] ts4 = scheduler.BuildTimesteps(4);

        Assert.Equal(4, ts4.Length);
        Assert.Equal(999, ts4[0]);
        Assert.Equal(749, ts4[1]);
        Assert.Equal(499, ts4[2]);
        Assert.Equal(249, ts4[3]);

        // Boundary scalings at t=999
        var (cSkip, cOut) = scheduler.GetBoundaryScalings(999);
        Assert.InRange(cSkip, 0.0f, 1.0f);
        Assert.InRange(cOut, 0.0f, 1.0f);
        Assert.True(cOut > cSkip, "At high timesteps, cOut should dominate cSkip");

        // Boundary scalings at t=0
        var (cSkip0, cOut0) = scheduler.GetBoundaryScalings(0);
        Assert.Equal(1.0f, cSkip0, precision: 3);
        Assert.Equal(0.0f, cOut0, precision: 3);
    }

    [Fact]
    public void LcmScheduler_Step_UpdatesSampleCorrectly()
    {
        var scheduler = new LcmScheduler(numTrainTimesteps: 1000);
        int size = 16;
        var sample = new float[size];
        var modelOut = new float[size];

        Array.Fill(sample, 1.0f);
        Array.Fill(modelOut, -0.5f);

        scheduler.Step(sample, modelOut, timestep: 500, prevTimestep: 0);

        // At prevTimestep=0, sample becomes exact predicted x_0
        var (cSkip, cOut) = scheduler.GetBoundaryScalings(500);
        float expectedX0 = cSkip * 1.0f + cOut * (-0.5f);

        for (int i = 0; i < size; i++)
        {
            Assert.Equal(expectedX0, sample[i], precision: 4);
        }
    }

    [Fact]
    public void StreamBatchPipeline_ShiftsAndPopsFramesContinuously()
    {
        int batchSize = 4;
        int latentElements = 32;
        int[] timesteps = [999, 749, 499, 249];

        var pipeline = new StreamBatchPipeline(batchSize, latentElements, timesteps);

        var newFrame = new float[latentElements];
        Array.Fill(newFrame, 1.5f);

        var batchedInput = new float[batchSize * latentElements];
        var batchedPreds = new float[batchSize * latentElements];
        var outputFrame = new float[latentElements];

        // Frame 1
        pipeline.PrepareBatchInput(newFrame, batchedInput);
        Array.Fill(batchedPreds, 0.1f);
        pipeline.StepAndPop(batchedPreds, outputFrame);

        Assert.True(pipeline.IsWarmedUp);
        Assert.Equal(latentElements, outputFrame.Length);

        // Frame 2 with R-CFG
        var uncondResidual = new float[latentElements];
        Array.Fill(uncondResidual, 0.05f);
        pipeline.SetUncondResidual(uncondResidual);

        pipeline.PrepareBatchInput(newFrame, batchedInput);
        pipeline.StepAndPop(batchedPreds, outputFrame, guidanceScale: 2.0f);

        Assert.Equal(latentElements, outputFrame.Length);
    }
}
