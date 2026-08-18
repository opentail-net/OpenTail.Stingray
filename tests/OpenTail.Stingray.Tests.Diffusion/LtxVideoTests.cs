using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Diffusion.LTXVideo;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class LtxVideoTests
{
    [Fact]
    public void LtxVideoRoPE_ComputeContinuous3DRoPE_ValidatesShapesAndRanges()
    {
        int numFrames = 3;
        int patchH = 16;
        int patchW = 16;
        int headDim = 64;

        var (cos, sin) = LtxVideoRoPE.ComputeContinuous3DRoPE(numFrames, patchH, patchW, headDim);

        int totalTokens = numFrames * patchH * patchW;
        Assert.Equal(totalTokens * headDim, cos.Length);
        Assert.Equal(totalTokens * headDim, sin.Length);

        for (int i = 0; i < cos.Length; i++)
        {
            Assert.InRange(cos[i], -1.0001f, 1.0001f);
            Assert.InRange(sin[i], -1.0001f, 1.0001f);
        }
    }

    [Fact]
    public void LtxVideoModel_Forward_ComputesVelocityPrediction()
    {
        var model = new LtxVideoModel(numLayers: 2, hiddenSize: 64, numHeads: 2, headDim: 32);
        int numFrames = 2;
        int patchH = 4;
        int patchW = 4;
        int inChannels = 128;
        int numTokens = numFrames * patchH * patchW;

        var latents = new float[numTokens * inChannels];
        for (int i = 0; i < latents.Length; i++)
            latents[i] = 0.05f * (i % 17 - 8);

        var context = new float[16 * 4096];
        for (int i = 0; i < context.Length; i++)
            context[i] = 0.01f * (i % 5);

        var velocity = model.Forward(latents, timestep: 500f, context, numFrames, patchH, patchW);

        Assert.Equal(numTokens * model.OutChannels, velocity.Length);
        for (int i = 0; i < velocity.Length; i++)
        {
            Assert.False(float.IsNaN(velocity[i]));
            Assert.False(float.IsInfinity(velocity[i]));
        }
    }

    [Fact]
    public void LtxVideoPipeline_GenerateVideo_ProducesValidFrames()
    {
        var model = new LtxVideoModel(numLayers: 2, hiddenSize: 64, numHeads: 2, headDim: 32);
        using var pipeline = new LtxVideoPipeline(model, temporalScale: 8, spatialScale: 32);

        var frames = pipeline.GenerateVideo(
            prompt: "A high speed car drifting on neon wet asphalt at night",
            width: 64,
            height: 64,
            numFrames: 1,
            steps: 2,
            seed: 42);

        Assert.NotEmpty(frames);
        Assert.Equal(3 * 64 * 64, frames[0].Length);
    }
}
