using OpenTail.Stingray.Diffusion.Flux2;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class Flux2ConformanceTests
{
    [Fact]
    public void Flux2RoPE_BuildContextFreqs_GeneratesOrthogonal3DFrequencies()
    {
        int nTokens = 4;
        int[] axesDim = [16, 56, 56]; // sum = 128 = HeadDim
        var positions = new int[nTokens * 3];

        positions[0] = 0; positions[1] = 0; positions[2] = 0; // Target (0, 0, 0)
        positions[3] = 0; positions[4] = 1; positions[5] = 2; // Target (0, 1, 2)
        positions[6] = 1; positions[7] = 0; positions[8] = 0; // Ref 1 (1, 0, 0)
        positions[9] = 2; positions[10] = 3; positions[11] = 4; // Ref 2 (2, 3, 4)

        var (cos, sin) = Flux2RoPE.BuildContextFreqs(positions, nTokens, axesDim);

        Assert.Equal(nTokens * 128, cos.Length);
        Assert.Equal(nTokens * 128, sin.Length);

        for (int i = 0; i < cos.Length; i++)
        {
            float mag = cos[i] * cos[i] + sin[i] * sin[i];
            Assert.True(MathF.Abs(mag - 1.0f) < 1e-4f, $"Magnitude at {i} was {mag}");
        }
    }

    [Fact]
    public void Flux2DiT_Forward_EvaluatesMultiReferenceConditioning()
    {
        var @params = new Flux2Params
        {
            HiddenSize = 64,
            NumHeads = 4,
            DepthDoubleBlocks = 1,
            DepthSingleBlocks = 1,
            AxesDim = [8, 4, 4], // sum = 16 = HeadDim
            InChannels = 8,
            OutChannels = 8,
            ContextInDim = 32,
            VecInDim = 16
        };

        var dit = new Flux2DiT(@params);

        int nTarget = 4;
        var targetLatent = new float[nTarget * @params.InChannels];
        var targetPos = new int[nTarget * 3];

        int nRef = 4;
        var refLatents = new List<float[]> { new float[nRef * @params.InChannels] };
        var refPositions = new List<int[]> { new int[nRef * 3] };

        var txtEmbeds = new float[8 * @params.ContextInDim];
        var pooledEmbed = new float[@params.VecInDim];

        float[] velocity = dit.Forward(
            targetLatent, targetPos,
            refLatents, refPositions,
            txtEmbeds, pooledEmbed,
            timestep: 0.5f,
            guidance: 3.5f);

        Assert.NotNull(velocity);
        Assert.Equal(nTarget * @params.OutChannels, velocity.Length);
    }

    [Fact]
    public void Flux2Pipeline_GeneratesImageWithReferences()
    {
        var @params = new Flux2Params
        {
            HiddenSize = 32,
            NumHeads = 2,
            DepthDoubleBlocks = 1,
            DepthSingleBlocks = 1,
            AxesDim = [4, 6, 6], // sum = 16 = HeadDim
            InChannels = 8,
            OutChannels = 8,
            ContextInDim = 16,
            VecInDim = 16
        };

        using var pipeline = new Flux2Pipeline(@params);

        string tempPng = Path.Combine(Path.GetTempPath(), $"flux2_test_{Guid.NewGuid():N}.png");
        try
        {
            var request = new Flux2GenerationRequest
            {
                Prompt = "A golden retriever sitting in a fantasy garden",
                ReferenceImagesRgb = new List<float[]> { new float[32 * 32 * 3] },
                Width = 32,
                Height = 32,
                Steps = 2,
                Guidance = 3.0f,
                OutputPath = tempPng
            };

            float[] rgb = pipeline.Generate(request);

            Assert.NotNull(rgb);
            Assert.Equal(32 * 32 * 3, rgb.Length);
            Assert.True(File.Exists(tempPng));
        }
        finally
        {
            if (File.Exists(tempPng)) File.Delete(tempPng);
        }
    }
}
