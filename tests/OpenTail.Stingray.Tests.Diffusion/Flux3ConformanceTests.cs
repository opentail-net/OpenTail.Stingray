using OpenTail.Stingray.Diffusion.Flux3;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class Flux3ConformanceTests
{
    [Fact]
    public void Flux3RoPE_BuildVideoFreqs_GeneratesCorrectDimensions()
    {
        int nTokens = 8;
        int[] axesDim = [32, 48, 48]; // sum = 128
        var positions = new int[nTokens * 3];
        for (int i = 0; i < nTokens; i++)
        {
            positions[i * 3 + 0] = i;      // t
            positions[i * 3 + 1] = i * 2;  // y
            positions[i * 3 + 2] = i * 3;  // x
        }

        var (cos, sin) = Flux3RoPE.BuildVideoFreqs(positions, nTokens, axesDim);

        Assert.Equal(nTokens * 128, cos.Length);
        Assert.Equal(nTokens * 128, sin.Length);

        // Cos^2 + Sin^2 should equal 1 for all entries
        for (int i = 0; i < cos.Length; i++)
        {
            float mag = cos[i] * cos[i] + sin[i] * sin[i];
            Assert.True(MathF.Abs(mag - 1.0f) < 1e-4f, $"RoPE magnitude at {i} was {mag}");
        }
    }

    [Fact]
    public void Flux3KvCache_StoresAndRetrievesLayerTensors()
    {
        var kvCache = new Flux3KvCache(numDoubleBlocks: 2, numSingleBlocks: 4);

        float[] k0 = [1f, 2f, 3f, 4f];
        float[] v0 = [5f, 6f, 7f, 8f];

        var layer0 = kvCache.GetDoubleLayer(0);
        Assert.False(layer0.HasCachedTokens);

        layer0.Store(k0, v0, numTokens: 2);
        Assert.True(layer0.HasCachedTokens);
        Assert.Equal(2, layer0.NumRefTokens);

        var (retK, retV) = layer0.Retrieve();
        Assert.Equal(k0, retK);
        Assert.Equal(v0, retV);

        kvCache.Clear();
        Assert.False(layer0.HasCachedTokens);
    }

    [Fact]
    public void Flux3SelfFlowScheduler_BuildsMonotonicTimestepSchedule()
    {
        var scheduler = new Flux3SelfFlowScheduler(shift: 3.0f);
        int steps = 10;
        float[] ts = scheduler.BuildTimesteps(steps);

        Assert.Equal(steps + 1, ts.Length);
        Assert.Equal(1.0f, ts[0], precision: 4);
        Assert.Equal(0.0f, ts[^1], precision: 4);

        // Verify strictly monotonic decrease
        for (int i = 0; i < steps; i++)
        {
            Assert.True(ts[i] > ts[i + 1], $"Timestep {i} ({ts[i]}) not greater than {i+1} ({ts[i+1]})");
        }
    }

    [Fact]
    public void Flux3Pipeline_GeneratesMultimodalVideoAndAudio()
    {
        var @params = new Flux3Params
        {
            HiddenSize = 128,
            NumHeads = 4,
            DepthDoubleBlocks = 1,
            DepthSingleBlocks = 1,
            VideoAxesDim = [16, 8, 8], // sum = 32 = HeadDim
            AudioAxesDim = [16, 16],
            InVideoChannels = 16,
            OutVideoChannels = 16,
            InAudioChannels = 8,
            OutAudioChannels = 8,
            ContextInDim = 64,
            VecInDim = 32
        };

        using var pipeline = new Flux3Pipeline(@params);

        string tempGif = Path.Combine(Path.GetTempPath(), $"flux3_test_{Guid.NewGuid():N}.gif");
        try
        {
            var request = new Flux3GenerationRequest
            {
                Prompt = "A cinematic waterfall flowing with natural water sounds",
                Width = 32,
                Height = 32,
                VideoFrames = 4,
                Steps = 2,
                GenerateAudio = true,
                OutputPath = tempGif
            };

            var (frames, audio) = pipeline.Generate(request);

            Assert.Equal(4, frames.Count);
            Assert.NotNull(audio);
            Assert.True(audio.Length > 0);
            Assert.True(File.Exists(tempGif));
        }
        finally
        {
            if (File.Exists(tempGif)) File.Delete(tempGif);
        }
    }
}
