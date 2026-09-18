using OpenTail.Stingray.Diffusion.Flux2;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class Flux2ConformanceTests
{
    [Fact]
    public void Flux2RoPE_BuildContextFreqs_GeneratesOrthogonal4DFrequencies()
    {
        // Real BFL scheme confirmed in-repo (examples/flux2/src/flux2/model.py): 4 axes [t,h,w,l],
        // not FLUX.1's 3-axis scheme -- see docs/087.
        int nTokens = 4;
        int[] axesDim = [8, 40, 40, 40]; // sum = 128 = HeadDim
        var positions = new int[nTokens * 4];

        positions[0] = 0; positions[1] = 0; positions[2] = 0; positions[3] = 0;   // img (t=0,h=0,w=0,l=0)
        positions[4] = 0; positions[5] = 1; positions[6] = 2; positions[7] = 0;   // img (t=0,h=1,w=2,l=0)
        positions[8] = 0; positions[9] = 0; positions[10] = 0; positions[11] = 0; // txt (t=0,h=0,w=0,l=0)
        positions[12] = 0; positions[13] = 0; positions[14] = 0; positions[15] = 1; // txt (t=0,h=0,w=0,l=1)

        var (cos, sin) = Flux2RoPE.BuildContextFreqs(positions, nTokens, axesDim, theta: 2000f);

        Assert.Equal(nTokens * 128, cos.Length);
        Assert.Equal(nTokens * 128, sin.Length);

        for (int i = 0; i < cos.Length; i++)
        {
            float mag = cos[i] * cos[i] + sin[i] * sin[i];
            Assert.True(MathF.Abs(mag - 1.0f) < 1e-4f, $"Magnitude at {i} was {mag}");
        }
    }

    [Fact]
    public void Flux2RoPE_BuildContextFreqs_RejectsWrongAxisCount()
    {
        // FLUX.1's old 3-axis scheme must be rejected now -- real FLUX.2 is 4-axis.
        int[] wrongAxesDim = [16, 56, 56];
        var positions = new int[3];
        Assert.Throws<ArgumentException>(() => Flux2RoPE.BuildContextFreqs(positions, 1, wrongAxesDim));
    }

    [Fact]
    public void Flux2DiT_Forward_EvaluatesTextToImageNoReferenceConditioning()
    {
        var @params = new Flux2Params
        {
            HiddenSize = 64,
            NumHeads = 4,
            DepthDoubleBlocks = 1,
            DepthSingleBlocks = 1,
            AxesDim = [4, 4, 4, 4], // sum = 16 = HeadDim
            InChannels = 8,
            OutChannels = 8,
            ContextInDim = 32,
            Theta = 2000f,
        };

        var dit = new Flux2DiT(@params);

        int nTarget = 4;
        var targetLatent = new float[nTarget * @params.InChannels];
        var targetPos = new int[nTarget * 4]; // (t=0,h,w,l=0) per token, all-zero is fine structurally

        int nTxt = 8;
        var txtEmbeds = new float[nTxt * @params.ContextInDim];
        var txtPos = new int[nTxt * 4];
        for (int i = 0; i < nTxt; i++) txtPos[i * 4 + 3] = i; // l = sequential index

        var pooledEmbed = Array.Empty<float>(); // FLUX.2 has no CLIP pooled conditioning (VecInDim=0)

        float[] velocity = dit.Forward(
            targetLatent, targetPos,
            refLatents: null, refPositions: null,
            txtEmbeds, txtPos,
            pooledEmbed,
            timestep: 0.5f,
            guidance: 3.5f);

        Assert.NotNull(velocity);
        Assert.Equal(nTarget * @params.OutChannels, velocity.Length);
    }

    [Fact]
    public void Flux2DiT_Forward_ThrowsOnReferenceImages_NotYetImplemented()
    {
        // Reference-image conditioning needs the real causal_attn_fn token-isolation attention,
        // a documented, not-yet-implemented gap (docs/087) -- must fail loudly, not silently run
        // the plain no-ref data flow and produce a wrong result.
        var @params = new Flux2Params
        {
            HiddenSize = 16,
            NumHeads = 2,
            DepthDoubleBlocks = 1,
            DepthSingleBlocks = 1,
            AxesDim = [2, 2, 2, 2],
            InChannels = 4,
            OutChannels = 4,
            ContextInDim = 8,
        };
        var dit = new Flux2DiT(@params);

        var targetLatent = new float[4 * @params.InChannels];
        var targetPos = new int[4 * 4];
        var refLatents = new List<float[]> { new float[4 * @params.InChannels] };
        var refPositions = new List<int[]> { new int[4 * 4] };
        var txtEmbeds = new float[2 * @params.ContextInDim];
        var txtPos = new int[2 * 4];

        Assert.Throws<NotSupportedException>(() => dit.Forward(
            targetLatent, targetPos,
            refLatents, refPositions,
            txtEmbeds, txtPos,
            Array.Empty<float>(),
            timestep: 0.5f));
    }

    [Fact]
    public void Flux2Pipeline_GeneratesImage_TextToImageOnly()
    {
        var @params = new Flux2Params
        {
            HiddenSize = 32,
            NumHeads = 2,
            DepthDoubleBlocks = 1,
            DepthSingleBlocks = 1,
            AxesDim = [4, 4, 4, 4], // sum = 16 = HeadDim
            InChannels = 8,
            OutChannels = 8,
            ContextInDim = 16,
            Theta = 2000f,
        };

        using var pipeline = new Flux2Pipeline(@params);

        string tempPng = Path.Combine(Path.GetTempPath(), $"flux2_test_{Guid.NewGuid():N}.png");
        try
        {
            var request = new Flux2GenerationRequest
            {
                Prompt = "A golden retriever sitting in a fantasy garden",
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
