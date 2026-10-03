using System;
using OpenTail.Stingray.Diffusion.DiffusionGemma;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed unsafe class DiffusionGemmaTests
{
    [Fact]
    public void Config_DimensionsAndHyperparameters_MatchUpstreamSpecification()
    {
        var config = new DiffusionGemmaConfig();

        Assert.Equal(30, config.NumLayers);
        Assert.Equal(2816, config.HiddenDim);
        Assert.Equal(16, config.SlidingNumQHeads);
        Assert.Equal(8, config.SlidingNumKvHeads);
        Assert.Equal(256, config.SlidingHeadDim);
        Assert.Equal(1024, config.SlidingWindowSize);

        Assert.Equal(16, config.FullNumQHeads);
        Assert.Equal(2, config.FullNumKvHeads);
        Assert.Equal(512, config.FullHeadDim);

        Assert.Equal(2112, config.DenseIntermediateDim);
        Assert.Equal(128, config.NumExperts);
        Assert.Equal(8, config.NumExpertsUsed);
        Assert.Equal(704, config.ExpertIntermediateDim);

        Assert.Equal(48, config.MaxDenoisingSteps);
        Assert.Equal(256, config.CanvasLength);
        Assert.Equal(0.8f, config.TemperatureMax);
        Assert.Equal(0.408f, config.TemperatureMin);
        Assert.Equal(0.1f, config.EntropyBudgetNats);
        Assert.Equal(0.005f, config.ConvergenceEntropyThreshold);

        // Verify full attention every 6th layer (0-indexed: 5, 11, 17, 23, 29)
        int[] expectedFull = [5, 11, 17, 23, 29];
        for (int l = 0; l < 30; l++)
        {
            bool isFull = Array.IndexOf(expectedFull, l) >= 0;
            Assert.Equal(isFull, config.IsFullAttention(l));
        }

        // Embedding scale factor: sqrt(2816) ~ 53.065996
        float expectedScale = MathF.Sqrt(2816.0f);
        Assert.Equal(expectedScale, config.EmbedScale, 1e-4f);
    }

    [Fact]
    public void Sampler_TemperatureDecay_FollowsLinearSchedule()
    {
        var config = new DiffusionGemmaConfig();
        var sampler = new DiffusionGemmaSampler(config);

        // Step 0: TemperatureMax = 0.8
        float t0 = sampler.GetTemperature(0);
        Assert.Equal(0.8f, t0, 1e-5f);

        // Step 47 (last step in 48-step schedule): TemperatureMin = 0.408
        float tLast = sampler.GetTemperature(47);
        Assert.Equal(0.408f, tLast, 1e-5f);

        // Monotonically non-increasing
        for (int step = 0; step < 47; step++)
        {
            float tCurrent = sampler.GetTemperature(step);
            float tNext = sampler.GetTemperature(step + 1);
            Assert.True(tCurrent >= tNext, $"Step {step}: {tCurrent} should be >= {tNext}");
        }
    }

    [Fact]
    public void Sampler_ShannonEntropy_ComputesCorrectNats()
    {
        // Uniform distribution over 2 outcomes: H = - 2 * (0.5 * ln(0.5)) = ln(2) ~ 0.693147 nats
        float[] pUniform2 = [0.5f, 0.5f];
        float hUniform2 = DiffusionGemmaSampler.ComputeEntropyNats(pUniform2);
        Assert.Equal(MathF.Log(2.0f), hUniform2, 1e-4f);

        // Uniform distribution over 4 outcomes: H = ln(4) = 2 * ln(2) ~ 1.386294 nats
        float[] pUniform4 = [0.25f, 0.25f, 0.25f, 0.25f];
        float hUniform4 = DiffusionGemmaSampler.ComputeEntropyNats(pUniform4);
        Assert.Equal(MathF.Log(4.0f), hUniform4, 1e-4f);

        // Deterministic distribution (one-hot): H = 0 nats
        float[] pDeterministic = [1.0f, 0.0f, 0.0f, 0.0f];
        float hDet = DiffusionGemmaSampler.ComputeEntropyNats(pDeterministic);
        Assert.Equal(0.0f, hDet, 1e-5f);
    }

    [Fact]
    public void SelfConditioning_SoftEmbeddings_WeightedAndScaled()
    {
        // Setup simple vocabulary: 3 tokens, hiddenDim = 4
        int vocabSize = 3;
        int hiddenDim = 4;
        float[] embedWeights =
        [
            1.0f, 0.0f, 0.0f, 0.0f, // token 0
            0.0f, 1.0f, 0.0f, 0.0f, // token 1
            0.0f, 0.0f, 1.0f, 0.0f  // token 2
        ];

        // Soft probabilities for 1 token: [0.5, 0.5, 0.0]
        float[] probs = [0.5f, 0.5f, 0.0f];
        float[] softEmbed = new float[hiddenDim];
        float scale = 2.0f; // test scale

        fixed (float* wPtr = embedWeights)
        {
            DiffusionGemmaSelfConditioning.ComputeSoftEmbedding(probs, wPtr, vocabSize, hiddenDim, scale, softEmbed);
        }

        // Expected soft embedding = (0.5 * e0 + 0.5 * e1 + 0.0 * e2) * 2.0
        // = [0.5, 0.5, 0.0, 0.0] * 2.0 = [1.0, 1.0, 0.0, 0.0]
        Assert.Equal(1.0f, softEmbed[0], 1e-5f);
        Assert.Equal(1.0f, softEmbed[1], 1e-5f);
        Assert.Equal(0.0f, softEmbed[2], 1e-5f);
        Assert.Equal(0.0f, softEmbed[3], 1e-5f);
    }

    [Fact]
    public void SelfConditioning_MlpProjectionAndCanvasAddition()
    {
        int hiddenDim = 4;
        int interDim = 8;
        float[] softEmbed = [1.0f, 1.0f, 1.0f, 1.0f];
        float[] wGate = new float[interDim * hiddenDim];
        float[] wUp = new float[interDim * hiddenDim];
        float[] wDown = new float[hiddenDim * interDim];

        Array.Fill(wGate, 0.1f);
        Array.Fill(wUp, 0.1f);
        Array.Fill(wDown, 0.05f);

        float[] scOut = new float[hiddenDim];
        float[] canvas = [0.5f, 0.5f, 0.5f, 0.5f];
        float[] originalCanvas = (float[])canvas.Clone();

        fixed (float* gPtr = wGate, uPtr = wUp, dPtr = wDown)
        {
            DiffusionGemmaSelfConditioning.ApplySelfCondMlp(
                softEmbed, gPtr, uPtr, dPtr, hiddenDim, interDim, scOut);
        }

        DiffusionGemmaSelfConditioning.InjectSelfConditioning(canvas, scOut);

        // Canvas should be modified in-place by adding projected self-conditioning embedding
        for (int i = 0; i < hiddenDim; i++)
        {
            Assert.NotEqual(originalCanvas[i], canvas[i]);
            Assert.True(float.IsFinite(canvas[i]));
        }
    }

    [Fact]
    public void Sampler_Step_AdvancesAndAcceptsTokens()
    {
        var config = new DiffusionGemmaConfig
        {
            CanvasLength = 4,
            VocabSize = 8,
            MaxDenoisingSteps = 10,
            EntropyBudgetNats = 1.0f
        };
        var sampler = new DiffusionGemmaSampler(config, seed: 1234);

        // Create synthetic logits: 4 positions x 8 vocab
        float[] logits = new float[config.CanvasLength * config.VocabSize];
        // Give position 0 very sharp preference for token 2
        logits[0 * config.VocabSize + 2] = 20.0f;
        // Position 1 has weak preference
        logits[1 * config.VocabSize + 5] = 1.0f;
        // Position 2 and 3 flat

        int[] prevTokens = new int[config.CanvasLength];
        bool[] prevAccepted = new bool[config.CanvasLength];

        var result = sampler.Step(0, logits, prevTokens, prevAccepted);

        Assert.Equal(config.CanvasLength, result.Accepted.Length);
        Assert.Equal(config.CanvasLength, result.Tokens.Length);
        // Position 0 had low entropy so it should be accepted
        Assert.True(result.Accepted[0]);
        Assert.Equal(2, result.Tokens[0]);
        Assert.True(result.MeanEntropy > 0f);
    }
}
