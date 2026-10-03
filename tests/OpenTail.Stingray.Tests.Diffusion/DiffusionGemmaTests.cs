using System;
using System.Collections.Generic;
using System.IO;
using OpenTail.Stingray.Core;
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
        Assert.Equal(2112, config.SelfCondIntermediateDim);
        Assert.Equal(262144, config.VocabSize);

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
    public void SelfConditioning_UsesPreNormAndWeightlessPostNorm()
    {
        int hiddenDim = 4;
        int interDim = 8;
        float[] softEmbed = [2.0f, -1.0f, 0.5f, 1.5f];
        float[] preNormW = [1.0f, 1.0f, 1.0f, 1.0f];
        float[] wGate = new float[interDim * hiddenDim];
        float[] wUp = new float[interDim * hiddenDim];
        float[] wDown = new float[hiddenDim * interDim];
        Array.Fill(wGate, 0.1f);
        Array.Fill(wUp, 0.1f);
        Array.Fill(wDown, 0.05f);

        float[] canvas = [1.0f, 2.0f, 3.0f, 4.0f];

        fixed (float* pNorm = preNormW, g = wGate, u = wUp, d = wDown)
        {
            DiffusionGemmaSelfConditioning.ApplySelfConditioningWithNorms(
                softEmbed, pNorm, g, u, d, hiddenDim, interDim, canvas);
        }

        // Verify output is post-normalized (RMS norm should be ~1.0)
        float sumSq = 0f;
        for (int i = 0; i < hiddenDim; i++)
        {
            sumSq += canvas[i] * canvas[i];
            Assert.True(float.IsFinite(canvas[i]));
        }
        float rms = MathF.Sqrt(sumSq / hiddenDim);
        Assert.Equal(1.0f, rms, 1e-4f);
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

    [Fact]
    public void Sampler_Stability_UsesArgmaxHistory_NotTheRenoisedCanvas()
    {
        // Positions 0,1 are sharp (accepted); positions 2,3 are flat so they stay un-accepted and get
        // re-noised with random tokens. Argmax predictions are identical across the two steps, so the
        // sampler must report stability and stop (mean entropy forced under the threshold).
        var config = new DiffusionGemmaConfig
        {
            CanvasLength = 4,
            VocabSize = 8,
            MaxDenoisingSteps = 10,
            EntropyBudgetNats = 1.0f,
            ConvergenceEntropyThreshold = 10.0f,
        };
        var sampler = new DiffusionGemmaSampler(config, seed: 7);

        float[] logits = new float[4 * 8];
        logits[0 * 8 + 2] = 20f;
        logits[1 * 8 + 5] = 20f;
        logits[2 * 8 + 1] = 0.5f;  // flat-ish but a definite argmax
        logits[3 * 8 + 6] = 0.5f;

        var first = sampler.Step(0, logits, [-1, -1, -1, -1], new bool[4]);
        Assert.False(first.ShouldStop);                        // no previous argmax => not stable
        Assert.Equal([2, 5, 1, 6], first.ArgmaxTokens);

        var second = sampler.Step(1, logits, first.ArgmaxTokens, first.Accepted);
        Assert.Equal([2, 5, 1, 6], second.ArgmaxTokens);
        Assert.True(second.ShouldStop);                        // stable on argmax even though the canvas was re-noised
    }

    [Fact]
    public void Pipeline_SyntheticForwardPass_RunsPromptAndDenoisingLoop()
    {
        const int hiddenDim = 16;
        const int numLayers = 2;
        const int numHeads = 2;
        const int headDim = 8;
        const int vocabSize = 16;
        const int canvasLen = 4;
        const int interDim = 16;

        var config = new DiffusionGemmaConfig
        {
            HiddenDim = hiddenDim,
            NumLayers = numLayers,
            SlidingNumQHeads = numHeads,
            SlidingNumKvHeads = numHeads,
            SlidingHeadDim = headDim,
            FullNumQHeads = numHeads,
            FullNumKvHeads = numHeads,
            FullHeadDim = headDim,
            DenseIntermediateDim = interDim,
            CanvasLength = canvasLen,
            MaxDenoisingSteps = 2,
            VocabSize = vocabSize,
            TemperatureMax = 0.8f,
            TemperatureMin = 0.4f,
        };

        string path = Path.Combine(Path.GetTempPath(), $"diffgemma_synthetic_{Guid.NewGuid():N}.gguf");
        try
        {
            var tensors = new Dictionary<string, (long[] shape, DType dtype, byte[] data)>();

            void AddTensor(string name, long[] shape, float fill = 0.05f)
            {
                long total = 1;
                foreach (var s in shape) total *= s;
                var floats = new float[total];
                for (int i = 0; i < total; i++) floats[i] = fill * MathF.Sin(i + 1);
                var bytes = new byte[total * 4];
                Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
                tensors[name] = (shape, DType.Float32, bytes);
            }

            AddTensor("token_embd.weight", [hiddenDim, vocabSize]);
            AddTensor("output_norm.weight", [hiddenDim]);
            AddTensor("output.weight", [hiddenDim, vocabSize]);

            for (int i = 0; i < numLayers; i++)
            {
                AddTensor($"blk.{i}.attn_norm.weight", [hiddenDim]);
                AddTensor($"blk.{i}.attn_q.weight", [hiddenDim, numHeads * headDim]);
                AddTensor($"blk.{i}.attn_k.weight", [hiddenDim, numHeads * headDim]);
                AddTensor($"blk.{i}.attn_v.weight", [hiddenDim, numHeads * headDim]);
                AddTensor($"blk.{i}.attn_output.weight", [numHeads * headDim, hiddenDim]);
                AddTensor($"blk.{i}.ffn_norm.weight", [hiddenDim]);
                AddTensor($"blk.{i}.ffn_gate.weight", [hiddenDim, interDim]);
                AddTensor($"blk.{i}.ffn_up.weight", [hiddenDim, interDim]);
                AddTensor($"blk.{i}.ffn_down.weight", [interDim, hiddenDim]);
            }

            var metadata = new Dictionary<string, object>
            {
                ["general.architecture"] = "diffusiongemma"
            };

            WriteGgufFile(path, metadata, tensors);

            using var model = GgufModel.Open(path);
            var forwardPass = new DiffusionGemmaForwardPass(model, config);
            Assert.Equal(vocabSize, forwardPass.VocabSize);
            Assert.Equal(hiddenDim, forwardPass.HiddenDim);

            var pipeline = new DiffusionGemmaPipeline(forwardPass, config, seed: 1234);

            int[] prompt = [2, 5];
            var generatedTokens = pipeline.Generate(prompt, maxBlocks: 2);

            Assert.Equal(canvasLen * 2, generatedTokens.Count);
            foreach (var tok in generatedTokens)
            {
                Assert.InRange(tok, 0, vocabSize - 1);
            }

            // Test reproducibility with same seed
            var pipeline2 = new DiffusionGemmaPipeline(forwardPass, config, seed: 1234);
            var generatedTokens2 = pipeline2.Generate(prompt, maxBlocks: 2);
            Assert.Equal(generatedTokens, generatedTokens2);
        }
        finally
        {
            try { if (File.Exists(path)) File.Delete(path); } catch { }
        }
    }

    private static void WriteGgufFile(
        string path,
        Dictionary<string, object> metadata,
        Dictionary<string, (long[] shape, DType dtype, byte[] data)> tensors)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        // Header: Magic "GGUF" (0x46554747)
        writer.Write((byte)'G');
        writer.Write((byte)'G');
        writer.Write((byte)'U');
        writer.Write((byte)'F');
        writer.Write((uint)3); // Version 3
        writer.Write((ulong)tensors.Count);
        writer.Write((ulong)metadata.Count);

        // Write metadata
        foreach (var (key, value) in metadata)
        {
            WriteGgufString(writer, key);
            WriteGgufValue(writer, value);
        }

        long runningOffset = 0;
        var tensorOffsets = new Dictionary<string, ulong>();
        foreach (var (name, (shape, dtype, data)) in tensors)
        {
            if (runningOffset % 32 != 0) runningOffset += (32 - (runningOffset % 32));
            tensorOffsets[name] = (ulong)runningOffset;
            runningOffset += data.Length;
        }

        foreach (var (name, (shape, dtype, data)) in tensors)
        {
            WriteGgufString(writer, name);
            writer.Write((uint)shape.Length);
            for (int d = 0; d < shape.Length; d++)
            {
                writer.Write((ulong)shape[d]);
            }
            writer.Write((uint)dtype);
            writer.Write(tensorOffsets[name]);
        }

        long currentPos = stream.Position;
        int pad = (int)((32 - (currentPos % 32)) % 32);
        for (int i = 0; i < pad; i++) writer.Write((byte)0);

        long dataBase = stream.Position;
        foreach (var (name, (shape, dtype, data)) in tensors)
        {
            long targetPos = dataBase + (long)tensorOffsets[name];
            while (stream.Position < targetPos) writer.Write((byte)0);
            writer.Write(data);
        }
    }

    private static void WriteGgufString(BinaryWriter writer, string s)
    {
        byte[] utf8 = System.Text.Encoding.UTF8.GetBytes(s);
        writer.Write((ulong)utf8.Length);
        writer.Write(utf8);
    }

    private static void WriteGgufValue(BinaryWriter writer, object value)
    {
        switch (value)
        {
            case uint u32:
                writer.Write((uint)4);
                writer.Write(u32);
                break;
            case int i32:
                writer.Write((uint)5);
                writer.Write(i32);
                break;
            case float f32:
                writer.Write((uint)6);
                writer.Write(f32);
                break;
            case bool b:
                writer.Write((uint)7);
                writer.Write((byte)(b ? 1 : 0));
                break;
            case string s:
                writer.Write((uint)8);
                WriteGgufString(writer, s);
                break;
            case ulong u64:
                writer.Write((uint)10);
                writer.Write(u64);
                break;
            case object[] arr:
                writer.Write((uint)9);
                writer.Write((uint)5);
                writer.Write((ulong)arr.Length);
                foreach (var item in arr)
                {
                    writer.Write(Convert.ToInt32(item));
                }
                break;
            default:
                throw new NotSupportedException($"Unsupported metadata type: {value.GetType()}");
        }
    }
}

public sealed class DiffusionGemmaRealCheckpointGuardTests
{
    [Fact]
    public void RealGemma4MoeLayout_IsRefusedInsteadOfRunningDegraded()
    {
        const string path = @"F:\_models\diffusiongemma\diffusiongemma-26B-A4B-it-Q4_K_M.gguf";
        string? env = Environment.GetEnvironmentVariable("STINGRAY_TEST_DIFFUSIONGEMMA_GGUF");
        string? gguf = env is { Length: > 0 } && File.Exists(env) ? env : File.Exists(path) ? path : null;
        Assert.SkipUnless(gguf != null, "DiffusionGemma GGUF not found (set STINGRAY_TEST_DIFFUSIONGEMMA_GGUF)");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(gguf!);
        var ex = Assert.Throws<NotSupportedException>(() =>
            new OpenTail.Stingray.Diffusion.DiffusionGemma.DiffusionGemmaForwardPass(
                model, new OpenTail.Stingray.Diffusion.DiffusionGemma.DiffusionGemmaConfig()));
        Assert.Contains("Gemma-4 MoE layer layout", ex.Message);
    }
}
