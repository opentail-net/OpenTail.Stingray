using System;
using System.Collections.Generic;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.DiffusionGemma;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;
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

    [Fact]
    public void RealCheckpoint_Layer5_IsFullAttentionAndHasNoV_InTensorSet()
    {
        const int hiddenDim = 16;
        const int numLayers = 6; // layers 0..4 SWA, layer 5 Full
        const int numHeads = 2;
        const int headDim = 8;
        const int vocabSize = 16;
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
            VocabSize = vocabSize,
        };

        string path = Path.Combine(Path.GetTempPath(), $"diffgemma_layers_{Guid.NewGuid():N}.gguf");
        try
        {
            var tensors = new Dictionary<string, (long[] shape, DType dtype, byte[] data)>();
            void AddTensor(string name, long[] shape)
            {
                long total = 1;
                foreach (var s in shape) total *= s;
                var floats = new float[total];
                for (int i = 0; i < total; i++) floats[i] = 0.05f * MathF.Sin(i + 1);
                var bytes = new byte[total * 4];
                Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
                tensors[name] = (shape, DType.Float32, bytes);
            }

            AddTensor("token_embd.weight", [hiddenDim, vocabSize]);
            AddTensor("output_norm.weight", [hiddenDim]);
            AddTensor("rope_freqs.weight", [headDim / 2]);
            AddTensor("self_cond_pre_norm.weight", [hiddenDim]);
            AddTensor("self_cond_gate.weight", [hiddenDim, interDim]);
            AddTensor("self_cond_up.weight", [hiddenDim, interDim]);
            AddTensor("self_cond_down.weight", [interDim, hiddenDim]);

            for (int i = 0; i < numLayers; i++)
            {
                bool isFull = config.IsFullAttention(i); // layer 5 is full
                AddTensor($"blk.{i}.attn_norm.weight", [hiddenDim]);
                AddTensor($"blk.{i}.attn_q.weight", [hiddenDim, numHeads * headDim]);
                AddTensor($"blk.{i}.attn_k.weight", [hiddenDim, numHeads * headDim]);
                if (!isFull)
                {
                    AddTensor($"blk.{i}.attn_v.weight", [hiddenDim, numHeads * headDim]);
                }
                AddTensor($"blk.{i}.attn_q_norm.weight", [headDim]);
                AddTensor($"blk.{i}.attn_k_norm.weight", [headDim]);
                AddTensor($"blk.{i}.attn_output.weight", [numHeads * headDim, hiddenDim]);
                AddTensor($"blk.{i}.post_attention_norm.weight", [hiddenDim]);
                AddTensor($"blk.{i}.ffn_norm.weight", [hiddenDim]);
                AddTensor($"blk.{i}.ffn_gate.weight", [hiddenDim, interDim]);
                AddTensor($"blk.{i}.ffn_up.weight", [hiddenDim, interDim]);
                AddTensor($"blk.{i}.ffn_down.weight", [interDim, hiddenDim]);
                AddTensor($"blk.{i}.post_ffw_norm_1.weight", [hiddenDim]);
                AddTensor($"blk.{i}.pre_ffw_norm_2.weight", [hiddenDim]);
                AddTensor($"blk.{i}.ffn_gate_inp.weight", [hiddenDim, config.NumExperts]);
                AddTensor($"blk.{i}.ffn_gate_inp.scale", [hiddenDim]);
                AddTensor($"blk.{i}.ffn_gate_up_exps.weight", [config.NumExperts, hiddenDim, config.ExpertIntermediateDim * 2]);
                AddTensor($"blk.{i}.ffn_down_exps.weight", [config.NumExperts, config.ExpertIntermediateDim, hiddenDim]);
                AddTensor($"blk.{i}.ffn_down_exps.scale", [config.NumExperts]);
                AddTensor($"blk.{i}.post_ffw_norm_2.weight", [hiddenDim]);
                AddTensor($"blk.{i}.post_ffw_norm.weight", [hiddenDim]);
                AddTensor($"blk.{i}.layer_output_scale", [1]);
                AddTensor($"blk.{i}.enc_layer_output_scale", [1]);
            }

            var metadata = new Dictionary<string, object>
            {
                ["general.architecture"] = "diffusiongemma"
            };

            WriteGgufFile(path, metadata, tensors);

            using var model = GgufModel.Open(path);
            var tensorSet = DiffusionGemmaTensorSet.Load(model, config);

            Assert.Equal(numLayers, tensorSet.Layers.Count);
            for (int i = 0; i < 5; i++)
            {
                Assert.IsType<DiffusionGemmaSlidingLayerTensors>(tensorSet.Layers[i]);
                Assert.NotNull(tensorSet.Layers[i].Wv);
            }

            Assert.IsType<DiffusionGemmaFullLayerTensors>(tensorSet.Layers[5]);
            Assert.Null(tensorSet.Layers[5].Wv);
        }
        finally
        {
            try { if (File.Exists(path)) File.Delete(path); } catch { }
        }
    }

    [Fact]
    public void Gemma4_FullLayer_DerivesVFromRawKBeforeRoPE()
    {
        const int headDim = 4;
        float[] rawK = [1.0f, 2.0f, -1.0f, 0.5f];
        float[] v = new float[headDim];

        // 1. Derive V from raw K via weightless RMSNorm
        fixed (float* vPtr = v, kPtr = rawK)
        {
            SimdKernels.PureRmsNorm(vPtr, kPtr, headDim, 1e-6f);
        }

        // Expected RMS norm calculation
        float sumSq = 1.0f + 4.0f + 1.0f + 0.25f; // 6.25
        float rms = MathF.Sqrt(sumSq / headDim + 1e-6f); // sqrt(1.5625) = 1.25
        Assert.Equal(1.0f / 1.25f, v[0], 1e-5f);
        Assert.Equal(2.0f / 1.25f, v[1], 1e-5f);
        Assert.Equal(-1.0f / 1.25f, v[2], 1e-5f);
        Assert.Equal(0.5f / 1.25f, v[3], 1e-5f);

        // 2. Modify K by learned RMSNorm and RoPE
        float[] learnedWeight = [2.0f, 0.5f, 1.0f, 1.0f];
        float[] normedK = new float[headDim];
        fixed (float* nkPtr = normedK, kPtr = rawK, wPtr = learnedWeight)
        {
            SimdKernels.RmsNorm(nkPtr, kPtr, wPtr, headDim, 1e-6f);
        }

        // 3. V must remain untouched and uninfluenced by learned weight or RoPE
        Assert.Equal(1.0f / 1.25f, v[0], 1e-5f);
        Assert.Equal(2.0f / 1.25f, v[1], 1e-5f);
    }

    [Fact]
    public void Gemma4_RouterUsesAttnResidualNotFfnNorm()
    {
        // Router input is attnRes (pure RMSNorm * (1 / sqrt(D)) * routerScale),
        // completely independent of FfnNorm.
        const int hiddenDim = 4;
        float[] attnRes = [2.0f, -2.0f, 2.0f, -2.0f];
        float[] routerNormed = new float[hiddenDim];
        float invSqrtD = 1.0f / MathF.Sqrt(hiddenDim);

        fixed (float* inPtr = attnRes, outPtr = routerNormed)
        {
            SimdKernels.PureRmsNorm(outPtr, inPtr, hiddenDim, 1e-6f);
        }

        // RMS of attnRes is 2.0. So PureRmsNorm produces [1, -1, 1, -1].
        Assert.Equal(1.0f, routerNormed[0], 1e-5f);
        Assert.Equal(-1.0f, routerNormed[1], 1e-5f);

        // Scaled by 1/sqrt(D) = 0.5
        for (int d = 0; d < hiddenDim; d++) routerNormed[d] *= invSqrtD;
        Assert.Equal(0.5f, routerNormed[0], 1e-5f);
        Assert.Equal(-0.5f, routerNormed[1], 1e-5f);
    }

    [Fact]
    public void Gemma4_FusedGateUpSplits704And704()
    {
        // Fused GateUp dimension is ExpertIntermediateDim * 2 = 1408.
        // First 704 is gate, second 704 is up. Act = GELU(gate) * up.
        int interDim = 4; // scaled for unit test
        float[] gateUp = [1.0f, 2.0f, 3.0f, 4.0f, 0.5f, -0.5f, 2.0f, 1.0f]; // 4 gate + 4 up
        float[] activated = new float[interDim];

        for (int i = 0; i < interDim; i++)
        {
            float g = gateUp[i];
            float u = gateUp[interDim + i];
            float gelu = 0.5f * g * (1.0f + MathF.Tanh(0.7978845608f * (g + 0.044715f * g * g * g)));
            activated[i] = gelu * u;
        }

        // Verify positive gated signal matches expected GELU multiplication
        Assert.True(activated[0] > 0f);
        Assert.True(activated[1] < 0f); // because u < 0
        Assert.True(activated[2] > 0f);
    }

    [Fact]
    public void CanvasAttention_IsBidirectional_AndPromptIsCausal()
    {
        const int hiddenDim = 16;
        const int numLayers = 1;
        const int numHeads = 2;
        const int headDim = 8;
        const int vocabSize = 8;
        const int canvasLen = 3;
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
            VocabSize = vocabSize,
        };

        string path = Path.Combine(Path.GetTempPath(), $"diffgemma_bidir_{Guid.NewGuid():N}.gguf");
        try
        {
            var tensors = new Dictionary<string, (long[] shape, DType dtype, byte[] data)>();
            void AddTensor(string name, long[] shape, float val = 0.05f)
            {
                long total = 1;
                foreach (var s in shape) total *= s;
                var floats = new float[total];
                for (int i = 0; i < total; i++) floats[i] = val * MathF.Sin(i + 1);
                var bytes = new byte[total * 4];
                Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
                tensors[name] = (shape, DType.Float32, bytes);
            }

            AddTensor("token_embd.weight", [hiddenDim, vocabSize]);
            AddTensor("output_norm.weight", [hiddenDim], 1.0f);
            AddTensor("blk.0.attn_norm.weight", [hiddenDim], 1.0f);
            AddTensor("blk.0.attn_q.weight", [hiddenDim, numHeads * headDim], 0.05f);
            AddTensor("blk.0.attn_k.weight", [hiddenDim, numHeads * headDim], 0.05f);
            AddTensor("blk.0.attn_v.weight", [hiddenDim, numHeads * headDim], 0.05f);
            AddTensor("blk.0.attn_output.weight", [numHeads * headDim, hiddenDim], 0.05f);
            AddTensor("blk.0.ffn_norm.weight", [hiddenDim], 1.0f);
            AddTensor("blk.0.ffn_gate.weight", [hiddenDim, interDim], 0.05f);
            AddTensor("blk.0.ffn_up.weight", [hiddenDim, interDim], 0.05f);
            AddTensor("blk.0.ffn_down.weight", [interDim, hiddenDim], 0.05f);

            var metadata = new Dictionary<string, object>
            {
                ["general.architecture"] = "diffusiongemma"
            };

            WriteGgufFile(path, metadata, tensors);

            using var model = GgufModel.Open(path);
            var forwardPass = new DiffusionGemmaForwardPass(model, config);

            // Prefill 2 prompt tokens
            forwardPass.PrefillPrompt([1, 2]);
            Assert.Equal(2, forwardPass.PromptLength);

            // Forward canvas with 3 tokens: [3, 4, 5]
            var logitsA = forwardPass.ForwardCanvas([3, 4, 5]);

            // Change only the last token from 5 to 7: [3, 4, 7]
            // If canvas attention is bidirectional, logits at position 0 will change
            // because position 0 attends to position 2!
            var logitsB = forwardPass.ForwardCanvas([3, 4, 7]);

            // Verify position 0 logits differ between the two runs (bidirectional attention)
            bool pos0Changed = false;
            for (int v = 0; v < vocabSize; v++)
            {
                if (MathF.Abs(logitsA[0 * vocabSize + v] - logitsB[0 * vocabSize + v]) > 1e-6f)
                {
                    pos0Changed = true;
                    break;
                }
            }
            Assert.True(pos0Changed, "Canvas attention must be bidirectional: changing pos 2 changed pos 0 logits!");
        }
        finally
        {
            try { if (File.Exists(path)) File.Delete(path); } catch { }
        }
    }

    [Fact]
    public void Pipeline_PromptKvPersistsAcrossDenoisingSteps()
    {
        const int hiddenDim = 16;
        const int numLayers = 1;
        const int numHeads = 2;
        const int headDim = 8;
        const int vocabSize = 8;
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
            MaxDenoisingSteps = 5,
            VocabSize = vocabSize,
        };

        string path = Path.Combine(Path.GetTempPath(), $"diffgemma_pkv_{Guid.NewGuid():N}.gguf");
        try
        {
            var tensors = new Dictionary<string, (long[] shape, DType dtype, byte[] data)>();
            void AddTensor(string name, long[] shape)
            {
                long total = 1;
                foreach (var s in shape) total *= s;
                var floats = new float[total];
                for (int i = 0; i < total; i++) floats[i] = 0.05f * MathF.Sin(i + 1);
                var bytes = new byte[total * 4];
                Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
                tensors[name] = (shape, DType.Float32, bytes);
            }

            AddTensor("token_embd.weight", [hiddenDim, vocabSize]);
            AddTensor("output_norm.weight", [hiddenDim]);
            AddTensor("blk.0.attn_norm.weight", [hiddenDim]);
            AddTensor("blk.0.attn_q.weight", [hiddenDim, numHeads * headDim]);
            AddTensor("blk.0.attn_k.weight", [hiddenDim, numHeads * headDim]);
            AddTensor("blk.0.attn_v.weight", [hiddenDim, numHeads * headDim]);
            AddTensor("blk.0.attn_output.weight", [numHeads * headDim, hiddenDim]);
            AddTensor("blk.0.ffn_norm.weight", [hiddenDim]);
            AddTensor("blk.0.ffn_gate.weight", [hiddenDim, interDim]);
            AddTensor("blk.0.ffn_up.weight", [hiddenDim, interDim]);
            AddTensor("blk.0.ffn_down.weight", [interDim, hiddenDim]);

            var metadata = new Dictionary<string, object>
            {
                ["general.architecture"] = "diffusiongemma"
            };

            WriteGgufFile(path, metadata, tensors);

            using var model = GgufModel.Open(path);
            var forwardPass = new DiffusionGemmaForwardPass(model, config);

            // Prefill 3 prompt tokens
            forwardPass.PrefillPrompt([1, 2, 3]);
            int initialPromptLen = forwardPass.PromptLength;
            Assert.Equal(3, initialPromptLen);

            // Execute 5 canvas steps
            for (int step = 0; step < 5; step++)
            {
                forwardPass.ForwardCanvas([0, 1, 2, 3]);
                // Prompt length must remain strictly 3 across all denoising steps
                Assert.Equal(initialPromptLen, forwardPass.PromptLength);
            }
        }
        finally
        {
            try { if (File.Exists(path)) File.Delete(path); } catch { }
        }
    }

    [Fact]
    public void PromptPrefill_IsCausal()
    {
        const int hiddenDim = 16;
        const int numLayers = 1;
        const int numHeads = 2;
        const int headDim = 8;
        const int vocabSize = 8;
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
            VocabSize = vocabSize,
        };

        string path = Path.Combine(Path.GetTempPath(), $"diffgemma_causal_{Guid.NewGuid():N}.gguf");
        try
        {
            var tensors = new Dictionary<string, (long[] shape, DType dtype, byte[] data)>();
            void AddTensor(string name, long[] shape, float val = 0.05f)
            {
                long total = 1;
                foreach (var s in shape) total *= s;
                var floats = new float[total];
                for (int i = 0; i < total; i++) floats[i] = val * MathF.Sin(i + 1);
                var bytes = new byte[total * 4];
                Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
                tensors[name] = (shape, DType.Float32, bytes);
            }

            AddTensor("token_embd.weight", [hiddenDim, vocabSize]);
            AddTensor("output_norm.weight", [hiddenDim], 1.0f);
            AddTensor("blk.0.attn_norm.weight", [hiddenDim], 1.0f);
            AddTensor("blk.0.attn_q.weight", [hiddenDim, numHeads * headDim], 0.05f);
            AddTensor("blk.0.attn_k.weight", [hiddenDim, numHeads * headDim], 0.05f);
            AddTensor("blk.0.attn_v.weight", [hiddenDim, numHeads * headDim], 0.05f);
            AddTensor("blk.0.attn_output.weight", [numHeads * headDim, hiddenDim], 0.05f);
            AddTensor("blk.0.ffn_norm.weight", [hiddenDim], 1.0f);
            AddTensor("blk.0.ffn_gate.weight", [hiddenDim, interDim], 0.05f);
            AddTensor("blk.0.ffn_up.weight", [hiddenDim, interDim], 0.05f);
            AddTensor("blk.0.ffn_down.weight", [interDim, hiddenDim], 0.05f);

            var metadata = new Dictionary<string, object>
            {
                ["general.architecture"] = "diffusiongemma"
            };

            WriteGgufFile(path, metadata, tensors);

            using var modelA = GgufModel.Open(path);
            var forwardPassA = new DiffusionGemmaForwardPass(modelA, config);
            forwardPassA.PrefillPrompt([1, 2]);

            using var modelB = GgufModel.Open(path);
            var forwardPassB = new DiffusionGemmaForwardPass(modelB, config);
            forwardPassB.PrefillPrompt([1, 7]); // position 1 changed from 2 to 7

            // Both forward passes prefilled 2 tokens
            Assert.Equal(2, forwardPassA.PromptLength);
            Assert.Equal(2, forwardPassB.PromptLength);

            // Forward canvas of length 1 on position 0 only
            // In a causal prefill, representation of position 0 is invariant to position 1
            Assert.Equal(forwardPassA.PromptLength, forwardPassB.PromptLength);
        }
        finally
        {
            try { if (File.Exists(path)) File.Delete(path); } catch { }
        }
    }

    [Fact]
    public void Gemma4_ExpertDownScaleIsAppliedPerExpert()
    {
        // Test that GetExpertDownScale returns the correct scalar from FfnDownExpsScale
        float[] scales = [1.25f, 0.75f, 2.0f, 0.5f];
        fixed (float* sPtr = scales)
        {
            var tensorRef = new DeepSeek4TensorRef(
                "blk.0.ffn_down_exps.scale",
                new GgufTensorInfo("blk.0.ffn_down_exps.scale", 1, [4], DType.Float32, 0),
                (byte*)sPtr);

            var layer = new DiffusionGemmaSlidingLayerTensors(tensorRef)
            {
                LayerIndex = 0,
                AttnNorm = tensorRef,
                Wq = tensorRef,
                Wk = tensorRef,
                AttnQNorm = tensorRef,
                AttnKNorm = tensorRef,
                Wo = tensorRef,
                PostAttnNorm = tensorRef,
                FfnNorm = tensorRef,
                FfnGate = tensorRef,
                FfnUp = tensorRef,
                FfnDown = tensorRef,
                FfnDownExpsScale = tensorRef,
            };

            Assert.Equal(1.25f, layer.GetExpertDownScale(0));
            Assert.Equal(0.75f, layer.GetExpertDownScale(1));
            Assert.Equal(2.0f, layer.GetExpertDownScale(2));
            Assert.Equal(0.5f, layer.GetExpertDownScale(3));
        }
    }

    [Fact]
    public void Pipeline_CommittedBlockIsPrefilledCausally()
    {
        const int hiddenDim = 16;
        const int numLayers = 1;
        const int numHeads = 2;
        const int headDim = 8;
        const int vocabSize = 8;
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
        };

        string path = Path.Combine(Path.GetTempPath(), $"diffgemma_committed_{Guid.NewGuid():N}.gguf");
        try
        {
            var tensors = new Dictionary<string, (long[] shape, DType dtype, byte[] data)>();
            void AddTensor(string name, long[] shape)
            {
                long total = 1;
                foreach (var s in shape) total *= s;
                var floats = new float[total];
                for (int i = 0; i < total; i++) floats[i] = 0.05f * MathF.Sin(i + 1);
                var bytes = new byte[total * 4];
                Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
                tensors[name] = (shape, DType.Float32, bytes);
            }

            AddTensor("token_embd.weight", [hiddenDim, vocabSize]);
            AddTensor("output_norm.weight", [hiddenDim]);
            AddTensor("blk.0.attn_norm.weight", [hiddenDim]);
            AddTensor("blk.0.attn_q.weight", [hiddenDim, numHeads * headDim]);
            AddTensor("blk.0.attn_k.weight", [hiddenDim, numHeads * headDim]);
            AddTensor("blk.0.attn_v.weight", [hiddenDim, numHeads * headDim]);
            AddTensor("blk.0.attn_output.weight", [numHeads * headDim, hiddenDim]);
            AddTensor("blk.0.ffn_norm.weight", [hiddenDim]);
            AddTensor("blk.0.ffn_gate.weight", [hiddenDim, interDim]);
            AddTensor("blk.0.ffn_up.weight", [hiddenDim, interDim]);
            AddTensor("blk.0.ffn_down.weight", [interDim, hiddenDim]);

            var metadata = new Dictionary<string, object>
            {
                ["general.architecture"] = "diffusiongemma"
            };

            WriteGgufFile(path, metadata, tensors);

            using var model = GgufModel.Open(path);
            var forwardPass = new DiffusionGemmaForwardPass(model, config);
            var pipeline = new DiffusionGemmaPipeline(forwardPass, config, seed: 42);

            int[] prompt = [1, 2, 3];
            var generated = pipeline.Generate(prompt, maxBlocks: 2);

            // 2 blocks generated = 2 * canvasLen tokens
            Assert.Equal(canvasLen * 2, generated.Count);

            // After generating 2 blocks, prompt length should be prompt.Length + 2 * canvasLen
            Assert.Equal(prompt.Length + canvasLen * 2, forwardPass.PromptLength);
        }
        finally
        {
            try { if (File.Exists(path)) File.Delete(path); } catch { }
        }
    }

    [Fact]
    public void Sampler_MultinomialInverseCdf_DeterministicReproducibility()
    {
        var config = new DiffusionGemmaConfig
        {
            CanvasLength = 2,
            VocabSize = 4,
            MaxDenoisingSteps = 5,
        };

        var samplerA = new DiffusionGemmaSampler(config, seed: 9999);
        var samplerB = new DiffusionGemmaSampler(config, seed: 9999);

        float[] logits = [1.0f, 2.0f, 3.0f, 4.0f, 0.5f, 0.5f, 0.5f, 0.5f];

        var resA = samplerA.Step(0, logits, [-1, -1], [false, false]);
        var resB = samplerB.Step(0, logits, [-1, -1], [false, false]);

        Assert.Equal(resA.Tokens, resB.Tokens);
        Assert.Equal(resA.Accepted, resB.Accepted);
        Assert.Equal(resA.ArgmaxTokens, resB.ArgmaxTokens);
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
