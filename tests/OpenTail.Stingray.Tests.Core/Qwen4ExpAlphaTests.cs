using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Unit tests for Qwen4Exp Alpha components (Qwen4ExpAlpha.cs):
/// GatedResidual hyper-connections, PLE signed-sqrt gating and dilated conv, and QSA primitives.
/// </summary>
public class Qwen4ExpAlphaTests
{
    [Fact]
    public void Hyperparams_LayerSchedule_FollowsThreeToOnePattern()
    {
        var meta = new Dictionary<string, object>
        {
            ["qwen4exp.full_attention_interval"] = 4,
            ["qwen4exp.ple.layers"] = new List<int> { 2 },
            ["qwen4exp.hyper_connection.count"] = 4,
            ["qwen4exp.hyper_connection.low_rank"] = 320,
            ["qwen4exp.expert_feed_forward_length"] = 640,
        };

        var hp = Qwen4ExpHyperparams.FromMetadata(meta, totalLayers: 48);

        Assert.Equal(4, hp.HyperConnectionCount);
        Assert.Equal(320, hp.HyperConnectionLowRank);
        Assert.Equal(640, hp.ExpertFeedForwardLength);

        // Pattern across 48 layers: layers 0, 1, 2 are GDN; layer 3 is QSA; repeat.
        for (int l = 0; l < 48; l++)
        {
            bool expectedRecurrent = (l + 1) % 4 != 0;
            Assert.Equal(expectedRecurrent, hp.IsRecurrentLayer(l));
        }

        // PLE is at layer 2 only
        Assert.False(hp.IsPleLayer(0));
        Assert.False(hp.IsPleLayer(1));
        Assert.True(hp.IsPleLayer(2));
        Assert.False(hp.IsPleLayer(3));
    }

    [Fact]
    public void GatedResidual_Combine_ZeroInjection_IsPlainResidualAdd()
    {
        const int hc = 4;
        const int embedDim = 8;
        float[] residual = new float[hc * embedDim];
        for (int i = 0; i < residual.Length; i++) residual[i] = 10f + i;

        float[] blockOut = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];
        float[] zeroInject = [0f, 0f, 0f, 0f]; // 2 * sigmoid(0) = 1.0

        float[] expected = (float[])residual.Clone();
        for (int c = 0; c < hc; c++)
        {
            for (int i = 0; i < embedDim; i++)
            {
                expected[c * embedDim + i] += blockOut[i];
            }
        }

        Qwen4ExpGatedResidual.Combine(residual, blockOut, zeroInject, hc, embedDim);

        for (int i = 0; i < residual.Length; i++)
        {
            Assert.Equal(expected[i], residual[i], 1e-5f);
        }
    }

    [Fact]
    public void GatedResidual_MixAndCombine_EndToEndConsistency()
    {
        const int hc = 4;
        const int embedDim = 16;
        const int hcLowRank = 8;
        int hcDim = hc * embedDim;

        float[] x = new float[hcDim];
        for (int i = 0; i < hcDim; i++) x[i] = 0.5f + (i % 7) * 0.1f;

        float[] wNorm = new float[hcDim];
        Array.Fill(wNorm, 1f);

        float[] wDown = new float[hcLowRank * hcDim];
        Array.Fill(wDown, 0.01f);

        float[] wUp = new float[hcDim * hcLowRank];
        Array.Fill(wUp, 0.01f);

        float[] wInject = new float[hc * hcDim];
        Array.Fill(wInject, 0.005f);

        float[] mixed = new float[embedDim];
        float[] inject = new float[hc];

        Qwen4ExpGatedResidual.Mix(
            x, wNorm, wDown, wUp, wInject,
            mixed, inject,
            hc, embedDim, hcLowRank, eps: 1e-6f);

        // All mixed elements must be positive and non-zero given positive inputs and gammas
        for (int i = 0; i < embedDim; i++)
        {
            Assert.True(mixed[i] > 0f, $"mixed[{i}] was {mixed[i]}, expected > 0");
        }

        // Injection weights must be computed
        for (int c = 0; c < hc; c++)
        {
            Assert.True(inject[c] > 0f, $"inject[{c}] was {inject[c]}, expected > 0");
        }

        // Test combine
        float[] residual = (float[])x.Clone();
        float[] blockOut = (float[])mixed.Clone();

        Qwen4ExpGatedResidual.Combine(residual, blockOut, inject, hc, embedDim);

        for (int c = 0; c < hc; c++)
        {
            float w = 2.0f / (1.0f + MathF.Exp(-inject[c] / hc));
            for (int i = 0; i < embedDim; i++)
            {
                float expectedVal = x[c * embedDim + i] + blockOut[i] * w;
                Assert.Equal(expectedVal, residual[c * embedDim + i], 1e-5f);
            }
        }
    }

    [Fact]
    public void Ple_ComputeGate_PositiveAndNegativeDotProducts()
    {
        const int hc = 2;
        const int embedDim = 4;

        // Positive dot product stream 0: key . query = 4.0 * sqrt(4) = 8.0 -> s = 4.0
        // sgn(4.0) * sqrt(4.0) = 2.0 -> sigmoid(2.0)
        // Negative dot product stream 1: key . query = -9.0 * sqrt(4) = -18.0 -> s = -9.0
        // sgn(-9.0) * sqrt(9.0) = -3.0 -> sigmoid(-3.0)

        float[] key = [2f, 2f, 2f, 2f, -3f, -3f, -3f, -3f];
        float[] query = [1f, 1f, 1f, 1f, 1.5f, 1.5f, 1.5f, 1.5f];
        float[] gate = new float[hc];

        Qwen4ExpPle.ComputePleGate(key, query, gate, hc, embedDim);

        float expectedGate0 = 1.0f / (1.0f + MathF.Exp(-2.0f));
        float expectedGate1 = 1.0f / (1.0f + MathF.Exp(-(-3.0f)));

        Assert.Equal(expectedGate0, gate[0], 1e-4f);
        Assert.Equal(expectedGate1, gate[1], 1e-4f);
    }

    [Fact]
    public void Ple_DilatedConv_AccessesExpectedTaps()
    {
        const int hcDim = 4;
        const int kernelSize = 3;
        const int dilation = 2;
        // hist = (3 - 1) * 2 = 4.
        // Taps:
        // k = 0: tapOffset = 4 - (3 - 1 - 0) * 2 = 0
        // k = 1: tapOffset = 4 - (3 - 1 - 1) * 2 = 2
        // k = 2: tapOffset = 4 - (3 - 1 - 2) * 2 = 4 (current slot)
        const int totalSlots = 5;

        float[] historyBuffer = new float[totalSlots * hcDim];
        // Set distinct values at slots 0, 2, 4
        for (int i = 0; i < hcDim; i++)
        {
            historyBuffer[0 * hcDim + i] = 1f;
            historyBuffer[2 * hcDim + i] = 10f;
            historyBuffer[4 * hcDim + i] = 100f;
        }

        float[] convWeights = new float[kernelSize * hcDim];
        for (int i = 0; i < hcDim; i++)
        {
            convWeights[0 * hcDim + i] = 0.1f; // weight for tap 0 (history slot 0)
            convWeights[1 * hcDim + i] = 0.2f; // weight for tap 1 (history slot 2)
            convWeights[2 * hcDim + i] = 0.3f; // weight for tap 2 (history slot 4)
        }

        float[] convOut = new float[hcDim];
        Qwen4ExpPle.StepDilatedConv(historyBuffer, convWeights, convOut, hcDim, kernelSize, dilation, totalSlots);

        // Expected linear sum: 1 * 0.1 + 10 * 0.2 + 100 * 0.3 = 0.1 + 2.0 + 30.0 = 32.1
        // SiLU(32.1) = 32.1 / (1 + exp(-32.1)) ~= 32.1
        float expectedLinear = 32.1f;
        float expectedSilu = expectedLinear / (1.0f + MathF.Exp(-expectedLinear));

        for (int i = 0; i < hcDim; i++)
        {
            Assert.Equal(expectedSilu, convOut[i], 1e-4f);
        }
    }

    [Fact]
    public void Qsa_PoolIndexerKeys_AveragesAndNormalizes()
    {
        const int kpool = 4;
        const int idxDim = 4;

        float[] blockKeys = new float[kpool * idxDim];
        for (int k = 0; k < kpool; k++)
        {
            for (int d = 0; d < idxDim; d++)
            {
                blockKeys[k * idxDim + d] = 2.0f; // constant 2.0 across all tokens
            }
        }

        float[] normWeight = [1f, 1f, 1f, 1f];
        float[] pooled = new float[idxDim];

        Qwen4ExpQsa.PoolIndexerKeys(blockKeys, normWeight, pooled, kpool, idxDim, eps: 1e-6f);

        // Average of 2.0 is 2.0.
        // RMS of [2, 2, 2, 2] is sqrt(16 / 4) = 2.0.
        // Normalized with weight 1.0: 2.0 / 2.0 = 1.0.
        for (int d = 0; d < idxDim; d++)
        {
            Assert.Equal(1.0f, pooled[d], 1e-5f);
        }
    }

    [Fact]
    public void Qsa_ComputeBlockScore_RectifiesAndScales()
    {
        const int numHeads = 2;
        const int idxDim = 4;

        // Query head 0 has positive dot product with pooled key: dot = 4.0
        // Query head 1 has negative dot product with pooled key: dot = -2.0 (rectified to 0)
        float[] query = [1f, 1f, 1f, 1f, -0.5f, -0.5f, -0.5f, -0.5f];
        float[] pooledKey = [1f, 1f, 1f, 1f];

        float score = Qwen4ExpQsa.ComputeBlockScore(query, pooledKey, numHeads, idxDim);

        // ReLU(dot0) + ReLU(dot1) = 4.0 + 0 = 4.0
        // Scaled by 1/sqrt(4) = 0.5: 4.0 * 0.5 = 2.0
        Assert.Equal(2.0f, score, 1e-5f);
    }

    [Fact]
    public void Qsa_SplitAndNormQGated_And_ApplyAttentionGate()
    {
        const int numHeads = 2;
        const int headDim = 2;
        // doubleHeadDim = 4. Total length = 2 * 4 = 8.
        // Head 0: Q = [3, 4] (rms = sqrt((9+16)/2) = sqrt(12.5) ~= 3.5355), Gate = [0.0, 0.0]
        // Head 1: Q = [1, 1], Gate = [-10.0, 10.0]
        float[] qFull =
        [
            3f, 4f, 0f, 0f,
            1f, 1f, -10f, 10f
        ];

        float[] qNormWeight = [1f, 1f];
        float[] qOut = new float[numHeads * headDim];
        float[] gateOut = new float[numHeads * headDim];

        Qwen4ExpQsa.SplitAndNormQGated(qFull, qNormWeight, qOut, gateOut, numHeads, headDim, eps: 1e-6f);

        Assert.Equal(0f, gateOut[0]);
        Assert.Equal(0f, gateOut[1]);
        Assert.Equal(-10f, gateOut[2]);
        Assert.Equal(10f, gateOut[3]);

        // Attention gate test: gate of 0 -> sigmoid(0) = 0.5 -> multiply attnOut by 0.5
        float[] attnOut = [10f, 20f, 30f, 40f];
        Qwen4ExpQsa.ApplyAttentionGate(attnOut, gateOut);

        Assert.Equal(5f, attnOut[0], 1e-4f);
        Assert.Equal(10f, attnOut[1], 1e-4f);
        // Head 1 gate[0] = -10 -> sigmoid(-10) ~= 0.0000454 -> 30 * 4.54e-5 ~= 0.00136
        Assert.True(attnOut[2] < 0.1f);
        // Head 1 gate[1] = +10 -> sigmoid(10) ~= 0.99995 -> 40 * 0.99995 ~= 39.998
        Assert.Equal(40f, attnOut[3], 0.1f);
    }

    [Fact]
    public unsafe void Qwen4Exp_SyntheticForwardPass_RunsEndToEnd_ProducesFiniteLogits()
    {
        const int embedDim = 16;
        const int hc = 4;
        const int hcDim = hc * embedDim; // 64
        const int hcLowRank = 8;
        const int vocabSize = 32;
        const int numLayers = 2;
        const int numHeads = 2;
        const int numHeadsKv = 1;
        const int headDim = 8;
        const int ssmDState = 8;
        const int ssmDInner = 16;
        const int ssmDConv = 3;
        const int ssmDtRank = 2;
        const int ssmNGroup = 2;
        int keyDim = ssmDState * ssmNGroup; // 16
        int valDim = ssmDState * ssmDtRank; // 16
        int qkvDim = keyDim * 2 + valDim;   // 48
        const int pleConvKernel = 3;
        const int pleNgramSize = 2;
        const int shExpDim = 32;

        var hp = new Qwen4ExpHyperparams
        {
            EmbedDim = embedDim,
            NumHeads = numHeads,
            NumHeadsKv = numHeadsKv,
            HeadDim = headDim,
            NumLayer = numLayers,
            VocabSize = vocabSize,
            ContextLength = 512,
            HyperConnectionCount = hc,
            HyperConnectionLowRank = hcLowRank,
            SsmStateSize = ssmDState,
            SsmInnerSize = ssmDInner,
            SsmConvKernel = ssmDConv,
            SsmDtRank = ssmDtRank,
            SsmGroupCount = ssmNGroup,
            PleConvKernel = pleConvKernel,
            PleNgramSize = pleNgramSize,
            PleLayers = [0],
            RecurrentLayers = [true, false], // layer 0 = GDN, layer 1 = QSA
            RmsNormEps = 1e-6f,
        };

        var allocatedPointers = new List<nint>();
        Qwen4ExpTensorRef CreateRef(string name, long[] dims, float fillVal = 0.01f)
        {
            long count = 1;
            foreach (var d in dims) count *= d;
            nint mem = (nint)System.Runtime.InteropServices.NativeMemory.Alloc((nuint)count, sizeof(float));
            allocatedPointers.Add(mem);
            float* p = (float*)mem;
            for (long i = 0; i < count; i++) p[i] = fillVal + (float)((i % 13) * 0.001);
            var info = new OpenTail.Stingray.Core.GgufTensorInfo(name, dims.Length, dims, OpenTail.Stingray.Core.DType.Float32, 0);
            return new Qwen4ExpTensorRef(name, info, (byte*)p);
        }

        try
        {
            var tokEmbd = CreateRef("token_embd.weight", [embedDim, vocabSize], 0.05f);
            var output = CreateRef("output.weight", [embedDim, vocabSize], 0.05f);
            var hcHeadNorm = CreateRef("hc_head_norm.weight", [embedDim, hc], 1.0f);
            var hcHeadDown = CreateRef("hc_head_down.weight", [hcDim, hcLowRank], 0.02f);
            var hcHeadUp = CreateRef("hc_head_up.weight", [hcLowRank, hcDim], 0.02f);

            var layer0 = new Qwen4ExpLayerTensors
            {
                LayerIndex = 0,
                IsRecurrent = true,
                IsPle = true,
                HcAttnNorm = CreateRef("blk.0.hc_attn_norm.weight", [embedDim, hc], 1.0f),
                HcAttnDown = CreateRef("blk.0.hc_attn_down.weight", [hcDim, hcLowRank], 0.02f),
                HcAttnUp = CreateRef("blk.0.hc_attn_up.weight", [hcLowRank, hcDim], 0.02f),
                HcAttnInject = CreateRef("blk.0.hc_attn_inject.weight", [hcDim, hc], 0.01f),
                HcFfnNorm = CreateRef("blk.0.hc_ffn_norm.weight", [embedDim, hc], 1.0f),
                HcFfnDown = CreateRef("blk.0.hc_ffn_down.weight", [hcDim, hcLowRank], 0.02f),
                HcFfnUp = CreateRef("blk.0.hc_ffn_up.weight", [hcLowRank, hcDim], 0.02f),
                HcFfnInject = CreateRef("blk.0.hc_ffn_inject.weight", [hcDim, hc], 0.01f),

                AttnQkv = CreateRef("blk.0.attn_qkv.weight", [embedDim, qkvDim], 0.02f),
                AttnGate = CreateRef("blk.0.attn_gate.weight", [embedDim, valDim], 0.02f),
                SsmConv1d = CreateRef("blk.0.ssm_conv1d.weight", [ssmDConv, qkvDim], 0.1f),
                SsmDt = CreateRef("blk.0.ssm_dt.bias", [ssmDtRank], 0.1f),
                SsmA = CreateRef("blk.0.ssm_a", [ssmDtRank], -1.0f),
                SsmBeta = CreateRef("blk.0.ssm_beta.weight", [embedDim, ssmDtRank], 0.05f),
                SsmAlpha = CreateRef("blk.0.ssm_alpha.weight", [embedDim, ssmDtRank], 0.05f),
                SsmNorm = CreateRef("blk.0.ssm_norm.weight", [ssmDState], 1.0f),
                SsmOut = CreateRef("blk.0.ssm_out.weight", [valDim, embedDim], 0.05f),

                PleKey = CreateRef("blk.0.ple_key.weight", [embedDim, hcDim], 0.02f),
                PleValue = CreateRef("blk.0.ple_value.weight", [embedDim, embedDim], 0.02f),
                PleNormKey = CreateRef("blk.0.ple_norm_key.weight", [embedDim, hc], 1.0f),
                PleNormQuery = CreateRef("blk.0.ple_norm_query.weight", [embedDim, hc], 1.0f),
                PleNormConv = CreateRef("blk.0.ple_norm_conv.weight", [embedDim, hc], 1.0f),
                PleConv1d = CreateRef("blk.0.ple_conv1d.weight", [pleConvKernel, hcDim], 0.1f),

                FfnGateShexp = CreateRef("blk.0.ffn_gate_shexp.weight", [embedDim, shExpDim], 0.02f),
                FfnUpShexp = CreateRef("blk.0.ffn_up_shexp.weight", [embedDim, shExpDim], 0.02f),
                FfnDownShexp = CreateRef("blk.0.ffn_down_shexp.weight", [shExpDim, embedDim], 0.02f),
                FfnGateInpShexp = CreateRef("blk.0.ffn_gate_inp_shexp.weight", [embedDim], 0.05f),
            };

            var layer1 = new Qwen4ExpLayerTensors
            {
                LayerIndex = 1,
                IsRecurrent = false,
                IsPle = false,
                HcAttnNorm = CreateRef("blk.1.hc_attn_norm.weight", [embedDim, hc], 1.0f),
                HcAttnDown = CreateRef("blk.1.hc_attn_down.weight", [hcDim, hcLowRank], 0.02f),
                HcAttnUp = CreateRef("blk.1.hc_attn_up.weight", [hcLowRank, hcDim], 0.02f),
                HcAttnInject = CreateRef("blk.1.hc_attn_inject.weight", [hcDim, hc], 0.01f),
                HcFfnNorm = CreateRef("blk.1.hc_ffn_norm.weight", [embedDim, hc], 1.0f),
                HcFfnDown = CreateRef("blk.1.hc_ffn_down.weight", [hcDim, hcLowRank], 0.02f),
                HcFfnUp = CreateRef("blk.1.hc_ffn_up.weight", [hcLowRank, hcDim], 0.02f),
                HcFfnInject = CreateRef("blk.1.hc_ffn_inject.weight", [hcDim, hc], 0.01f),

                AttnQ = CreateRef("blk.1.attn_q.weight", [embedDim, numHeads * headDim * 2], 0.02f),
                AttnK = CreateRef("blk.1.attn_k.weight", [embedDim, numHeadsKv * headDim], 0.02f),
                AttnV = CreateRef("blk.1.attn_v.weight", [embedDim, numHeadsKv * headDim], 0.02f),
                AttnOut = CreateRef("blk.1.attn_output.weight", [numHeads * headDim, embedDim], 0.05f),
                AttnQNorm = CreateRef("blk.1.attn_q_norm.weight", [headDim], 1.0f),
                AttnKNorm = CreateRef("blk.1.attn_k_norm.weight", [headDim], 1.0f),

                FfnGateShexp = CreateRef("blk.1.ffn_gate_shexp.weight", [embedDim, shExpDim], 0.02f),
                FfnUpShexp = CreateRef("blk.1.ffn_up_shexp.weight", [embedDim, shExpDim], 0.02f),
                FfnDownShexp = CreateRef("blk.1.ffn_down_shexp.weight", [shExpDim, embedDim], 0.02f),
                FfnGateInpShexp = CreateRef("blk.1.ffn_gate_inp_shexp.weight", [embedDim], 0.05f),
            };

            var tensorSet = new Qwen4ExpTensorSet(
                tokEmbd, output, hcHeadNorm, hcHeadDown, hcHeadUp, null, [layer0, layer1]);

            using var forwardPass = new Qwen4ExpForwardPass(null!, hp, tensorSet);

            Assert.Equal(vocabSize, forwardPass.VocabSize);
            Assert.Equal(512, forwardPass.MaxSeqLen);

            // Step 0 forward
            var logits0 = forwardPass.Forward(token: 5, position: 0).ToArray();
            Assert.Equal(vocabSize, logits0.Length);
            for (int i = 0; i < vocabSize; i++)
            {
                Assert.False(float.IsNaN(logits0[i]), $"NaN at logit {i}");
                Assert.False(float.IsInfinity(logits0[i]), $"Infinity at logit {i}");
            }

            // Step 1 forward
            var logits1 = forwardPass.Forward(token: 12, position: 1).ToArray();
            Assert.Equal(vocabSize, logits1.Length);
            for (int i = 0; i < vocabSize; i++)
            {
                Assert.False(float.IsNaN(logits1[i]), $"NaN at logit {i}");
                Assert.False(float.IsInfinity(logits1[i]), $"Infinity at logit {i}");
            }

            // Logits should differ between steps as state and context evolve
            bool anyDiff = false;
            for (int i = 0; i < vocabSize; i++)
            {
                if (Math.Abs(logits0[i] - logits1[i]) > 1e-5f)
                {
                    anyDiff = true;
                    break;
                }
            }
            Assert.True(anyDiff, "Step 1 logits must differ from Step 0 due to recurrent state and attention");

            // ResetCache test
            forwardPass.ResetCache();
            var logitsReset = forwardPass.Forward(token: 5, position: 0).ToArray();
            for (int i = 0; i < vocabSize; i++)
            {
                Assert.Equal(logits0[i], logitsReset[i], 1e-4f);
            }
        }
        finally
        {
            foreach (var p in allocatedPointers)
            {
                System.Runtime.InteropServices.NativeMemory.Free((void*)p);
            }
        }
    }

    [Fact]
    public unsafe void Qwen4Exp_SyntheticForwardPass_RunsWithQsaSparseSelection_AndPleTable()
    {
        const int embedDim = 32;
        const int hc = 4;
        const int hcDim = hc * embedDim;
        const int hcLowRank = 8;
        const int numHeads = 4;
        const int numHeadsKv = 2;
        const int headDim = 8;
        const int vocabSize = 32;
        const int shExpDim = 16;
        const int ssmDState = 8;
        const int ssmDtRank = 4;
        const int ssmNGroup = 2;
        const int ssmDConv = 2;
        int keyDim = ssmDState * ssmNGroup;
        int valDim = ssmDState * ssmDtRank;
        int qkvDim = keyDim * 2 + valDim;
        const int pleConvKernel = 2;
        const int pleNgramSize = 2;
        const int indexerTopK = 8;
        const int indexerKPool = 4;
        const int pleTableRows = 16;

        var hp = new Qwen4ExpHyperparams
        {
            EmbedDim = embedDim,
            HyperConnectionCount = hc,
            HyperConnectionLowRank = hcLowRank,
            NumHeads = numHeads,
            NumHeadsKv = numHeadsKv,
            HeadDim = headDim,
            NumLayer = 2,
            VocabSize = vocabSize,
            ContextLength = 512,
            SsmStateSize = ssmDState,
            SsmInnerSize = embedDim,
            SsmConvKernel = ssmDConv,
            SsmDtRank = ssmDtRank,
            SsmGroupCount = ssmNGroup,
            PleConvKernel = pleConvKernel,
            PleNgramSize = pleNgramSize,
            PleLayers = [0],
            PleLayerMultipliers = [1UL, 31UL],
            PleHeadOffsets = [0U, 8U],
            PleHeadVocabSizes = [8U, 8U],
            IndexerHeadCount = 2,
            IndexerKeyLength = headDim,
            IndexerTopK = indexerTopK,
            IndexerKPool = indexerKPool,
            RopeDimensionSections = [4, 2, 2, 0],
            RecurrentLayers = [true, false],
            RmsNormEps = 1e-6f,
        };

        var allocatedPointers = new List<nint>();
        Qwen4ExpTensorRef CreateRef(string name, long[] dims, float fillVal = 0.01f)
        {
            long count = 1;
            foreach (var d in dims) count *= d;
            nint mem = (nint)System.Runtime.InteropServices.NativeMemory.Alloc((nuint)count, sizeof(float));
            allocatedPointers.Add(mem);
            float* p = (float*)mem;
            for (long i = 0; i < count; i++) p[i] = fillVal + (float)((i % 13) * 0.001);
            var info = new OpenTail.Stingray.Core.GgufTensorInfo(name, dims.Length, dims, OpenTail.Stingray.Core.DType.Float32, 0);
            return new Qwen4ExpTensorRef(name, info, (byte*)p);
        }

        try
        {
            var tokEmbd = CreateRef("token_embd.weight", [embedDim, vocabSize], 0.05f);
            var output = CreateRef("output.weight", [embedDim, vocabSize], 0.05f);
            var hcHeadNorm = CreateRef("hc_head_norm.weight", [embedDim, hc], 1.0f);
            var hcHeadDown = CreateRef("hc_head_down.weight", [hcDim, hcLowRank], 0.02f);
            var hcHeadUp = CreateRef("hc_head_up.weight", [hcLowRank, hcDim], 0.02f);
            int pleHeadDim = embedDim / 2;
            var perLayerTokEmbd = CreateRef("per_layer_token_embd.weight", [pleHeadDim, pleTableRows], 0.03f);

            var layer0 = new Qwen4ExpLayerTensors
            {
                LayerIndex = 0,
                IsRecurrent = true,
                IsPle = true,
                HcAttnNorm = CreateRef("blk.0.hc_attn_norm.weight", [embedDim, hc], 1.0f),
                HcAttnDown = CreateRef("blk.0.hc_attn_down.weight", [hcDim, hcLowRank], 0.02f),
                HcAttnUp = CreateRef("blk.0.hc_attn_up.weight", [hcLowRank, hcDim], 0.02f),
                HcAttnInject = CreateRef("blk.0.hc_attn_inject.weight", [hcDim, hc], 0.01f),
                HcFfnNorm = CreateRef("blk.0.hc_ffn_norm.weight", [embedDim, hc], 1.0f),
                HcFfnDown = CreateRef("blk.0.hc_ffn_down.weight", [hcDim, hcLowRank], 0.02f),
                HcFfnUp = CreateRef("blk.0.hc_ffn_up.weight", [hcLowRank, hcDim], 0.02f),
                HcFfnInject = CreateRef("blk.0.hc_ffn_inject.weight", [hcDim, hc], 0.01f),

                AttnQkv = CreateRef("blk.0.attn_qkv.weight", [embedDim, qkvDim], 0.02f),
                AttnGate = CreateRef("blk.0.attn_gate.weight", [embedDim, valDim], 0.02f),
                SsmConv1d = CreateRef("blk.0.ssm_conv1d.weight", [ssmDConv, qkvDim], 0.1f),
                SsmDt = CreateRef("blk.0.ssm_dt.bias", [ssmDtRank], 0.1f),
                SsmA = CreateRef("blk.0.ssm_a", [ssmDtRank], -1.0f),
                SsmBeta = CreateRef("blk.0.ssm_beta.weight", [embedDim, ssmDtRank], 0.05f),
                SsmAlpha = CreateRef("blk.0.ssm_alpha.weight", [embedDim, ssmDtRank], 0.05f),
                SsmNorm = CreateRef("blk.0.ssm_norm.weight", [ssmDState], 1.0f),
                SsmOut = CreateRef("blk.0.ssm_out.weight", [valDim, embedDim], 0.05f),

                PleKey = CreateRef("blk.0.ple_key.weight", [embedDim, hcDim], 0.02f),
                PleValue = CreateRef("blk.0.ple_value.weight", [embedDim, embedDim], 0.02f),
                PleNormKey = CreateRef("blk.0.ple_norm_key.weight", [embedDim, hc], 1.0f),
                PleNormQuery = CreateRef("blk.0.ple_norm_query.weight", [embedDim, hc], 1.0f),
                PleNormConv = CreateRef("blk.0.ple_norm_conv.weight", [embedDim, hc], 1.0f),
                PleConv1d = CreateRef("blk.0.ple_conv1d.weight", [pleConvKernel, hcDim], 0.1f),

                FfnGateShexp = CreateRef("blk.0.ffn_gate_shexp.weight", [embedDim, shExpDim], 0.02f),
                FfnUpShexp = CreateRef("blk.0.ffn_up_shexp.weight", [embedDim, shExpDim], 0.02f),
                FfnDownShexp = CreateRef("blk.0.ffn_down_shexp.weight", [shExpDim, embedDim], 0.02f),
                FfnGateInpShexp = CreateRef("blk.0.ffn_gate_inp_shexp.weight", [embedDim], 0.05f),
            };

            var layer1 = new Qwen4ExpLayerTensors
            {
                LayerIndex = 1,
                IsRecurrent = false,
                IsPle = false,
                HcAttnNorm = CreateRef("blk.1.hc_attn_norm.weight", [embedDim, hc], 1.0f),
                HcAttnDown = CreateRef("blk.1.hc_attn_down.weight", [hcDim, hcLowRank], 0.02f),
                HcAttnUp = CreateRef("blk.1.hc_attn_up.weight", [hcLowRank, hcDim], 0.02f),
                HcAttnInject = CreateRef("blk.1.hc_attn_inject.weight", [hcDim, hc], 0.01f),
                HcFfnNorm = CreateRef("blk.1.hc_ffn_norm.weight", [embedDim, hc], 1.0f),
                HcFfnDown = CreateRef("blk.1.hc_ffn_down.weight", [hcDim, hcLowRank], 0.02f),
                HcFfnUp = CreateRef("blk.1.hc_ffn_up.weight", [hcLowRank, hcDim], 0.02f),
                HcFfnInject = CreateRef("blk.1.hc_ffn_inject.weight", [hcDim, hc], 0.01f),

                AttnQ = CreateRef("blk.1.attn_q.weight", [embedDim, numHeads * headDim * 2], 0.02f),
                AttnK = CreateRef("blk.1.attn_k.weight", [embedDim, numHeadsKv * headDim], 0.02f),
                AttnV = CreateRef("blk.1.attn_v.weight", [embedDim, numHeadsKv * headDim], 0.02f),
                AttnOut = CreateRef("blk.1.attn_output.weight", [numHeads * headDim, embedDim], 0.05f),
                AttnQNorm = CreateRef("blk.1.attn_q_norm.weight", [headDim], 1.0f),
                AttnKNorm = CreateRef("blk.1.attn_k_norm.weight", [headDim], 1.0f),

                IndexQProj = CreateRef("blk.1.index_q_proj.weight", [embedDim, 2 * headDim], 0.02f),
                IndexKProj = CreateRef("blk.1.index_k_proj.weight", [embedDim, headDim], 0.02f),
                IndexQNorm = CreateRef("blk.1.index_q_norm.weight", [headDim], 1.0f),
                IndexKNorm = CreateRef("blk.1.index_k_norm.weight", [headDim], 1.0f),

                FfnGateShexp = CreateRef("blk.1.ffn_gate_shexp.weight", [embedDim, shExpDim], 0.02f),
                FfnUpShexp = CreateRef("blk.1.ffn_up_shexp.weight", [embedDim, shExpDim], 0.02f),
                FfnDownShexp = CreateRef("blk.1.ffn_down_shexp.weight", [shExpDim, embedDim], 0.02f),
                FfnGateInpShexp = CreateRef("blk.1.ffn_gate_inp_shexp.weight", [embedDim], 0.05f),
            };

            var tensorSet = new Qwen4ExpTensorSet(
                tokEmbd, output, hcHeadNorm, hcHeadDown, hcHeadUp, perLayerTokEmbd, [layer0, layer1]);

            using var forwardPass = new Qwen4ExpForwardPass(null!, hp, tensorSet);

            // Execute 6 sequential forward steps straddling K-pool boundary (kpool=4)
            // Step 0..3: pool 0 accumulates; at step 3 (count=4), pool 0 completes and gets pooled/RoPE'd.
            // Step 4..5: pool 1 begins as active tail.
            for (int step = 0; step < 6; step++)
            {
                var logits = forwardPass.Forward(token: 3 + step, position: step).ToArray();
                Assert.Equal(vocabSize, logits.Length);
                for (int i = 0; i < vocabSize; i++)
                {
                    Assert.False(float.IsNaN(logits[i]), $"NaN at step {step} logit {i}");
                    Assert.False(float.IsInfinity(logits[i]), $"Infinity at step {step} logit {i}");
                }
            }
        }
        finally
        {
            foreach (var p in allocatedPointers)
            {
                System.Runtime.InteropServices.NativeMemory.Free((void*)p);
            }
        }
    }

    [Fact]
    public void PleHasher_KnownTokenSequence_CalculatesXorHashAndHeadIndices()
    {
        // 3 tokens, ngram=3, multipliers=[3, 5, 7]
        // heads=2, offsets=[100, 200], vocabSizes=[1000, 2000]
        var hp = new Qwen4ExpHyperparams
        {
            PleNgramSize = 3,
            PleLayerMultipliers = [3UL, 5UL, 7UL],
            PleHeadOffsets = [100U, 200U],
            PleHeadVocabSizes = [1000U, 2000U],
        };

        var hasher = new Qwen4ExpPleHasher(hp);
        Span<long> rowIndices = stackalloc long[2];

        // Step 0: token 10
        // k=0: tok=10, mult=3 -> 30. hash=30.
        // head 0: 100 + (30 % 1000) = 130.
        // head 1: 200 + (30 % 2000) = 230.
        hasher.PushToken(10);
        hasher.ComputeRowIndices(rowIndices);
        Assert.Equal(130, rowIndices[0]);
        Assert.Equal(230, rowIndices[1]);

        // Step 1: token 20. history: [10, 20]
        // k=0: tok=20, mult=3 -> 60.
        // k=1: tok=10, mult=5 -> 50.
        // 60 = 0b00111100, 50 = 0b00110010 -> 60 ^ 50 = 0b00001110 = 14.
        // Note: With addition it would be 60 + 50 = 110!
        hasher.PushToken(20);
        hasher.ComputeRowIndices(rowIndices);
        Assert.Equal(100 + 14, rowIndices[0]);
        Assert.Equal(200 + 14, rowIndices[1]);

        // Step 2: token 30. history: [10, 20, 30]
        // Head 0 is in bigram group (n = 2):
        // k=0: tok=30, mult=3 -> 90.
        // k=1: tok=20, mult=5 -> 100. (90 ^ 100 = 62)
        // row[0] = 100 + 62 = 162.
        // Head 1 is in trigram group (n = 3):
        // k=0: tok=30, mult=3 -> 90.
        // k=1: tok=20, mult=5 -> 100. (90 ^ 100 = 62)
        // k=2: tok=10, mult=7 -> 70. (62 ^ 70 = 120)
        // row[1] = 200 + 120 = 320.
        hasher.PushToken(30);
        hasher.ComputeRowIndices(rowIndices);
        Assert.Equal(100 + 62, rowIndices[0]);
        Assert.Equal(200 + 120, rowIndices[1]);
    }

    [Fact]
    public void PleHasher_ChunkedVsSingleShot_HistoryContinuity()
    {
        var hp = new Qwen4ExpHyperparams
        {
            PleNgramSize = 4,
            PleLayerMultipliers = [17UL, 31UL, 53UL, 97UL],
            PleHeadOffsets = [0U, 1024U, 2048U, 4096U],
            PleHeadVocabSizes = [1024U, 1024U, 2048U, 4096U],
        };

        var singleShot = new Qwen4ExpPleHasher(hp);
        var chunked = new Qwen4ExpPleHasher(hp);

        int[] tokens = [42, 101, 7, 888, 1234, 55];

        // Single shot
        for (int i = 0; i < tokens.Length; i++)
        {
            singleShot.PushToken(tokens[i]);
        }

        // Chunked: push 2 tokens, then 4 tokens
        chunked.PushTokens(tokens.AsSpan(0, 2));
        chunked.PushTokens(tokens.AsSpan(2, 4));

        Span<long> singleIndices = stackalloc long[4];
        Span<long> chunkedIndices = stackalloc long[4];

        singleShot.ComputeRowIndices(singleIndices);
        chunked.ComputeRowIndices(chunkedIndices);

        for (int h = 0; h < 4; h++)
        {
            Assert.Equal(singleIndices[h], chunkedIndices[h]);
        }
    }

    [Fact]
    public void Qsa_SelectTopKPools_ExcludesLowScoringPoolsAndRetainsTail()
    {
        // 5 pools total: scores [0.1, 5.0, 0.2, 8.0, 0.3]
        // topK=2 pools -> pool 3 (score 8.0) and pool 1 (score 5.0) should be selected.
        float[] poolScores = [0.1f, 5.0f, 0.2f, 8.0f, 0.3f];
        Span<int> selected = stackalloc int[poolScores.Length];

        int count = Qwen4ExpQsa.SelectTopKPools(poolScores, topKPoolCount: 2, selected);

        Assert.Equal(2, count);
        Assert.True(selected[0] == 3 || selected[1] == 3);
        Assert.True(selected[0] == 1 || selected[1] == 1);
        Assert.DoesNotContain(0, selected.Slice(0, count).ToArray());
        Assert.DoesNotContain(2, selected.Slice(0, count).ToArray());
        Assert.DoesNotContain(4, selected.Slice(0, count).ToArray());
    }

    [Fact]
    public void Qwen4ExpRope_FourSectionImRope_RotatesSectionsCorrectly()
    {
        // 4 sections: s0=1, s1=1, s2=1, s3=1. Total pairs = 4 -> headDim = 8, halfDim = 4.
        // In NeoX-style RoPE, pair i consists of elements (i, i + halfDim).
        // Pairs 0, 1, 2 correspond to sections 0, 1, 2 (rotated).
        // Pair 3 (elements 3 and 7) corresponds to section 3 (strictly unrotated per IMRoPE spec).
        int[] sections = [1, 1, 1, 1];
        float[] vec = [1f, 2f, 3f, 99f, 5f, 6f, 7f, 100f];
        float[] original = (float[])vec.Clone();

        Qwen4ExpRope.ApplyImRope(vec, pos: 10, numHeads: 1, headDim: 8, sections);

        // Pairs 0, 1, 2 (indices 0, 1, 2 and 4, 5, 6) must have rotated
        for (int i = 0; i < 3; i++)
        {
            Assert.NotEqual(original[i], vec[i]);
            Assert.NotEqual(original[i + 4], vec[i + 4]);
        }

        // Pair 3 (indices 3 and 7) must remain strictly identical
        Assert.Equal(original[3], vec[3]);
        Assert.Equal(original[7], vec[7]);
    }

    [Fact]
    public void PleHasher_BigramVsTrigram_16HeadGroups()
    {
        // Actual Qwen4Exp geometry: ngram_size = 3, heads_per_ngram = 8 -> 16 total heads
        // Heads 0..7 are bigrams (n=2), Heads 8..15 are trigrams (n=3)
        uint[] offsets = new uint[16];
        uint[] vocabSizes = new uint[16];
        for (int h = 0; h < 16; h++)
        {
            offsets[h] = (uint)(h * 1000);
            vocabSizes[h] = 10000U;
        }

        var hp = new Qwen4ExpHyperparams
        {
            PleNgramSize = 3,
            PleHeadsPerNgram = 8,
            PleLayerMultipliers = [3UL, 5UL, 7UL],
            PleHeadOffsets = offsets,
            PleHeadVocabSizes = vocabSizes,
        };

        var hasher = new Qwen4ExpPleHasher(hp);
        Assert.Equal(16, hasher.NumHeads);
        Assert.Equal(8, hasher.HeadsPerNgram);

        // Sequence: [10, 20, 30]
        hasher.PushTokens([10, 20, 30]);

        Span<long> rows = stackalloc long[16];
        hasher.ComputeRowIndices(rows);

        // Bigram hash for [20, 30]: 30 * 3 ^ 20 * 5 = 90 ^ 100 = 62
        for (int h = 0; h < 8; h++)
        {
            long expected = offsets[h] + 62;
            Assert.Equal(expected, rows[h]);
        }

        // Trigram hash for [10, 20, 30]: 30 * 3 ^ 20 * 5 ^ 10 * 7 = 62 ^ 70 = 120
        for (int h = 8; h < 16; h++)
        {
            long expected = offsets[h] + 120;
            Assert.Equal(expected, rows[h]);
        }
    }

    [Fact]
    public void PleHasher_BoundaryAndEosSemantics()
    {
        const int eosId = 248044;
        var hp = new Qwen4ExpHyperparams
        {
            PleNgramSize = 3,
            PleHeadsPerNgram = 1,
            PleEosTokenId = eosId,
            PleLayerMultipliers = [2UL, 3UL, 5UL],
            PleHeadOffsets = [0U, 1000000U],
            PleHeadVocabSizes = [10000000U, 10000000U],
        };

        var hasher = new Qwen4ExpPleHasher(hp);
        Span<long> rows = stackalloc long[2];

        // 1. [first token]: token 10
        // Bigram: 10 * 2 ^ eos * 3
        // Trigram: 10 * 2 ^ eos * 3 ^ eos * 5
        hasher.PushToken(10);
        hasher.ComputeRowIndices(rows);
        ulong expBi1 = (10UL * 2UL) ^ ((ulong)eosId * 3UL);
        ulong expTri1 = expBi1 ^ ((ulong)eosId * 5UL);
        Assert.Equal((long)expBi1, rows[0]);
        Assert.Equal(1000000L + (long)expTri1, rows[1]);

        // 2. [token after normal token]: token 20 -> history [10, 20]
        // Bigram: 20 * 2 ^ 10 * 3
        // Trigram: 20 * 2 ^ 10 * 3 ^ eos * 5
        hasher.PushToken(20);
        hasher.ComputeRowIndices(rows);
        ulong expBi2 = (20UL * 2UL) ^ (10UL * 3UL);
        ulong expTri2 = expBi2 ^ ((ulong)eosId * 5UL);
        Assert.Equal((long)expBi2, rows[0]);
        Assert.Equal(1000000L + (long)expTri2, rows[1]);

        // 3. [current token == EOS]: token 248044 -> history [10, 20, eos]
        // Current token is EOS but it does not erase its own context!
        // Bigram: eos * 2 ^ 20 * 3
        // Trigram: eos * 2 ^ 20 * 3 ^ 10 * 5
        hasher.PushToken(eosId);
        hasher.ComputeRowIndices(rows);
        ulong expBi3 = ((ulong)eosId * 2UL) ^ (20UL * 3UL);
        ulong expTri3 = expBi3 ^ (10UL * 5UL);
        Assert.Equal((long)expBi3, rows[0]);
        Assert.Equal(1000000L + (long)expTri3, rows[1]);

        // 4. [token after EOS]: token 30 -> history [10, 20, eos, 30]
        // Predecessor is EOS; older token (20) is cut off by the intervening EOS and replaced with EOS!
        // Bigram: 30 * 2 ^ eos * 3
        // Trigram: 30 * 2 ^ eos * 3 ^ eos * 5
        hasher.PushToken(30);
        hasher.ComputeRowIndices(rows);
        ulong expBi4 = (30UL * 2UL) ^ ((ulong)eosId * 3UL);
        ulong expTri4 = expBi4 ^ ((ulong)eosId * 5UL);
        Assert.Equal((long)expBi4, rows[0]);
        Assert.Equal(1000000L + (long)expTri4, rows[1]);
    }

    [Fact]
    public void RoPE_Qwen38Geometry_PartialRotaryFactor_Theta10M()
    {
        // Qwen3.8 configuration: headDim = 128, partial_rotary_factor = 0.25 -> ropeDim = 32
        // sections = [11, 11, 10]
        // ropeTheta = 10,000,000
        const int headDim = 128;
        const int ropeDim = 32;
        int[] sections = [11, 11, 10];

        float[] vec = new float[headDim];
        for (int i = 0; i < headDim; i++) vec[i] = 1.0f + i * 0.1f;
        float[] original = (float[])vec.Clone();

        Qwen4ExpRope.ApplyImRope(vec, pos: 5, numHeads: 1, headDim: headDim, sections: sections, ropeDim: ropeDim, ropeTheta: 10000000.0f);

        // First ropeDim (0..31) are rotated
        int halfRope = ropeDim / 2; // 16
        for (int i = 0; i < halfRope; i++)
        {
            Assert.NotEqual(original[i], vec[i]);
            Assert.NotEqual(original[i + halfRope], vec[i + halfRope]);
        }

        // Remaining dimensions [32..127] must be strictly unrotated pass-through!
        for (int i = ropeDim; i < headDim; i++)
        {
            Assert.Equal(original[i], vec[i]);
        }
    }

    [Fact]
    public void Qsa_CandidateSelection_AlwaysRetainsActiveTail()
    {
        // L = 6 tokens, kpool = 4
        // Complete pool 0: tokens 0..3
        // Incomplete tail: tokens 4..5
        // Even if pool 0 has negative score, active tail tokens [4, 5] must always be in selected set.
        const int kpool = 4;
        const int totalTokens = 6;
        int completedPools = 1;
        float[] poolScores = [-100.0f]; // Very bad score

        Span<int> selectedPools = stackalloc int[1];
        int nSelected = Qwen4ExpQsa.SelectTopKPools(poolScores, topKPoolCount: 1, selectedPools);

        var selectedTokenIndices = new HashSet<int>();
        for (int i = 0; i < nSelected; i++)
        {
            int p = selectedPools[i];
            int startToken = p * kpool;
            for (int t = 0; t < kpool; t++) selectedTokenIndices.Add(startToken + t);
        }

        // Active tail tokens are unconditionally added per Qwen4Exp spec
        int tailStart = completedPools * kpool;
        for (int t = tailStart; t < totalTokens; t++)
        {
            selectedTokenIndices.Add(t);
        }

        // Verify active tail tokens 4 and 5 are unconditionally retained
        Assert.Contains(4, selectedTokenIndices);
        Assert.Contains(5, selectedTokenIndices);
        Assert.Equal(6, selectedTokenIndices.Count);
    }

    [Fact]
    public unsafe void Ple_Concatenation_AssemblyGeometry()
    {
        // 4 heads, pleHeadDim = 4 -> embedDim = 16.
        // Head 0 gets row 0 filled with 1.0f
        // Head 1 gets row 1 filled with 2.0f
        // Head 2 gets row 2 filled with 3.0f
        // Head 3 gets row 3 filled with 4.0f
        // Concatenation must produce [1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4].
        // If averaged, it would produce [2.5, 2.5, 2.5, 2.5].
        const int numHeads = 4;
        const int pleHeadDim = 4;
        const int embedDim = numHeads * pleHeadDim; // 16
        const int totalRows = 4;

        float[] table = new float[pleHeadDim * totalRows];
        for (int r = 0; r < totalRows; r++)
        {
            for (int d = 0; d < pleHeadDim; d++)
            {
                table[r * pleHeadDim + d] = (float)(r + 1);
            }
        }

        fixed (float* pTable = table)
        {
            var info = new OpenTail.Stingray.Core.GgufTensorInfo(
                "per_layer_token_embd.weight", 2, [pleHeadDim, totalRows], OpenTail.Stingray.Core.DType.Float32, 0);
            var perLayerRef = new Qwen4ExpTensorRef("per_layer_token_embd.weight", info, (byte*)pTable);

            // Mock pleHeadRows mapping: head h -> row h
            long[] headRows = [0L, 1L, 2L, 3L];

            Span<float> pleEmb = stackalloc float[embedDim];
            pleEmb.Clear();

            int bytesPerRow = pleHeadDim * sizeof(float);
            for (int h = 0; h < numHeads; h++)
            {
                long row = headRows[h];
                byte* rowPtr = perLayerRef.DataPtr + row * bytesPerRow;
                var headSlice = pleEmb.Slice(h * pleHeadDim, pleHeadDim);
                fixed (float* pHead = headSlice)
                {
                    OpenTail.Stingray.Cpu.SimdKernels.DequantRow(rowPtr, pHead, pleHeadDim, info.DType);
                }
            }

            // Verify concatenation: slice 0 has 1.0, slice 1 has 2.0, slice 2 has 3.0, slice 3 has 4.0
            for (int h = 0; h < numHeads; h++)
            {
                float expected = (float)(h + 1);
                for (int d = 0; d < pleHeadDim; d++)
                {
                    Assert.Equal(expected, pleEmb[h * pleHeadDim + d]);
                }
            }
        }
    }
}

