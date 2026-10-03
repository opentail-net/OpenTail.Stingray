using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class Glm5NextSyntheticTests : IDisposable
{
    private readonly List<string> _tempFiles = [];

    public void Dispose()
    {
        foreach (var file in _tempFiles)
        {
            try { if (File.Exists(file)) File.Delete(file); } catch { }
        }
    }

    [Fact]
    public void Glm5NextHyperparams_FromGgufMetadata_ParsesExpectedKeys()
    {
        var metadata = new Dictionary<string, object>
        {
            ["glm5next.leading_dense_block_count"] = (ulong)3,
            ["glm5next.expert_weights_scale"] = 2.5f,
            ["glm5next.expert_weights_norm"] = true,
            ["glm5next.expert_shared_count"] = (ulong)1,
            ["glm5next.expert_feed_forward_length"] = (ulong)2048,
            ["glm5next.attention.q_lora_rank"] = (ulong)1536,
            ["glm5next.attention.kv_lora_rank"] = (ulong)512,
            ["glm5next.attention.key_length_mla"] = (ulong)128,
            ["glm5next.attention.value_length_mla"] = (ulong)128,
            ["glm5next.kda.head_dim"] = (ulong)128,
            ["glm5next.ssm.conv_kernel"] = (ulong)4,
            ["glm5next.attention.indexer.head_count"] = (ulong)32,
            ["glm5next.attention.indexer.key_length"] = (ulong)128,
            ["glm5next.attention.indexer.top_k"] = (ulong)2048,
            ["glm5next.attention.indexer.kpool"] = (ulong)4,
            ["glm5next.hyper_connection.count"] = (ulong)4,
            ["glm5next.hyper_connection.sinkhorn_iterations"] = (ulong)20,
            ["glm5next.hyper_connection.epsilon"] = 1e-6f,
            ["glm5next.swiglu_clamp_exp"] = 10.0f,
            ["glm5next.expert_gating_func"] = (ulong)2,
        };

        var hp = Glm5NextHyperparams.FromGgufMetadata(
            metadata, "glm5next", numLayerAll: 45, embedDim: 4096, numHeads: 32, headDim: 128,
            numExperts: 288, numExpertsUsed: 8);

        Assert.Equal(45, hp.NumLayer);
        Assert.Equal(3, hp.LeadingDenseBlockCount);
        Assert.Equal(2.5f, hp.ExpertWeightsScale);
        Assert.True(hp.ExpertWeightsNorm);
        Assert.Equal(1, hp.ExpertSharedCount);
        Assert.Equal(1536, hp.QLoraRank);
        Assert.Equal(512, hp.KvLoraRank);
        Assert.Equal(128, hp.HeadDimKda);
        Assert.Equal(4, hp.SsmDConv);
        Assert.Equal(32, hp.IndexerNumHeads);
        Assert.Equal(128, hp.IndexerHeadSize);
        Assert.Equal(2048, hp.IndexerTopK);
        Assert.Equal(4, hp.IndexerKPool);
        Assert.Equal(4, hp.HcMult);
        Assert.Equal(20, hp.HcSinkhornIters);
        Assert.Equal(1e-6f, hp.HcEps);
        Assert.Equal(10.0f, hp.SwiGluClampExp);
        Assert.Equal(2, hp.ExpertGatingFunc);
    }

    [Fact]
    public void Glm5Next_TrunkPattern_FollowsStrict3To1RecurrentMlaPattern()
    {
        var hp = new Glm5NextHyperparams { NumLayerAll = 45 };

        // 3 KDA followed by 1 MLA pattern
        Assert.True(hp.IsRecurrent(0));
        Assert.True(hp.IsRecurrent(1));
        Assert.True(hp.IsRecurrent(2));
        Assert.False(hp.IsRecurrent(3)); // MLA

        Assert.True(hp.IsRecurrent(4));
        Assert.True(hp.IsRecurrent(5));
        Assert.True(hp.IsRecurrent(6));
        Assert.False(hp.IsRecurrent(7)); // MLA

        Assert.True(hp.IsRecurrent(40));
        Assert.True(hp.IsRecurrent(41));
        Assert.True(hp.IsRecurrent(42));
        Assert.False(hp.IsRecurrent(43)); // MLA
        Assert.True(hp.IsRecurrent(44));  // KDA
    }

    [Fact]
    public void Glm5Next_SyntheticForwardPass_ExecutesBothKdaAndMlaWithMhc()
    {
        const int embedDim = 32;
        const int numHeads = 2;
        const int headDimKda = 16;
        const int qLoraRank = 16;
        const int kvLoraRank = 16;
        const int headDimMla = 16;
        const int dConv = 4;
        const int dInner = numHeads * headDimKda; // 32
        const int indexerNumHeads = 2;
        const int indexerHeadDim = 16;
        const int kpool = 4;
        const int numExperts = 4;
        const int numExpertsUsed = 2;
        const int expertFfn = 32;
        const int denseFfn = 32;
        const int vocabSize = 32;
        const int hcMult = 4;
        const int hcMixDim = (2 + hcMult) * hcMult; // 24
        const int numLayers = 4; // layers 0, 1, 2 are KDA; layer 3 is MLA

        string path = Path.Combine(Path.GetTempPath(), $"glm5next_synthetic_{Guid.NewGuid():N}.gguf");
        _tempFiles.Add(path);

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

        AddTensor("token_embd.weight", [embedDim, vocabSize]);
        AddTensor("output_norm.weight", [embedDim]);
        AddTensor("output.weight", [embedDim, vocabSize]);

        for (int i = 0; i < numLayers; i++)
        {
            AddTensor($"blk.{i}.attn_norm.weight", [embedDim]);
            AddTensor($"blk.{i}.ffn_norm.weight", [embedDim]);

            // mHC
            AddTensor($"blk.{i}.hc_attn_fn.weight", [hcMult * embedDim, hcMixDim]);
            AddTensor($"blk.{i}.hc_attn_base.weight", [hcMixDim]);
            AddTensor($"blk.{i}.hc_attn_scale.weight", [3]);
            AddTensor($"blk.{i}.hc_ffn_fn.weight", [hcMult * embedDim, hcMixDim]);
            AddTensor($"blk.{i}.hc_ffn_base.weight", [hcMixDim]);
            AddTensor($"blk.{i}.hc_ffn_scale.weight", [3]);

            if (i < 3) // KDA layers (0, 1, 2)
            {
                AddTensor($"blk.{i}.ssm_conv1d_q.weight", [dConv, dInner]);
                AddTensor($"blk.{i}.ssm_conv1d_k.weight", [dConv, dInner]);
                AddTensor($"blk.{i}.ssm_conv1d_v.weight", [dConv, dInner]);
                AddTensor($"blk.{i}.attn_q.weight", [embedDim, dInner]);
                AddTensor($"blk.{i}.attn_k.weight", [embedDim, dInner]);
                AddTensor($"blk.{i}.attn_v.weight", [embedDim, dInner]);
                AddTensor($"blk.{i}.ssm_f_a.weight", [embedDim, headDimKda]);
                AddTensor($"blk.{i}.ssm_f_b.weight", [headDimKda, dInner]);
                AddTensor($"blk.{i}.ssm_beta.weight", [embedDim, numHeads]);
                AddTensor($"blk.{i}.ssm_a.weight", [numHeads]);
                AddTensor($"blk.{i}.ssm_dt.bias", [dInner]);
                AddTensor($"blk.{i}.ssm_g_a.weight", [embedDim, headDimKda]);
                AddTensor($"blk.{i}.ssm_g_b.weight", [headDimKda, dInner]);
                AddTensor($"blk.{i}.ssm_norm.weight", [dInner]);
                AddTensor($"blk.{i}.attn_output.weight", [dInner, embedDim]);
            }
            else // MLA layer (3)
            {
                AddTensor($"blk.{i}.attn_q_a_norm.weight", [qLoraRank]);
                AddTensor($"blk.{i}.attn_kv_a_norm.weight", [kvLoraRank]);
                AddTensor($"blk.{i}.attn_q_a.weight", [embedDim, qLoraRank]);
                AddTensor($"blk.{i}.attn_q_b.weight", [qLoraRank, numHeads * headDimMla]);
                AddTensor($"blk.{i}.attn_kv_a_mqa.weight", [embedDim, kvLoraRank]);
                AddTensor($"blk.{i}.attn_k_b.weight", [headDimMla, kvLoraRank, numHeads]);
                AddTensor($"blk.{i}.attn_v_b.weight", [kvLoraRank, headDimMla, numHeads]);
                AddTensor($"blk.{i}.attn_output.weight", [numHeads * headDimMla, embedDim]);

                AddTensor($"blk.{i}.indexer_k_norm.weight", [indexerHeadDim]);
                AddTensor($"blk.{i}.indexer_k_norm.bias", [indexerHeadDim]);
                AddTensor($"blk.{i}.indexer_proj.weight", [embedDim, indexerNumHeads]);
                AddTensor($"blk.{i}.indexer_attn_k.weight", [embedDim, indexerHeadDim]);
                AddTensor($"blk.{i}.indexer_attn_q_b.weight", [qLoraRank, indexerNumHeads * indexerHeadDim]);
                AddTensor($"blk.{i}.indexer_kpool_gate.weight", [embedDim, indexerHeadDim]);
                AddTensor($"blk.{i}.indexer_kpool_ape.weight", [indexerHeadDim, kpool]);
            }

            if (i == 0)
            {
                AddTensor($"blk.{i}.ffn_gate.weight", [embedDim, denseFfn]);
                AddTensor($"blk.{i}.ffn_up.weight", [embedDim, denseFfn]);
                AddTensor($"blk.{i}.ffn_down.weight", [denseFfn, embedDim]);
            }
            else
            {
                AddTensor($"blk.{i}.ffn_gate_inp.weight", [embedDim, numExperts]);
                AddTensor($"blk.{i}.ffn_gate_exps.weight", [embedDim, expertFfn, numExperts]);
                AddTensor($"blk.{i}.ffn_up_exps.weight", [embedDim, expertFfn, numExperts]);
                AddTensor($"blk.{i}.ffn_down_exps.weight", [expertFfn, embedDim, numExperts]);
                AddTensor($"blk.{i}.ffn_gate_shexp.weight", [embedDim, expertFfn]);
                AddTensor($"blk.{i}.ffn_up_shexp.weight", [embedDim, expertFfn]);
                AddTensor($"blk.{i}.ffn_down_shexp.weight", [expertFfn, embedDim]);
            }
        }

        var metadata = new Dictionary<string, object>
        {
            ["general.architecture"] = "glm5next",
            ["glm5next.leading_dense_block_count"] = (ulong)1,
            ["glm5next.expert_weights_scale"] = 2.5f,
            ["glm5next.expert_weights_norm"] = true,
            ["glm5next.expert_shared_count"] = (ulong)1,
            ["glm5next.expert_feed_forward_length"] = (ulong)expertFfn,
            ["glm5next.attention.q_lora_rank"] = (ulong)qLoraRank,
            ["glm5next.attention.kv_lora_rank"] = (ulong)kvLoraRank,
            ["glm5next.attention.key_length_mla"] = (ulong)headDimMla,
            ["glm5next.attention.value_length_mla"] = (ulong)headDimMla,
            ["glm5next.kda.head_dim"] = (ulong)headDimKda,
            ["glm5next.ssm.conv_kernel"] = (ulong)dConv,
            ["glm5next.attention.indexer.head_count"] = (ulong)indexerNumHeads,
            ["glm5next.attention.indexer.key_length"] = (ulong)indexerHeadDim,
            ["glm5next.attention.indexer.top_k"] = (ulong)16,
            ["glm5next.attention.indexer.kpool"] = (ulong)kpool,
            ["glm5next.hyper_connection.count"] = (ulong)hcMult,
            ["glm5next.hyper_connection.sinkhorn_iterations"] = (ulong)20,
            ["glm5next.hyper_connection.epsilon"] = 1e-6f,
            ["glm5next.swiglu_clamp_exp"] = 10.0f,
            ["glm5next.expert_gating_func"] = (ulong)2,
        };

        WriteGgufFile(path, metadata, tensors);

        using var model = GgufModel.Open(path);
        var hp = Glm5NextHyperparams.FromGgufMetadata(
            model.Metadata, "glm5next", numLayerAll: numLayers, embedDim: embedDim, numHeads: numHeads,
            headDim: headDimKda, numExperts: numExperts, numExpertsUsed: numExpertsUsed);

        var forwardPass = new Glm5NextForwardPass(model, hp);
        Assert.Equal(vocabSize, forwardPass.VocabSize);

        // Step 0
        var logits0 = forwardPass.Forward(token: 3, position: 0);
        Assert.Equal(vocabSize, logits0.Length);
        foreach (var l in logits0)
        {
            Assert.False(float.IsNaN(l));
            Assert.False(float.IsInfinity(l));
        }

        // Step 1
        var logits1 = forwardPass.Forward(token: 7, position: 1);
        Assert.Equal(vocabSize, logits1.Length);
        foreach (var l in logits1)
        {
            Assert.False(float.IsNaN(l));
            Assert.False(float.IsInfinity(l));
        }
    }

    private static void WriteGgufFile(
        string path,
        Dictionary<string, object> metadata,
        Dictionary<string, (long[] shape, DType dtype, byte[] data)> tensors)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        writer.Write((byte)'G');
        writer.Write((byte)'G');
        writer.Write((byte)'U');
        writer.Write((byte)'F');
        writer.Write((uint)3);
        writer.Write((ulong)tensors.Count);
        writer.Write((ulong)metadata.Count);

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
            default:
                throw new NotSupportedException($"Value type {value.GetType()} not supported in synthetic writer.");
        }
    }
}
