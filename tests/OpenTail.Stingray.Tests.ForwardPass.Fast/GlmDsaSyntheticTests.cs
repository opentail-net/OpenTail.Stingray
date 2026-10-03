using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class GlmDsaSyntheticTests : IDisposable
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
    public void GlmDsaHyperparams_FromGgufMetadata_ParsesExpectedKeys()
    {
        var metadata = new Dictionary<string, object>
        {
            ["glm-dsa.leading_dense_block_count"] = (ulong)3,
            ["glm-dsa.expert_weights_scale"] = 2.5f,
            ["glm-dsa.expert_weights_norm"] = true,
            ["glm-dsa.expert_shared_count"] = (ulong)1,
            ["glm-dsa.attention.q_lora_rank"] = (ulong)2048,
            ["glm-dsa.attention.kv_lora_rank"] = (ulong)512,
            ["glm-dsa.attention.key_length_mla"] = (ulong)256,
            ["glm-dsa.attention.value_length_mla"] = (ulong)256,
            ["glm-dsa.attention.indexer.head_count"] = (ulong)32,
            ["glm-dsa.attention.indexer.key_length"] = (ulong)128,
            ["glm-dsa.attention.indexer.top_k"] = (ulong)2048,
            ["glm-dsa.expert_gating_func"] = (ulong)2,
            ["glm-dsa.attention.indexer.types"] = new object[] { 1, 1, 1, 0, 0, 0, 1, 0 },
        };

        var hp = GlmDsaHyperparams.FromGgufMetadata(
            metadata, "glm-dsa", numLayerAll: 78, embedDim: 7168, numHeads: 64, headDim: 256,
            ropeDim: 64, numExperts: 256, numExpertsUsed: 8);

        Assert.Equal(78, hp.NumLayer);
        Assert.Equal(3, hp.LeadingDenseBlockCount);
        Assert.Equal(2.5f, hp.ExpertWeightsScale);
        Assert.True(hp.ExpertWeightsNorm);
        Assert.Equal(1, hp.ExpertSharedCount);
        Assert.Equal(2048, hp.QLoraRank);
        Assert.Equal(512, hp.KvLoraRank);
        Assert.Equal(256, hp.EffectiveHeadDimK);
        Assert.Equal(256, hp.EffectiveHeadDimV);
        Assert.Equal(32, hp.IndexerNumHeads);
        Assert.Equal(128, hp.IndexerHeadSize);
        Assert.Equal(2048, hp.IndexerTopK);
        Assert.Equal(2, hp.ExpertGatingFunc);

        // Custom indexer types schedule
        Assert.True(hp.IsIndexerFull(0));
        Assert.True(hp.IsIndexerFull(1));
        Assert.True(hp.IsIndexerFull(2));
        Assert.False(hp.IsIndexerFull(3));
        Assert.False(hp.IsIndexerFull(4));
        Assert.False(hp.IsIndexerFull(5));
        Assert.True(hp.IsIndexerFull(6));
        Assert.False(hp.IsIndexerFull(7));
    }

    [Fact]
    public void GlmDsa_DefaultIndexerSchedule_FollowsGlm52Pattern()
    {
        var hp = new GlmDsaHyperparams();
        Assert.True(hp.IsIndexerFull(0));
        Assert.True(hp.IsIndexerFull(1));
        Assert.True(hp.IsIndexerFull(2));
        Assert.False(hp.IsIndexerFull(3));
        Assert.False(hp.IsIndexerFull(4));
        Assert.False(hp.IsIndexerFull(5));
        Assert.True(hp.IsIndexerFull(6));
        Assert.False(hp.IsIndexerFull(7));
        Assert.False(hp.IsIndexerFull(8));
        Assert.False(hp.IsIndexerFull(9));
        Assert.True(hp.IsIndexerFull(10));
    }

    [Fact]
    public unsafe void GlmDsa_HadamardRotation_IsInvolutoryAndOrthonormal()
    {
        const int blockSize = 128;
        float[] original = new float[blockSize];
        float[] vector = new float[blockSize];
        for (int i = 0; i < blockSize; i++)
        {
            original[i] = (float)Math.Sin(i * 0.17f);
            vector[i] = original[i];
        }

        fixed (float* ptr = vector)
        {
            // First transform
            PrismHadamard.ApplySylvesterHadamard(ptr, blockSize, blockSize);

            // Energy conservation (orthonormality: ||H x|| = ||x||)
            float originalNorm = 0f, transformedNorm = 0f;
            for (int i = 0; i < blockSize; i++)
            {
                originalNorm += original[i] * original[i];
                transformedNorm += vector[i] * vector[i];
            }
            Assert.Equal(MathF.Sqrt(originalNorm), MathF.Sqrt(transformedNorm), 4);

            // Second transform: H(H(x)) = x because H is symmetric and orthonormal
            PrismHadamard.ApplySylvesterHadamard(ptr, blockSize, blockSize);
            for (int i = 0; i < blockSize; i++)
            {
                Assert.Equal(original[i], vector[i], 4);
            }
        }
    }

    [Fact]
    public void GlmDsa_SyntheticForwardPass_ExecutesCorrectly()
    {
        const int embedDim = 32;
        const int numHeads = 2;
        const int qLoraRank = 16;
        const int kvLoraRank = 16;
        const int headDimK = 16;
        const int headDimV = 16;
        const int ropeDim = 8;
        const int indexerNumHeads = 2;
        const int indexerHeadDim = 128; // must be 128 for Hadamard
        const int numExperts = 4;
        const int numExpertsUsed = 2;
        const int expertFfn = 32;
        const int denseFfn = 32;
        const int vocabSize = 32;
        const int numLayers = 3; // layer 0 dense, layers 1 and 2 MoE

        string path = Path.Combine(Path.GetTempPath(), $"glm_dsa_synthetic_{Guid.NewGuid():N}.gguf");
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
            AddTensor($"blk.{i}.attn_q_a_norm.weight", [qLoraRank]);
            AddTensor($"blk.{i}.attn_kv_a_norm.weight", [kvLoraRank]);
            AddTensor($"blk.{i}.attn_q_a.weight", [embedDim, qLoraRank]);
            AddTensor($"blk.{i}.attn_q_b.weight", [qLoraRank, numHeads * headDimK]);
            AddTensor($"blk.{i}.attn_kv_a_mqa.weight", [embedDim, kvLoraRank + ropeDim]);
            AddTensor($"blk.{i}.attn_k_b.weight", [headDimK - ropeDim, kvLoraRank, numHeads]);
            AddTensor($"blk.{i}.attn_v_b.weight", [kvLoraRank, headDimV, numHeads]);
            AddTensor($"blk.{i}.attn_output.weight", [numHeads * headDimV, embedDim]);
            AddTensor($"blk.{i}.ffn_norm.weight", [embedDim]);

            // Indexer
            AddTensor($"blk.{i}.indexer_k_norm.weight", [indexerHeadDim]);
            AddTensor($"blk.{i}.indexer_k_norm.bias", [indexerHeadDim]);
            AddTensor($"blk.{i}.indexer_proj.weight", [embedDim, indexerNumHeads]);
            AddTensor($"blk.{i}.indexer_attn_k.weight", [embedDim, indexerHeadDim]);
            AddTensor($"blk.{i}.indexer_attn_q_b.weight", [qLoraRank, indexerNumHeads * indexerHeadDim]);

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
            ["general.architecture"] = "glm-dsa",
            ["glm-dsa.leading_dense_block_count"] = (ulong)1,
            ["glm-dsa.expert_weights_scale"] = 2.5f,
            ["glm-dsa.expert_weights_norm"] = true,
            ["glm-dsa.expert_shared_count"] = (ulong)1,
            ["glm-dsa.attention.q_lora_rank"] = (ulong)qLoraRank,
            ["glm-dsa.attention.kv_lora_rank"] = (ulong)kvLoraRank,
            ["glm-dsa.attention.key_length_mla"] = (ulong)headDimK,
            ["glm-dsa.attention.value_length_mla"] = (ulong)headDimV,
            ["glm-dsa.attention.indexer.head_count"] = (ulong)indexerNumHeads,
            ["glm-dsa.attention.indexer.key_length"] = (ulong)indexerHeadDim,
            ["glm-dsa.attention.indexer.top_k"] = (ulong)16,
            ["glm-dsa.expert_gating_func"] = (ulong)2,
        };

        WriteGgufFile(path, metadata, tensors);

        using var model = GgufModel.Open(path);
        var hp = GlmDsaHyperparams.FromGgufMetadata(
            model.Metadata, "glm-dsa", numLayerAll: numLayers, embedDim: embedDim, numHeads: numHeads,
            headDim: headDimK, ropeDim: ropeDim, numExperts: numExperts, numExpertsUsed: numExpertsUsed);

        var forwardPass = new GlmDsaForwardPass(model, hp);
        Assert.Equal(vocabSize, forwardPass.VocabSize);

        // Step 0
        var logits0 = forwardPass.Forward(token: 5, position: 0);
        Assert.Equal(vocabSize, logits0.Length);
        foreach (var l in logits0)
        {
            Assert.False(float.IsNaN(l));
            Assert.False(float.IsInfinity(l));
        }

        // Step 1
        var logits1 = forwardPass.Forward(token: 12, position: 1);
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

        // Calculate tensor data offsets
        long tensorInfoStart = stream.Position;
        // Placeholder for tensor infos: we need to write tensor infos first
        // Each tensor info: name, n_dims, dims..., dtype, offset
        long runningOffset = 0;
        var tensorOffsets = new Dictionary<string, ulong>();
        foreach (var (name, (shape, dtype, data)) in tensors)
        {
            // Align offset to 32 bytes
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

        // Align stream to 32 bytes for data block
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
                writer.Write((uint)4); // GGUF_TYPE_UINT32
                writer.Write(u32);
                break;
            case int i32:
                writer.Write((uint)5); // GGUF_TYPE_INT32
                writer.Write(i32);
                break;
            case float f32:
                writer.Write((uint)6); // GGUF_TYPE_FLOAT32
                writer.Write(f32);
                break;
            case bool b:
                writer.Write((uint)7); // GGUF_TYPE_BOOL
                writer.Write((byte)(b ? 1 : 0));
                break;
            case string s:
                writer.Write((uint)8); // GGUF_TYPE_STRING
                WriteGgufString(writer, s);
                break;
            case ulong u64:
                writer.Write((uint)10); // GGUF_TYPE_UINT64
                writer.Write(u64);
                break;
            case object[] arr:
                writer.Write((uint)9); // GGUF_TYPE_ARRAY
                writer.Write((uint)5); // element type INT32
                writer.Write((ulong)arr.Length);
                foreach (var item in arr)
                {
                    writer.Write(Convert.ToInt32(item));
                }
                break;
            default:
                throw new NotSupportedException($"Value type {value.GetType()} not supported in synthetic writer.");
        }
    }
}
