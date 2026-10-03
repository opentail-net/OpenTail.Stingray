using System;
using System.Collections.Generic;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class DeepSeek41SyntheticTests : IDisposable
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
    public void DeepSeek41_SyntheticForwardPass_ExecutesWithMhcAndEngram()
    {
        const int embedDim = 32;
        const int numHeads = 2;
        const int headDim = 16;
        const int ropeDim = 8;
        const int qLoraRank = 16;
        const int outLoraRank = 16;
        const int outGroupCount = 2;
        const int numExperts = 4;
        const int numExpertsUsed = 2;
        const int expertFfn = 16;
        const int vocabSize = 32;
        const int compressedVocab = 16;
        const int engramHeads = 2;
        const int engramHeadDim = 8;
        const int engramEmbeddingDim = engramHeads * engramHeadDim; // 16
        const int numLayers = 2;

        string path = Path.Combine(Path.GetTempPath(), $"deepseek41_synthetic_{Guid.NewGuid():N}.gguf");
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

            // Attention tensors
            AddTensor($"blk.{i}.attn_q_a.weight", [embedDim, qLoraRank]);
            AddTensor($"blk.{i}.attn_q_a_norm.weight", [qLoraRank]);
            AddTensor($"blk.{i}.attn_q_b.weight", [qLoraRank, numHeads * headDim]);
            AddTensor($"blk.{i}.attn_kv.weight", [embedDim, headDim]);
            AddTensor($"blk.{i}.attn_kv_a_norm.weight", [headDim]);
            AddTensor($"blk.{i}.attn_output_a.weight", [numHeads * headDim, outLoraRank]);
            AddTensor($"blk.{i}.attn_output_b.weight", [outLoraRank, embedDim]);

            // MoE tensors
            AddTensor($"blk.{i}.ffn_gate_inp.weight", [embedDim, numExperts]);
            AddTensor($"blk.{i}.ffn_gate_exps.weight", [embedDim, expertFfn, numExperts]);
            AddTensor($"blk.{i}.ffn_up_exps.weight", [embedDim, expertFfn, numExperts]);
            AddTensor($"blk.{i}.ffn_down_exps.weight", [expertFfn, embedDim, numExperts]);
            AddTensor($"blk.{i}.ffn_gate_shexp.weight", [embedDim, expertFfn]);
            AddTensor($"blk.{i}.ffn_up_shexp.weight", [embedDim, expertFfn]);
            AddTensor($"blk.{i}.ffn_down_shexp.weight", [expertFfn, embedDim]);

            // Engram on layer 1
            if (i == 1)
            {
                AddTensor($"blk.{i}.engram.weight", [compressedVocab, engramEmbeddingDim]);
                AddTensor($"blk.{i}.engram.proj.weight", [engramEmbeddingDim, embedDim]);
            }
        }

        var metadata = new Dictionary<string, object>
        {
            ["general.architecture"] = "deepseek41",
            ["deepseek41.block_count"] = (ulong)numLayers,
            ["deepseek41.nextn_predict_layers"] = (ulong)0,
            ["deepseek41.embedding_length"] = (ulong)embedDim,
            ["deepseek41.attention.head_count"] = (ulong)numHeads,
            ["deepseek41.attention.key_length"] = (ulong)headDim,
            ["deepseek41.rope.dimension_count"] = (ulong)ropeDim,
            ["deepseek41.attention.q_lora_rank"] = (ulong)qLoraRank,
            ["deepseek41.attention.output_lora_rank"] = (ulong)outLoraRank,
            ["deepseek41.attention.output_group_count"] = (ulong)outGroupCount,
            ["deepseek41.expert_count"] = (ulong)numExperts,
            ["deepseek41.expert_used_count"] = (ulong)numExpertsUsed,
            ["deepseek41.expert_weights_scale"] = 1.5f,
            ["deepseek41.expert_weights_norm"] = true,
            ["deepseek41.engram.max_ngram"] = (ulong)4,
            ["deepseek41.engram.head_count"] = (ulong)engramHeads,
            ["deepseek41.engram.head_dim"] = (ulong)engramHeadDim,
            ["deepseek41.engram.compressed_vocab_size"] = (ulong)compressedVocab,
        };

        WriteGgufFile(path, metadata, tensors);

        using var model = GgufModel.Open(path);
        var hp = DeepSeek41Hyperparams.FromGgufMetadata(
            model.Metadata, "deepseek41");

        var forwardPass = new DeepSeek41ForwardPass(model, hp);
        Assert.Equal(vocabSize, forwardPass.VocabSize);

        // Step 0 forward pass
        var logits0 = forwardPass.Forward(token: 5, position: 0);
        Assert.Equal(vocabSize, logits0.Length);
        foreach (var l in logits0)
        {
            Assert.False(float.IsNaN(l));
            Assert.False(float.IsInfinity(l));
        }

        // Step 1 forward pass (activates 2-gram engram lookup)
        var logits1 = forwardPass.Forward(token: 9, position: 1);
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
