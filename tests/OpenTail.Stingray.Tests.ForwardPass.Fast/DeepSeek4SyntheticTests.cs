using System;
using System.Collections.Generic;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

// Synthetic deepseek4 models: tiny dimensions, deterministic weights, no checkpoint.
// These check that every attention variant runs and builds its compressed streams. They are not a
// numeric oracle: the CSA overlap and indexer selection still need the b10306 comparison (plan Phase 2).
public sealed class DeepSeek4SyntheticTests : IDisposable
{
    private const int EmbedDim = 32;
    private const int NumHeads = 2;
    private const int HeadDim = 16;
    private const int RopeDim = 8;
    private const int QLoraRank = 16;
    private const int OutLoraRank = 16;
    private const int OutGroupCount = 2;
    private const int NumExperts = 4;
    private const int NumExpertsUsed = 2;
    private const int ExpertFfn = 16;
    private const int VocabSize = 32;
    private const int HcMult = 4;
    private const int HcMixDim = (2 + HcMult) * HcMult; // 24
    private const int IndexerNumHeads = 2;
    private const int IndexerHeadSize = 8;
    private const int IndexerTopK = 2;

    private readonly List<string> _tempFiles = [];

    public void Dispose()
    {
        foreach (var file in _tempFiles)
        {
            try { if (File.Exists(file)) File.Delete(file); } catch { }
        }
    }

    [Fact]
    public void DeepSeek4_SyntheticForwardPass_ExecutesWithMhcAndGroupedOutput()
    {
        // Two raw (ratio 0) layers: the baseline that the compressed variants are compared with.
        string path = WriteSyntheticModel([0, 0]);
        using var model = GgufModel.Open(path);
        var forwardPass = new DeepSeek4ForwardPass(model, HyperparamsFor(model, 2));
        Assert.Equal(VocabSize, forwardPass.VocabSize);

        AssertFinite(forwardPass.Forward(token: 5, position: 0));
        AssertFinite(forwardPass.Forward(token: 9, position: 1));
    }

    [Fact]
    public void DeepSeek4_CsaAndHcaLayers_BuildTheirCompressedStreams()
    {
        // Layer 0 is CSA (ratio 4), layer 1 is HCA (ratio 128). After 130 tokens, CSA has finalised
        // floor(130 / 4) = 32 blocks and HCA has finalised floor(130 / 128) = 1 block.
        string path = WriteSyntheticModel([4, 128]);
        using var model = GgufModel.Open(path);
        var forwardPass = new DeepSeek4ForwardPass(model, HyperparamsFor(model, 2));

        for (int pos = 0; pos < 130; pos++)
        {
            AssertFinite(forwardPass.Forward(token: pos % VocabSize, position: pos));
        }

        Assert.Equal(32, forwardPass.CompressedState.Csa(0)!.BlockCount);
        Assert.Equal(1, forwardPass.CompressedState.Hca(1)!.BlockCount);
    }

    [Fact]
    public void DeepSeek4_CompressedAttention_ChangesTheOutputOnlyOnceABlockExists()
    {
        // Same weights for both models; only the attention variant differs. CSA (ratio 4) finalises
        // its first block at position 3, so for positions 0 to 2 both models attend over raw keys only
        // and must agree exactly. From position 3 on, the compressed key must change the logits.
        float[][] compressed = LogitsPerPosition([4, 128], 4);
        float[][] raw = LogitsPerPosition([0, 0], 4);

        for (int p = 0; p < 3; p++)
        {
            Assert.True(compressed[p].AsSpan().SequenceEqual(raw[p]), $"position {p} differs before any compressed block exists.");
        }

        float scale = 0f;
        foreach (var l in raw[3]) scale = MathF.Max(scale, MathF.Abs(l));
        float maxDiff = 0f;
        for (int i = 0; i < raw[3].Length; i++)
        {
            maxDiff = MathF.Max(maxDiff, MathF.Abs(compressed[3][i] - raw[3][i]));
        }
        // Measured 2026-11-08: max diff 3.66e-7 on a logit scale of 1.0e-3 (relative 3.7e-4). The
        // threshold is 1e-4 relative, well above float32 noise and below the measured value.
        Assert.True(maxDiff > 1e-4f * scale, $"compressed and raw logits differ by only {maxDiff} at position 3 (scale {scale}); the compressed key had no effect.");
    }

    [Fact]
    public void DeepSeek4_CsaLayerWithoutItsCompressor_IsRefusedAtConstruction()
    {
        // The checkpoint declares a CSA layer but omits its compressor: refuse with the tensor's name,
        // not with a null dereference on the first token.
        string path = WriteSyntheticModel([4, 128], omit: ["blk.0.attn_compressor_kv.weight"]);
        using var model = GgufModel.Open(path);

        var ex = Assert.Throws<InvalidOperationException>(() => new DeepSeek4ForwardPass(model, HyperparamsFor(model, 2)));
        Assert.Contains("blk.0.attn_compressor_kv.weight", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DeepSeek4_CompressedBlockRope_UsesTheBlocksFirstRawToken()
    {
        // Block k covers raw tokens ratio*k .. ratio*k + ratio - 1 and is roped at ratio*k, as the reference's
        // state_write_pos is. Block 0 is at position 0 under either formula, so only later blocks catch a regression.
        // This pins the convention; it is not a numeric check against the reference.
        Assert.Equal(0, DeepSeek4ForwardPass.CompressedBlockPosition(0, 4));
        Assert.Equal(12, DeepSeek4ForwardPass.CompressedBlockPosition(3, 4));
        Assert.Equal(128, DeepSeek4ForwardPass.CompressedBlockPosition(1, 128));
    }

    private float[][] LogitsPerPosition(int[] ratios, int tokens)
    {
        string path = WriteSyntheticModel(ratios);
        using var model = GgufModel.Open(path);
        var forwardPass = new DeepSeek4ForwardPass(model, HyperparamsFor(model, ratios.Length));
        var result = new float[tokens][];
        for (int pos = 0; pos < tokens; pos++)
        {
            result[pos] = forwardPass.Forward(token: pos % VocabSize, position: pos).ToArray();
            AssertFinite(result[pos]);
        }
        return result;
    }

    private static DeepSeek4Hyperparams HyperparamsFor(GgufModel model, int numLayers) =>
        DeepSeek4Hyperparams.FromGgufMetadata(
            model.Metadata, "deepseek4", numLayerAll: numLayers, embedDim: EmbedDim, numHeads: NumHeads,
            headDim: HeadDim, ropeDim: RopeDim, numExperts: NumExperts, numExpertsUsed: NumExpertsUsed);

    private static void AssertFinite(ReadOnlySpan<float> logits)
    {
        Assert.Equal(VocabSize, logits.Length);
        foreach (var l in logits)
        {
            Assert.False(float.IsNaN(l));
            Assert.False(float.IsInfinity(l));
        }
    }

    // Writes a synthetic GGUF with one layer per compress ratio and returns its path (removed on Dispose).
    private string WriteSyntheticModel(int[] compressRatios, string[]? omit = null)
    {
        int numLayers = compressRatios.Length;
        string path = Path.Combine(Path.GetTempPath(), $"deepseek4_synthetic_{Guid.NewGuid():N}.gguf");
        _tempFiles.Add(path);

        var tensors = new Dictionary<string, (long[] shape, DType dtype, byte[] data)>();

        void AddTensor(string name, long[] shape, float fill = 0.05f)
        {
            if (omit is not null && Array.IndexOf(omit, name) >= 0) return;
            long total = 1;
            foreach (var s in shape) total *= s;
            var floats = new float[total];
            for (int i = 0; i < total; i++) floats[i] = fill * MathF.Sin(i + 1);
            var bytes = new byte[total * 4];
            Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
            tensors[name] = (shape, DType.Float32, bytes);
        }

        AddTensor("token_embd.weight", [EmbedDim, VocabSize]);
        AddTensor("output_norm.weight", [EmbedDim]);
        AddTensor("output.weight", [EmbedDim, VocabSize]);
        AddTensor("output_hc_fn.weight", [HcMult * EmbedDim, HcMult]);
        AddTensor("output_hc_base.weight", [HcMult]);
        AddTensor("output_hc_scale.weight", [1]);

        for (int i = 0; i < numLayers; i++)
        {
            AddTensor($"blk.{i}.attn_norm.weight", [EmbedDim]);
            AddTensor($"blk.{i}.ffn_norm.weight", [EmbedDim]);

            // HyperConnection tensors
            AddTensor($"blk.{i}.hc_attn_fn.weight", [HcMult * EmbedDim, HcMixDim]);
            AddTensor($"blk.{i}.hc_attn_base.weight", [HcMixDim]);
            AddTensor($"blk.{i}.hc_attn_scale.weight", [3]);
            AddTensor($"blk.{i}.hc_ffn_fn.weight", [HcMult * EmbedDim, HcMixDim]);
            AddTensor($"blk.{i}.hc_ffn_base.weight", [HcMixDim]);
            AddTensor($"blk.{i}.hc_ffn_scale.weight", [3]);

            // Attention tensors
            AddTensor($"blk.{i}.attn_q_a.weight", [EmbedDim, QLoraRank]);
            AddTensor($"blk.{i}.attn_q_a_norm.weight", [QLoraRank]);
            AddTensor($"blk.{i}.attn_q_b.weight", [QLoraRank, NumHeads * HeadDim]);
            AddTensor($"blk.{i}.attn_kv.weight", [EmbedDim, HeadDim]);
            AddTensor($"blk.{i}.attn_kv_a_norm.weight", [HeadDim]);
            AddTensor($"blk.{i}.attn_output_a.weight", [NumHeads * HeadDim, OutLoraRank]);
            AddTensor($"blk.{i}.attn_output_b.weight", [OutLoraRank, EmbedDim]);

            // MoE tensors
            AddTensor($"blk.{i}.ffn_gate_inp.weight", [EmbedDim, NumExperts]);
            AddTensor($"blk.{i}.ffn_gate_exps.weight", [EmbedDim, ExpertFfn, NumExperts]);
            AddTensor($"blk.{i}.ffn_up_exps.weight", [EmbedDim, ExpertFfn, NumExperts]);
            AddTensor($"blk.{i}.ffn_down_exps.weight", [ExpertFfn, EmbedDim, NumExperts]);
            AddTensor($"blk.{i}.ffn_gate_shexp.weight", [EmbedDim, ExpertFfn]);
            AddTensor($"blk.{i}.ffn_up_shexp.weight", [EmbedDim, ExpertFfn]);
            AddTensor($"blk.{i}.ffn_down_shexp.weight", [ExpertFfn, EmbedDim]);

            // Compressor tensors for a non-zero ratio: CSA (4) overlaps, so its width is 2 * headDim;
            // HCA (128) is 1 * headDim. The APE table has one row per position in a block.
            int ratio = compressRatios[i];
            if (ratio != 0)
            {
                int coff = ratio == 4 ? 2 * HeadDim : HeadDim;
                AddTensor($"blk.{i}.attn_compressor_kv.weight", [EmbedDim, coff]);
                AddTensor($"blk.{i}.attn_compressor_gate.weight", [EmbedDim, coff]);
                AddTensor($"blk.{i}.attn_compressor_ape.weight", [coff, ratio]);
                AddTensor($"blk.{i}.attn_compressor_norm.weight", [HeadDim]);
            }

            // CSA also has the lightning indexer, whose compressor is overlapped like CSA's.
            if (ratio == 4)
            {
                AddTensor($"blk.{i}.indexer_compressor_kv.weight", [EmbedDim, 2 * IndexerHeadSize]);
                AddTensor($"blk.{i}.indexer_compressor_gate.weight", [EmbedDim, 2 * IndexerHeadSize]);
                AddTensor($"blk.{i}.indexer_compressor_ape.weight", [2 * IndexerHeadSize, ratio]);
                AddTensor($"blk.{i}.indexer_compressor_norm.weight", [IndexerHeadSize]);
                AddTensor($"blk.{i}.indexer.proj.weight", [EmbedDim, IndexerNumHeads]);
                AddTensor($"blk.{i}.indexer.attn_q_b.weight", [QLoraRank, IndexerNumHeads * IndexerHeadSize]);
            }
        }

        var ratiosMeta = new object[compressRatios.Length];
        for (int i = 0; i < compressRatios.Length; i++) ratiosMeta[i] = (ulong)compressRatios[i];

        var metadata = new Dictionary<string, object>
        {
            ["general.architecture"] = "deepseek4",
            ["deepseek4.block_count"] = (ulong)numLayers,
            ["deepseek4.nextn_predict_layers"] = (ulong)0,
            ["deepseek4.embedding_length"] = (ulong)EmbedDim,
            ["deepseek4.attention.head_count"] = (ulong)NumHeads,
            ["deepseek4.attention.key_length"] = (ulong)HeadDim,
            ["deepseek4.rope.dimension_count"] = (ulong)RopeDim,
            ["deepseek4.attention.q_lora_rank"] = (ulong)QLoraRank,
            ["deepseek4.attention.output_lora_rank"] = (ulong)OutLoraRank,
            ["deepseek4.attention.output_group_count"] = (ulong)OutGroupCount,
            ["deepseek4.expert_count"] = (ulong)NumExperts,
            ["deepseek4.expert_used_count"] = (ulong)NumExpertsUsed,
            ["deepseek4.expert_weights_scale"] = 1.0f,
            ["deepseek4.expert_weights_norm"] = true,
            ["deepseek4.attention.compress_ratios"] = ratiosMeta,
            ["deepseek4.attention.indexer.head_count"] = (ulong)IndexerNumHeads,
            ["deepseek4.attention.indexer.key_length"] = (ulong)IndexerHeadSize,
            ["deepseek4.attention.indexer.top_k"] = (ulong)IndexerTopK,
        };

        WriteGgufFile(path, metadata, tensors);
        return path;
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
            case Array arr:
                writer.Write((uint)9); // GGUF Array type
                writer.Write((uint)10); // Element type: UINT64
                writer.Write((ulong)arr.Length);
                for (int i = 0; i < arr.Length; i++)
                {
                    writer.Write((ulong)arr.GetValue(i)!);
                }
                break;
            default:
                throw new NotSupportedException($"Value type {value.GetType()} not supported in synthetic writer.");
        }
    }
}
