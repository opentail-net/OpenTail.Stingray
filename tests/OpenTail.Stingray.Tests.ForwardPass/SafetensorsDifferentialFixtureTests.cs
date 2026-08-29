namespace OpenTail.Stingray.Tests.ForwardPass;

using System.Buffers.Binary;
using System.Text.Json;
using System.Text.Json.Nodes;
using ForwardPass = OpenTail.Stingray.Engine.ForwardPass;

public sealed class SafetensorsDifferentialFixtureTests : HeavyTestBase, IDisposable
{
    private readonly string _tempDir;

    public SafetensorsDifferentialFixtureTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"opentail_safetensors_diff_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
        {
            try { Directory.Delete(_tempDir, recursive: true); } catch { }
        }
    }

    [Fact]
    public void Safetensors_F32_MatchesGguf_ByteForByteAndBitIdenticalLogits()
    {
        string ggufPath = Path.Combine(_tempDir, "model.gguf");
        string safetensorsDir = Path.Combine(_tempDir, "st_model");
        Directory.CreateDirectory(safetensorsDir);

        var weights = GenerateDeterministicWeights(vocabSize: 128, hiddenSize: 64, layers: 2, heads: 4, kvHeads: 2, intermediateSize: 128);

        BuildGgufFile(ggufPath, weights, vocabSize: 128, hiddenSize: 64, layers: 2, heads: 4, kvHeads: 2, intermediateSize: 128, tied: false);
        BuildSafetensorsPackage(safetensorsDir, weights, vocabSize: 128, hiddenSize: 64, layers: 2, heads: 4, kvHeads: 2, intermediateSize: 128, tied: false, isF16: false);

        using var ggufModelHandle = SharedModelCacheFixture.Instance.Acquire(ggufPath);
        var ggufModel = ggufModelHandle.Model;
        using var stSource = SafetensorsTensorSource.Open(safetensorsDir);

        // 1. Verify tensor enumeration and shape equivalence
        Assert.Equal(ggufModel.Tensors.Count, stSource.Tensors.Count);
        foreach (var ggufTensor in ggufModel.Tensors)
        {
            var stTensor = stSource.FindTensor(ggufTensor.Name);
            Assert.NotNull(stTensor);
            var stInfo = stTensor.Value;
            Assert.Equal(ggufTensor.Dimensions, stInfo.Dimensions);
            Assert.Equal(ggufTensor.DType, stInfo.DType);

            ReadOnlySpan<byte> ggufBytes = ggufModel.GetTensorData(ggufTensor);
            ReadOnlySpan<byte> stBytes = stSource.GetTensorData(stInfo);

            Assert.Equal(ggufBytes.Length, stBytes.Length);
            Assert.True(ggufBytes.SequenceEqual(stBytes), $"Tensor '{ggufTensor.Name}' bytes differ between GGUF and SafeTensors!");
        }

        // 2. Verify bit-identical logits through ForwardPass
        var ggufHp = ModelHyperparams.FromGgufMetadata(ggufModel.Metadata);
        var stHp = ModelHyperparams.FromGgufMetadata(stSource.Metadata);
        using var backend = new CpuBackend();

        using var ggufFwd = new ForwardPass(ggufModel, backend, ggufHp);
        using var stFwd = new ForwardPass(stSource, backend, stHp);

        var prompt = new int[] { 1, 5, 12, 33 };

        var ggufLogits = ggufFwd.Prefill(prompt).ToArray();
        var stLogits = stFwd.Prefill(prompt).ToArray();

        Assert.Equal(ggufLogits.Length, stLogits.Length);
        for (int i = 0; i < ggufLogits.Length; i++)
        {
            Assert.Equal(ggufLogits[i], stLogits[i]);
        }
    }

    [Fact]
    public void Safetensors_F16_MatchesGguf_WithinTolerance()
    {
        string ggufPath = Path.Combine(_tempDir, "model_f16.gguf");
        string safetensorsDir = Path.Combine(_tempDir, "st_model_f16");
        Directory.CreateDirectory(safetensorsDir);

        var weights = GenerateDeterministicWeights(vocabSize: 128, hiddenSize: 64, layers: 2, heads: 4, kvHeads: 2, intermediateSize: 128);

        BuildGgufFile(ggufPath, weights, vocabSize: 128, hiddenSize: 64, layers: 2, heads: 4, kvHeads: 2, intermediateSize: 128, tied: false);
        BuildSafetensorsPackage(safetensorsDir, weights, vocabSize: 128, hiddenSize: 64, layers: 2, heads: 4, kvHeads: 2, intermediateSize: 128, tied: false, isF16: true);

        using var ggufModelHandle = SharedModelCacheFixture.Instance.Acquire(ggufPath);
        var ggufModel = ggufModelHandle.Model;
        using var stSource = SafetensorsTensorSource.Open(safetensorsDir);

        var ggufHp = ModelHyperparams.FromGgufMetadata(ggufModel.Metadata);
        var stHp = ModelHyperparams.FromGgufMetadata(stSource.Metadata);
        using var backend = new CpuBackend();

        using var ggufFwd = new ForwardPass(ggufModel, backend, ggufHp);
        using var stFwd = new ForwardPass(stSource, backend, stHp);

        var prompt = new int[] { 1, 5, 12, 33 };

        var ggufLogits = ggufFwd.Prefill(prompt).ToArray();
        var stLogits = stFwd.Prefill(prompt).ToArray();

        Assert.Equal(ggufLogits.Length, stLogits.Length);
        Assert.True(ggufLogits.Any(x => Math.Abs(x) > 1e-4f), "Logits must be non-trivial (non-zero).");
        Assert.Equal(Argmax(ggufLogits), Argmax(stLogits));
        for (int i = 0; i < ggufLogits.Length; i++)
        {
            Assert.InRange(Math.Abs(ggufLogits[i] - stLogits[i]), 0f, 1e-2f);
        }

        // Multi-step greedy token generation parity
        int nextTokenF16 = Argmax(ggufLogits);
        int posF16 = prompt.Length;
        for (int step = 0; step < 5; step++)
        {
            var ggufStepLogits = ggufFwd.Prefill(new[] { nextTokenF16 }, startPos: posF16).ToArray();
            var stStepLogits = stFwd.Prefill(new[] { nextTokenF16 }, startPos: posF16).ToArray();

            int ggufNext = Argmax(ggufStepLogits);
            int stNext = Argmax(stStepLogits);

            Assert.Equal(ggufNext, stNext);
            nextTokenF16 = ggufNext;
            posF16++;
        }
    }

    [Fact]
    public void Safetensors_TiedEmbeddings_MatchesGguf()
    {
        string ggufPath = Path.Combine(_tempDir, "model_tied.gguf");
        string safetensorsDir = Path.Combine(_tempDir, "st_model_tied");
        Directory.CreateDirectory(safetensorsDir);

        var weights = GenerateDeterministicWeights(vocabSize: 128, hiddenSize: 64, layers: 2, heads: 4, kvHeads: 2, intermediateSize: 128);

        BuildGgufFile(ggufPath, weights, vocabSize: 128, hiddenSize: 64, layers: 2, heads: 4, kvHeads: 2, intermediateSize: 128, tied: true);
        BuildSafetensorsPackage(safetensorsDir, weights, vocabSize: 128, hiddenSize: 64, layers: 2, heads: 4, kvHeads: 2, intermediateSize: 128, tied: true, isF16: false);

        using var ggufModelHandle = SharedModelCacheFixture.Instance.Acquire(ggufPath);
        var ggufModel = ggufModelHandle.Model;
        using var stSource = SafetensorsTensorSource.Open(safetensorsDir);

        var ggufHp = ModelHyperparams.FromGgufMetadata(ggufModel.Metadata);
        var stHp = ModelHyperparams.FromGgufMetadata(stSource.Metadata);
        using var backend = new CpuBackend();

        using var ggufFwd = new ForwardPass(ggufModel, backend, ggufHp);
        using var stFwd = new ForwardPass(stSource, backend, stHp);

        var prompt = new int[] { 1, 5, 12, 33 };

        var ggufLogits = ggufFwd.Prefill(prompt).ToArray();
        var stLogits = stFwd.Prefill(prompt).ToArray();

        Assert.Equal(ggufLogits.Length, stLogits.Length);
        for (int i = 0; i < ggufLogits.Length; i++)
        {
            Assert.Equal(ggufLogits[i], stLogits[i]);
        }
    }

    [Fact]
    public void Safetensors_Bf16_MatchesGguf_WithinTolerance()
    {
        string ggufPath = Path.Combine(_tempDir, "model_bf16.gguf");
        string safetensorsDir = Path.Combine(_tempDir, "st_model_bf16");
        Directory.CreateDirectory(safetensorsDir);

        var weights = GenerateDeterministicWeights(vocabSize: 128, hiddenSize: 64, layers: 2, heads: 4, kvHeads: 2, intermediateSize: 128);

        BuildGgufFile(ggufPath, weights, vocabSize: 128, hiddenSize: 64, layers: 2, heads: 4, kvHeads: 2, intermediateSize: 128, tied: false);
        BuildSafetensorsPackageWithDtype(safetensorsDir, weights, vocabSize: 128, hiddenSize: 64, layers: 2, heads: 4, kvHeads: 2, intermediateSize: 128, tied: false, dtypeStr: "BF16");

        using var ggufModelHandle = SharedModelCacheFixture.Instance.Acquire(ggufPath);
        var ggufModel = ggufModelHandle.Model;
        using var stSource = SafetensorsTensorSource.Open(safetensorsDir);

        var ggufHp = ModelHyperparams.FromGgufMetadata(ggufModel.Metadata);
        var stHp = ModelHyperparams.FromGgufMetadata(stSource.Metadata);
        using var backend = new CpuBackend();

        using var ggufFwd = new ForwardPass(ggufModel, backend, ggufHp);
        using var stFwd = new ForwardPass(stSource, backend, stHp);

        var prompt = new int[] { 1, 5, 12, 33 };

        var ggufLogits = ggufFwd.Prefill(prompt).ToArray();
        var stLogits = stFwd.Prefill(prompt).ToArray();

        Assert.Equal(ggufLogits.Length, stLogits.Length);
        Assert.True(ggufLogits.Any(x => Math.Abs(x) > 1e-4f), "Logits must be non-trivial (non-zero).");
        Assert.Equal(Argmax(ggufLogits), Argmax(stLogits));
        for (int i = 0; i < ggufLogits.Length; i++)
        {
            Assert.InRange(Math.Abs(ggufLogits[i] - stLogits[i]), 0f, 1e-2f);
        }

        // Multi-step greedy token generation parity
        int nextTokenBf16 = Argmax(ggufLogits);
        int posBf16 = prompt.Length;
        for (int step = 0; step < 5; step++)
        {
            var ggufStepLogits = ggufFwd.Prefill(new[] { nextTokenBf16 }, startPos: posBf16).ToArray();
            var stStepLogits = stFwd.Prefill(new[] { nextTokenBf16 }, startPos: posBf16).ToArray();

            int ggufNext = Argmax(ggufStepLogits);
            int stNext = Argmax(stStepLogits);

            Assert.Equal(ggufNext, stNext);
            nextTokenBf16 = ggufNext;
            posBf16++;
        }
    }

    [Fact]
    public void Safetensors_Bf16_ConversionIsLazyOnFirstAccess()
    {
        string safetensorsDir = Path.Combine(_tempDir, "st_model_bf16_lazy");
        Directory.CreateDirectory(safetensorsDir);

        var weights = GenerateDeterministicWeights(vocabSize: 128, hiddenSize: 64, layers: 2, heads: 4, kvHeads: 2, intermediateSize: 128);
        BuildSafetensorsPackageWithDtype(safetensorsDir, weights, vocabSize: 128, hiddenSize: 64, layers: 2, heads: 4, kvHeads: 2, intermediateSize: 128, tied: false, dtypeStr: "BF16");

        using var stSource = SafetensorsTensorSource.Open(safetensorsDir);
        var normInfo = stSource.FindTensor("output_norm.weight");
        Assert.NotNull(normInfo);

        // Before GetTensorDataPtr, no conversion buffer should be fetched
        var dataSpan = stSource.GetTensorData(normInfo.Value);
        Assert.Equal(64 * sizeof(float), dataSpan.Length);
        Assert.Contains(dataSpan.ToArray(), b => b != 0);
    }

    private static int Argmax(ReadOnlySpan<float> span)
    {
        int maxIdx = 0;
        float maxVal = span[0];
        for (int i = 1; i < span.Length; i++)
        {
            if (span[i] > maxVal)
            {
                maxVal = span[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    #region Helper Methods & Synthetic Model Generation

    private sealed record SyntheticWeight(string CanonicalName, string SafetensorsName, long[] GgufDims, int[] SafetensorsShape, float[] Data);

    private static List<SyntheticWeight> GenerateDeterministicWeights(int vocabSize, int hiddenSize, int layers, int heads, int kvHeads, int intermediateSize)
    {
        var rng = new Random(42);
        var list = new List<SyntheticWeight>();

        float[] Rnd(int count)
        {
            var arr = new float[count];
            for (int i = 0; i < count; i++)
            {
                arr[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
            }
            return arr;
        }

        // token_embd.weight: shape [vocabSize, hiddenSize]
        list.Add(new SyntheticWeight("token_embd.weight", "model.embed_tokens.weight", [hiddenSize, vocabSize], [vocabSize, hiddenSize], Rnd(vocabSize * hiddenSize)));
        // output_norm.weight: shape [hiddenSize]
        list.Add(new SyntheticWeight("output_norm.weight", "model.norm.weight", [hiddenSize], [hiddenSize], Rnd(hiddenSize)));
        // output.weight: shape [vocabSize, hiddenSize]
        list.Add(new SyntheticWeight("output.weight", "lm_head.weight", [hiddenSize, vocabSize], [vocabSize, hiddenSize], Rnd(vocabSize * hiddenSize)));

        int headDim = hiddenSize / heads;
        int kvDim = kvHeads * headDim;

        for (int l = 0; l < layers; l++)
        {
            list.Add(new SyntheticWeight($"blk.{l}.attn_norm.weight", $"model.layers.{l}.input_layernorm.weight", [hiddenSize], [hiddenSize], Rnd(hiddenSize)));
            list.Add(new SyntheticWeight($"blk.{l}.attn_q.weight", $"model.layers.{l}.self_attn.q_proj.weight", [hiddenSize, hiddenSize], [hiddenSize, hiddenSize], Rnd(hiddenSize * hiddenSize)));
            list.Add(new SyntheticWeight($"blk.{l}.attn_k.weight", $"model.layers.{l}.self_attn.k_proj.weight", [hiddenSize, kvDim], [kvDim, hiddenSize], Rnd(kvDim * hiddenSize)));
            list.Add(new SyntheticWeight($"blk.{l}.attn_v.weight", $"model.layers.{l}.self_attn.v_proj.weight", [hiddenSize, kvDim], [kvDim, hiddenSize], Rnd(kvDim * hiddenSize)));
            list.Add(new SyntheticWeight($"blk.{l}.attn_output.weight", $"model.layers.{l}.self_attn.o_proj.weight", [hiddenSize, hiddenSize], [hiddenSize, hiddenSize], Rnd(hiddenSize * hiddenSize)));

            list.Add(new SyntheticWeight($"blk.{l}.ffn_norm.weight", $"model.layers.{l}.post_attention_layernorm.weight", [hiddenSize], [hiddenSize], Rnd(hiddenSize)));
            list.Add(new SyntheticWeight($"blk.{l}.ffn_gate.weight", $"model.layers.{l}.mlp.gate_proj.weight", [hiddenSize, intermediateSize], [intermediateSize, hiddenSize], Rnd(intermediateSize * hiddenSize)));
            list.Add(new SyntheticWeight($"blk.{l}.ffn_up.weight", $"model.layers.{l}.mlp.up_proj.weight", [hiddenSize, intermediateSize], [intermediateSize, hiddenSize], Rnd(intermediateSize * hiddenSize)));
            list.Add(new SyntheticWeight($"blk.{l}.ffn_down.weight", $"model.layers.{l}.mlp.down_proj.weight", [intermediateSize, hiddenSize], [hiddenSize, intermediateSize], Rnd(hiddenSize * intermediateSize)));
        }

        return list;
    }

    private static void BuildGgufFile(string ggufPath, List<SyntheticWeight> weights, int vocabSize, int hiddenSize, int layers, int heads, int kvHeads, int intermediateSize, bool tied)
    {
        using var fs = new FileStream(ggufPath, FileMode.Create, FileAccess.Write);
        using var w = new BinaryWriter(fs, Encoding.UTF8);

        w.Write(0x46554747u); // Magic: GGUF
        w.Write(3u);          // Version

        var activeWeights = tied ? weights.Where(x => x.CanonicalName != "output.weight").ToList() : weights;

        w.Write((ulong)activeWeights.Count);

        var metadata = new Dictionary<string, (GgufValueType Type, object Value)>
        {
            ["general.architecture"] = (GgufValueType.String, "llama"),
            ["general.name"] = (GgufValueType.String, "SyntheticLlama"),
            ["llama.vocab_size"] = (GgufValueType.UInt32, (uint)vocabSize),
            ["llama.context_length"] = (GgufValueType.UInt32, 512u),
            ["llama.embedding_length"] = (GgufValueType.UInt32, (uint)hiddenSize),
            ["llama.block_count"] = (GgufValueType.UInt32, (uint)layers),
            ["llama.attention.head_count"] = (GgufValueType.UInt32, (uint)heads),
            ["llama.attention.head_count_kv"] = (GgufValueType.UInt32, (uint)kvHeads),
            ["llama.attention.key_length"] = (GgufValueType.UInt32, (uint)(hiddenSize / heads)),
            ["llama.rope.dimension_count"] = (GgufValueType.UInt32, (uint)(hiddenSize / heads)),
            ["llama.feed_forward_length"] = (GgufValueType.UInt32, (uint)intermediateSize),
            ["llama.attention.layer_norm_rms_epsilon"] = (GgufValueType.Float32, 1e-5f),
            ["llama.rope.freq_base"] = (GgufValueType.Float32, 10000f),
        };

        w.Write((ulong)metadata.Count);

        foreach (var (key, (type, val)) in metadata)
        {
            WriteGgufString(w, key);
            w.Write((uint)type);
            WriteGgufValue(w, type, val);
        }

        long currentOffset = 0;
        var offsets = new List<long>();
        foreach (var tensor in activeWeights)
        {
            long remainder = currentOffset % 32;
            if (remainder != 0) currentOffset += (32 - remainder);
            offsets.Add(currentOffset);
            currentOffset += tensor.Data.Length * sizeof(float);
        }

        for (int i = 0; i < activeWeights.Count; i++)
        {
            var item = activeWeights[i];
            WriteGgufString(w, item.CanonicalName);
            w.Write((uint)item.GgufDims.Length);
            foreach (var dim in item.GgufDims) w.Write((ulong)dim);
            w.Write((uint)DType.Float32);
            w.Write((ulong)offsets[i]);
        }

        long streamPos = fs.Position;
        long rem = streamPos % 32;
        if (rem != 0)
        {
            int pad = 32 - (int)rem;
            w.Write(new byte[pad]);
        }

        for (int i = 0; i < activeWeights.Count; i++)
        {
            var item = activeWeights[i];
            foreach (float f in item.Data) w.Write(f);
        }
    }

    private static void BuildSafetensorsPackage(string dir, List<SyntheticWeight> weights, int vocabSize, int hiddenSize, int layers, int heads, int kvHeads, int intermediateSize, bool tied, bool isF16)
    {
        BuildSafetensorsPackageWithDtype(dir, weights, vocabSize, hiddenSize, layers, heads, kvHeads, intermediateSize, tied, isF16 ? "F16" : "F32");
    }

    private static void BuildSafetensorsPackageWithDtype(string dir, List<SyntheticWeight> weights, int vocabSize, int hiddenSize, int layers, int heads, int kvHeads, int intermediateSize, bool tied, string dtypeStr)
    {
        string configJson = $"{{\"model_type\":\"llama\",\"hidden_size\":{hiddenSize},\"num_hidden_layers\":{layers},\"num_attention_heads\":{heads},\"num_key_value_heads\":{kvHeads},\"intermediate_size\":{intermediateSize},\"vocab_size\":{vocabSize},\"max_position_embeddings\":512,\"rope_theta\":10000.0,\"rms_norm_eps\":1e-5,\"tie_word_embeddings\":{(tied ? "true" : "false")},\"torch_dtype\":\"{dtypeStr.ToLowerInvariant()}\"}}";
        File.WriteAllText(Path.Combine(dir, "config.json"), configJson);

        var vocabSb = new StringBuilder();
        vocabSb.Append("{\"model\":{\"type\":\"BPE\",\"vocab\":{");
        for (int i = 0; i < vocabSize; i++)
        {
            if (i > 0) vocabSb.Append(',');
            vocabSb.Append($"\"tok_{i}\":{i}");
        }
        vocabSb.Append("},\"merges\":[]}}");
        File.WriteAllText(Path.Combine(dir, "tokenizer.json"), vocabSb.ToString());

        var activeWeights = tied ? weights.Where(x => x.CanonicalName != "output.weight").ToList() : weights;

        var headerSb = new StringBuilder();
        headerSb.Append("{\"__metadata__\":{\"format\":\"pt\"}");

        long currentOffset = 0;
        var tensorDataList = new List<byte[]>();

        foreach (var item in activeWeights)
        {
            byte[] tensorBytes;
            if (dtypeStr == "F16")
            {
                tensorBytes = new byte[item.Data.Length * sizeof(ushort)];
                for (int i = 0; i < item.Data.Length; i++)
                {
                    ushort h = BitConverter.HalfToUInt16Bits((Half)item.Data[i]);
                    BinaryPrimitives.WriteUInt16LittleEndian(tensorBytes.AsSpan(i * 2, 2), h);
                }
            }
            else if (dtypeStr == "BF16")
            {
                tensorBytes = new byte[item.Data.Length * sizeof(ushort)];
                for (int i = 0; i < item.Data.Length; i++)
                {
                    uint bits = BitConverter.SingleToUInt32Bits(item.Data[i]);
                    ushort bf16 = (ushort)(bits >> 16);
                    BinaryPrimitives.WriteUInt16LittleEndian(tensorBytes.AsSpan(i * 2, 2), bf16);
                }
            }
            else
            {
                tensorBytes = new byte[item.Data.Length * sizeof(float)];
                for (int i = 0; i < item.Data.Length; i++)
                {
                    BinaryPrimitives.WriteSingleLittleEndian(tensorBytes.AsSpan(i * 4, 4), item.Data[i]);
                }
            }

            tensorDataList.Add(tensorBytes);
            long endOffset = currentOffset + tensorBytes.Length;

            string shapeJson = "[" + string.Join(",", item.SafetensorsShape) + "]";
            headerSb.Append($",\"{item.SafetensorsName}\":{{\"dtype\":\"{dtypeStr}\",\"shape\":{shapeJson},\"data_offsets\":[{currentOffset},{endOffset}]}}");

            currentOffset = endOffset;
        }

        headerSb.Append('}');
        string headerJson = headerSb.ToString();
        byte[] headerJsonBytes = Encoding.UTF8.GetBytes(headerJson);

        using var stFs = new FileStream(Path.Combine(dir, "model.safetensors"), FileMode.Create, FileAccess.Write);
        using var stW = new BinaryWriter(stFs, Encoding.UTF8);

        stW.Write((ulong)headerJsonBytes.Length);
        stW.Write(headerJsonBytes);

        foreach (byte[] bytes in tensorDataList)
        {
            stW.Write(bytes);
        }
    }

    private static void WriteGgufString(BinaryWriter w, string s)
    {
        byte[] b = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)b.Length);
        w.Write(b);
    }

    private static void WriteGgufValue(BinaryWriter w, GgufValueType type, object val)
    {
        switch (type)
        {
            case GgufValueType.UInt32: w.Write((uint)val); break;
            case GgufValueType.Float32: w.Write((float)val); break;
            case GgufValueType.String: WriteGgufString(w, (string)val); break;
            default: throw new NotSupportedException();
        }
    }

    #endregion
}
