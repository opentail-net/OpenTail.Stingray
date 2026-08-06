using System.Buffers.Binary;
using System.Text;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Tests.Core;

public sealed class SafetensorsTextModelPackageTests
{
    [Fact]
    public void Open_DenseLlamaPackage_ReportsExternalAssetsAndDtype()
    {
        using var package = TestPackage.Create();

        var model = SafetensorsTextModelPackage.Open(package.Directory);

        Assert.Equal("llama", model.ModelType);
        Assert.Equal(8, model.HiddenSize);
        Assert.Equal(1, model.NumHiddenLayers);
        Assert.Equal(2, model.NumAttentionHeads);
        Assert.Equal(2, model.NumKeyValueHeads);
        Assert.Equal(16, model.IntermediateSize);
        Assert.Equal(32, model.VocabSize);
        Assert.Equal(128, model.ContextLength);
        Assert.Equal(500_000f, model.RopeTheta);
        Assert.Equal(1e-6f, model.RmsNormEps);
        Assert.EndsWith("tokenizer.json", model.TokenizerPath);
        Assert.Equal(["F16"], model.WeightDtypes);
    }

    [Fact]
    public void Open_MapsConfigAndTensorNamesToTheCanonicalLlamaContract()
    {
        using var package = TestPackage.Create();

        var model = SafetensorsTextModelPackage.Open(package.Directory);
        var hp = ModelHyperparams.FromGgufMetadata(model.ToOpenTailMetadata());

        Assert.Equal(8, hp.EmbeddingDim);
        Assert.Equal(1, hp.NumLayers);
        Assert.Equal(2, hp.NumHeads);
        Assert.Equal(2, hp.NumKvHeads);
        Assert.Equal(4, hp.HeadDim);
        Assert.Equal(4, hp.RopeDim);
        Assert.Equal(16, hp.IntermediateDim);
        Assert.Equal(128, hp.ContextLength);
        Assert.Equal("token_embd.weight", SafetensorsTextModelPackage.TryMapToOpenTailTensorName("model.embed_tokens.weight"));
        Assert.Equal("blk.0.attn_q.weight", SafetensorsTextModelPackage.TryMapToOpenTailTensorName("model.layers.0.self_attn.q_proj.weight"));
        Assert.Equal("blk.12.ffn_down.weight", SafetensorsTextModelPackage.TryMapToOpenTailTensorName("model.layers.12.mlp.down_proj.weight"));
        Assert.Null(SafetensorsTextModelPackage.TryMapToOpenTailTensorName("model.layers.0.self_attn.q_proj.bias"));
    }

    [Fact]
    public void Open_UnsupportedWeightDtype_FailsBeforeInference()
    {
        using var package = TestPackage.Create(dtype: "I64");

        var ex = Assert.Throws<NotSupportedException>(() => SafetensorsTextModelPackage.Open(package.Directory));

        Assert.Contains("F32, F16, or BF16", ex.Message);
    }

    [Fact]
    public void Open_MissingProjection_FailsWithTheTensorName()
    {
        using var package = TestPackage.Create(omit: "model.layers.0.self_attn.o_proj.weight");

        var ex = Assert.Throws<InvalidDataException>(() => SafetensorsTextModelPackage.Open(package.Directory));

        Assert.Contains("self_attn.o_proj.weight", ex.Message);
    }

    [Fact]
    public void Open_IncompatibleTensorShape_FailsBeforeInference()
    {
        using var package = TestPackage.Create(shapeOverrideName: "model.layers.0.self_attn.k_proj.weight");

        var ex = Assert.Throws<InvalidDataException>(() => SafetensorsTextModelPackage.Open(package.Directory));

        Assert.Contains("self_attn.k_proj.weight", ex.Message);
        Assert.Contains("expected [8, 8]", ex.Message);
    }

    [Fact]
    public void CanonicalLoader_ExposesOriginalWeightsThroughOpenTailNames()
    {
        using var package = TestPackage.Create();
        using var weights = SafetensorsLlamaWeightLoader.Open(package.Directory);

        Assert.True(weights.Contains("token_embd.weight"));
        Assert.True(weights.Contains("blk.0.attn_output.weight"));
        Assert.False(weights.Contains("model.embed_tokens.weight"));
        Assert.Equal("F16", weights.GetDtype("blk.0.ffn_gate.weight"));
        Assert.Equal([16, 8], weights.GetShape("blk.0.ffn_gate.weight"));
    }

    [Fact]
    public void Open_MixedSupportedDtypes_AreAcceptedAndReported()
    {
        using var package = TestPackage.Create(dtype: "F16", alternateDtype: "BF16");

        var model = SafetensorsTextModelPackage.Open(package.Directory);

        Assert.Equal(["BF16", "F16"], model.WeightDtypes);
    }

    [Fact]
    public void Open_ValidTwoShardPackage_UsesTheIndex()
    {
        using var package = TestPackage.CreateSharded();

        var model = SafetensorsTextModelPackage.Open(package.Directory);

        Assert.Equal("llama", model.ModelType);
        Assert.EndsWith("model.safetensors.index.json", model.WeightsPath);
        Assert.Equal(["F16"], model.WeightDtypes);
    }

    private sealed class TestPackage : IDisposable
    {
        public string Directory { get; }

        private TestPackage(string directory) => Directory = directory;

        public static TestPackage Create(string dtype = "F16", string? alternateDtype = null,
            string? omit = null, string? shapeOverrideName = null)
        {
            string directory = Path.Combine(Path.GetTempPath(), "opentail-st-package-" + Guid.NewGuid().ToString("N"));
            System.IO.Directory.CreateDirectory(directory);
            File.WriteAllText(Path.Combine(directory, "config.json"), """
                {
                  "model_type": "llama",
                  "hidden_size": 8,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 2,
                  "intermediate_size": 16,
                  "vocab_size": 32,
                  "max_position_embeddings": 128,
                  "rope_theta": 500000.0,
                  "rms_norm_eps": 0.000001,
                  "hidden_act": "silu",
                  "attention_bias": false,
                  "mlp_bias": false
                }
                """);
            File.WriteAllText(Path.Combine(directory, "tokenizer.json"), "{}");

            string[] names =
            [
                "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
                "model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.v_proj.weight",
                "model.layers.0.self_attn.o_proj.weight", "model.layers.0.post_attention_layernorm.weight",
                "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight",
                "model.layers.0.mlp.down_proj.weight"
            ];
            // Every tensor gets a real, non-overlapping byte range matching its shape and dtype, and the
            // file carries the bytes to back it. The fixture used to declare shapes with
            // "data_offsets":[0,0] and write no data at all; SafetensorsLoader now rejects a header whose
            // declared extent disagrees with shape x dtype, and it is right to — a package that lies
            // about its own layout is exactly what the hardening pass exists to refuse.
            var header = new StringBuilder("{");
            bool first = true;
            long cursor = 0;
            foreach (string name in names)
                if (!string.Equals(name, omit, StringComparison.Ordinal))
                {
                    string tensorDtype = alternateDtype is not null && name == "model.norm.weight"
                        ? alternateDtype : dtype;
                    int[] dims = ShapeDims(name, shapeOverrideName);
                    long elements = 1;
                    foreach (int d in dims) elements *= d;
                    long end = cursor + (elements * BytesPerElement(tensorDtype));

                    if (!first) header.Append(',');
                    header.Append('"').Append(name).Append("\":{\"dtype\":\"").Append(tensorDtype)
                        .Append("\",\"shape\":[").AppendJoin(',', dims)
                        .Append("],\"data_offsets\":[").Append(cursor).Append(',').Append(end).Append("]}");
                    first = false;
                    cursor = end;
                }
            header.Append('}');
            byte[] json = Encoding.UTF8.GetBytes(header.ToString());
            using var file = new FileStream(Path.Combine(directory, "model.safetensors"), FileMode.CreateNew);
            Span<byte> headerLength = stackalloc byte[sizeof(ulong)];
            BinaryPrimitives.WriteUInt64LittleEndian(headerLength, (ulong)json.Length);
            file.Write(headerLength);
            file.Write(json);
            file.Write(new byte[cursor]);
            return new TestPackage(directory);
        }

        public static TestPackage CreateSharded()
        {
            var package = Create();
            File.Delete(Path.Combine(package.Directory, "model.safetensors"));
            string[] names =
            [
                "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
                "model.layers.0.input_layernorm.weight", "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.v_proj.weight",
                "model.layers.0.self_attn.o_proj.weight", "model.layers.0.post_attention_layernorm.weight",
                "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight",
                "model.layers.0.mlp.down_proj.weight"
            ];
            WriteShard(Path.Combine(package.Directory, "model-00001.safetensors"), names[..6]);
            WriteShard(Path.Combine(package.Directory, "model-00002.safetensors"), names[6..]);
            string map = string.Join(',', names.Select((name, i) =>
                $"\"{name}\":\"model-0000{(i < 6 ? 1 : 2)}.safetensors\""));
            File.WriteAllText(Path.Combine(package.Directory, "model.safetensors.index.json"),
                "{\"weight_map\":{" + map + "}}");
            return package;
        }

        private static void WriteShard(string path, IReadOnlyList<string> names)
        {
            var header = new StringBuilder("{");
            long cursor = 0;
            for (int i = 0; i < names.Count; i++)
            {
                int[] dims = ShapeDims(names[i], null);
                long elements = 1;
                foreach (int dim in dims) elements *= dim;
                long end = cursor + elements * BytesPerElement("F16");
                if (i > 0) header.Append(',');
                header.Append('"').Append(names[i]).Append("\":{\"dtype\":\"F16\",\"shape\":[")
                    .AppendJoin(',', dims).Append("],\"data_offsets\":[").Append(cursor).Append(',').Append(end).Append("]}");
                cursor = end;
            }
            header.Append('}');
            byte[] json = Encoding.UTF8.GetBytes(header.ToString());
            using var file = new FileStream(path, FileMode.CreateNew);
            file.Write(BitConverter.GetBytes((ulong)json.Length));
            file.Write(json);
            file.Write(new byte[cursor]);
        }

        private static int BytesPerElement(string dtype) => dtype switch
        {
            "F32" => 4,
            "F16" or "BF16" => 2,
            "I64" => 8,
            _ => throw new InvalidOperationException($"Fixture has no element width for dtype '{dtype}'."),
        };

        private static int[] ShapeDims(string name, string? overrideName)
        {
            if (string.Equals(name, overrideName, StringComparison.Ordinal)) return [7, 8];
            return name switch
            {
                "model.embed_tokens.weight" or "lm_head.weight" => [32, 8],
                "model.norm.weight" => [8],
                _ when name.EndsWith("input_layernorm.weight", StringComparison.Ordinal)
                    || name.EndsWith("post_attention_layernorm.weight", StringComparison.Ordinal) => [8],
                _ when name.EndsWith("self_attn.q_proj.weight", StringComparison.Ordinal)
                    || name.EndsWith("self_attn.k_proj.weight", StringComparison.Ordinal)
                    || name.EndsWith("self_attn.v_proj.weight", StringComparison.Ordinal)
                    || name.EndsWith("self_attn.o_proj.weight", StringComparison.Ordinal) => [8, 8],
                _ when name.EndsWith("mlp.gate_proj.weight", StringComparison.Ordinal)
                    || name.EndsWith("mlp.up_proj.weight", StringComparison.Ordinal) => [16, 8],
                _ when name.EndsWith("mlp.down_proj.weight", StringComparison.Ordinal) => [8, 16],
                _ => throw new InvalidOperationException($"Unexpected fixture tensor '{name}'."),
            };
        }

        public void Dispose()
        {
            try { System.IO.Directory.Delete(Directory, recursive: true); }
            catch { }
        }
    }
}
