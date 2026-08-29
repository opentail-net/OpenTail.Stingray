
namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Phase 0 of the SafeTensors plan: support must be decidable from the files on disk, without
/// constructing a forward pass, and every refusal must name the offending asset or setting.
/// </summary>
public sealed class ModelPackageInspectorTests
{
    [Fact]
    public void Inspect_ValidDenseLlamaPackage_IsSupportedAndReportsTheRoute()
    {
        using var pkg = Package.Create();

        var report = ModelPackageInspector.Inspect(pkg.Directory);

        Assert.True(report.IsSupported, report.ToSummary());
        Assert.Empty(report.Rejections);
        Assert.Equal("dense-llama-cpu", report.ProfileId);
        Assert.Equal("llama", report.ArchitectureId);
        Assert.Equal(["F16"], report.SourceDtypes);
        Assert.Equal(ModelPackageTokenizerFamily.HuggingFaceJson, report.TokenizerFamily);
        Assert.Equal(ModelPackageBackends.Cpu, report.AvailableBackends);

        // Memory must be reportable BEFORE load, from header arithmetic alone.
        Assert.NotNull(report.EstimatedWeightBytes);
        Assert.Equal(Package.ExpectedElements * 2, report.EstimatedWeightBytes);
    }

    [Fact]
    public void Inspect_MissingConfig_NamesConfigJson()
    {
        using var pkg = Package.Create(deleteConfig: true);

        var report = ModelPackageInspector.Inspect(pkg.Directory);

        Assert.False(report.IsSupported);
        var r = Assert.Single(report.Rejections, x => x.Kind == ModelPackageRejectionKind.MissingConfig);
        Assert.Equal("config.json", r.Subject);
    }

    [Fact]
    public void Inspect_MissingTokenizer_NamesTheTokenizerAsset()
    {
        using var pkg = Package.Create(deleteTokenizer: true);

        var report = ModelPackageInspector.Inspect(pkg.Directory);

        Assert.False(report.IsSupported);
        Assert.Contains(report.Rejections, x => x.Kind == ModelPackageRejectionKind.MissingTokenizer);
        Assert.Equal(ModelPackageTokenizerFamily.Unknown, report.TokenizerFamily);
    }

    [Fact]
    public void Inspect_SentencePieceOnly_IsRefusedByAProfileThatWantsHuggingFaceJson()
    {
        using var pkg = Package.Create(deleteTokenizer: true, sentencePiece: true);

        var report = ModelPackageInspector.Inspect(pkg.Directory);

        Assert.False(report.IsSupported);
        Assert.Equal(ModelPackageTokenizerFamily.SentencePiece, report.TokenizerFamily);
        Assert.Contains(report.Rejections,
            x => x.Kind == ModelPackageRejectionKind.MissingTokenizer && x.Subject == "SentencePiece");
    }

    [Fact]
    public void Inspect_UnsupportedArchitecture_NamesTheModelType()
    {
        using var pkg = Package.Create(modelType: "gpt2");

        var report = ModelPackageInspector.Inspect(pkg.Directory);

        Assert.False(report.IsSupported);
        var r = Assert.Single(report.Rejections, x => x.Kind == ModelPackageRejectionKind.UnsupportedArchitecture);
        Assert.Equal("gpt2", r.Subject);
    }

    /// <summary>
    /// Tied embeddings (tie_word_embeddings: true) are supported and aliased to token_embd.weight.
    /// This tests that config tie_word_embeddings=true is accepted.
    /// </summary>
    [Fact]
    public void Inspect_TiedWordEmbeddings_IsSupported()
    {
        using var pkg = Package.Create(extraConfig: "\"tie_word_embeddings\": true,");

        var report = ModelPackageInspector.Inspect(pkg.Directory);

        Assert.True(report.IsSupported);
        Assert.Empty(report.Rejections);
    }

    [Theory]
    [InlineData("\"attention_bias\": true,", "attention_bias")]
    [InlineData("\"mlp_bias\": true,", "mlp_bias")]
    [InlineData("\"rope_scaling\": {\"type\":\"linear\",\"factor\":2.0},", "rope_scaling")]
    [InlineData("\"hidden_act\": \"gelu\",", "hidden_act")]
    public void Inspect_UnfamiliarConfigSemantics_AreRefusedNotIgnored(string extra, string subject)
    {
        using var pkg = Package.Create(extraConfig: extra, includeHiddenAct: subject != "hidden_act");

        var report = ModelPackageInspector.Inspect(pkg.Directory);

        Assert.False(report.IsSupported);
        Assert.Contains(report.Rejections,
            x => x.Kind == ModelPackageRejectionKind.UnsupportedConfig && x.Subject == subject);
    }

    [Fact]
    public void Inspect_UnsupportedDtype_NamesTheDtype()
    {
        using var pkg = Package.Create(dtype: "I64");

        var report = ModelPackageInspector.Inspect(pkg.Directory);

        Assert.False(report.IsSupported);
        var r = Assert.Single(report.Rejections, x => x.Kind == ModelPackageRejectionKind.UnsupportedDtype);
        Assert.Equal("I64", r.Subject);
    }

    [Fact]
    public void Inspect_NoWeights_NamesTheMissingWeights()
    {
        using var pkg = Package.Create(deleteWeights: true);

        var report = ModelPackageInspector.Inspect(pkg.Directory);

        Assert.False(report.IsSupported);
        Assert.Contains(report.Rejections, x => x.Kind == ModelPackageRejectionKind.MissingWeights);
    }

    [Fact]
    public void Inspect_ShardIndexReferencingAMissingFile_NamesTheShard()
    {
        using var pkg = Package.Create(deleteWeights: true);
        File.WriteAllText(Path.Combine(pkg.Directory, "model.safetensors.index.json"),
            """{"weight_map":{"model.norm.weight":"model-00001-of-00002.safetensors"}}""");

        var report = ModelPackageInspector.Inspect(pkg.Directory);

        Assert.False(report.IsSupported);
        var r = Assert.Single(report.Rejections, x => x.Kind == ModelPackageRejectionKind.MissingShard);
        Assert.Equal("model-00001-of-00002.safetensors", r.Subject);
    }

    /// <summary>Shard indexes are untrusted metadata and must not reach outside the package root.</summary>
    [Fact]
    public void Inspect_ShardIndexEscapingThePackageRoot_IsRefused()
    {
        using var pkg = Package.Create(deleteWeights: true);
        File.WriteAllText(Path.Combine(pkg.Directory, "model.safetensors.index.json"),
            """{"weight_map":{"model.norm.weight":"../../escaped.safetensors"}}""");

        var report = ModelPackageInspector.Inspect(pkg.Directory);

        Assert.False(report.IsSupported);
        Assert.Contains(report.Rejections,
            x => x.Kind == ModelPackageRejectionKind.MalformedPackage && x.Detail.Contains("outside the package root"));
    }

    [Fact]
    public void Inspect_MalformedConfigJson_IsReportedNotThrown()
    {
        using var pkg = Package.Create();
        File.WriteAllText(Path.Combine(pkg.Directory, "config.json"), "{ this is not json ");

        var report = ModelPackageInspector.Inspect(pkg.Directory);

        Assert.False(report.IsSupported);
        Assert.Contains(report.Rejections,
            x => x.Kind == ModelPackageRejectionKind.MalformedPackage && x.Subject == "config.json");
    }

    [Fact]
    public void Inspect_NonObjectConfigJson_IsReportedNotThrown()
    {
        using var pkg = Package.Create();
        File.WriteAllText(Path.Combine(pkg.Directory, "config.json"), "[]");

        var report = ModelPackageInspector.Inspect(pkg.Directory);

        Assert.False(report.IsSupported);
        Assert.Contains(report.Rejections,
            x => x.Kind == ModelPackageRejectionKind.MalformedPackage && x.Subject == "config.json");
    }

    /// <summary>
    /// Callers print every reason at once, so inspection must not stop at the first fault.
    /// </summary>
    [Fact]
    public void Inspect_SeveralFaults_ReportsAllOfThem()
    {
        using var pkg = Package.Create(modelType: "gpt2", dtype: "I64", deleteTokenizer: true);

        var report = ModelPackageInspector.Inspect(pkg.Directory);

        Assert.False(report.IsSupported);
        Assert.Contains(report.Rejections, x => x.Kind == ModelPackageRejectionKind.UnsupportedArchitecture);
        Assert.Contains(report.Rejections, x => x.Kind == ModelPackageRejectionKind.UnsupportedDtype);
        Assert.Contains(report.Rejections, x => x.Kind == ModelPackageRejectionKind.MissingTokenizer);
        Assert.True(report.Rejections.Count >= 3, report.ToSummary());
    }

    [Fact]
    public void Inspect_NonExistentPath_IsReportedNotThrown()
    {
        string missing = Path.Combine(Path.GetTempPath(), "opentail-absent-" + Guid.NewGuid().ToString("N"), "sub");

        var report = ModelPackageInspector.Inspect(missing);

        Assert.False(report.IsSupported);
        Assert.Contains(report.Rejections, x => x.Kind == ModelPackageRejectionKind.MalformedPackage);
    }

    [Fact]
    public void UnsupportedPackage_AdvertisesNoBackends()
    {
        using var pkg = Package.Create(modelType: "gpt2");

        var report = ModelPackageInspector.Inspect(pkg.Directory);

        Assert.Equal(ModelPackageBackends.None, report.AvailableBackends);
    }

    [Fact]
    public void CapabilityProfile_IsNarrowAndVersioned()
    {
        var profile = ModelPackageCapability.DenseLlamaCpu;

        Assert.Equal(ModelPackageCapability.CurrentSchemaVersion, profile.SchemaVersion);
        Assert.Equal(ModelPackageBackends.Cpu, profile.Backends);
        Assert.False(profile.SupportsBatching);
        Assert.False(profile.SupportsSessions);
        Assert.False(profile.SupportsMultimodal);
        Assert.DoesNotContain(profile.Exclusions, e => e.Contains("Tied output embeddings"));
        Assert.Contains(profile.Exclusions, e => e.Contains("CUDA"));
    }

    [Fact]
    public void RenderCapabilityTable_PublishesEveryProfileAsARow()
    {
        string table = ModelPackageInspector.RenderCapabilityTable();

        Assert.Contains("| Profile |", table);
        foreach (var c in ModelPackageCapability.All)
            Assert.Contains(c.ProfileId, table);
    }

    private sealed class Package : IDisposable
    {
        // 12 tensors of the 8-wide / 32-vocab / 16-intermediate fixture below.
        public const int ExpectedElements = (32 * 8) + 8 + (32 * 8)      // embed, norm, lm_head
                                          + 8 + 8                        // two layer norms
                                          + (8 * 8) * 4                  // q,k,v,o
                                          + (16 * 8) * 2 + (8 * 16);     // gate,up,down

        public string Directory { get; }

        private Package(string directory) => Directory = directory;

        public static Package Create(
            string dtype = "F16",
            string modelType = "llama",
            string extraConfig = "",
            bool includeHiddenAct = true,
            bool deleteConfig = false,
            bool deleteTokenizer = false,
            bool deleteWeights = false,
            bool sentencePiece = false)
        {
            string dir = Path.Combine(Path.GetTempPath(), "opentail-cap-" + Guid.NewGuid().ToString("N"));
            System.IO.Directory.CreateDirectory(dir);

            if (!deleteConfig)
            {
                string act = includeHiddenAct ? "\"hidden_act\": \"silu\"," : "";
                File.WriteAllText(Path.Combine(dir, "config.json"), $$"""
                    {
                      "model_type": "{{modelType}}",
                      {{extraConfig}}
                      {{act}}
                      "hidden_size": 8,
                      "num_hidden_layers": 1,
                      "num_attention_heads": 2,
                      "num_key_value_heads": 2,
                      "intermediate_size": 16,
                      "vocab_size": 32,
                      "max_position_embeddings": 128,
                      "rope_theta": 500000.0,
                      "rms_norm_eps": 0.000001
                    }
                    """);
            }

            if (!deleteTokenizer) File.WriteAllText(Path.Combine(dir, "tokenizer.json"), "{}");
            else if (sentencePiece) File.WriteAllBytes(Path.Combine(dir, "tokenizer.model"), [1, 2, 3]);

            if (!deleteWeights) WriteWeights(Path.Combine(dir, "model.safetensors"), dtype);
            return new Package(dir);
        }

        private static void WriteWeights(string path, string dtype)
        {
            (string Name, int[] Shape)[] tensors =
            [
                ("model.embed_tokens.weight", [32, 8]),
                ("model.norm.weight", [8]),
                ("lm_head.weight", [32, 8]),
                ("model.layers.0.input_layernorm.weight", [8]),
                ("model.layers.0.self_attn.q_proj.weight", [8, 8]),
                ("model.layers.0.self_attn.k_proj.weight", [8, 8]),
                ("model.layers.0.self_attn.v_proj.weight", [8, 8]),
                ("model.layers.0.self_attn.o_proj.weight", [8, 8]),
                ("model.layers.0.post_attention_layernorm.weight", [8]),
                ("model.layers.0.mlp.gate_proj.weight", [16, 8]),
                ("model.layers.0.mlp.up_proj.weight", [16, 8]),
                ("model.layers.0.mlp.down_proj.weight", [8, 16]),
            ];

            int width = dtype switch { "F32" => 4, "I64" => 8, _ => 2 };
            var header = new StringBuilder("{");
            long offset = 0;
            for (int i = 0; i < tensors.Length; i++)
            {
                var (name, shape) = tensors[i];
                int elements = 1;
                foreach (int d in shape) elements *= d;
                long end = offset + (long)elements * width;
                if (i > 0) header.Append(',');
                header.Append('"').Append(name).Append("\":{\"dtype\":\"").Append(dtype)
                      .Append("\",\"shape\":[").Append(string.Join(',', shape))
                      .Append("],\"data_offsets\":[").Append(offset).Append(',').Append(end).Append("]}");
                offset = end;
            }
            header.Append('}');

            byte[] json = Encoding.UTF8.GetBytes(header.ToString());
            using var file = new FileStream(path, FileMode.CreateNew);
            file.Write(BitConverter.GetBytes((ulong)json.Length));
            file.Write(json);
            file.Write(new byte[offset]);
        }

        public void Dispose()
        {
            try { System.IO.Directory.Delete(Directory, recursive: true); }
            catch { }
        }
    }
}
