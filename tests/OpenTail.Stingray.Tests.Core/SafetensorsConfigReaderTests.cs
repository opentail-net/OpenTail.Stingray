
namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Phase 1 item 4: package configuration must be parsed explicitly, and anything not fully
/// understood must be refused rather than defaulted.
/// </summary>
public sealed class SafetensorsConfigReaderTests
{
    [Fact]
    public void Read_WellFormedLlamaConfig_MapsCanonicalHyperparameters()
    {
        using var dir = new TempDir();
        dir.WriteConfig();

        var result = SafetensorsConfigReader.Read(dir.ConfigPath);

        Assert.True(result.IsUsable, string.Join("; ", result.Rejections));
        var hp = result.Hyperparams!;
        Assert.Equal(32, hp.VocabSize);
        Assert.Equal(128, hp.ContextLength);
        Assert.Equal(8, hp.EmbeddingDim);
        Assert.Equal(1, hp.NumLayers);
        Assert.Equal(2, hp.NumHeads);
        Assert.Equal(2, hp.NumKvHeads);
        Assert.Equal(16, hp.IntermediateDim);
        Assert.Equal(4, hp.HeadDim);
        Assert.Equal(4, hp.RopeDim);
        Assert.Equal(500_000f, hp.RopeTheta);
        Assert.Equal(1e-6f, hp.RmsNormEps);
    }

    /// <summary>
    /// The core rule: an unread setting may change the arithmetic, so unknown keys are refused.
    /// </summary>
    [Fact]
    public void Read_UnknownConfigKey_IsRefusedNotIgnored()
    {
        using var dir = new TempDir();
        dir.WriteConfig(extra: "\"sliding_window\": 4096,");

        var result = SafetensorsConfigReader.Read(dir.ConfigPath);

        Assert.False(result.IsUsable);
        Assert.Null(result.Hyperparams);
        var r = Assert.Single(result.Rejections, x => x.Subject == "sliding_window");
        Assert.Equal(ModelPackageRejectionKind.UnsupportedConfig, r.Kind);
    }

    [Fact]
    public void Read_BenignProvenanceKeys_AreAccepted()
    {
        using var dir = new TempDir();
        dir.WriteConfig(extra:
            "\"architectures\": [\"LlamaForCausalLM\"], \"torch_dtype\": \"bfloat16\", " +
            "\"transformers_version\": \"4.44.0\", \"use_cache\": true,");

        var result = SafetensorsConfigReader.Read(dir.ConfigPath);

        Assert.True(result.IsUsable, string.Join("; ", result.Rejections));
    }

    /// <summary>
    /// An explicit head_dim must win over hidden_size/heads: a model whose head_dim differs is a
    /// different model, and deriving it silently would produce plausible-looking output.
    /// </summary>
    [Fact]
    public void Read_ExplicitHeadDim_IsHonouredOverTheDerivedValue()
    {
        using var dir = new TempDir();
        dir.WriteConfig(extra: "\"head_dim\": 6,");

        var result = SafetensorsConfigReader.Read(dir.ConfigPath);

        Assert.True(result.IsUsable, string.Join("; ", result.Rejections));
        Assert.Equal(6, result.Hyperparams!.HeadDim);
        Assert.Equal(6, result.Hyperparams!.RopeDim);
    }

    [Fact]
    public void Read_HiddenSizeNotDivisibleByHeads_WithoutExplicitHeadDim_IsRefused()
    {
        using var dir = new TempDir();
        dir.WriteConfig(hiddenSize: 9);

        var result = SafetensorsConfigReader.Read(dir.ConfigPath);

        Assert.False(result.IsUsable);
        Assert.Contains(result.Rejections, x => x.Subject == "hidden_size");
    }

    [Fact]
    public void Read_HeadsNotAMultipleOfKvHeads_IsRefused()
    {
        using var dir = new TempDir();
        dir.WriteConfig(heads: 4, kvHeads: 3);

        var result = SafetensorsConfigReader.Read(dir.ConfigPath);

        Assert.False(result.IsUsable);
        Assert.Contains(result.Rejections, x => x.Subject == "num_key_value_heads");
    }

    [Theory]
    [InlineData("\"rope_scaling\": {\"type\":\"linear\",\"factor\":2.0},", "rope_scaling")]
    [InlineData("\"attention_bias\": true,", "attention_bias")]
    [InlineData("\"mlp_bias\": true,", "mlp_bias")]
    public void Read_SemanticsOutsideTheProfile_AreRefused(string extra, string subject)
    {
        using var dir = new TempDir();
        dir.WriteConfig(extra: extra);

        var result = SafetensorsConfigReader.Read(dir.ConfigPath);

        Assert.False(result.IsUsable);
        Assert.Contains(result.Rejections, x => x.Subject == subject);
    }

    [Fact]
    public void Read_TieWordEmbeddings_IsAccepted()
    {
        using var dir = new TempDir();
        dir.WriteConfig(extra: "\"tie_word_embeddings\": true,");

        var result = SafetensorsConfigReader.Read(dir.ConfigPath);

        Assert.True(result.IsUsable);
        Assert.Empty(result.Rejections);
    }

    [Fact]
    public void Read_NonSiluActivation_IsRefused()
    {
        using var dir = new TempDir();
        dir.WriteConfig(activation: "gelu");

        var result = SafetensorsConfigReader.Read(dir.ConfigPath);

        Assert.False(result.IsUsable);
        Assert.Contains(result.Rejections, x => x.Subject == "hidden_act");
    }

    [Fact]
    public void Read_MissingContextLength_IsRefusedRatherThanDefaulted()
    {
        using var dir = new TempDir();
        dir.WriteConfig(includeContext: false);

        var result = SafetensorsConfigReader.Read(dir.ConfigPath);

        Assert.False(result.IsUsable);
        Assert.Contains(result.Rejections, x => x.Subject == "max_position_embeddings");
    }

    [Fact]
    public void Read_MissingRequiredDimension_NamesIt()
    {
        using var dir = new TempDir();
        dir.WriteConfig(includeIntermediate: false);

        var result = SafetensorsConfigReader.Read(dir.ConfigPath);

        Assert.False(result.IsUsable);
        Assert.Contains(result.Rejections,
            x => x.Subject == "intermediate_size" && x.Kind == ModelPackageRejectionKind.UnsupportedConfig);
    }

    [Fact]
    public void Read_MalformedJson_IsReportedNotThrown()
    {
        using var dir = new TempDir();
        File.WriteAllText(dir.ConfigPath, "{ nope ");

        var result = SafetensorsConfigReader.Read(dir.ConfigPath);

        Assert.False(result.IsUsable);
        Assert.Contains(result.Rejections, x => x.Kind == ModelPackageRejectionKind.MalformedPackage);
    }

    [Theory]
    [InlineData("[]")]
    [InlineData("null")]
    [InlineData("\"not an object\"")]
    public void Read_NonObjectConfigJson_IsReportedNotThrown(string payload)
    {
        using var dir = new TempDir();
        File.WriteAllText(dir.ConfigPath, payload);

        var result = SafetensorsConfigReader.Read(dir.ConfigPath);

        Assert.False(result.IsUsable);
        Assert.Contains(result.Rejections, x => x.Kind == ModelPackageRejectionKind.MalformedPackage);
    }

    [Fact]
    public void Read_MissingFile_IsReportedNotThrown()
    {
        using var dir = new TempDir();

        var result = SafetensorsConfigReader.Read(Path.Combine(dir.Path, "absent.json"));

        Assert.False(result.IsUsable);
        Assert.Contains(result.Rejections, x => x.Kind == ModelPackageRejectionKind.MissingConfig);
    }

    [Fact]
    public void Read_SpecialTokenIds_AreCarriedThrough()
    {
        using var dir = new TempDir();
        dir.WriteConfig(extra: "\"bos_token_id\": 1, \"eos_token_id\": 2, \"pad_token_id\": 0,");

        var result = SafetensorsConfigReader.Read(dir.ConfigPath);

        Assert.True(result.IsUsable, string.Join("; ", result.Rejections));
        Assert.Equal(1, result.Generation.BosTokenId);
        Assert.Equal(2, result.Generation.EosTokenId);
        Assert.Equal(0, result.Generation.PadTokenId);
    }

    [Fact]
    public void ReadGenerationDefaults_OverlaysSamplingDefaults()
    {
        using var dir = new TempDir();
        dir.WriteConfig(extra: "\"eos_token_id\": 2,");
        File.WriteAllText(Path.Combine(dir.Path, "generation_config.json"),
            """{"temperature":0.6,"top_p":0.95,"top_k":20,"bos_token_id":7}""");

        var config = SafetensorsConfigReader.Read(dir.ConfigPath);
        var generation = SafetensorsConfigReader.ReadGenerationDefaults(dir.Path, config.Generation);

        Assert.Equal(0.6f, generation.Temperature);
        Assert.Equal(0.95f, generation.TopP);
        Assert.Equal(20, generation.TopK);
        Assert.Equal(7, generation.BosTokenId);
        Assert.Equal(2, generation.EosTokenId);   // config.json value survives
    }

    /// <summary>Defaults are not semantics: a broken generation_config must not block loading.</summary>
    [Fact]
    public void ReadGenerationDefaults_MalformedFile_FallsBackToConfigValues()
    {
        using var dir = new TempDir();
        dir.WriteConfig(extra: "\"eos_token_id\": 2,");
        File.WriteAllText(Path.Combine(dir.Path, "generation_config.json"), "{ broken ");

        var config = SafetensorsConfigReader.Read(dir.ConfigPath);
        var generation = SafetensorsConfigReader.ReadGenerationDefaults(dir.Path, config.Generation);

        Assert.Equal(2, generation.EosTokenId);
        Assert.Null(generation.Temperature);
    }

    [Fact]
    public void ReadGenerationDefaults_AbsentFile_IsNotAnError()
    {
        using var dir = new TempDir();
        dir.WriteConfig();

        var config = SafetensorsConfigReader.Read(dir.ConfigPath);
        var generation = SafetensorsConfigReader.ReadGenerationDefaults(dir.Path, config.Generation);

        Assert.Null(generation.Temperature);
    }

    private sealed class TempDir : IDisposable
    {
        public string Path { get; } = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(), "opentail-cfg-" + Guid.NewGuid().ToString("N"));

        public string ConfigPath => System.IO.Path.Combine(Path, "config.json");

        public TempDir() => Directory.CreateDirectory(Path);

        public void WriteConfig(
            string extra = "",
            string activation = "silu",
            int hiddenSize = 8,
            int heads = 2,
            int kvHeads = 2,
            bool includeContext = true,
            bool includeIntermediate = true)
        {
            string context = includeContext ? "\"max_position_embeddings\": 128," : "";
            string intermediate = includeIntermediate ? "\"intermediate_size\": 16," : "";
            File.WriteAllText(ConfigPath, $$"""
                {
                  "model_type": "llama",
                  {{extra}}
                  "hidden_size": {{hiddenSize}},
                  "num_hidden_layers": 1,
                  "num_attention_heads": {{heads}},
                  "num_key_value_heads": {{kvHeads}},
                  {{intermediate}}
                  "vocab_size": 32,
                  {{context}}
                  "rope_theta": 500000.0,
                  "rms_norm_eps": 0.000001,
                  "hidden_act": "{{activation}}"
                }
                """);
        }

        public void Dispose()
        {
            try { Directory.Delete(Path, recursive: true); } catch { }
        }
    }
}
