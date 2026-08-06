using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Exercises the tooling against a real downloaded Hugging Face package rather than a fixture.
/// </summary>
/// <remarks>
/// <para>Skipped when the package is absent, so the suite stays hermetic and CI does not depend on a
/// network fetch. Present locally it is the only test that sees what real packages actually contain —
/// which is how the tied-embedding, BF16 and unknown-config-key gaps were found.</para>
///
/// <para>Package: <c>HuggingFaceTB/SmolLM2-135M-Instruct</c> under <c>models/</c>.</para>
/// </remarks>
public sealed class RealPackageTouchpointTests
{
    private static string? PackagePath()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "models", "SmolLM2-135M-Instruct");
            if (File.Exists(Path.Combine(candidate, "config.json"))) return candidate;
        }
        return null;
    }

    [Fact]
    public void Inspect_RealSmolLm2Package_IsReadCorrectlyAndSupported()
    {
        string? path = PackagePath();
        Assert.SkipWhen(path is null, "SmolLM2-135M-Instruct package not present under models/.");

        var report = ModelPackageInspector.Inspect(path!);

        // Everything the inspector reads from disk must be right.
        Assert.Equal("llama", report.ArchitectureId);
        Assert.Equal(["BF16"], report.SourceDtypes);
        Assert.Equal(ModelPackageTokenizerFamily.HuggingFaceJson, report.TokenizerFamily);
        Assert.NotNull(report.EstimatedWeightBytes);
        Assert.True(report.EstimatedWeightBytes > 200L * 1024 * 1024);

        // Tied embeddings and BF16 execution are now fully supported.
        Assert.True(report.IsSupported);
        Assert.Empty(report.Rejections);
        Assert.Equal(ModelPackageBackends.Cpu, report.AvailableBackends);
    }

    [Fact]
    public void ConfigReader_RealSmolLm2Config_HasNoRejections()
    {
        string? path = PackagePath();
        Assert.SkipWhen(path is null, "SmolLM2-135M-Instruct package not present under models/.");

        var result = SafetensorsConfigReader.Read(Path.Combine(path!, "config.json"));

        Assert.Empty(result.Rejections);
    }

    [Fact]
    public void TokenizerSource_RealSmolLm2Tokenizer_LoadsThroughTheSharedConstructionPath()
    {
        string? path = PackagePath();
        Assert.SkipWhen(path is null, "SmolLM2-135M-Instruct package not present under models/.");

        var loaded = HuggingFaceTokenizerSource.Load(path!);
        Assert.True(loaded.IsUsable, string.Join("; ", loaded.Rejections));

        var source = loaded.Source!;
        Assert.Equal(49152, source.Tokens.Length);
        Assert.NotEmpty(source.Merges);

        var tokenizer = GgufTokenizer.FromSource(source);
        Assert.Equal(49152, tokenizer.VocabSize);

        // A real round trip through the real vocabulary.
        var ids = tokenizer.Encode("Hello world");
        Assert.NotEmpty(ids);
        Assert.Contains("Hello", tokenizer.Decode(ids));
    }
}
