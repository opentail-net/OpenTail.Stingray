using System.Text.RegularExpressions;

namespace OpenTail.Stingray.Tests.Cli;

public sealed class StaticPlanConfigurationTests
{
    [Fact]
    public void Resolve_UsesCliThenProfileThenEnvironmentThenDefault_AndReportsConflict()
    {
        var settings = new StaticPlanCommand.Settings { Backend = "cuda", MaxBatchSize = 4 };
        var profile = new StaticPlanProfile(Backend: "vulkan", MaxBatchSize: 2, KvType: "bf16");
        string? Environment(string name) => name switch
        {
            "STINGRAY_BACKEND" => "cpu",
            "STINGRAY_MAX_BATCH" => "8",
            "STINGRAY_KV_DTYPE" => "q8_0",
            _ => null,
        };

        var result = StaticPlanConfiguration.Resolve(settings, profile, Environment);

        Assert.Equal("cuda", result.Get<string>("backend"));
        Assert.Equal("cli", result.Values["backend"].Source);
        Assert.Equal(4, result.Get<int>("max_batch"));
        Assert.Equal("bf16", result.Get<string>("kv_type"));
        Assert.Equal("profile", result.Values["kv_type"].Source);
        Assert.Equal("auto", result.Get<string>("spec_type"));
        Assert.Contains(result.Diagnostics, x => x.Field == "backend" && x.Kind == "conflict");
    }

    [Fact]
    public void Resolve_EnvironmentIsUsedWhenNoHigherPriorityValueExists()
    {
        var result = StaticPlanConfiguration.Resolve(new StaticPlanCommand.Settings(), null,
            name => name == "STINGRAY_TQ" ? "true" : null);

        Assert.True(result.Get<bool>("turbo_quant"));
        Assert.Equal("environment", result.Values["turbo_quant"].Source);
        Assert.Equal(System.Text.Json.JsonValueKind.True, result.Values["turbo_quant"].Value.ValueKind);
    }

    [Fact]
    public void Resolve_JsonSnapshotPreservesPrimitiveValueTypes()
    {
        var result = StaticPlanConfiguration.Resolve(
            new StaticPlanCommand.Settings { MaxBatchSize = 8, ToolGrammar = false }, null, _ => null);

        Assert.Equal(System.Text.Json.JsonValueKind.Number, result.Values["max_batch"].Value.ValueKind);
        Assert.Equal(System.Text.Json.JsonValueKind.False, result.Values["tool_grammar"].Value.ValueKind);
    }

    [Fact]
    public void Profile_RejectsUnknownFields()
    {
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, "{\"backend\":\"cpu\",\"typoed_setting\":true}");
            Assert.Throws<System.Text.Json.JsonException>(() => StaticPlanProfile.Load(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Profile_AcceptsThePublishedSnakeCaseSchemaNames()
    {
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllText(path, "{\"gpu_layers\":-1,\"turbo_quant\":true,\"snap_kv_budget\":0}");
            var profile = StaticPlanProfile.Load(path);

            Assert.Equal(-1, profile!.GpuLayers);
            Assert.True(profile.TurboQuant);
            Assert.Equal(0, profile.SnapKvBudget);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Theory]
    [InlineData("cpu.json")]
    [InlineData("cuda-dense.json")]
    [InlineData("hybrid-moe.json")]
    [InlineData("local-server.json")]
    [InlineData("vulkan.json")]
    public void Profile_CheckedInRecommendedProfile_IsAcceptedByTheStrictLoader(string filename)
    {
        string root = FindRepositoryRoot();
        string path = Path.Combine(root, "docs", "profiles", filename);

        Assert.NotNull(StaticPlanProfile.Load(path));
    }

    [Fact]
    public void Resolve_InvalidEnvironmentValueFallsBackWithDiagnostic()
    {
        var result = StaticPlanConfiguration.Resolve(new StaticPlanCommand.Settings(), null,
            name => name == "STINGRAY_MAX_BATCH" ? "not-a-number" : null);

        Assert.Equal(1, result.Get<int>("max_batch"));
        Assert.Equal("default_after_invalid_environment", result.Values["max_batch"].Source);
        Assert.Contains(result.Diagnostics, x => x.Kind == "invalid" && x.Field == "max_batch");
    }

    [Fact]
    public void Resolve_DoesNotInventEnvironmentInputsThatRunDoesNotRead()
    {
        var result = StaticPlanConfiguration.Resolve(new StaticPlanCommand.Settings(), null,
            name => name switch
            {
                "STINGRAY_CTX_SIZE" => "4096",
                "STINGRAY_SPEC_TYPE" => "mtp",
                _ => null,
            });

        Assert.Equal(0, result.Get<int>("context_size"));
        Assert.Equal("auto", result.Get<string>("spec_type"));
        Assert.Equal("default", result.Values["context_size"].Source);
        Assert.Equal("default", result.Values["spec_type"].Source);
    }

    [Fact]
    public void ResolvedProfile_RoundTripsTheResolvedPlanningValues()
    {
        var configuration = StaticPlanConfiguration.Resolve(
            new StaticPlanCommand.Settings { Backend = "cpu", MaxBatchSize = 3, ToolGrammar = true }, null, _ => null);

        var profile = StaticPlanProfile.FromEffectiveConfiguration(configuration);

        Assert.Equal("cpu", profile.Backend);
        Assert.Equal(3, profile.MaxBatchSize);
        Assert.True(profile.ToolGrammar);
        Assert.Null(profile.SnapKvBudget); // -1 means unset, not an explicit profile value.
    }

    /// <summary>
    /// The option inventory is deliberately a human-classified document, but its declared source
    /// count must never silently become stale. This lightweight guard keeps the refresh work
    /// visible without pretending that a raw attribute scan can classify a public option.
    /// </summary>
    [Fact]
    public void CliOptionInventory_DeclaredCountMatchesSource()
    {
        string root = FindRepositoryRoot();
        int sourceCount = Directory.EnumerateFiles(Path.Combine(root, "src", "OpenTail.Stingray.Cli"), "*.cs", SearchOption.AllDirectories)
            .Sum(path => File.ReadLines(path).Count(line => line.Contains("[CommandOption", StringComparison.Ordinal)));

        string inventory = File.ReadAllText(Path.Combine(root, "docs", "cli-option-inventory.md"));
        Match count = Regex.Match(inventory, @"\*\*(\d+) option declarations\*\*");
        Assert.True(count.Success, "CLI option inventory must declare its current source option count.");
        Assert.Equal(sourceCount, int.Parse(count.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture));

        // The declared count alone was not enough: it tracked source correctly while the TABLE
        // silently fell to 96 rows against 149 options, which is exactly the drift the document
        // warned about. Assert the rows too, so the inventory cannot be half-current again.
        int rows = Regex.Matches(inventory, @"(?m)^\| `[^`]+` \|").Count;
        Assert.True(rows == sourceCount,
            $"cli-option-inventory.md has {rows} option rows but src declares {sourceCount}. " +
            "Regenerate with scripts/gen-cli-option-inventory.ps1.");
    }

    private static string FindRepositoryRoot()
    {
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
        {
            if (File.Exists(Path.Combine(directory.FullName, "OpenTail.Stingray.slnx")))
                return directory.FullName;
        }

        throw new DirectoryNotFoundException("Could not find the OpenTail.Stingray repository root.");
    }
}
