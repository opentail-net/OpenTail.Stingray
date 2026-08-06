using OpenTail.Stingray.Cli;

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
}
