
namespace OpenTail.Stingray.Tests.Server.Fast;

public sealed class ServerEnvironmentOverridesTests
{
    [Fact]
    public void Apply_UsesValidEnvironmentValuesAndReturnsTheirReceipt()
    {
        var options = new OpenTailStingrayServerOptions
        {
            ModelPath = "from-config.gguf",
            MaxBatchSize = 1,
            Backend = ServerBackend.Cpu,
        };
        string? Environment(string name) => name switch
        {
            "STINGRAY_MODEL" => "from-env.gguf",
            "STINGRAY_MAX_BATCH" => "4",
            "STINGRAY_BACKEND" => "cuda",
            "STINGRAY_TQ" => "true",
            "STINGRAY_TOOL_GRAMMAR" => "1",
            _ => null,
        };

        var applied = ServerEnvironmentOverrides.Apply(options, Environment);

        Assert.Equal("from-env.gguf", options.ModelPath);
        Assert.Equal(4, options.MaxBatchSize);
        Assert.Equal(ServerBackend.Cuda, options.Backend);
        Assert.True(options.TurboQuant);
        Assert.True(options.ToolGrammar);
        Assert.Equal(["STINGRAY_MODEL", "STINGRAY_MAX_BATCH", "STINGRAY_BACKEND", "STINGRAY_TQ", "STINGRAY_TOOL_GRAMMAR"], applied);
    }

    [Fact]
    public void Apply_IgnoresMalformedAndDisabledValues()
    {
        var options = new OpenTailStingrayServerOptions { MaxBatchSize = 7, Backend = ServerBackend.Vulkan };
        string? Environment(string name) => name switch
        {
            "STINGRAY_MAX_BATCH" => "zero",
            "STINGRAY_BACKEND" => "metal",
            "STINGRAY_TQ" => "false",
            _ => null,
        };

        var applied = ServerEnvironmentOverrides.Apply(options, Environment);

        Assert.Empty(applied);
        Assert.Equal(7, options.MaxBatchSize);
        Assert.Equal(ServerBackend.Vulkan, options.Backend);
        Assert.False(options.TurboQuant);
    }

    [Fact]
    public void Receipt_SortsNamesAndDoesNotExposeValues()
    {
        var receipt = new ServerEnvironmentOverrideReceipt();

        receipt.Record(["STINGRAY_TQ", "STINGRAY_MODEL"]);

        Assert.Equal(["STINGRAY_MODEL", "STINGRAY_TQ"], receipt.Names);
    }

    [Fact]
    public void Apply_BudgetsAndThinkingOverrides_AppliesExpectedOptionsAndRecordsReceipt()
    {
        // docs/3-product-and-runtime/04-quality-of-life-improvements-plan.md Item 2:
        // Memory budgets, prefill chunks, thinking controls, and mmproj path.
        var options = new OpenTailStingrayServerOptions();
        string? Environment(string name) => name switch
        {
            "STINGRAY_MMPROJ" => "mmproj.gguf",
            "STINGRAY_MAX_QUEUE" => "64",
            "STINGRAY_MAX_CONCURRENT" => "16",
            "STINGRAY_PREFILL_CHUNK" => "512",
            "STINGRAY_KV_BUDGET_MB" => "2048",
            "STINGRAY_PREFIX_CACHE_MB" => "1024",
            "STINGRAY_PREFILL_DEQUANT_MB" => "256",
            "STINGRAY_N_GPU_LAYERS" => "33",
            "STINGRAY_KV_DTYPE" => "q8_0",
            "STINGRAY_TQ_MODE" => "int8",
            "STINGRAY_NO_THINKING" => "true",
            "STINGRAY_PRESERVE_THINKING" => "1",
            _ => null,
        };

        var applied = ServerEnvironmentOverrides.Apply(options, Environment);

        Assert.Equal("mmproj.gguf", options.MmprojPath);
        Assert.Equal(64, options.MaxQueuedRequests);
        Assert.Equal(16, options.MaxConcurrentRequests);
        Assert.Equal(512, options.PrefillChunkTokens);
        Assert.Equal(2048L, options.KvBudgetMb);
        Assert.Equal(1024L, options.PrefixCacheMb);
        Assert.Equal(256L, options.PrefillDequantCacheMb);
        Assert.Equal(33, options.NGpuLayers);
        Assert.Equal("q8_0", options.KvType);
        Assert.Equal("int8", options.TqMode);
        Assert.True(options.DisableThinking);
        Assert.True(options.PreserveThinking);

        var receipt = new ServerEnvironmentOverrideReceipt();
        receipt.Record(applied);

        Assert.Equal(
            [
                "STINGRAY_KV_BUDGET_MB",
                "STINGRAY_KV_DTYPE",
                "STINGRAY_MAX_CONCURRENT",
                "STINGRAY_MAX_QUEUE",
                "STINGRAY_MMPROJ",
                "STINGRAY_NO_THINKING",
                "STINGRAY_N_GPU_LAYERS",
                "STINGRAY_PREFILL_CHUNK",
                "STINGRAY_PREFILL_DEQUANT_MB",
                "STINGRAY_PREFIX_CACHE_MB",
                "STINGRAY_PRESERVE_THINKING",
                "STINGRAY_TQ_MODE",
            ],
            receipt.Names);
    }
}

