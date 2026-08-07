using OpenTail.Stingray.Server;

namespace OpenTail.Stingray.Tests.Server;

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
}
