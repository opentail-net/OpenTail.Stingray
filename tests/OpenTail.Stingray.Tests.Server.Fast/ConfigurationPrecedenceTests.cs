using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Options;
using Xunit;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Pins the configuration precedence that the server host depends on.
///
/// <para><b>Why this exists.</b> <c>AddOpenTailStingray(configuration, configure)</c> binds the
/// <c>OpenTail.Stingray</c> section first and then runs the inline <c>configure</c> delegate, so the
/// delegate wins. The host uses exactly that ordering to layer ~17 <c>STINGRAY_*</c> overrides
/// on top of <c>appsettings.json</c> — which means <b>environment beats host configuration</b> in
/// the shipped product.</para>
///
/// <para>That order was enforced only by the sequence of statements in two files. A routine
/// refactor could reverse it and no test would fail, while every deployment that sets those
/// variables alongside an <c>appsettings.json</c> would silently start losing to the file.</para>
///
/// <para><b>This test does not endorse the order</b>, it records it. Note that
/// <c>quality-of-life-improvements-plan.md</c> §7.3 currently specifies the OPPOSITE chain
/// (<c>host configuration > environment</c>); that conflict is logged in the plan as a decision to
/// be made deliberately. If the order is changed on purpose, change this test in the same commit —
/// that is the point of pinning it.</para>
/// </summary>
public sealed class ConfigurationPrecedenceTests
{
    private static IConfiguration ConfigWith(params (string Key, string Value)[] entries) =>
        new ConfigurationBuilder()
            .AddInMemoryCollection(entries.Select(e =>
                new KeyValuePair<string, string?>(e.Key, e.Value)))
            .Build();

    private static OpenTailStingrayServerOptions Resolve(
        IConfiguration configuration, Action<OpenTailStingrayServerOptions>? configure = null)
    {
        var services = new ServiceCollection();
        services.AddOpenTailStingray(configuration, configure);
        return services.BuildServiceProvider()
            .GetRequiredService<IOptions<OpenTailStingrayServerOptions>>().Value;
    }

    [Fact]
    public void ConfigurationSectionBindsWhenNothingOverridesIt()
    {
        var options = Resolve(ConfigWith(("OpenTail.Stingray:ModelPath", "from-config.gguf")));
        Assert.Equal("from-config.gguf", options.ModelPath);
    }

    /// <summary>
    /// The load-bearing case: the inline delegate runs after the bind, so it wins. This is the
    /// mechanism by which the host's environment-variable overrides beat <c>appsettings.json</c>.
    /// </summary>
    [Fact]
    public void InlineConfigureOverridesTheConfigurationSection()
    {
        var options = Resolve(
            ConfigWith(("OpenTail.Stingray:ModelPath", "from-config.gguf")),
            opts => opts.ModelPath = "from-inline.gguf");

        Assert.Equal("from-inline.gguf", options.ModelPath);
    }

    /// <summary>
    /// The host only assigns an override when the variable is actually set, so an absent
    /// environment variable must leave the configured value intact rather than blanking it. A
    /// delegate that assigned unconditionally would erase config for every unset variable.
    /// </summary>
    [Fact]
    public void InlineConfigureLeavesUntouchedKeysAtTheirConfiguredValue()
    {
        var options = Resolve(
            ConfigWith(("OpenTail.Stingray:ModelPath", "from-config.gguf"),
                       ("OpenTail.Stingray:MaxBatchSize", "7")),
            opts => opts.ModelPath = "from-inline.gguf");

        Assert.Equal("from-inline.gguf", options.ModelPath);
        Assert.Equal(7, options.MaxBatchSize);
    }

    /// <summary>Several overrides in one delegate must all land, not just the first.</summary>
    [Fact]
    public void MultipleInlineOverridesAllApply()
    {
        var options = Resolve(
            ConfigWith(("OpenTail.Stingray:ModelPath", "a.gguf"), ("OpenTail.Stingray:MaxBatchSize", "1")),
            opts => { opts.ModelPath = "b.gguf"; opts.MaxBatchSize = 9; });

        Assert.Equal("b.gguf", options.ModelPath);
        Assert.Equal(9, options.MaxBatchSize);
    }
}
