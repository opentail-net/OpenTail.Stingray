using System.ComponentModel;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// Compatibility tests for the hand-rolled option binder that replaced Spectre.Console.Cli.
/// The production settings classes use <c>{ get; init; }</c>, so these deliberately do too —
/// binding relies on init-only setters being reachable through reflection (init is enforced by
/// the compiler, not the runtime), and that assumption is load-bearing for all 86 real options.
/// </summary>
public sealed class OptionBinderTests
{
    private sealed class TestSettings : CommandSettings
    {
        [CommandOption("-m|--model")]
        [Description("Path to model")]
        public string? Model { get; init; }

        [CommandOption("-n|--n-predict")]
        [DefaultValue(512)]
        public int Predict { get; init; }

        [CommandOption("--temp")]
        public float Temp { get; init; }

        [CommandOption("-g|--ngl|--n-gpu-layers")]
        public int GpuLayers { get; init; }

        [CommandOption("--budget")]
        public long Budget { get; init; }

        [CommandOption("--verbose")]
        [DefaultValue(false)]
        public bool Verbose { get; init; }

        // Placeholder present, so this takes an explicit value rather than being a switch.
        [CommandOption("--gpu-moe-prefill <BOOL>")]
        public bool? MoePrefill { get; init; }

        [CommandOption("--image <PATH>")]
        public string[]? Images { get; init; }
    }

    private static (TestSettings Settings, string? Error) Bind(params string[] args)
    {
        var settings = new TestSettings();
        bool ok = OptionBinder.TryBind(
            settings, OptionModel.Describe<TestSettings>(), args, out string? error);
        return (settings, ok ? null : error);
    }

    // ── The load-bearing assumption ─────────────────────────────────────────

    [Fact]
    public void InitOnlyProperties_AreSettableViaReflection()
    {
        var (s, error) = Bind("--model", "a.gguf");
        Assert.Null(error);
        Assert.Equal("a.gguf", s.Model);
    }

    // ── Aliases ─────────────────────────────────────────────────────────────

    [Theory]
    [InlineData("-m")]
    [InlineData("--model")]
    public void ShortAndLongAliases_BindTheSameProperty(string alias)
    {
        var (s, error) = Bind(alias, "x.gguf");
        Assert.Null(error);
        Assert.Equal("x.gguf", s.Model);
    }

    [Theory]
    [InlineData("-g")]
    [InlineData("--ngl")]
    [InlineData("--n-gpu-layers")]
    public void EveryAliasInAMultiAliasTemplate_IsAccepted(string alias)
    {
        var (s, error) = Bind(alias, "20");
        Assert.Null(error);
        Assert.Equal(20, s.GpuLayers);
    }

    // ── Values that look like options ───────────────────────────────────────

    [Fact]
    public void NegativeValue_IsConsumedAsAValueNotAnOption()
    {
        // "-g -1" is the documented way to offload all layers; a parser that refuses to
        // consume a '-'-prefixed token as a value would break it.
        var (s, error) = Bind("-g", "-1");
        Assert.Null(error);
        Assert.Equal(-1, s.GpuLayers);
    }

    [Fact]
    public void NegativeFloat_IsConsumedAsAValue()
    {
        var (s, error) = Bind("--temp", "-0.5");
        Assert.Null(error);
        Assert.Equal(-0.5f, s.Temp);
    }

    // ── --name=value ────────────────────────────────────────────────────────

    [Fact]
    public void InlineEqualsForm_BindsTheValue()
    {
        var (s, error) = Bind("--model=b.gguf");
        Assert.Null(error);
        Assert.Equal("b.gguf", s.Model);
    }

    [Fact]
    public void InlineEqualsForm_PreservesEqualsSignsInsideTheValue()
    {
        // A JSON schema or key=value payload must survive intact.
        var (s, error) = Bind("--model=a=b=c");
        Assert.Null(error);
        Assert.Equal("a=b=c", s.Model);
    }

    // ── Flags ───────────────────────────────────────────────────────────────

    [Fact]
    public void Flag_IsTrueByPresenceWithNoValue()
    {
        var (s, error) = Bind("--verbose");
        Assert.Null(error);
        Assert.True(s.Verbose);
    }

    [Fact]
    public void Flag_DoesNotSwallowTheFollowingToken()
    {
        var (s, error) = Bind("--verbose", "--model", "c.gguf");
        Assert.Null(error);
        Assert.True(s.Verbose);
        Assert.Equal("c.gguf", s.Model);
    }

    [Fact]
    public void Flag_CanBeExplicitlyDisabledWithEqualsFalse()
    {
        var (s, error) = Bind("--verbose=false");
        Assert.Null(error);
        Assert.False(s.Verbose);
    }

    [Fact]
    public void OptionWithBoolPlaceholder_RequiresAnExplicitValue()
    {
        // "--gpu-moe-prefill <BOOL>" declares a placeholder, so it is NOT a presence flag.
        var (s, error) = Bind("--gpu-moe-prefill", "false");
        Assert.Null(error);
        Assert.False(s.MoePrefill);
    }

    [Theory]
    [InlineData("true", true)]
    [InlineData("1", true)]
    [InlineData("yes", true)]
    [InlineData("on", true)]
    [InlineData("false", false)]
    [InlineData("0", false)]
    [InlineData("no", false)]
    [InlineData("off", false)]
    public void BoolValues_AcceptTheSpellingsUsersActuallyType(string raw, bool expected)
    {
        var (s, error) = Bind("--gpu-moe-prefill", raw);
        Assert.Null(error);
        Assert.Equal(expected, s.MoePrefill);
    }

    // ── Defaults ────────────────────────────────────────────────────────────

    [Fact]
    public void DefaultValue_IsAppliedWhenTheOptionIsAbsent()
    {
        var (s, error) = Bind();
        Assert.Null(error);
        Assert.Equal(512, s.Predict);
    }

    [Fact]
    public void DefaultValue_IsOverriddenByAnExplicitArgument()
    {
        var (s, error) = Bind("-n", "64");
        Assert.Null(error);
        Assert.Equal(64, s.Predict);
    }

    // ── Repeatable options ──────────────────────────────────────────────────

    [Fact]
    public void RepeatableOption_AccumulatesEveryOccurrenceInOrder()
    {
        var (s, error) = Bind("--image", "a.png", "--image", "b.png", "--image", "c.png");
        Assert.Null(error);
        Assert.NotNull(s.Images);
        Assert.Equal(["a.png", "b.png", "c.png"], s.Images);
    }

    [Fact]
    public void RepeatableOption_IsNullWhenNeverSupplied()
    {
        var (s, error) = Bind();
        Assert.Null(error);
        Assert.Null(s.Images);
    }

    [Fact]
    public void RepeatableOption_WorksWithASingleOccurrence()
    {
        var (s, error) = Bind("--image", "only.png");
        Assert.Null(error);
        Assert.NotNull(s.Images);
        Assert.Equal(["only.png"], s.Images);
    }

    // ── Type conversion ─────────────────────────────────────────────────────

    [Fact]
    public void LongValues_BindBeyondIntRange()
    {
        var (s, error) = Bind("--budget", "5000000000");
        Assert.Null(error);
        Assert.Equal(5_000_000_000L, s.Budget);
    }

    [Fact]
    public void FloatParsing_IsInvariantCulture()
    {
        // A machine with a comma decimal separator must still parse "0.7" from the CLI.
        var (s, error) = Bind("--temp", "0.7");
        Assert.Null(error);
        Assert.Equal(0.7f, s.Temp);
    }

    // ── Errors ──────────────────────────────────────────────────────────────

    [Fact]
    public void UnknownOption_IsRejectedWithTheOffendingName()
    {
        var (_, error) = Bind("--bogus");
        Assert.NotNull(error);
        Assert.Contains("--bogus", error, StringComparison.Ordinal);
    }

    [Fact]
    public void MissingValueAtEndOfArgs_IsRejected()
    {
        var (_, error) = Bind("--model");
        Assert.NotNull(error);
        Assert.Contains("requires a value", error, StringComparison.Ordinal);
    }

    [Fact]
    public void NonNumericValueForNumericOption_IsRejected()
    {
        var (_, error) = Bind("-n", "abc");
        Assert.NotNull(error);
        Assert.Contains("integer", error, StringComparison.Ordinal);
    }

    [Fact]
    public void NonNumericValueForFloatOption_IsRejected()
    {
        var (_, error) = Bind("--temp", "hot");
        Assert.NotNull(error);
        Assert.Contains("number", error, StringComparison.Ordinal);
    }

    [Fact]
    public void UnexpectedPositionalArgument_IsRejected()
    {
        var (_, error) = Bind("stray");
        Assert.NotNull(error);
        Assert.Contains("stray", error, StringComparison.Ordinal);
    }

    [Fact]
    public void EmptyArgs_BindCleanly()
    {
        var (_, error) = Bind();
        Assert.Null(error);
    }

    // ── Option model shape ──────────────────────────────────────────────────

    [Fact]
    public void Describe_FindsEveryAnnotatedPropertyAndIgnoresNothingElse()
    {
        var options = OptionModel.Describe<TestSettings>();
        Assert.Equal(8, options.Count);
    }

    [Fact]
    public void Describe_StripsThePlaceholderFromAliases()
    {
        var options = OptionModel.Describe<TestSettings>();
        var image = options.Single(o => o.Aliases.Contains("--image"));
        Assert.Equal("PATH", image.Placeholder);
        Assert.Equal(["--image"], image.Aliases);
    }

    [Fact]
    public void Describe_MarksPlaceholderlessBoolsAsFlagsAndOthersAsValued()
    {
        var options = OptionModel.Describe<TestSettings>();
        Assert.True(options.Single(o => o.Aliases.Contains("--verbose")).IsFlag);
        Assert.False(options.Single(o => o.Aliases.Contains("--gpu-moe-prefill")).IsFlag);
        Assert.False(options.Single(o => o.Aliases.Contains("--model")).IsFlag);
    }
}
