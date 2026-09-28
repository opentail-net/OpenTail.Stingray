using System.ComponentModel;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary><see cref="CommandOptionAttribute.Positional"/>: <c>setup chat</c> as well as <c>setup --task chat</c>.</summary>
public sealed class PositionalOptionTests
{
    private sealed class PositionalSettings : CommandSettings
    {
        [CommandOption("-t|--task <TASK>", Positional = true)]
        public string? Task { get; init; }

        [CommandOption("-y|--yes")]
        [DefaultValue(false)]
        public bool Yes { get; init; }
    }

    private static (PositionalSettings Settings, string? Error) Bind(params string[] args)
    {
        var settings = new PositionalSettings();
        bool ok = OptionBinder.TryBind(settings, OptionModel.Describe<PositionalSettings>(), args, out string? error);
        return (settings, ok ? null : error);
    }

    [Fact]
    public void BareArgumentBindsToPositionalOption()
    {
        var (s, error) = Bind("chat", "--yes");
        Assert.Null(error);
        Assert.Equal("chat", s.Task);
        Assert.True(s.Yes);
    }

    [Fact]
    public void NamedFormStillWorks()
    {
        var (s, error) = Bind("--yes", "-t", "speak");
        Assert.Null(error);
        Assert.Equal("speak", s.Task);
    }

    [Fact]
    public void SecondBareArgumentIsRejected()
    {
        var (_, error) = Bind("chat", "speak");
        Assert.Equal("unexpected argument 'speak'.", error);
    }
}
