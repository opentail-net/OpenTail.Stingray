using System.ComponentModel;
using OpenTail.Stingray.Cli.CommandLine;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// Dispatch and help/version behaviour for the host that replaced Spectre's CommandApp.
/// </summary>
public sealed class CommandAppTests
{
    // Commands record what they saw so dispatch can be asserted without running real work.
    private sealed class Recorder
    {
        internal static string? LastCommand;
        internal static string? LastModel;
        internal static int ExitCode;

        internal static void Reset()
        {
            LastCommand = null;
            LastModel = null;
            ExitCode = 0;
        }
    }

    private sealed class RootSettings : CommandSettings
    {
        [CommandOption("-m|--model")]
        [Description("Path to GGUF model file")]
        public string? Model { get; init; }

        [CommandOption("-n|--n-predict")]
        [Description("Number of tokens to predict (default: 512)")]
        [DefaultValue(512)]
        public int Predict { get; init; }

        [CommandOption("--budget")]
        [Description("Memory budget")]
        [DefaultValue(64)]
        public int Budget { get; init; }
    }

    private sealed class RootCommand : Command<RootSettings>
    {
        protected override int Execute(RootSettings settings, CancellationToken cancellation)
        {
            Recorder.LastCommand = "root";
            Recorder.LastModel = settings.Model;
            return Recorder.ExitCode;
        }
    }

    private sealed class SubSettings : CommandSettings
    {
        [CommandOption("-m|--model")]
        [Description("Path to GGUF model file")]
        public string? Model { get; init; }
    }

    private sealed class SubCommand : Command<SubSettings>
    {
        protected override int Execute(SubSettings settings, CancellationToken cancellation)
        {
            Recorder.LastCommand = "sub";
            Recorder.LastModel = settings.Model;
            return Recorder.ExitCode;
        }
    }

    private sealed class FailingSettings : CommandSettings
    {
        [CommandOption("--value")]
        public string? Value { get; init; }

        public override string? Validate() =>
            Value == "bad" ? "value must not be 'bad'" : null;
    }

    private sealed class FailingCommand : Command<FailingSettings>
    {
        protected override int Execute(FailingSettings settings, CancellationToken cancellation)
        {
            Recorder.LastCommand = "failing";
            return 0;
        }
    }

    private static CommandApp<RootCommand> BuildApp()
    {
        Recorder.Reset();
        var app = new CommandApp<RootCommand>();
        app.Configure(config =>
        {
            config.SetApplicationName("test-cli");
            config.SetApplicationVersion("1.2.3");
            config.AddCommand<SubCommand>("sub").WithDescription("A sub command");
            config.AddCommand<FailingCommand>("failing").WithDescription("Validation demo");
        });
        return app;
    }

    /// <summary>Run the app with stdout captured.</summary>
    private static (int Exit, string Output) Run(CommandApp<RootCommand> app, params string[] args)
    {
        var original = Console.Out;
        var buffer = new StringWriter();
        Console.SetOut(buffer);
        try
        {
            int exit = app.Run(args);
            return (exit, buffer.ToString());
        }
        finally
        {
            Console.SetOut(original);
        }
    }

    // ── Dispatch ────────────────────────────────────────────────────────────

    [Fact]
    public void NoSubcommand_RunsTheDefaultCommand()
    {
        var app = BuildApp();
        var (exit, _) = Run(app, "-m", "a.gguf");
        Assert.Equal(0, exit);
        Assert.Equal("root", Recorder.LastCommand);
        Assert.Equal("a.gguf", Recorder.LastModel);
    }

    [Fact]
    public void LeadingSubcommandName_RoutesToThatCommand()
    {
        var app = BuildApp();
        var (exit, _) = Run(app, "sub", "-m", "b.gguf");
        Assert.Equal(0, exit);
        Assert.Equal("sub", Recorder.LastCommand);
        Assert.Equal("b.gguf", Recorder.LastModel);
    }

    [Fact]
    public void SubcommandNameIsNotTreatedAsAnOptionValue()
    {
        // The name is stripped before binding, so the sub-command must not see it.
        var app = BuildApp();
        Run(app, "sub");
        Assert.Equal("sub", Recorder.LastCommand);
        Assert.Null(Recorder.LastModel);
    }

    [Fact]
    public void CommandExitCode_IsPropagated()
    {
        var app = BuildApp();
        Recorder.ExitCode = 3;
        var (exit, _) = Run(app, "-m", "a.gguf");
        Assert.Equal(3, exit);
    }

    // ── Validation ──────────────────────────────────────────────────────────

    [Fact]
    public void FailedValidation_ReportsTheMessageAndDoesNotRunTheCommand()
    {
        var app = BuildApp();
        var (exit, output) = Run(app, "failing", "--value", "bad");
        Assert.Equal(1, exit);
        Assert.Null(Recorder.LastCommand);
        Assert.Contains("must not be 'bad'", output, StringComparison.Ordinal);
    }

    [Fact]
    public void PassingValidation_RunsTheCommand()
    {
        var app = BuildApp();
        var (exit, _) = Run(app, "failing", "--value", "fine");
        Assert.Equal(0, exit);
        Assert.Equal("failing", Recorder.LastCommand);
    }

    [Fact]
    public void BindingError_ReportsAndDoesNotRunTheCommand()
    {
        var app = BuildApp();
        var (exit, output) = Run(app, "--nope");
        Assert.Equal(1, exit);
        Assert.Null(Recorder.LastCommand);
        Assert.Contains("--nope", output, StringComparison.Ordinal);
    }

    // ── Version ─────────────────────────────────────────────────────────────

    [Theory]
    [InlineData("--version")]
    [InlineData("-v")]
    public void VersionFlag_PrintsTheConfiguredVersionAndDoesNotRun(string flag)
    {
        var app = BuildApp();
        var (exit, output) = Run(app, flag);
        Assert.Equal(0, exit);
        Assert.Contains("1.2.3", output, StringComparison.Ordinal);
        Assert.Null(Recorder.LastCommand);
    }

    // ── Help ────────────────────────────────────────────────────────────────

    [Theory]
    [InlineData("--help")]
    [InlineData("-h")]
    public void HelpFlag_PrintsUsageAndDoesNotRun(string flag)
    {
        var app = BuildApp();
        var (exit, output) = Run(app, flag);
        Assert.Equal(0, exit);
        Assert.Contains("USAGE", output, StringComparison.Ordinal);
        Assert.Contains("test-cli", output, StringComparison.Ordinal);
        Assert.Null(Recorder.LastCommand);
    }

    [Fact]
    public void RootHelp_ListsRegisteredSubcommands()
    {
        var app = BuildApp();
        var (_, output) = Run(app, "--help");
        Assert.Contains("sub", output, StringComparison.Ordinal);
        Assert.Contains("A sub command", output, StringComparison.Ordinal);
    }

    [Fact]
    public void RootHelp_ListsTheDefaultCommandsOptions()
    {
        var app = BuildApp();
        var (_, output) = Run(app, "--help");
        Assert.Contains("--model", output, StringComparison.Ordinal);
        Assert.Contains("Path to GGUF model file", output, StringComparison.Ordinal);
    }

    [Fact]
    public void SubcommandHelp_ShowsThatSubcommandNotTheRoot()
    {
        var app = BuildApp();
        var (exit, output) = Run(app, "sub", "--help");
        Assert.Equal(0, exit);
        Assert.Contains("sub", output, StringComparison.Ordinal);
        Assert.Null(Recorder.LastCommand);
    }

    // ── Default rendering in help ───────────────────────────────────────────

    [Fact]
    public void Help_DoesNotRepeatADefaultTheDescriptionAlreadyStates()
    {
        // "--n-predict"'s description already ends with "(default: 512)"; appending the
        // [DefaultValue] again rendered it twice.
        var app = BuildApp();
        var (_, output) = Run(app, "--help");

        int occurrences = output.Split("default: 512").Length - 1;
        Assert.Equal(1, occurrences);
    }

    [Fact]
    public void Help_StillAppendsADefaultWhenTheDescriptionOmitsIt()
    {
        var app = BuildApp();
        var (_, output) = Run(app, "--help");
        Assert.Contains("default: 64", output, StringComparison.Ordinal);
    }
}
