using OpenTail.Stingray.Cli.Terminal;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// Errors belong on stderr so a script can separate them from results on stdout, and exit codes are named so
/// "fix the invocation" (64) is distinguishable from "the run failed" (1).
/// </summary>
public sealed class ErrorStreamAndExitCodeTests
{
    private static (string Out, string Err) Capture(Action action)
    {
        var originalOut = Console.Out;
        var originalErr = Console.Error;
        var stdout = new StringWriter();
        var stderr = new StringWriter();
        Console.SetOut(stdout);
        Console.SetError(stderr);
        try { action(); return (stdout.ToString(), stderr.ToString()); }
        finally { Console.SetOut(originalOut); Console.SetError(originalErr); }
    }

    [Fact]
    public void ErrorLine_WritesToStderrOnly()
    {
        var (o, e) = Capture(() => AnsiConsole.ErrorLine("[red]Error:[/] boom"));
        Assert.Contains("boom", e, StringComparison.Ordinal);
        Assert.Equal("", o);
    }

    [Fact]
    public void MarkupLine_StillWritesToStdoutOnly()
    {
        var (o, e) = Capture(() => AnsiConsole.MarkupLine("[green]fine[/]"));
        Assert.Contains("fine", o, StringComparison.Ordinal);
        Assert.Equal("", e);
    }

    [Fact]
    public void NoCommandReportsItsErrorsOnStdout()
    {
        // Every "[red]Error" report must go through ErrorLine; a stray MarkupLine would put an error on stdout.
        string root = AppContext.BaseDirectory;
        while (root is not null && !File.Exists(Path.Combine(root, "OpenTail.Stingray.slnx"))) root = Path.GetDirectoryName(root)!;
        Assert.NotNull(root);
        var offenders = Directory.EnumerateFiles(Path.Combine(root, "src", "OpenTail.Stingray.Cli"), "*.cs", SearchOption.AllDirectories)
            .Where(p => !p.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}"))
            .SelectMany(p => File.ReadAllText(p).Split('\n').Select((line, i) => (Path: p, Line: i + 1, Text: line)))
            .Where(x => x.Text.Contains("AnsiConsole.MarkupLine(", StringComparison.Ordinal) && x.Text.Contains("[red]", StringComparison.Ordinal)
                         && !IsRedReportLine(x.Text))
            .Select(x => $"{Path.GetFileName(x.Path)}:{x.Line}")
            .ToList();
        Assert.True(offenders.Count == 0, "errors written to stdout: " + string.Join(", ", offenders));
    }

    // Red text that is part of a REPORT (a verdict or a status table), not an error: these stay on stdout.
    private static bool IsRedReportLine(string line) =>
        line.Contains("[red]REJECT:", StringComparison.Ordinal)
        || line.Contains("[red]NOT SUPPORTED", StringComparison.Ordinal)
        || line.Contains("[red]-[/]", StringComparison.Ordinal)
        || line.Contains("non-finite NLL", StringComparison.Ordinal);

    [Fact]
    public void GgufTransplant_MissingArguments_ReturnsTheUsageExitCode()
    {
        var command = (ICommand)new GgufTransplantCommand();
        int exit = 0;
        Capture(() => exit = command.Run([], CancellationToken.None));
        Assert.Equal(64, exit);
    }
}
