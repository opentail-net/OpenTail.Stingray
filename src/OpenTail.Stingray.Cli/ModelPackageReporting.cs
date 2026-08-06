using OpenTail.Stingray.Cli.CommandLine;
using OpenTail.Stingray.Cli.Terminal;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// Renders <see cref="ModelPackageCapabilityReport"/> for the CLI surfaces named in Phase 0 of the
/// SafeTensors plan: <c>inspect</c>, <c>doctor</c>, and the published capability table.
/// </summary>
internal static class ModelPackageReporting
{
    /// <summary>
    /// True when <paramref name="path"/> should be treated as a SafeTensors model package rather than
    /// a GGUF file — i.e. a directory, or a <c>.safetensors</c> file whose package root is its parent.
    /// </summary>
    /// <remarks>
    /// The product contract is that the input is a model DIRECTORY, not merely a weights file. A bare
    /// <c>.safetensors</c> path is accepted as a convenience and resolved to its containing package,
    /// because that is what a user who tab-completed the weights file meant.
    /// </remarks>
    public static bool LooksLikePackage(string path) =>
        Directory.Exists(path)
        || path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)
        || path.EndsWith(".safetensors.index.json", StringComparison.OrdinalIgnoreCase);

    /// <summary>Prints the capability verdict for one package. Returns a process exit code.</summary>
    public static int PrintReport(string packagePath)
    {
        var report = ModelPackageInspector.Inspect(packagePath);

        AnsiConsole.MarkupLine($"[bold]{Markup.Escape(report.PackagePath)}[/]");
        AnsiConsole.MarkupLine($"  profile        : {Markup.Escape(report.ProfileId)}");
        AnsiConsole.MarkupLine($"  architecture   : {Markup.Escape(report.ArchitectureId ?? "unknown")}");
        AnsiConsole.MarkupLine($"  source dtypes  : {Markup.Escape(report.SourceDtypes.Count == 0 ? "-" : string.Join(", ", report.SourceDtypes))}");
        AnsiConsole.MarkupLine($"  tokenizer      : {report.TokenizerFamily}");
        AnsiConsole.MarkupLine($"  backends       : {report.AvailableBackends}");
        if (report.EstimatedWeightBytes is { } bytes)
            AnsiConsole.MarkupLine($"  weight bytes   : {bytes / (1024.0 * 1024.0):F1} MiB (from headers; not loaded)");
        if (report.EstimatedWorkingSetBytes is { } workingBytes && workingBytes != report.EstimatedWeightBytes)
            AnsiConsole.MarkupLine($"  working set    : ~{workingBytes / (1024.0 * 1024.0):F1} MiB (CPU execution working set)");

        if (report.IsSupported)
        {
            AnsiConsole.MarkupLine("  [green]SUPPORTED[/]");
            return 0;
        }

        AnsiConsole.MarkupLine("  [red]NOT SUPPORTED[/]");
        foreach (var rejection in report.Rejections)
            AnsiConsole.MarkupLine($"    [red]-[/] [yellow]{rejection.Kind}[/] " +
                $"[[{Markup.Escape(rejection.Subject)}]]: {Markup.Escape(rejection.Detail)}");

        // GGUF is the recommended deployment route, and saying so here is more useful than a bare
        // refusal — but only when the problem is the format/profile, not a broken package.
        if (report.Rejections.Any(r => r.Kind is ModelPackageRejectionKind.UnsupportedArchitecture
                                            or ModelPackageRejectionKind.UnsupportedDtype
                                            or ModelPackageRejectionKind.UnsupportedConfig))
        {
            AnsiConsole.MarkupLine("  [dim]A GGUF build of this model is the supported deployment route.[/]");
        }
        return 1;
    }

    /// <summary>Prints the published capability rows.</summary>
    public static void PrintCapabilityTable()
    {
        AnsiConsole.MarkupLine("[bold]Model package capability profiles[/]");
        AnsiConsole.WriteLine();
        foreach (string line in ModelPackageInspector.RenderCapabilityTable().Split(Environment.NewLine))
            AnsiConsole.WriteLine(line);
        AnsiConsole.WriteLine();
        foreach (var profile in ModelPackageCapability.All)
        {
            AnsiConsole.MarkupLine($"[bold]{Markup.Escape(profile.ProfileId)}[/] — {Markup.Escape(profile.Description)}");
            foreach (string exclusion in profile.Exclusions)
                AnsiConsole.MarkupLine($"  [dim]excluded:[/] {Markup.Escape(exclusion)}");
        }
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[dim]Anything absent from a row is unsupported. GGUF remains the preferred local deployment format.[/]");
    }
}
