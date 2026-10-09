using OpenTail.Stingray.Core.Catalog;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// <c>stingray setup &lt;task|id&gt;</c>: shows what a catalog bundle costs (size, licence, RAM,
/// speed), asks, then downloads and SHA-256-checks it into the model home. Front-door step 2
/// (<c>docs/3-product-and-runtime/103-front-door-design.md</c>).
/// </summary>
public sealed class SetupCommand : Command<SetupCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-t|--task <TARGET>", Positional = true)]
        [Description("A task (chat, speak, transcribe) for its default model, or a catalog id from `stingray models`.")]
        public string? Target { get; init; }

        [CommandOption("-y|--yes")]
        [Description("Do not ask before downloading.")]
        public bool Yes { get; init; }

        [CommandOption("--accept-licence")]
        [Description("Accept a non-permissive licence without asking (needed with --yes for such bundles).")]
        public bool AcceptLicence { get; init; }
    }

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        if (string.IsNullOrWhiteSpace(settings.Target))
        {
            AnsiConsole.MarkupLine("Usage: [yellow]stingray setup <task|id>[/], for example [yellow]stingray setup chat[/]. See [yellow]stingray models[/].");
            return 1;
        }

        var entry = ModelCatalog.Find(settings.Target);
        if (entry is null)
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] '{Markup.Escape(settings.Target)}' is neither a task ({string.Join(", ", ModelCatalog.Tasks)}) nor a catalog id.");
            AnsiConsole.MarkupLine("Run [yellow]stingray models[/] to see the catalog, or [yellow]stingray pull -r <repo>[/] for any other GGUF.");
            return 1;
        }

        var home = ModelHome.Default();
        using var http = ModelDownloader.CreateClient("OpenTail.Stingray/setup");
        var outcome = SetupFlow.Run(entry, home, http, settings.Yes, settings.AcceptLicence, interactive: !Console.IsInputRedirected, new ConsolePrompt(), cancellation, feasibility: SetupFlow.DefaultFeasibility());
        if (!outcome.Installed) return 1;
        AnsiConsole.MarkupLine($"Run it: [yellow]{Markup.Escape(entry.RunCommand(home))}[/]");
        return 0;
    }

    internal static void PrintSummary(CatalogEntry entry, ModelHome home, long remaining)
    {
        void Row(string label, string value) => AnsiConsole.MarkupLine($"  [bold]{label,-9}[/] {Markup.Escape(value)}");
        Row("Model", $"{entry.Id} ({entry.Task}): {entry.Why}");
        Row("Download", remaining == 0
            ? $"{ConsoleDownloadProgress.FormatBytes(entry.TotalSize)}, already downloaded"
            : $"{ConsoleDownloadProgress.FormatBytes(remaining)} of {ConsoleDownloadProgress.FormatBytes(entry.TotalSize)} to fetch");
        Row("Files", string.Join(", ", entry.Files.Select(f => $"{f.Repo}/{f.RepoPath}")));
        Row("Licence", entry.Licence);
        Row("Needs", entry.Hardware);
        Row("Speed", entry.Speed);
        Row("Saved to", home.Root);
        AnsiConsole.WriteLine();
    }
}
