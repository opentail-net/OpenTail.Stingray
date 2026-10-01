using OpenTail.Stingray.Core.Catalog;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// <c>stingray models [task]</c>: which tasks are ready, what is installed in the model home, and
/// the one command that fixes each gap. With a task, lists that task's catalog options.
/// (<c>list-models</c> is the separate "which GGUF files are in this folder" listing.)
/// </summary>
public sealed class ModelsCommand : Command<ModelsCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-t|--task <TASK>", Positional = true)]
        [Description("Show the catalog options for one task (chat, speak, transcribe).")]
        public string? Task { get; init; }
    }

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        var home = ModelHome.Default();

        if (settings.Task is { } task)
        {
            var options = ModelCatalog.ForTask(task).ToList();
            if (options.Count == 0)
            {
                AnsiConsole.ErrorLine($"[red]Error:[/] unknown task '{Markup.Escape(task)}'. Tasks: {string.Join(", ", ModelCatalog.Tasks)}.");
                return 1;
            }
            for (int i = 0; i < options.Count; i++)
            {
                var e = options[i];
                AnsiConsole.MarkupLine($"[bold]{Markup.Escape(e.Id)}[/]{(i == 0 ? " (default)" : "")}  {Describe(home.StateOf(e))}");
                SetupCommand.PrintSummary(e, home, home.RemainingBytes(e));
            }
            return 0;
        }

        AnsiConsole.MarkupLine($"Model home: [yellow]{Markup.Escape(home.Root)}[/] (set {ModelHome.EnvironmentVariable} to move it)");
        AnsiConsole.WriteLine();

        var table = new Table().AddColumn("Task").AddColumn("Model").AddColumn("Size").AddColumn("State").AddColumn("Next step");
        foreach (string t in ModelCatalog.Tasks)
        {
            // A task is ready if any of its options is installed; show that one, else the default.
            var entries = ModelCatalog.ForTask(t).ToList();
            var shown = entries.FirstOrDefault(e => home.StateOf(e) == InstallState.Installed) ?? entries[0];
            var state = home.StateOf(shown);
            string next = state == InstallState.Installed
                ? shown.RunCommand(home)
                : $"stingray setup {t}";
            table.AddRow(
                Markup.Escape(t),
                Markup.Escape(shown.Id),
                ConsoleDownloadProgress.FormatBytes(shown.TotalSize),
                Describe(state),
                $"[yellow]{Markup.Escape(next)}[/]");
        }
        AnsiConsole.Write(table);
        AnsiConsole.MarkupLine("[dim]`stingray models <task>` lists a task's options. Any other GGUF: `stingray pull -r <repo>`.[/]");
        return 0;
    }

    private static string Describe(InstallState state) => state switch
    {
        InstallState.Installed => "[green]installed[/]",
        InstallState.Partial => "[yellow]partial[/]",
        _ => "[dim]not installed[/]",
    };
}
