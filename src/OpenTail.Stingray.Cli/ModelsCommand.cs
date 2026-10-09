using OpenTail.Stingray.Cli.CommandLine;
using OpenTail.Stingray.Core.Catalog;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// <c>stingray models [task]</c>: which tasks are ready, what is installed in the model home, and
/// the one command that fixes each gap. With a task, lists that task's catalog options.
/// With <c>--local</c>, shows catalogue state plus local GGUF models on disk (plan P4).
/// With <c>use &lt;task&gt; &lt;id&gt;</c>, configures a persistent favourite for that task (plan P3).
/// (<c>list-models</c> is the separate "which GGUF files are in this folder" listing.)
/// </summary>
public sealed class ModelsCommand : Command<ModelsCommand.Settings>, ICommand
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-t|--task <TASK>", Positional = true)]
        [Description("Show the catalog options for one task (chat, speak, transcribe).")]
        public string? Task { get; init; }

        [CommandOption("-l|--local")]
        [Description("Show catalogue state plus local GGUF models on disk.")]
        public bool Local { get; init; }

        [CommandOption("--verify")]
        [Description("With --local: re-hash installed catalogue files to verify integrity against published SHA-256 (slower).")]
        public bool Verify { get; init; }

        [CommandOption("--clear")]
        [Description("With 'models use <task>': clear the configured favourite for this task.")]
        public bool Clear { get; init; }

        public override string? Validate()
        {
            if (Task is not null && (Local || Verify))
                return "--task and --local/--verify cannot be combined.";
            if (Clear)
                return "--clear only applies with 'models use <task>'.";
            return null;
        }
    }

    public int Run(string[] args, CancellationToken cancellation = default) => ((ICommand)this).Run(args, cancellation);

    int ICommand.Run(string[] args, CancellationToken cancellation)
    {
        if (args.Length > 0 && string.Equals(args[0], "use", StringComparison.OrdinalIgnoreCase))
        {
            return ExecuteUse(args[1..], cancellation);
        }

        var settings = new Settings();
        if (!OptionBinder.TryBind(settings, OptionModel.Describe<Settings>(), args, out string? error))
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] {Markup.Escape(error!)}");
            return 1;
        }

        if (settings.Validate() is { } validationError)
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] {Markup.Escape(validationError)}");
            return 1;
        }

        return Execute(settings, cancellation);
    }

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        if (settings.Local || settings.Verify)
        {
            return ExecuteLocalInventory(settings, cancellation);
        }

        var home = ModelHome.Default();
        Favourites.TryGetAll(out var favourites, out _);

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
                bool isFav = favourites.TryGetValue(task, out var fid) && string.Equals(e.Id, fid, StringComparison.OrdinalIgnoreCase);
                string badge = isFav ? " (favourite)" : (i == 0 ? " (default)" : "");
                AnsiConsole.MarkupLine($"[bold]{Markup.Escape(e.Id)}[/]{badge}  {Describe(home.StateOf(e))}");
                SetupCommand.PrintSummary(e, home, home.RemainingBytes(e));
            }
            return 0;
        }

        AnsiConsole.MarkupLine($"Model home: [yellow]{Markup.Escape(home.Root)}[/] (set {ModelHome.EnvironmentVariable} to move it)");
        AnsiConsole.WriteLine();

        var table = new Table().AddColumn("Task").AddColumn("Model").AddColumn("Size").AddColumn("State").AddColumn("Next step");
        foreach (string t in ModelCatalog.Tasks)
        {
            var entries = ModelCatalog.ForTask(t).ToList();
            string? favId = favourites.TryGetValue(t, out var f) ? f : null;
            var favEntry = favId is not null ? ModelCatalog.Find(favId) : null;

            // A task is ready if its favourite or any option is installed; show that one, else favourite or default.
            var shown = (favEntry != null && home.StateOf(favEntry) == InstallState.Installed)
                ? favEntry
                : (entries.FirstOrDefault(e => home.StateOf(e) == InstallState.Installed) ?? favEntry ?? entries[0]);
            var state = home.StateOf(shown);
            string next = state == InstallState.Installed
                ? shown.RunCommand(home)
                : $"stingray setup {t}";

            bool isFav = favId is not null && string.Equals(shown.Id, favId, StringComparison.OrdinalIgnoreCase);
            string modelDisplay = isFav ? $"{shown.Id} (favourite)" : shown.Id;

            table.AddRow(
                Markup.Escape(t),
                Markup.Escape(modelDisplay),
                ConsoleDownloadProgress.FormatBytes(shown.TotalSize),
                Describe(state),
                $"[yellow]{Markup.Escape(next)}[/]");
        }
        AnsiConsole.Write(table);
        AnsiConsole.MarkupLine("[dim]`stingray models <task>` lists a task's options. `stingray models use <task> <id>` sets a favourite.[/]");
        return 0;
    }

    private static int ExecuteUse(string[] args, CancellationToken cancellation)
    {
        if (args.Length == 0)
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] 'models use' requires a task (e.g. 'models use chat <id>' or 'models use chat --clear'). Tasks: {string.Join(", ", ModelCatalog.Tasks)}.");
            return 1;
        }

        bool clear = args.Any(a => a is "--clear" or "-c");
        var nonFlags = args.Where(a => !a.StartsWith('-')).ToList();

        if (nonFlags.Count == 0)
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] 'models use' requires a task (e.g. 'models use chat <id>' or 'models use chat --clear'). Tasks: {string.Join(", ", ModelCatalog.Tasks)}.");
            return 1;
        }

        string task = nonFlags[0];
        if (!ModelCatalog.Tasks.Contains(task, StringComparer.OrdinalIgnoreCase))
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] unknown task '{Markup.Escape(task)}'. Tasks: {string.Join(", ", ModelCatalog.Tasks)}.");
            return 1;
        }

        if (clear)
        {
            try
            {
                Favourites.Clear(task);
                AnsiConsole.MarkupLine($"Cleared favourite for [bold]{Markup.Escape(task)}[/].");
                return 0;
            }
            catch (Exception ex)
            {
                AnsiConsole.ErrorLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
                return 1;
            }
        }

        if (nonFlags.Count < 2)
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] specify a model id for task '{Markup.Escape(task)}' (e.g. 'models use {Markup.Escape(task)} <id>'), or --clear to remove it. Options: {string.Join(", ", ModelCatalog.ForTask(task).Select(e => e.Id))}.");
            return 1;
        }

        string id = nonFlags[1];
        var entry = ModelCatalog.Find(id);
        if (entry is null)
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] unknown model '{Markup.Escape(id)}'. Available catalog models for {Markup.Escape(task)}: {string.Join(", ", ModelCatalog.ForTask(task).Select(e => e.Id))}.");
            return 1;
        }

        if (!entry.Task.Equals(task, StringComparison.OrdinalIgnoreCase))
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] model '{Markup.Escape(id)}' is for task '{Markup.Escape(entry.Task)}', not '{Markup.Escape(task)}'.");
            return 1;
        }

        try
        {
            Favourites.Set(task, entry.Id);
            AnsiConsole.MarkupLine($"Configured [bold]{Markup.Escape(entry.Id)}[/] as favourite for [bold]{Markup.Escape(task)}[/].");
            return 0;
        }
        catch (Exception ex)
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] {Markup.Escape(ex.Message)}");
            return 1;
        }
    }

    private static int ExecuteLocalInventory(Settings settings, CancellationToken cancellation)
    {
        var home = ModelHome.Default();
        var report = LocalInventory.Scan(home: home, verify: settings.Verify, ct: cancellation);
        Favourites.TryGetAll(out var favourites, out _);

        AnsiConsole.MarkupLine($"Model home: [yellow]{Markup.Escape(report.Home.Root)}[/] (set {ModelHome.EnvironmentVariable} to move it)");
        if (Environment.GetEnvironmentVariable(LocalInventory.ModelDirsEnvVar) is { Length: > 0 } envDirs)
        {
            AnsiConsole.MarkupLine($"STINGRAY_MODEL_DIRS: [yellow]{Markup.Escape(envDirs)}[/]");
        }
        AnsiConsole.WriteLine();

        AnsiConsole.MarkupLine("[bold]Catalogue Models[/]");
        var catTable = new Table()
            .AddColumn("Task")
            .AddColumn("Model")
            .AddColumn("Size")
            .AddColumn("State");
        if (settings.Verify)
            catTable.AddColumn("Integrity");
        catTable.AddColumn("Next step");

        foreach (var item in report.CatalogItems)
        {
            string stateStr = Describe(item.State);
            string next = item.State == InstallState.Installed
                ? item.Entry.RunCommand(home)
                : $"stingray setup {item.Entry.Task}";

            bool isFav = favourites.TryGetValue(item.Entry.Task, out var favId) && string.Equals(item.Entry.Id, favId, StringComparison.OrdinalIgnoreCase);
            string modelDisplay = isFav ? $"{item.Entry.Id} (favourite)" : item.Entry.Id;

            var row = new List<string>
            {
                Markup.Escape(item.Entry.Task),
                Markup.Escape(modelDisplay),
                ConsoleDownloadProgress.FormatBytes(item.Entry.TotalSize),
                stateStr,
            };
            if (settings.Verify)
            {
                string integrityStr = item.Integrity switch
                {
                    IntegrityState.Intact => "[green]intact[/]",
                    IntegrityState.Damaged => "[red]damaged[/]",
                    _ => "[dim]-[/]",
                };
                row.Add(integrityStr);
            }
            row.Add($"[yellow]{Markup.Escape(next)}[/]");
            catTable.AddRow(row.ToArray());
        }
        AnsiConsole.Write(catTable);
        AnsiConsole.WriteLine();

        AnsiConsole.MarkupLine("[bold]Local GGUF Models[/]");
        if (report.LocalGgufs.Count == 0)
        {
            AnsiConsole.MarkupLine("[dim]No .gguf files found in scanned directories.[/]");
        }
        else
        {
            var ggufTable = new Table()
                .AddColumn("File")
                .AddColumn("Size")
                .AddColumn("Architecture")
                .AddColumn("Support")
                .AddColumn("Quant")
                .AddColumn("Fit");

            foreach (var g in report.LocalGgufs)
            {
                string supportStr = g.AdmissionStatus == "admitted" ? "[green]admitted[/]" : "[dim]not supported[/]";
                string fitStr = g.FitStatus switch
                {
                    "fits" => "[green]fits[/]",
                    "does not fit" => "[red]does not fit[/]",
                    "unknown" => "[yellow]unknown[/]",
                    _ => "[dim]n/a[/]",
                };
                string archStr = g.IsReadable ? Markup.Escape(g.Architecture) : "[red]unreadable[/]";

                ggufTable.AddRow(
                    Markup.Escape(g.FileName),
                    ConsoleDownloadProgress.FormatBytes(g.SizeBytes),
                    archStr,
                    supportStr,
                    Markup.Escape(g.Quantization),
                    fitStr);
            }
            AnsiConsole.Write(ggufTable);
        }

        return 0;
    }

    private static string Describe(InstallState state) => state switch
    {
        InstallState.Installed => "[green]installed[/]",
        InstallState.Partial => "[yellow]partial[/]",
        _ => "[dim]not installed[/]",
    };
}
