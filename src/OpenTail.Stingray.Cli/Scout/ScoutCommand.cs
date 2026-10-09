namespace OpenTail.Stingray.Cli.Scout;

/// <summary>
/// <c>stingray scout -m model.gguf</c>: a read-only, evidence-citing dossier on a GGUF checkpoint (plan:
/// docs/3-product-and-runtime/2026-10-09-checkpoint-scout-and-ai-admission-plan.md). Opens the header/metadata/tensor index only;
/// constructs no forward pass, reads no tensor values, downloads nothing. It is decision support, never an admission authority.
/// </summary>
public sealed class ScoutCommand : Command<ScoutCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-m|--model <PATH>")]
        [Description("Path to a GGUF model file")]
        public string? ModelPath { get; init; }

        [CommandOption("--format <FORMAT>")]
        [Description("Output format: text (default) or json")]
        public string Format { get; init; } = "text";

        [CommandOption("-o|--output <PATH>")]
        [Description("Write the JSON report to this file (in addition to the chosen stdout format)")]
        public string? OutputPath { get; init; }

        [CommandOption("--budget <SIZE>")]
        [Description("Host RAM budget for execution feasibility, e.g. 64G. Without it, feasibility is not assessed.")]
        public string? Budget { get; init; }

        [CommandOption("--reserve <SIZE>")]
        [Description("Headroom kept free under the budget (default 8G)")]
        public string? Reserve { get; init; }

        public override string? Validate()
        {
            if (string.IsNullOrWhiteSpace(ModelPath)) return "Use -m <model.gguf>.";
            if (Format is not ("text" or "json")) return "--format must be 'text' or 'json'.";
            if (Budget is not null && !ScoutSize.TryParse(Budget, out _)) return $"--budget '{Budget}' is not a size (try 64G).";
            if (Reserve is not null && !ScoutSize.TryParse(Reserve, out _)) return $"--reserve '{Reserve}' is not a size (try 8G).";
            return null;
        }
    }

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        if (!ModelPathResolver.TryRequireModelFile(settings.ModelPath, out int modelFileExit))
            return modelFileExit;
        string path = settings.ModelPath!;

        long? budget = settings.Budget is null ? null : (ScoutSize.TryParse(settings.Budget, out long b) ? b : null);
        long reserve = settings.Reserve is not null && ScoutSize.TryParse(settings.Reserve, out long r) ? r : ScoutOptions.DefaultReserveBytes;
        var options = new ScoutOptions(StingrayBuildVersion.Value, budget, reserve);

        string fileName = Path.GetFileName(path);
        long? fileBytes = new FileInfo(path).Length;

        ScoutReport report;
        try
        {
            using var model = GgufModel.Open(path);
            report = ScoutAnalyzer.Analyze(new ScoutInput(fileName, fileBytes, model.Header.Version, model.Header.TensorCount,
                model.Header.MetadataKvCount, model.Metadata, model.Tensors), options);
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or NotSupportedException or UnauthorizedAccessException or ArgumentException or OverflowException)
        {
            report = ScoutAnalyzer.Unreadable(fileName, fileBytes, $"{ex.GetType().Name}: {ex.Message}", options);
        }

        string json = JsonSerializer.Serialize(report, ScoutJsonContext.Default.ScoutReport);
        if (settings.OutputPath is { Length: > 0 } outPath)
        {
            File.WriteAllText(outPath, json + "\n");
            Console.Error.WriteLine($"Wrote scout report: {Path.GetFileName(outPath)}");
        }

        if (settings.Format == "json") Console.WriteLine(json);
        else ScoutTextRenderer.Write(report);

        return report.Stages[0].State == StageState.Failed ? ExitCodes.Failure : ExitCodes.Success;
    }
}

internal static class ScoutTextRenderer
{
    public static void Write(ScoutReport r)
    {
        var a = r.Artifact;
        AnsiConsole.MarkupLine($"[bold]{Markup.Escape(a.FileName)}[/]  GGUF v{a.GgufVersion?.ToString() ?? "?"}  |  " +
            $"{a.TensorCount?.ToString() ?? "?"} tensors  |  {a.MetadataKeyCount?.ToString() ?? "?"} metadata keys  |  {Bytes(a.FileBytes)}  |  shards {a.ShardCount?.ToString() ?? "?"}");
        AnsiConsole.MarkupLine($"[dim]scout schema v{r.SchemaVersion}, build {Markup.Escape(r.ScoutBuild)}; static only: no weights read, no forward pass[/]");

        if (r.Architecture is { } arch)
        {
            AnsiConsole.WriteLine();
            string status = arch.Resolved ? $"{arch.DescriptorId} ({arch.Status})" : "[yellow]not registered[/]";
            AnsiConsole.MarkupLine($"[bold]Architecture:[/] declared '{Markup.Escape(arch.Declared ?? "(none)")}' -> {status}");
            if (arch.RefusalReason is not null) AnsiConsole.MarkupLine($"  {Markup.Escape(arch.RefusalReason)}");
        }

        if (r.Tensors is { } t)
        {
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[bold]Tensors:[/] {t.Count}, {Bytes(t.TotalBytes)}" + (t.LayerCount is int lc ? $", {lc} layers (blk)" : ""));
            AnsiConsole.MarkupLine("  " + Markup.Escape(string.Join("  ", t.ByDType.Select(d => $"{d.DType} x{d.Count} ({Bytes(d.Bytes)})"))));
            AnsiConsole.MarkupLine($"  {t.Patterns.Count} distinct tensor patterns (see --format json for the full list)");
            foreach (var irr in t.Irregularities)
                AnsiConsole.MarkupLine($"  [yellow]irregular[/] {Markup.Escape(irr.Pattern)}: {Markup.Escape(irr.Kind)} - {Markup.Escape(irr.Detail)}");
        }

        if (r.Metadata?.Tokenizer is { } tok)
        {
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"[bold]Tokenizer:[/] model={Markup.Escape(tok.Model ?? "(absent)")} merges={tok.HasMerges} scores={tok.HasScores} " +
                $"chat_template={tok.HasChatTemplate} vocab={tok.VocabSize?.ToString() ?? "?"}");
        }

        foreach (var f in r.Findings)
            AnsiConsole.MarkupLine($"[dim]finding[/] {Markup.Escape(f.Id)} ({f.Certainty}): {Markup.Escape(f.Summary)}");

        AnsiConsole.WriteLine();
        if (r.Blockers.Count == 0) AnsiConsole.MarkupLine("[green]No blockers found by static checks.[/] (This is not admission.)");
        foreach (var b in r.Blockers)
        {
            string color = b.Kind == BlockerKind.Confirmed ? "red" : "yellow";
            AnsiConsole.MarkupLine($"[{color}]{b.Kind.ToString().ToLowerInvariant()}[/] {Markup.Escape(b.Id)}: {Markup.Escape(b.Summary)}");
            foreach (var e in b.Evidence.Take(3))
                AnsiConsole.MarkupLine($"    [dim]{Markup.Escape(e.Kind)} {Markup.Escape(e.Name)} {Markup.Escape(e.Value ?? "")}[/]");
        }

        var res = r.Resources;
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine($"[bold]Resources:[/] file {Bytes(res.FileBytes)}; host working set {res.HostWorkingSet.Certainty.ToString().ToLowerInvariant()}; " +
            $"execution {res.ExecutionDecision}" + (res.BudgetBytes is long bb ? $" (budget {Bytes(bb)}, reserve {Bytes(res.ReserveBytes)})" : ""));
        AnsiConsole.MarkupLine($"  [dim]{Markup.Escape(res.Reason)}[/]");

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[bold]Next:[/]");
        foreach (var n in r.NextActions)
            AnsiConsole.MarkupLine($"  {n.Order}. {Markup.Escape(n.Command)}  [dim]{Markup.Escape(n.Why)}[/]");
    }

    private static string Bytes(long? b) => b is null ? "?" : b < (1L << 20) ? $"{b} B" : b < (1L << 30) ? $"{b / 1048576.0:F1} MiB" : $"{b / 1073741824.0:F2} GiB";
}
