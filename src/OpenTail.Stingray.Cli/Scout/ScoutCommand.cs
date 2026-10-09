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

        [CommandOption("-c|--ctx-size <N>")]
        [Description("Context length the memory estimate assumes (default 4096, capped at the model's own limit)")]
        public int? ContextSize { get; init; }

        [CommandOption("--signatures <DIR>")]
        [Description("Extra reference signatures (*.signature.json) to rank against, in addition to the built-in admitted set")]
        public string? SignaturesDir { get; init; }

        [CommandOption("--no-builtin-signatures")]
        [Description("Rank only against --signatures <dir>, ignoring the signatures shipped in the binary (for leave-one-out checks and contributor sets)")]
        public bool NoBuiltinSignatures { get; init; }

        [CommandOption("--emit-signature <PATH>")]
        [Description("Write this file's structural signature as JSON. Admitted architectures only; hashes the file (cached beside it); adds an origin if the target already holds the same structure")]
        public string? EmitSignaturePath { get; init; }

        [CommandOption("--origin-repo <REPO>")]
        [Description("With --emit-signature: Hugging Face repo id the file came from, recorded as provenance (never guessed)")]
        public string? OriginRepo { get; init; }

        [CommandOption("--origin-revision <REV>")]
        [Description("With --emit-signature: repo revision/commit the file came from")]
        public string? OriginRevision { get; init; }

        public override string? Validate()
        {
            if (string.IsNullOrWhiteSpace(ModelPath)) return "Use -m <model.gguf>.";
            if (Format is not ("text" or "json")) return "--format must be 'text' or 'json'.";
            if (Budget is not null && !ScoutSize.TryParse(Budget, out _)) return $"--budget '{Budget}' is not a size (try 64G).";
            if (Reserve is not null && !ScoutSize.TryParse(Reserve, out _)) return $"--reserve '{Reserve}' is not a size (try 8G).";
            if (ContextSize is <= 0) return "--ctx-size must be positive.";
            if (SignaturesDir is not null && !Directory.Exists(SignaturesDir)) return $"--signatures directory '{SignaturesDir}' does not exist.";
            return null;
        }
    }

    /// <summary>
    /// Writes (or extends) a reference signature. This is a deliberate contributor action, so unlike a normal scout run it hashes the file:
    /// evidence without a hash cannot be tied to one conversion. The hash is cached beside the model exactly as <c>stingray hash</c> does.
    /// If the target file already holds the same structure, the new file is added to its origins instead of overwriting it.
    /// </summary>
    private static int EmitSignature(Settings settings, string path, ScoutInput? input, ScoutReport report, string sigPath)
    {
        if (input is null)
        {
            Console.Error.WriteLine("error: no signature written: the file could not be read");
            return ExitCodes.Failure;
        }
        // Check admission before hashing: a refused file must not cost minutes of hashing.
        var origin = new SigOrigin(input.FileName, input.FileBytes, null, Blank(settings.OriginRepo), Blank(settings.OriginRevision), StingrayBuildVersion.Value);
        var sig = SignatureBuilder.TryBuild(input, report, origin, out string refusal);
        if (sig is null)
        {
            Console.Error.WriteLine("error: no signature written: " + refusal);
            return ExitCodes.Failure;
        }
        var fp = OpenTail.Stingray.Engine.Verification.ModelFingerprinter.Compute(path);
        sig = sig with { Origins = [origin with { Sha256 = fp.Sha256 }] };
        if (File.Exists(sigPath))
        {
            ArchSignature? existing;
            try { existing = JsonSerializer.Deserialize(File.ReadAllText(sigPath), ScoutJsonContext.Default.ArchSignature); }
            catch (JsonException ex) { Console.Error.WriteLine($"error: {Path.GetFileName(sigPath)} exists but is not a signature: {ex.Message}"); return ExitCodes.Failure; }
            var merged = existing is null ? null : SignatureBuilder.TryMerge(existing, sig, out refusal);
            if (merged is null)
            {
                Console.Error.WriteLine($"error: not overwriting {Path.GetFileName(sigPath)}: {(existing is null ? "it is empty" : refusal)}");
                return ExitCodes.Failure;
            }
            sig = merged;
        }
        File.WriteAllText(sigPath, JsonSerializer.Serialize(sig, ScoutJsonContext.Default.ArchSignature) + "\n");
        Console.Error.WriteLine($"Wrote signature for {sig.ArchitectureId} (structure {sig.StructureId[..12]}, {sig.Origins.Count} origin file(s)): {Path.GetFileName(sigPath)}");
        return ExitCodes.Success;
    }

    private static string? Blank(string? s) => string.IsNullOrWhiteSpace(s) ? null : s.Trim();

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        if (!ModelPathResolver.TryRequireModelFile(settings.ModelPath, out int modelFileExit))
            return modelFileExit;
        string path = settings.ModelPath!;

        long? budget = settings.Budget is null ? null : (ScoutSize.TryParse(settings.Budget, out long b) ? b : null);
        long reserve = settings.Reserve is not null && ScoutSize.TryParse(settings.Reserve, out long r) ? r : ScoutOptions.DefaultReserveBytes;
        var signatures = settings.NoBuiltinSignatures ? new List<ArchSignature>() : new List<ArchSignature>(SignatureStore.LoadEmbedded());
        if (settings.SignaturesDir is { Length: > 0 } dir)
        {
            var problems = new List<string>();
            signatures.AddRange(SignatureStore.LoadDirectory(dir, problems));
            foreach (string p in problems) Console.Error.WriteLine("warning: signature skipped: " + p);
        }
        var options = new ScoutOptions(StingrayBuildVersion.Value, budget, reserve, signatures, settings.ContextSize ?? ScoutOptions.DefaultContextTokens);

        string fileName = Path.GetFileName(path);
        long? fileBytes = new FileInfo(path).Length;

        ScoutReport report;
        ScoutInput? input = null;
        try
        {
            using var model = GgufModel.Open(path);
            input = new ScoutInput(fileName, fileBytes, model.Header.Version, model.Header.TensorCount,
                model.Header.MetadataKvCount, model.Metadata, model.Tensors);
            report = ScoutAnalyzer.Analyze(input, options);
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or NotSupportedException or UnauthorizedAccessException or ArgumentException or OverflowException)
        {
            report = ScoutAnalyzer.Unreadable(fileName, fileBytes, $"{ex.GetType().Name}: {ex.Message}", options);
        }

        if (settings.EmitSignaturePath is { Length: > 0 } sigPath)
        {
            int emit = EmitSignature(settings, path, input, report, sigPath);
            if (emit != ExitCodes.Success) return emit;
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
            if (arch.CandidatesState == "computed" && arch.Candidates.Count > 0)
            {
                AnsiConsole.MarkupLine("[bold]Nearest admitted structural parents[/] [dim](structure only; a lead for a golden run, not admission)[/]");
                foreach (var cand in arch.Candidates)
                {
                    AnsiConsole.MarkupLine($"  {Markup.Escape(cand.Id)} [dim]({Markup.Escape(cand.Status)}, ref {Markup.Escape(string.Join(", ", cand.ReferenceFiles.Take(2)))}{(cand.ReferenceFiles.Count > 2 ? $" +{cand.ReferenceFiles.Count - 2}" : "")})[/]: " +
                        (cand.DifferenceCount == 0 ? "[green]identical structure[/]" : $"[yellow]{cand.DifferenceCount} difference(s)[/]"));
                    foreach (var d in cand.Differing.Take(3)) AnsiConsole.MarkupLine($"      [dim]{Markup.Escape(d)}[/]");
                }
            }
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
        foreach (var p in res.WorkingSetComponents)
            AnsiConsole.MarkupLine($"    [dim]{Markup.Escape(p.Name),-16} {(p.Bytes is long pb ? Bytes(pb) : "unknown"),10}  {p.Certainty.ToString().ToLowerInvariant()}[/]");
        if (res.HostWorkingSet.Bytes is long total)
            AnsiConsole.MarkupLine($"    [dim]{"estimated peak",-16} {Bytes(total),10}  upper bound, CPU run, {res.ContextTokens} tokens[/]");

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine("[bold]Next:[/]");
        foreach (var n in r.NextActions)
            AnsiConsole.MarkupLine($"  {n.Order}. {Markup.Escape(n.Command)}  [dim]{Markup.Escape(n.Why)}[/]");
    }

    private static string Bytes(long? b) => b is null ? "?" : b < (1L << 20) ? $"{b} B" : b < (1L << 30) ? $"{b / 1048576.0:F1} MiB" : $"{b / 1073741824.0:F2} GiB";
}
