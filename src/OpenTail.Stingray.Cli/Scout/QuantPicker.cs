using OpenTail.Stingray.Engine.Scout;
using System.Text.RegularExpressions;
using OpenTail.Stingray.Core.Net;

namespace OpenTail.Stingray.Cli.Scout;

public enum QuantVerdict
{
    /// <summary>The upper-bound CPU estimate plus reserve fits the budget.</summary>
    Fits,
    /// <summary>A known estimate that exceeds the budget.</summary>
    TooBig,
    /// <summary>The memory estimate could not be established, so nothing is promised.</summary>
    Unknown,
    /// <summary>A confirmed blocker unrelated to memory: an architecture or storage type this engine cannot run.</summary>
    Unsupported,
    /// <summary>The index could not be read (over the byte cap, access refused, malformed).</summary>
    NotInspected,
}

public sealed record QuantRow(
    string File,
    string Quant,
    int Shards,
    long? SizeBytes,
    string? DominantDType,
    long? EstimatedPeakBytes,
    QuantVerdict Verdict,
    string Why,
    // Hub-published SHA-256 (single files only); the Hub's claim, not verified here.
    string? Sha256,
    // A command that fetches exactly this file at exactly the inspected commit.
    string PullCommand);

public sealed record QuantReport(
    int SchemaVersion,
    string ScoutBuild,
    string Repo,
    string Revision,
    string? License,
    string? Gated,
    long? Downloads30Days,
    string? HubArchitecture,
    string? Architecture,
    string? ArchitectureStatus,
    long BudgetBytes,
    // "given" (--budget) or "detected_ram" (this machine's RAM).
    string BudgetSource,
    long ReserveBytes,
    int ContextTokens,
    // File name of the largest quant that fits; null when none does. A heuristic (larger is usually closer to the original), not a measured quality ranking.
    string? Recommended,
    IReadOnlyList<QuantRow> Rows,
    IReadOnlyList<string> Notes,
    NetworkUse Network)
{
    public const int CurrentSchemaVersion = 1;
}

/// <summary>
/// Which quantisations of a hosted model can this machine run? Reads each file's index remotely (the weights are never downloaded), runs the same analysis and
/// memory gate as <c>scout</c>, and prints the verdict per file with a pinned <c>pull</c> command. The memory figure is the CPU-run upper bound from
/// <see cref="HostMemoryEstimator"/>; nothing here says anything about GPU placement.
/// </summary>
public static partial class QuantPicker
{
    public const int DefaultMaxFiles = 40;

    [GeneratedRegex(@"(?<![A-Za-z0-9])(IQ\d+_[A-Za-z0-9]+|Q\d+(?:_K)?(?:_[A-Za-z0-9]+)*|BF16|F16|F32|MXFP4|NVFP4|TQ\d_\d)(?![A-Za-z0-9])", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex QuantLabel();

    public static async Task<(QuantReport? Report, string? Message)> RunAsync(
        ExternalHttpClient http, string repoId, string? revision, long maxIndexBytes, int maxFiles,
        ScoutOptions options, string budgetSource, Action<string>? progress, CancellationToken ct)
    {
        var (repo, failure) = await RemoteScout.TryGetRepoAsync(http, repoId, revision, ct).ConfigureAwait(false);
        if (repo is null) return (null, failure);

        var notes = new List<string>();
        var all = HubClient.GroupModels(repo.Files).Where(m => !HubClient.IsProjector(m)).ToList();
        foreach (var broken in all.Where(m => !HubClient.IsComplete(m)).ToList())
        {
            notes.Add($"Skipped '{broken.Name}': a split model with shards missing on the Hub.");
            all.Remove(broken);
        }
        if (all.Count == 0) return (null, $"'{repo.Id}' has no complete GGUF model files.");

        // Largest first: those are the ones that may not fit, and the order the table is shown in.
        var ordered = all.OrderByDescending(m => m.TotalSize ?? -1).ThenBy(m => m.Name, StringComparer.Ordinal).ToList();
        if (ordered.Count > maxFiles)
        {
            notes.Add($"The repo has {ordered.Count} models; only the first {maxFiles} (largest first) were inspected. Raise --max-quants to see the rest.");
            ordered = ordered.Take(maxFiles).ToList();
        }

        var rows = new List<QuantRow>();
        ArchitectureResolution? arch = null;
        int i = 0;
        foreach (var model in ordered)
        {
            progress?.Invoke($"[{++i}/{ordered.Count}] {System.IO.Path.GetFileName(model.Shards[0].Path)}");
            var r = await RemoteScout.InspectAsync(http, repo, model, maxIndexBytes, options, ct).ConfigureAwait(false);
            var report = r.Report!;
            arch ??= report.Architecture;
            rows.Add(ToRow(repo, model, report));
        }

        if (rows.Count > 0 && rows.All(x => x.Verdict == QuantVerdict.Unsupported))
            notes.Add("No quantisation of this model can run here: " + rows[0].Why);

        string? recommended = rows.Where(x => x.Verdict == QuantVerdict.Fits).Select(x => x.File).FirstOrDefault();
        var report0 = new QuantReport(QuantReport.CurrentSchemaVersion, options.BuildId, repo.Id, repo.Revision, repo.License, repo.Gated, repo.Downloads30Days,
            repo.Gguf?.Architecture, arch?.Declared, arch?.Status, options.BudgetBytes ?? 0, budgetSource, options.ReserveBytes, options.ContextTokens,
            recommended, rows, notes, new NetworkUse(http.Log.Requests, http.Log.Hosts, http.Log.BytesReceived));
        return (report0, null);
    }

    private static QuantRow ToRow(HubRepo repo, HubModelFile model, ScoutReport report)
    {
        string first = System.IO.Path.GetFileName(model.Shards[0].Path);
        string display = model.IsSplit ? $"{StemOf(first)} ({model.Shards.Count} shards)" : first;
        string quant = QuantLabel().Match(first) is { Success: true } m ? m.Value.ToUpperInvariant() : "?";
        string hint = model.IsSplit ? StemOf(first) : first;
        string pull = $"stingray pull -r {repo.Id} --revision {repo.Revision} -q \"{hint}\"";

        var dominant = report.Tensors?.ByDType.OrderByDescending(d => d.Bytes).FirstOrDefault()?.DType;
        long? est = report.Resources.HostWorkingSet.Bytes;

        QuantVerdict verdict;
        string why;
        var confirmed = report.Blockers.FirstOrDefault(b => b.Kind == BlockerKind.Confirmed);
        if (report.Blockers.Any(b => b.Id == "scout.remote.unsupported_storage"))
        {
            // The loader refused the file's own tensor types: a verdict about this checkpoint (retired or newer GGML formats), not a read problem.
            verdict = QuantVerdict.Unsupported;
            why = confirmed!.Summary;
        }
        else if (report.Stages[0].State == StageState.Failed)
        {
            verdict = QuantVerdict.NotInspected;
            why = confirmed?.Summary ?? "The index could not be read.";
        }
        else if (confirmed is not null)
        {
            verdict = QuantVerdict.Unsupported;
            why = confirmed.Summary;
        }
        else
        {
            switch (report.Resources.ExecutionDecision)
            {
                case "allowed": verdict = QuantVerdict.Fits; why = $"peak about {Gib(est)} + reserve fits"; break;
                case "blocked" when est is not null: verdict = QuantVerdict.TooBig; why = report.Resources.Reason; break;
                default: verdict = QuantVerdict.Unknown; why = report.Resources.Reason; break;
            }
        }

        return new QuantRow(display, quant, model.Shards.Count, model.TotalSize, dominant, est, verdict, why, model.Sha256, pull);
    }

    private static string StemOf(string shardName) => Regex.Replace(shardName, @"-\d{5}-of-\d{5}\.gguf$", "", RegexOptions.IgnoreCase);

    internal static string Gib(long? bytes) => bytes is long b ? $"{b / 1073741824.0:F1} GiB" : "?";
}

internal static class QuantTextRenderer
{
    public static void Write(QuantReport r)
    {
        AnsiConsole.MarkupLine($"[bold]{Markup.Escape(r.Repo)}[/] @ {Markup.Escape(r.Revision[..Math.Min(12, r.Revision.Length)])}  " +
            $"[dim]licence {Markup.Escape(r.License ?? "not stated")}{(r.Gated is null ? "" : $", gated ({Markup.Escape(r.Gated)})")}; " +
            $"architecture {Markup.Escape(r.Architecture ?? r.HubArchitecture ?? "?")}" +
            $"{(r.ArchitectureStatus is null ? "" : $" ({Markup.Escape(r.ArchitectureStatus)})")}[/]");
        string src = r.BudgetSource == "detected_ram" ? "this machine's RAM" : "from --budget";
        AnsiConsole.MarkupLine($"[dim]Budget {QuantPicker.Gib(r.BudgetBytes)} ({src}), reserve {QuantPicker.Gib(r.ReserveBytes)}, context {r.ContextTokens} tokens. CPU run only: GPU placement is not estimated. Weights were not downloaded.[/]");
        AnsiConsole.WriteLine();

        var table = new Table().Border(TableBorder.Simple)
            .AddColumn("Quant").AddColumn("File").AddColumn(new TableColumn("Size").RightAligned())
            .AddColumn(new TableColumn("Est. peak RAM").RightAligned()).AddColumn("Verdict");
        foreach (var row in r.Rows)
        {
            string color = row.Verdict switch { QuantVerdict.Fits => "green", QuantVerdict.TooBig => "yellow", QuantVerdict.Unknown => "yellow", _ => "red" };
            string label = row.Verdict switch
            {
                QuantVerdict.Fits => "fits", QuantVerdict.TooBig => "too big", QuantVerdict.Unknown => "unknown",
                QuantVerdict.Unsupported => "unsupported", _ => "not inspected",
            };
            table.AddRow(Markup.Escape(row.Quant), Markup.Escape(row.File), Bytes(row.SizeBytes), row.EstimatedPeakBytes is long e ? QuantPicker.Gib(e) : "-", $"[{color}]{label}[/]");
        }
        AnsiConsole.Write(table);

        foreach (var row in r.Rows.Where(x => x.Verdict is QuantVerdict.Unsupported or QuantVerdict.NotInspected or QuantVerdict.Unknown).GroupBy(x => x.Why).Take(4))
            AnsiConsole.MarkupLine($"  [dim]{Markup.Escape(string.Join(", ", row.Select(x => x.Quant).Distinct()))}: {Markup.Escape(row.Key)}[/]");
        foreach (string n in r.Notes) AnsiConsole.MarkupLine($"[yellow]note:[/] {Markup.Escape(n)}");

        AnsiConsole.WriteLine();
        if (r.Recommended is null)
        {
            bool anyTooBig = r.Rows.Any(x => x.Verdict == QuantVerdict.TooBig);
            bool anyUnknown = r.Rows.Any(x => x.Verdict == QuantVerdict.Unknown);
            if (anyTooBig)
                AnsiConsole.MarkupLine("[bold]No quantisation fits.[/] Raise --budget if this machine has more usable memory, lower --ctx-size, or choose a smaller model.");
            else if (anyUnknown)
                AnsiConsole.MarkupLine("[bold]Cannot tell whether any fits.[/] The memory estimate for this model family is not modelled yet, so nothing is promised either way. This is not a 'too big' result.");
            else
                AnsiConsole.MarkupLine("[bold]None of these can run here[/] (see the reasons above).");
        }
        else
        {
            var best = r.Rows.First(x => x.File == r.Recommended);
            AnsiConsole.MarkupLine($"[bold]Largest that fits:[/] {Markup.Escape(best.File)}  [dim](larger is usually closer to the original model; this is not a measured quality ranking)[/]");
            AnsiConsole.MarkupLine($"  [yellow]{Markup.Escape(best.PullCommand)}[/]");
        }
        AnsiConsole.MarkupLine($"[dim]network: {r.Network.Requests} request(s), {Bytes(r.Network.BytesReceived)} received from {Markup.Escape(string.Join(", ", r.Network.Hosts))}[/]");
    }

    private static string Bytes(long? b) => ScoutTextRenderer.Bytes(b);
}

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower, WriteIndented = true,
    UseStringEnumConverter = true, DefaultIgnoreCondition = JsonIgnoreCondition.Never)]
[JsonSerializable(typeof(QuantReport))]
internal partial class QuantJsonContext : JsonSerializerContext;
