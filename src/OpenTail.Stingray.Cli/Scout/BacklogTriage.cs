using OpenTail.Stingray.Engine.Scout;
using OpenTail.Stingray.Core.Net;

namespace OpenTail.Stingray.Cli.Scout;

/// <summary>One backlog family after a remote scout of its most downloaded repo. Evidence for a human, never an admission decision.</summary>
public sealed record TriageRow(
    string Architecture,
    string Repo,
    string Outcome,
    string? NearestRelative,
    string? RelativeKind,
    int? Differences,
    string? EstimatedPeak,
    string Detail);

public sealed record BacklogTriageReport(BacklogReport Backlog, IReadOnlyList<TriageRow> Triage);

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower, WriteIndented = true,
    UseStringEnumConverter = true, DefaultIgnoreCondition = JsonIgnoreCondition.Never)]
[JsonSerializable(typeof(BacklogTriageReport))]
internal partial class BacklogTriageJsonContext : JsonSerializerContext;

/// <summary>
/// <c>stingray scout --backlog --triage N</c>: remote-scouts (index only, a few MiB each) the most downloaded repo of each of the top N
/// unregistered families and tabulates the nearest admitted relative and the memory estimate. The ranking is advisory: it neither
/// admits anything nor changes how an architecture is handled (CLAUDE.md rule 15).
/// </summary>
public static class BacklogTriage
{
    public static async Task<IReadOnlyList<TriageRow>> RunAsync(ExternalHttpClient http, BacklogReport backlog, int families, ScoutOptions options, CancellationToken ct)
    {
        var rows = new List<TriageRow>();
        foreach (var entry in backlog.Backlog.Where(e => e.Status == "unregistered" && e.Repos.Count > 0).Take(families))
        {
            ct.ThrowIfCancellationRequested();
            string repo = entry.Repos[0];
            try
            {
                var r = await RemoteScout.RunAsync(http, new RemoteScoutRequest(repo, null, null, RemoteGgufReader.DefaultMaxIndexBytes), options, ct).ConfigureAwait(false);
                string note = "";
                if (r.Report is null && r.Choices.Count > 1 && r.Choices.Where(c => c.TotalSize is not null).MinBy(c => c.TotalSize) is { } smallest)
                {
                    // Several quantisations: the structure is the same in each, so inspect the smallest (least to read, same tensor layout).
                    int offered = r.Choices.Count;
                    r = await RemoteScout.RunAsync(http, new RemoteScoutRequest(repo, smallest.Name, null, RemoteGgufReader.DefaultMaxIndexBytes), options, ct).ConfigureAwait(false);
                    note = $"smallest of {offered} files: {smallest.Name}; ";
                }
                if (r.Report is null)
                {
                    rows.Add(new(entry.Architecture, repo, "not_inspected", null, null, null, null, r.Message ?? (r.Choices.Count > 0 ? "several GGUF files; pick one with scout -r" : "no report")));
                    continue;
                }
                var best = r.Report.Architecture?.Candidates.FirstOrDefault();
                string? peak = r.Report.Resources.HostWorkingSet.Bytes is long b ? $"{b / (double)(1L << 30):F1} GiB" : null;
                rows.Add(new(entry.Architecture, repo, "inspected", best?.Id, best?.Kind, best?.DifferenceCount, peak, note + r.Report.Resources.Reason));
            }
            catch (ExternalAccessDeniedException) { throw; }
            catch (HttpRequestException ex) { rows.Add(new(entry.Architecture, repo, "network_error", null, null, null, null, ex.Message)); }
        }
        return rows;
    }

    public static void Write(IReadOnlyList<TriageRow> rows)
    {
        AnsiConsole.MarkupLine("[bold]Triage[/] (index-only remote scout of each family's most downloaded repo; advisory)");
        var table = new Table().AddColumn("Architecture").AddColumn("Repo").AddColumn("Outcome").AddColumn("Nearest admitted").AddColumn("Peak (CPU)");
        foreach (var r in rows)
            table.AddRow(Markup.Escape(r.Architecture), Markup.Escape(r.Repo), Markup.Escape(r.Outcome),
                r.NearestRelative is null ? "-" : Markup.Escape($"{r.NearestRelative} ({r.RelativeKind}, {r.Differences} differences)"),
                Markup.Escape(r.EstimatedPeak ?? "unknown"));
        AnsiConsole.Write(table);
    }
}
