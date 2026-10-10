using OpenTail.Stingray.Engine.Scout;
using OpenTail.Stingray.Core.Net;

namespace OpenTail.Stingray.Cli.Scout;

public sealed record BacklogEntry(
    string Architecture,
    string Status,
    long Downloads30Days,
    int RepoCount,
    int GatedCount,
    IReadOnlyList<string> Repos);

public sealed record BacklogReport(
    int SchemaVersion,
    string ScoutBuild,
    int Limit,
    int TotalReposInspected,
    IReadOnlyList<BacklogEntry> Backlog,
    IReadOnlyList<BacklogEntry> Admitted,
    IReadOnlyList<BacklogEntry> All,
    NetworkUse Network)
{
    public const int CurrentSchemaVersion = 1;
}

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower, WriteIndented = true,
    UseStringEnumConverter = true, DefaultIgnoreCondition = JsonIgnoreCondition.Never)]
[JsonSerializable(typeof(BacklogReport))]
internal partial class BacklogJsonContext : JsonSerializerContext;

/// <summary>
/// <c>stingray scout --backlog [--limit N]</c>: queries Hugging Face for the most downloaded GGUF architectures
/// and prints the backlog of unregistered/unadmitted families.
/// </summary>
public static class Backlog
{
    public const int DefaultLimit = 50;
    public const int HardCapLimit = 200;

    /// <summary>
    /// Resolves the public status of an architecture: 'admitted', 'not admitted', or 'unregistered'.
    /// Ported-but-unverified architectures (both NotAdmitted and Experimental) are masked as 'unregistered' per CLAUDE.md rule 14.
    /// </summary>
    public static string ResolveStatus(string? architecture)
    {
        if (string.IsNullOrWhiteSpace(architecture))
            return "unregistered";

        var desc = ArchitectureRegistry.Find(architecture);
        if (desc is null)
            return "unregistered";

        if (desc.Status == AdmissionStatus.Admitted)
            return "admitted";

        // CLAUDE.md rule 14: ported-but-unverified families stay internal;
        // both NotAdmitted and Experimental show as unregistered in public output.
        if (desc.Status is AdmissionStatus.NotAdmitted or AdmissionStatus.Experimental)
            return "unregistered";

        return "not admitted";
    }

    public static async Task<BacklogReport> RunAsync(
        ExternalHttpClient http,
        int? limit,
        string buildId,
        CancellationToken ct)
    {
        int effectiveLimit = Math.Clamp(limit ?? DefaultLimit, 1, HardCapLimit);
        string urlStr = $"https://huggingface.co/api/models?filter=gguf&pipeline_tag=text-generation&sort=downloads&direction=-1&limit={effectiveLimit}&expand[]=gguf&expand[]=downloads&expand[]=gated";
        var uri = new Uri(urlStr);

        string json = await http.GetStringAsync(uri, ct).ConfigureAwait(false);
        return Parse(json, effectiveLimit, buildId, new NetworkUse(http.Log.Requests, http.Log.Hosts, http.Log.BytesReceived));
    }

    public static BacklogReport Parse(string json, int limit, string buildId, NetworkUse network)
    {
        using var doc = JsonDocument.Parse(json);
        if (doc.RootElement.ValueKind != JsonValueKind.Array)
            throw new InvalidDataException("Hub models response is not a JSON array.");

        var repos = new List<HubModelSummary>();
        foreach (var element in doc.RootElement.EnumerateArray())
        {
            if (element.ValueKind != JsonValueKind.Object)
                continue;

            try
            {
                if (!element.TryGetProperty("id", out var idProp) || idProp.ValueKind != JsonValueKind.String)
                    continue;

                string id = idProp.GetString()!;
                long downloads = element.TryGetProperty("downloads", out var dlProp) && dlProp.ValueKind == JsonValueKind.Number && dlProp.TryGetInt64(out long dl) ? dl : 0;

                bool gated = false;
                if (element.TryGetProperty("gated", out var gProp))
                {
                    if (gProp.ValueKind == JsonValueKind.True) gated = true;
                    else if (gProp.ValueKind == JsonValueKind.String)
                    {
                        string? gStr = gProp.GetString();
                        if (!string.IsNullOrEmpty(gStr) && !gStr.Equals("false", StringComparison.OrdinalIgnoreCase))
                            gated = true;
                    }
                }

                string arch = "no architecture declared";
                if (element.TryGetProperty("gguf", out var ggufProp) && ggufProp.ValueKind == JsonValueKind.Object)
                {
                    if (ggufProp.TryGetProperty("architecture", out var archProp) && archProp.ValueKind == JsonValueKind.String)
                    {
                        string? a = archProp.GetString();
                        if (!string.IsNullOrWhiteSpace(a))
                        {
                            a = a.Trim();
                            if (a.Equals("(none)", StringComparison.OrdinalIgnoreCase))
                                arch = "no architecture declared";
                            else if (a.Equals("clip", StringComparison.OrdinalIgnoreCase))
                                arch = "projector";
                            else
                                arch = a;
                        }
                    }
                }

                repos.Add(new HubModelSummary(id, downloads, gated, arch));
            }
            catch (Exception ex) when (ex is JsonException or InvalidOperationException or FormatException)
            {
                // Malformed JSON entries are skipped
            }
        }

        // Group by architecture (case-insensitive)
        var grouped = repos
            .GroupBy(r => r.Architecture, StringComparer.OrdinalIgnoreCase)
            .Select(g =>
            {
                string arch = g.Key;
                string status = ResolveStatus(arch);
                long downloads = g.Sum(x => x.Downloads);
                int repoCount = g.Count();
                int gatedCount = g.Count(x => x.Gated);
                var sortedRepos = g.OrderByDescending(x => x.Downloads).ThenBy(x => x.Id, StringComparer.Ordinal).Select(x => x.Id).ToList();

                return new BacklogEntry(arch, status, downloads, repoCount, gatedCount, sortedRepos);
            })
            .OrderByDescending(e => e.Downloads30Days)
            .ThenBy(e => e.Architecture, StringComparer.OrdinalIgnoreCase)
            .ToList();

        var backlog = grouped.Where(e => !e.Status.Equals("admitted", StringComparison.OrdinalIgnoreCase)).ToList();
        var admitted = grouped.Where(e => e.Status.Equals("admitted", StringComparison.OrdinalIgnoreCase)).ToList();

        return new BacklogReport(BacklogReport.CurrentSchemaVersion, buildId, limit, repos.Count, backlog, admitted, grouped, network);
    }

    private sealed record HubModelSummary(string Id, long Downloads, bool Gated, string Architecture);
}

internal static class BacklogTextRenderer
{
    public static void Write(BacklogReport report)
    {
        AnsiConsole.MarkupLine("[bold]Hugging Face GGUF Architecture Backlog[/]");
        AnsiConsole.MarkupLine("[dim]Downloads are 30-day counts per repository, not per file or inference use. Weights were not downloaded.[/]");
        AnsiConsole.WriteLine();

        if (report.Backlog.Count == 0)
        {
            AnsiConsole.MarkupLine("[green]All architectures in the inspected repositories are already admitted.[/]");
        }
        else
        {
            var table = new Table().Border(TableBorder.Simple)
                .AddColumn(new TableColumn("Rank").RightAligned())
                .AddColumn("Architecture")
                .AddColumn("Status")
                .AddColumn(new TableColumn("30d Downloads").RightAligned())
                .AddColumn(new TableColumn("Repos").RightAligned())
                .AddColumn(new TableColumn("Gated").RightAligned())
                .AddColumn("Top Repository");

            int rank = 1;
            foreach (var item in report.Backlog)
            {
                string statusColor = item.Status == "admitted" ? "green" : "yellow";
                table.AddRow(
                    rank++.ToString(),
                    Markup.Escape(item.Architecture),
                    $"[{statusColor}]{Markup.Escape(item.Status)}[/]",
                    $"{item.Downloads30Days:N0}",
                    item.RepoCount.ToString(),
                    item.GatedCount > 0 ? $"[yellow]{item.GatedCount}[/]" : "0",
                    Markup.Escape(item.Repos.Count > 0 ? item.Repos[0] : "-"));
            }
            AnsiConsole.Write(table);
        }

        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine($"[dim]Inspected {report.TotalReposInspected} repositories ({report.Admitted.Count} admitted architecture(s): {string.Join(", ", report.Admitted.Select(a => a.Architecture))}).[/]");
        if (report.Network.Requests > 0)
        {
            AnsiConsole.MarkupLine($"[dim]network: {report.Network.Requests} request(s) to {string.Join(", ", report.Network.Hosts)}, {ScoutTextRenderer.Bytes(report.Network.BytesReceived)} received[/]");
        }
    }
}
