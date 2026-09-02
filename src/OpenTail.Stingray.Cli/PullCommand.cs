using System.Net.Http;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// Downloads a GGUF model (and, if sharded, its sibling parts) directly from a Hugging Face repo
/// id, e.g. <c>stingray pull -r bartowski/Qwen2.5-7B-Instruct-GGUF</c>.
///
/// <para>Closes a real gap against the project's own stated goal ("Run any GGUF from Hugging Face",
/// <c>docs/00-current-work.md</c>): every session so far fetched checkpoints by hand outside this
/// tool before running them. This is intentionally the minimum useful slice — repo resolution, quant
/// selection, and a resumable download — not a full model-store/manifest/alias system (see
/// <c>ListModelsCommand</c>'s doc comment for why that's explicitly out of scope here too).</para>
/// </summary>
public sealed class PullCommand : Command<PullCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-r|--repo <REPO>")]
        [Description("Hugging Face repo id, e.g. bartowski/Qwen2.5-7B-Instruct-GGUF (a full https://huggingface.co/... URL also works)")]
        public string Repo { get; init; } = "";

        [CommandOption("-q|--quant <SUBSTRING>")]
        [Description("Case-insensitive substring to pick among multiple .gguf files (e.g. Q4_K_M). Default: prefer Q4_K_M, then Q4_K_S, Q5_K_M, Q8_0, else the first listed.")]
        public string? Quant { get; init; }

        [CommandOption("-o|--out <DIR>")]
        [Description("Destination directory (default: ./models)")]
        public string OutDir { get; init; } = "models";

        [CommandOption("--list")]
        [Description("List available .gguf files in the repo and exit, without downloading")]
        public bool ListOnly { get; init; }
    }

    private static readonly string[] s_preferredQuantOrder = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q4_0", "Q8_0"];

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        string repo = NormalizeRepo(settings.Repo);
        if (string.IsNullOrWhiteSpace(repo))
        {
            AnsiConsole.MarkupLine("[red]Error:[/] give a Hugging Face repo id, e.g. `stingray pull bartowski/Qwen2.5-7B-Instruct-GGUF`");
            return 1;
        }

        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
        http.DefaultRequestHeaders.UserAgent.ParseAdd("OpenTail.Stingray/pull");

        List<(string Name, long? Size)> files;
        try
        {
            files = ListGgufFiles(http, repo, cancellation);
        }
        catch (Exception ex) when (ex is HttpRequestException or JsonException or TaskCanceledException)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] could not fetch repo listing for '{Markup.Escape(repo)}': {Markup.Escape(ex.Message)}");
            AnsiConsole.MarkupLine("If this repo is gated, accept its terms on huggingface.co first and set HF_TOKEN — anonymous access is used otherwise.");
            return 1;
        }

        if (files.Count == 0)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] no .gguf files found in [yellow]{Markup.Escape(repo)}[/].");
            return 1;
        }

        if (settings.ListOnly)
        {
            foreach (var (name, size) in files.OrderBy(f => f.Name, StringComparer.OrdinalIgnoreCase))
                AnsiConsole.MarkupLine($"  {Markup.Escape(name)}  {(size is { } s ? FormatBytes(s) : "?")}");
            return 0;
        }

        var selected = SelectFiles(files, settings.Quant);
        if (selected.Count == 0)
        {
            AnsiConsole.MarkupLine($"[red]Error:[/] no .gguf file matched --quant '{Markup.Escape(settings.Quant ?? "")}'.");
            AnsiConsole.MarkupLine("Available files:");
            foreach (var (name, _) in files) AnsiConsole.MarkupLine($"  {Markup.Escape(name)}");
            return 1;
        }

        Directory.CreateDirectory(settings.OutDir);

        foreach (var (name, size) in selected)
        {
            string destPath = Path.Combine(settings.OutDir, name);
            string url = $"https://huggingface.co/{repo}/resolve/main/{Uri.EscapeDataString(name).Replace("%2F", "/")}?download=true";
            AnsiConsole.MarkupLine($"[bold]Downloading[/] {Markup.Escape(name)} {(size is { } s ? $"({FormatBytes(s)})" : "")}");
            try
            {
                DownloadWithResume(http, url, destPath, size, cancellation);
            }
            catch (Exception ex) when (ex is IOException or HttpRequestException or TaskCanceledException)
            {
                AnsiConsole.MarkupLine($"[red]Error:[/] download failed: {Markup.Escape(ex.Message)}");
                AnsiConsole.MarkupLine($"Partial file kept at {Markup.Escape(destPath)} — rerun `pull` to resume.");
                return 1;
            }
            AnsiConsole.MarkupLine($"[green]Saved[/] {Markup.Escape(Path.GetFullPath(destPath))}");
        }

        if (selected.Count == 1)
        {
            AnsiConsole.WriteLine();
            AnsiConsole.MarkupLine($"Run it: [yellow]stingray -m {Markup.Escape(Path.Combine(settings.OutDir, selected[0].Name))} -p \"Hello\"[/]");
        }

        return 0;
    }

    /// <summary>Accepts either a bare "owner/name" repo id or a full huggingface.co URL.</summary>
    private static string NormalizeRepo(string input)
    {
        input = input.Trim();
        if (input.Length == 0) return input;
        if (Uri.TryCreate(input, UriKind.Absolute, out var uri) && uri.Host.Contains("huggingface.co", StringComparison.OrdinalIgnoreCase))
        {
            var segments = uri.AbsolutePath.Trim('/').Split('/');
            if (segments.Length >= 2) return $"{segments[0]}/{segments[1]}";
        }
        return input;
    }

    private static List<(string Name, long? Size)> ListGgufFiles(HttpClient http, string repo, CancellationToken ct)
    {
        string apiUrl = $"https://huggingface.co/api/models/{repo}";
        using var request = new HttpRequestMessage(HttpMethod.Get, apiUrl);
        string? token = Environment.GetEnvironmentVariable("HF_TOKEN");
        if (!string.IsNullOrEmpty(token))
            request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);

        using var response = http.Send(request, ct);
        response.EnsureSuccessStatusCode();
        using var stream = response.Content.ReadAsStream(ct);
        using var doc = JsonDocument.Parse(stream);

        var result = new List<(string, long?)>();
        if (!doc.RootElement.TryGetProperty("siblings", out var siblings)) return result;
        foreach (var sib in siblings.EnumerateArray())
        {
            if (!sib.TryGetProperty("rfilename", out var nameEl)) continue;
            string? name = nameEl.GetString();
            if (name is null || !name.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)) continue;
            // The default (non-"expand") HF models API response omits "size" on siblings; a
            // per-file HEAD request on download resolves the real size regardless, so this is a
            // best-effort hint only.
            long? size = sib.TryGetProperty("size", out var sizeEl) && sizeEl.TryGetInt64(out long sz) ? sz
                : sib.TryGetProperty("lfs", out var lfsEl) && lfsEl.TryGetProperty("size", out var lfsSizeEl) && lfsSizeEl.TryGetInt64(out long lfsSz) ? lfsSz
                : null;
            result.Add((name, size));
        }
        return result;
    }

    /// <summary>
    /// Picks one file (or, for a sharded checkpoint named like
    /// <c>model-00001-of-00003.gguf</c>, every shard sharing that base name) to download.
    /// </summary>
    internal static List<(string Name, long? Size)> SelectFiles(List<(string Name, long? Size)> files, string? quantHint)
    {
        IEnumerable<(string Name, long? Size)> candidates = files;
        if (!string.IsNullOrEmpty(quantHint))
            candidates = files.Where(f => f.Name.Contains(quantHint, StringComparison.OrdinalIgnoreCase));
        else if (files.Count > 1)
        {
            foreach (string preferred in s_preferredQuantOrder)
            {
                var match = files.Where(f => f.Name.Contains(preferred, StringComparison.OrdinalIgnoreCase)).ToList();
                if (match.Count > 0) { candidates = match; break; }
            }
        }

        var picked = candidates.OrderBy(f => f.Name, StringComparer.OrdinalIgnoreCase).FirstOrDefault();
        if (picked.Name is null) return [];

        // Sharded GGUF naming: name-00001-of-00005.gguf. Pull every shard once one is selected.
        var shardMatch = System.Text.RegularExpressions.Regex.Match(picked.Name, @"^(.*-)(\d{5})-of-(\d{5})(\.gguf)$", System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        if (!shardMatch.Success) return [picked];

        string prefix = shardMatch.Groups[1].Value;
        string suffix = shardMatch.Groups[4].Value;
        string totalStr = shardMatch.Groups[3].Value;
        return files
            .Where(f => f.Name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)
                        && f.Name.EndsWith(suffix, StringComparison.OrdinalIgnoreCase)
                        && f.Name.Contains($"-of-{totalStr}", StringComparison.OrdinalIgnoreCase))
            .OrderBy(f => f.Name, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    /// <summary>
    /// Streams the download to <paramref name="destPath"/>. If a same-sized or larger partial file
    /// already exists it is treated as complete and skipped (best-effort — HF resolve URLs don't
    /// reliably echo a stable ETag across CDN nodes, so this is a size check, not a hash check);
    /// otherwise any partial bytes present are used as a Range-resume starting offset.
    /// </summary>
    private static void DownloadWithResume(HttpClient http, string url, string destPath, long? expectedSize, CancellationToken ct)
    {
        long existing = File.Exists(destPath) ? new FileInfo(destPath).Length : 0;
        if (existing > 0 && expectedSize is { } exp && existing >= exp)
        {
            AnsiConsole.MarkupLine("  [dim]already present, skipping[/]");
            return;
        }

        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        string? token = Environment.GetEnvironmentVariable("HF_TOKEN");
        if (!string.IsNullOrEmpty(token))
            request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);
        if (existing > 0)
            request.Headers.Range = new System.Net.Http.Headers.RangeHeaderValue(existing, null);

        using var response = http.Send(request, HttpCompletionOption.ResponseHeadersRead, ct);
        bool resumed = existing > 0 && response.StatusCode == System.Net.HttpStatusCode.PartialContent;
        if (existing > 0 && !resumed)
            existing = 0; // Server ignored the Range request; restart from scratch.
        response.EnsureSuccessStatusCode();

        long? total = response.Content.Headers.ContentLength is { } cl ? cl + existing : expectedSize;
        using var contentStream = response.Content.ReadAsStream(ct);
        using var fileStream = new FileStream(destPath, resumed ? FileMode.Append : FileMode.Create, FileAccess.Write);

        byte[] buffer = new byte[1024 * 1024];
        long downloaded = existing;
        int lastPercent = -1;
        int read;
        while ((read = contentStream.Read(buffer, 0, buffer.Length)) > 0)
        {
            ct.ThrowIfCancellationRequested();
            fileStream.Write(buffer, 0, read);
            downloaded += read;
            if (total is { } t and > 0)
            {
                int percent = (int)(downloaded * 100 / t);
                if (percent != lastPercent && percent % 5 == 0)
                {
                    Console.Write($"\r  {percent,3}%  {FormatBytes(downloaded)} / {FormatBytes(t)}   ");
                    lastPercent = percent;
                }
            }
        }
        Console.WriteLine();
    }

    private static string FormatBytes(long bytes)
    {
        if (bytes < 1024) return $"{bytes} B";
        if (bytes < 1024L * 1024) return $"{bytes / 1024.0:F1} KiB";
        if (bytes < 1024L * 1024 * 1024) return $"{bytes / (1024.0 * 1024):F1} MiB";
        return $"{bytes / (1024.0 * 1024 * 1024):F2} GiB";
    }
}
