using OpenTail.Stingray.Engine.Verification;
using System.Net.Http;
using OpenTail.Stingray.Core.Catalog;

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
            AnsiConsole.ErrorLine("[red]Error:[/] give a Hugging Face repo id, e.g. `stingray pull bartowski/Qwen2.5-7B-Instruct-GGUF`");
            return 1;
        }

        var access = OpenTail.Stingray.Core.Net.ExternalAccess.Evaluate();
        if (!access.Allowed)
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] not contacting Hugging Face: {Markup.Escape(access.Reason)}. To allow it: {Markup.Escape(access.HowToEnable ?? "see STINGRAY_ALLOW_EXTERNAL")}.");
            return ExitCodes.Failure;
        }

        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
        http.DefaultRequestHeaders.UserAgent.ParseAdd("OpenTail.Stingray/pull");

        var publishedSha256 = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        List<(string Name, long? Size)> files;
        try
        {
            files = ListGgufFiles(http, repo, cancellation, publishedSha256);
        }
        catch (Exception ex) when (ex is HttpRequestException or JsonException or TaskCanceledException)
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] could not fetch repo listing for '{Markup.Escape(repo)}': {Markup.Escape(ex.Message)}");
            AnsiConsole.MarkupLine("If this repo is gated, accept its terms on huggingface.co first and set HF_TOKEN — anonymous access is used otherwise.");
            return 1;
        }

        if (files.Count == 0)
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] no .gguf files found in [yellow]{Markup.Escape(repo)}[/].");
            return 1;
        }

        if (settings.ListOnly)
        {
            foreach (var (name, size) in files.OrderBy(f => f.Name, StringComparer.OrdinalIgnoreCase))
                AnsiConsole.MarkupLine($"  {Markup.Escape(name)}  {(size is { } s ? ConsoleDownloadProgress.FormatBytes(s) : "?")}");
            return 0;
        }

        var selected = SelectFiles(files, settings.Quant);
        if (selected.Count == 0)
        {
            AnsiConsole.ErrorLine($"[red]Error:[/] no .gguf file matched --quant '{Markup.Escape(settings.Quant ?? "")}'.");
            AnsiConsole.MarkupLine("Available files:");
            foreach (var (name, _) in files) AnsiConsole.MarkupLine($"  {Markup.Escape(name)}");
            return 1;
        }

        Directory.CreateDirectory(settings.OutDir);

        foreach (var (name, size) in selected)
        {
            // Repo file names come from the Hugging Face listing; never let one escape --out
            // (a "../" or rooted name). Hugging Face does not allow them, but check locally anyway.
            string outRoot = Path.GetFullPath(settings.OutDir);
            string destPath = Path.GetFullPath(Path.Combine(outRoot, name));
            if (Path.IsPathRooted(name) || !destPath.StartsWith(outRoot + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            {
                AnsiConsole.ErrorLine($"[red]Skipping {Markup.Escape(name)}:[/] it would be written outside {Markup.Escape(outRoot)}.");
                continue;
            }
            Directory.CreateDirectory(Path.GetDirectoryName(destPath)!);   // sharded repos keep shards in a quant subfolder
            string url = $"https://huggingface.co/{repo}/resolve/main/{Uri.EscapeDataString(name).Replace("%2F", "/")}?download=true";
            AnsiConsole.MarkupLine($"[bold]Downloading[/] {Markup.Escape(name)} {(size is { } s ? $"({ConsoleDownloadProgress.FormatBytes(s)})" : "")}");
            try
            {
                DownloadWithResume(http, url, destPath, size, cancellation);
            }
            catch (Exception ex) when (ex is IOException or HttpRequestException or TaskCanceledException)
            {
                AnsiConsole.ErrorLine($"[red]Error:[/] download failed: {Markup.Escape(ex.Message)}");
                AnsiConsole.MarkupLine($"Partial file kept at {Markup.Escape(destPath)} — rerun `pull` to resume.");
                return 1;
            }
            AnsiConsole.MarkupLine($"[green]Saved[/] {Markup.Escape(Path.GetFullPath(destPath))}");
            if (publishedSha256.TryGetValue(name, out string? expectedSha))
            {
                AnsiConsole.MarkupLine("Verifying SHA-256...");
                var fp = ModelFingerprinter.Compute(destPath);
                if (!string.Equals(fp.Sha256, expectedSha, StringComparison.OrdinalIgnoreCase))
                {
                    AnsiConsole.ErrorLine($"[red]SHA-256 MISMATCH[/] for {Markup.Escape(name)}: Hugging Face publishes {expectedSha}, the file on disk is {fp.Sha256}. " +
                        "The download is corrupt or incomplete; delete it and rerun `pull`.");
                    return 1;
                }
                AnsiConsole.MarkupLine($"[green]SHA-256 verified[/] ({fp.Sha256[..12]}...)");
            }
            else AnsiConsole.MarkupLine("[dim]No SHA-256 published for this file; not verified.[/]");
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
        if (Uri.TryCreate(input, UriKind.Absolute, out var uri) && (uri.Host.Equals("huggingface.co", StringComparison.OrdinalIgnoreCase)
                || uri.Host.Equals("www.huggingface.co", StringComparison.OrdinalIgnoreCase)
                || uri.Host.Equals("hf.co", StringComparison.OrdinalIgnoreCase)))
        {
            var segments = uri.AbsolutePath.Trim('/').Split('/');
            if (segments.Length >= 2) return $"{segments[0]}/{segments[1]}";
        }
        return input;
    }

    private static List<(string Name, long? Size)> ListGgufFiles(HttpClient http, string repo, CancellationToken ct, Dictionary<string, string>? sha256ByName = null)
    {
        // ?blobs=true makes the API return each file's size and (for LFS files) its SHA-256; the default response carries neither.
        string apiUrl = $"https://huggingface.co/api/models/{repo}?blobs=true";
        using var request = new HttpRequestMessage(HttpMethod.Get, apiUrl);
        string? token = Environment.GetEnvironmentVariable("HF_TOKEN");
        if (!string.IsNullOrEmpty(token))
            request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);

        using var response = http.Send(request, ct);
        response.EnsureSuccessStatusCode();
        using var stream = response.Content.ReadAsStream(ct);
        using var doc = JsonDocument.Parse(stream);
        return ParseGgufListing(doc.RootElement, sha256ByName);
    }

    /// <summary>Reads the .gguf files (and, when published, their LFS SHA-256) from a Hugging Face model API response.</summary>
    internal static List<(string Name, long? Size)> ParseGgufListing(JsonElement root, Dictionary<string, string>? sha256ByName = null)
    {
        var result = new List<(string, long?)>();
        if (!root.TryGetProperty("siblings", out var siblings)) return result;
        foreach (var sib in siblings.EnumerateArray())
        {
            if (!sib.TryGetProperty("rfilename", out var nameEl)) continue;
            string? name = nameEl.GetString();
            if (name is null || !name.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)) continue;
            // Without ?blobs=true the API omits "size" on siblings; a per-file HEAD request on download resolves the real size regardless,
            // so this is a best-effort hint only.
            long? size = sib.TryGetProperty("size", out var sizeEl) && sizeEl.TryGetInt64(out long sz) ? sz
                : sib.TryGetProperty("lfs", out var lfsEl) && lfsEl.TryGetProperty("size", out var lfsSizeEl) && lfsSizeEl.TryGetInt64(out long lfsSz) ? lfsSz
                : null;
            if (sha256ByName is not null && sib.TryGetProperty("lfs", out var lfs) && lfs.TryGetProperty("sha256", out var shaEl)
                && shaEl.GetString() is { Length: 64 } sha)
                sha256ByName[name] = sha.ToLowerInvariant();
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
        {
            // An exact file name wins. Otherwise a hint that does not ask for a projector skips mmproj-* files: the
            // projector's name usually contains the model's, so "Qwen3VL-2B-Instruct-Q8_0" also matched
            // "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf" and picked it (sorted first) instead of the model.
            var exact = files.Where(f => f.Name.Equals(quantHint, StringComparison.OrdinalIgnoreCase)
                                         || f.Name.Equals(quantHint + ".gguf", StringComparison.OrdinalIgnoreCase)).ToList();
            var matching = files.Where(f => f.Name.Contains(quantHint, StringComparison.OrdinalIgnoreCase)).ToList();
            var nonProjector = matching.Where(f => !f.Name.StartsWith("mmproj", StringComparison.OrdinalIgnoreCase)).ToList();
            candidates = exact.Count > 0 ? exact
                : !quantHint.Contains("mmproj", StringComparison.OrdinalIgnoreCase) && nonProjector.Count > 0 ? nonProjector
                : matching;
        }
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
    /// Downloads via <see cref="ModelDownloader.DownloadAsync"/> (resumes a partial file; an existing file of the
    /// listed size counts as complete, a size check rather than a hash check) with console progress.
    /// </summary>
    private static void DownloadWithResume(HttpClient http, string url, string destPath, long? expectedSize, CancellationToken ct)
    {
        var printer = new ConsoleDownloadProgress();
        var outcome = ModelDownloader.DownloadAsync(http, url, destPath, expectedSize, printer.Report, ct).GetAwaiter().GetResult();
        printer.Finish();
        if (outcome == DownloadOutcome.AlreadyComplete)
            AnsiConsole.MarkupLine("  [dim]already present, skipping[/]");
    }
}
