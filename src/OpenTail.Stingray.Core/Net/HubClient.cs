using System.Text.Json;
using System.Text.RegularExpressions;

namespace OpenTail.Stingray.Core.Net;

/// <summary>One file in a Hub repository. <see cref="Sha256"/> is what the Hub PUBLISHES (Git LFS); it is not a hash anyone here computed.</summary>
public sealed record HubFile(string Path, long? Size, string? Sha256);

/// <summary>The Hub's own summary of a GGUF repo (from the model API's <c>gguf</c> field). Declared by the Hub, not read from the file by us.</summary>
public sealed record HubGgufSummary(string? Architecture, long? ParameterCount, long? ContextLength, long? TotalFileSize);

/// <summary>
/// What the Hub says about a repository at one commit. <see cref="Revision"/> is the immutable commit SHA: every later request is pinned to it,
/// so the metadata and the bytes of a file always describe the same revision.
/// <see cref="Downloads30Days"/> is the repository's download count over the last 30 days (the API's <c>downloads</c>); it is per repository, not per file.
/// </summary>
public sealed record HubRepo(
    string Id,
    string Revision,
    bool Private,
    // null when not gated; otherwise the Hub's value ("auto" or "manual").
    string? Gated,
    long? Downloads30Days,
    string? License,
    IReadOnlyList<string> Tags,
    HubGgufSummary? Gguf,
    IReadOnlyList<HubFile> Files);

/// <summary>A model file as the user thinks of it: one .gguf, or all shards of a split one.</summary>
public sealed record HubModelFile(string Name, IReadOnlyList<HubFile> Shards)
{
    public long? TotalSize => Shards.All(s => s.Size is not null) ? Shards.Sum(s => s.Size!.Value) : null;
    public bool IsSplit => Shards.Count > 1;
    public string? Sha256 => Shards.Count == 1 ? Shards[0].Sha256 : null;
}

public static partial class HubClient
{
    private static readonly Uri s_api = new("https://huggingface.co/api/models/");

    [GeneratedRegex(@"^(?<stem>.+)-(?<i>\d{5})-of-(?<n>\d{5})\.gguf$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant)]
    private static partial Regex SplitName();

    /// <summary>Validates and normalises "owner/name" or a huggingface.co URL. Returns null for anything else (no path traversal, no extra segments).</summary>
    public static string? NormalizeRepoId(string? input)
    {
        if (string.IsNullOrWhiteSpace(input)) return null;
        input = input.Trim();
        if (Uri.TryCreate(input, UriKind.Absolute, out var uri))
        {
            if (!ExternalAccess.IsAllowedHost(uri.Host)) return null;
            var parts = uri.AbsolutePath.Trim('/').Split('/', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 2) return null;
            input = parts[0] + "/" + parts[1];
        }
        var seg = input.Split('/');
        if (seg.Length != 2 || !seg.All(IsSafeSegment)) return null;
        return input;
    }

    private static bool IsSafeSegment(string s) =>
        s.Length is > 0 and <= 96 && s != "." && s != ".." && s.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '_' or '.');

    /// <summary>Fetches the repo at its current head (or at <paramref name="revision"/>, a commit/branch/tag), including per-file size and published SHA-256.</summary>
    public static async Task<HubRepo> GetRepoAsync(ExternalHttpClient http, string repo, string? revision, CancellationToken ct)
    {
        string id = NormalizeRepoId(repo) ?? throw new ArgumentException($"'{repo}' is not a Hugging Face repo id (owner/name).", nameof(repo));
        string path = revision is { Length: > 0 } ? $"{id}/revision/{Uri.EscapeDataString(revision)}" : id;
        string json = await http.GetStringAsync(new Uri(s_api, path + "?blobs=true"), ct).ConfigureAwait(false);
        using var doc = JsonDocument.Parse(json);
        return ParseRepo(doc.RootElement, id);
    }

    /// <summary>Parses a model-API response. Tolerant: an absent field is null, never a guess; a malformed file entry is skipped, not fatal.</summary>
    public static HubRepo ParseRepo(JsonElement root, string fallbackId)
    {
        string sha = Str(root, "sha") ?? throw new InvalidDataException("The Hub response has no commit SHA; cannot pin the revision.");
        var tags = root.TryGetProperty("tags", out var t) && t.ValueKind == JsonValueKind.Array
            ? t.EnumerateArray().Select(x => x.GetString()).OfType<string>().ToArray() : [];
        string? gated = root.TryGetProperty("gated", out var g)
            ? g.ValueKind switch { JsonValueKind.String => g.GetString(), JsonValueKind.True => "true", _ => null }
            : null;
        string? license = root.TryGetProperty("cardData", out var card) && card.ValueKind == JsonValueKind.Object ? Str(card, "license") : null;
        license ??= tags.FirstOrDefault(x => x.StartsWith("license:", StringComparison.Ordinal))?["license:".Length..];

        HubGgufSummary? gguf = null;
        if (root.TryGetProperty("gguf", out var gg) && gg.ValueKind == JsonValueKind.Object)
            gguf = new HubGgufSummary(Str(gg, "architecture"), Num(gg, "total"), Num(gg, "context_length"), Num(gg, "totalFileSize"));

        var files = new List<HubFile>();
        if (root.TryGetProperty("siblings", out var sib) && sib.ValueKind == JsonValueKind.Array)
        {
            foreach (var s in sib.EnumerateArray())
            {
                string? name = Str(s, "rfilename");
                if (name is null) continue;
                long? size = Num(s, "size");
                string? hash = null;
                if (s.TryGetProperty("lfs", out var lfs) && lfs.ValueKind == JsonValueKind.Object)
                {
                    size ??= Num(lfs, "size");
                    if (Str(lfs, "sha256") is { Length: 64 } h) hash = h.ToLowerInvariant();
                }
                files.Add(new HubFile(name, size, hash));
            }
        }

        return new HubRepo(Str(root, "id") ?? fallbackId, sha, root.TryGetProperty("private", out var p) && p.ValueKind == JsonValueKind.True,
            gated, Num(root, "downloads"), license, tags, gguf, files);
    }

    private static string? Str(JsonElement e, string name) => e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
    private static long? Num(JsonElement e, string name) => e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number && v.TryGetInt64(out long n) ? n : null;

    /// <summary>
    /// Groups the repo's .gguf files into models: a single file is one model; <c>name-00001-of-00003.gguf</c> and its siblings are one split model.
    /// Projector files (<c>mmproj*</c>) are returned too, flagged by name, so a caller can decide whether they matter. A split model with a missing
    /// shard is returned with the shards that exist and <see cref="HubModelFile.IsSplit"/> true; check completeness with <see cref="IsComplete"/>.
    /// </summary>
    public static IReadOnlyList<HubModelFile> GroupModels(IEnumerable<HubFile> files)
    {
        var singles = new List<HubModelFile>();
        var splits = new Dictionary<string, List<(int Index, int Total, HubFile File)>>(StringComparer.Ordinal);
        foreach (var f in files.Where(f => f.Path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)))
        {
            var m = SplitName().Match(System.IO.Path.GetFileName(f.Path));
            if (!m.Success) { singles.Add(new HubModelFile(f.Path, [f])); continue; }
            string key = (System.IO.Path.GetDirectoryName(f.Path)?.Replace('\\', '/') is { Length: > 0 } d ? d + "/" : "") + m.Groups["stem"].Value + "|" + m.Groups["n"].Value;
            if (!splits.TryGetValue(key, out var list)) splits[key] = list = [];
            list.Add((int.Parse(m.Groups["i"].Value), int.Parse(m.Groups["n"].Value), f));
        }
        foreach (var (key, list) in splits)
        {
            var ordered = list.OrderBy(x => x.Index).Select(x => x.File).ToArray();
            string stem = key[..key.LastIndexOf('|')];
            singles.Add(new HubModelFile(stem + $"-{list[0].Total:D5}-shards.gguf", ordered));
        }
        return singles.OrderBy(m => m.Name, StringComparer.Ordinal).ToArray();
    }

    /// <summary>True when a split model has every shard its names promise (a single file is always complete).</summary>
    public static bool IsComplete(HubModelFile model)
    {
        if (!model.IsSplit) return true;
        var m = SplitName().Match(System.IO.Path.GetFileName(model.Shards[0].Path));
        if (!m.Success) return true;
        int total = int.Parse(m.Groups["n"].Value);
        var present = model.Shards.Select(s => int.Parse(SplitName().Match(System.IO.Path.GetFileName(s.Path)).Groups["i"].Value)).ToHashSet();
        return Enumerable.Range(1, total).All(present.Contains);
    }

    public static bool IsProjector(HubModelFile model) =>
        System.IO.Path.GetFileName(model.Shards[0].Path).StartsWith("mmproj", StringComparison.OrdinalIgnoreCase);

    /// <summary>URL of a file pinned to a commit. Path segments are escaped individually so a '/' in a sub-folder stays a separator.</summary>
    public static Uri ResolveUrl(string repo, string revision, string path)
    {
        string id = NormalizeRepoId(repo) ?? throw new ArgumentException($"'{repo}' is not a Hugging Face repo id.", nameof(repo));
        string escaped = string.Join('/', path.Split('/').Select(Uri.EscapeDataString));
        return new Uri($"https://huggingface.co/{id}/resolve/{Uri.EscapeDataString(revision)}/{escaped}");
    }
}
