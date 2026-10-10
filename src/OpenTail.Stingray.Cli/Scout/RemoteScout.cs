using OpenTail.Stingray.Engine.Scout;
using System.Net;
using OpenTail.Stingray.Core.Net;

namespace OpenTail.Stingray.Cli.Scout;

public sealed record RemoteScoutRequest(string Repo, string? File, string? Revision, long MaxIndexBytes);

/// <summary>
/// Outcome of a remote scout. Exactly one of these holds: a <see cref="Report"/> (possibly an "inspection failed" one), a list of <see cref="Choices"/>
/// the user must pick from, or only a <see cref="Message"/> (repo missing, restricted, nothing to inspect).
/// </summary>
public sealed record RemoteScoutResult(
    ScoutReport? Report, ScoutInput? Input, SigOrigin? Origin, HubRepo? Repo,
    IReadOnlyList<HubModelFile> Choices, string? Message);

/// <summary>
/// <c>scout -r owner/repo</c>: resolve the repo to a commit, pick one model file, read its index with bounded Range requests, and run the same
/// analysis as a local file. The weights are never downloaded. Everything reported about the source is what the Hub said at that commit.
/// </summary>
public static class RemoteScout
{
    public static async Task<RemoteScoutResult> RunAsync(ExternalHttpClient http, RemoteScoutRequest req, ScoutOptions options, CancellationToken ct)
    {
        var (found, failure) = await TryGetRepoAsync(http, req.Repo, req.Revision, ct).ConfigureAwait(false);
        if (found is null) return Only(failure!);
        var repo = found;
        var models = HubClient.GroupModels(repo.Files);
        var candidates = models.Where(m => !HubClient.IsProjector(m)).ToArray();

        HubModelFile? chosen;
        if (req.File is { Length: > 0 } wanted)
        {
            chosen = models.FirstOrDefault(m => Matches(m, wanted));
            if (chosen is null)
                return new(null, null, null, repo, candidates, $"No GGUF in '{repo.Id}' matches '{wanted}'.");
        }
        else if (candidates.Length == 1) chosen = candidates[0];
        else if (candidates.Length == 0) return new(null, null, null, repo, [], $"'{repo.Id}' has no GGUF model files.");
        else return new(null, null, null, repo, candidates, $"'{repo.Id}' has {candidates.Length} GGUF models; choose one with -f.");

        if (!HubClient.IsComplete(chosen))
            return new(null, null, null, repo, [], $"'{chosen.Name}' is a split model with missing shards on the Hub; it cannot be loaded as published.");

        return await InspectAsync(http, repo, chosen, req.MaxIndexBytes, options, ct).ConfigureAwait(false);
    }

    /// <summary>
    /// Inspects one model file of an already-fetched repo: bounded index read, then the full analysis, with provenance and network usage attached.
    /// The repo is passed in so a caller inspecting many files (the quant picker) asks the Hub about the repo once.
    /// </summary>
    public static async Task<RemoteScoutResult> InspectAsync(ExternalHttpClient http, HubRepo repo, HubModelFile chosen, long maxIndexBytes, ScoutOptions options, CancellationToken ct)
    {
        var paths = chosen.Shards.Select(s => s.Path).ToArray();
        var idx = await RemoteGgufReader.ReadAsync(http, repo.Id, repo.Revision, paths, maxIndexBytes, ct).ConfigureAwait(false);

        string fileName = System.IO.Path.GetFileName(paths[0]);
        long? listedBytes = chosen.TotalSize;
        string? sha = chosen.Sha256;
        var source = new ArtifactSource("huggingface", repo.Id, repo.Revision, paths, sha, sha is null ? null : "published_by_huggingface",
            repo.Gated, repo.Private, repo.License, repo.Downloads30Days, repo.Gguf?.Architecture, idx.BytesRead);
        var net = new NetworkUse(http.Log.Requests, http.Log.Hosts, http.Log.BytesReceived);
        string shaState = sha is null ? "not_computed" : "published_by_huggingface";

        if (idx.Outcome != RemoteIndexOutcome.Complete)
        {
            string id = idx.Outcome switch
            {
                RemoteIndexOutcome.IncompleteOverCap => "scout.remote.index_over_cap",
                RemoteIndexOutcome.AccessRestricted => "scout.remote.access_restricted",
                RemoteIndexOutcome.UnsupportedStorage => "scout.remote.unsupported_storage",
                RemoteIndexOutcome.NotFound => "scout.remote.not_found",
                _ => "scout.remote.failed",
            };
            var failed = ScoutAnalyzer.Unreadable(fileName, listedBytes, idx.Detail, options);
            failed = failed with
            {
                Artifact = failed.Artifact with { Source = source, Sha256State = shaState, ShardCount = paths.Length },
                Blockers = [new ScoutBlocker(id, BlockerKind.Confirmed, idx.Detail, [new EvidenceItem("repo", repo.Id, repo.Revision)])],
                Network = net,
            };
            return new(failed, null, null, repo, [], null);
        }

        var index = idx.Index!;
        long total = idx.ShardSizes.Sum();
        var input = new ScoutInput(fileName, total, index.Header.Version, index.Header.TensorCount, index.Header.MetadataKvCount, index.Metadata, index.Tensors);
        var report = ScoutAnalyzer.Analyze(input, options);

        var findings = report.Findings.ToList();
        var blockers = report.Blockers.ToList();
        if (listedBytes is long lb && lb != total)
            blockers.Add(new ScoutBlocker("scout.source.size_mismatch", BlockerKind.Suspected,
                $"The Hub's file listing says {lb} bytes but the server reports {total}; the repository may have changed under the pinned commit.",
                [new EvidenceItem("repo", repo.Id, repo.Revision)]));
        string? declared = index.Metadata.TryGetValue("general.architecture", out var a) ? Convert.ToString(a, System.Globalization.CultureInfo.InvariantCulture) : null;
        if (repo.Gguf?.Architecture is { Length: > 0 } hubArch && declared is not null && !string.Equals(hubArch, declared, StringComparison.OrdinalIgnoreCase))
            blockers.Add(new ScoutBlocker("scout.source.hub_architecture_disagrees", BlockerKind.Suspected,
                $"The Hub declares architecture '{hubArch}' but the file itself says '{declared}'. Trust the file; check the repo for a mixed or mislabelled upload.",
                [new EvidenceItem("hub", "gguf.architecture", hubArch), new EvidenceItem("metadata", "general.architecture", declared)]));
        if (repo.Gated is not null)
            findings.Add(new ScoutFinding("source.gated", $"The repository is gated ({repo.Gated}): downloading the weights needs accepted terms and HF_TOKEN.", Certainty.Known,
                [new EvidenceItem("hub", "gated", repo.Gated)], "Scout read only the index. Accepting a licence is a decision for a person; this tool never does it."));

        report = report with
        {
            Artifact = report.Artifact with { Source = source, Sha256State = shaState },
            Findings = findings.OrderBy(f => f.Id, StringComparer.Ordinal).ToArray(),
            Blockers = blockers.OrderBy(b => b.Kind).ThenBy(b => b.Id, StringComparer.Ordinal).ToArray(),
            Network = net,
        };

        var origin = new SigOrigin(fileName, total, sha, repo.Id, repo.Revision, options.BuildId, sha is null ? null : "published_by_huggingface");
        return new(report, input, origin, repo, [], null);
    }

    /// <summary>
    /// Fetches the repo's metadata at a pinned commit, turning every expected failure into a message a person can act on. Shared by single-file
    /// remote scout and the quant picker. A refused external-access policy is NOT a failure here: it propagates as <see cref="ExternalAccessDeniedException"/>.
    /// </summary>
    public static async Task<(HubRepo? Repo, string? Failure)> TryGetRepoAsync(ExternalHttpClient http, string repo, string? revision, CancellationToken ct)
    {
        try { return (await HubClient.GetRepoAsync(http, repo, revision, ct).ConfigureAwait(false), null); }
        catch (HttpRequestException ex) when (ex.StatusCode is HttpStatusCode.NotFound)
        { return (null, $"'{repo}'{(revision is null ? "" : $" at revision '{revision}'")} was not found on Hugging Face."); }
        catch (HttpRequestException ex) when (ex.StatusCode is HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden)
        { return (null, $"Hugging Face refused access to '{repo}' ({(int)ex.StatusCode}). The Hub answers the same way for a private repo, a gated one, and one that does not exist. If it is gated, accept its terms on huggingface.co and set HF_TOKEN."); }
        catch (HttpRequestException ex) { return (null, $"Could not reach Hugging Face: {ex.Message}"); }
        catch (InvalidDataException ex) { return (null, $"Hugging Face returned something this tool cannot use: {ex.Message}"); }
        catch (System.Text.Json.JsonException ex) { return (null, $"Hugging Face returned something this tool cannot use: {ex.Message}"); }
    }
    private static RemoteScoutResult Only(string message) => new(null, null, null, null, [], message);

    /// <summary>A model matches by full path, by file name, or (for a split model) by any shard name or its stem, ignoring case.</summary>
    internal static bool Matches(HubModelFile m, string wanted)
    {
        string w = wanted.Replace('\\', '/').Trim();
        if (m.Name.Equals(w, StringComparison.OrdinalIgnoreCase)) return true;
        foreach (var s in m.Shards)
        {
            if (s.Path.Equals(w, StringComparison.OrdinalIgnoreCase)) return true;
            string fn = System.IO.Path.GetFileName(s.Path);
            if (fn.Equals(w, StringComparison.OrdinalIgnoreCase) || fn.Equals(w + ".gguf", StringComparison.OrdinalIgnoreCase)) return true;
        }
        // A split model is also known by its stem ("big" or "big.gguf" for big-00001-of-00003.gguf), with or without its folder.
        if (m.IsSplit)
        {
            var parts = System.Text.RegularExpressions.Regex.Match(m.Shards[0].Path, @"^(?<stem>.+)-\d{5}-of-\d{5}\.gguf$", System.Text.RegularExpressions.RegexOptions.IgnoreCase);
            if (parts.Success)
            {
                string stem = parts.Groups["stem"].Value;
                string stemName = System.IO.Path.GetFileName(stem);
                foreach (var cand in new[] { stem, stemName })
                    if (cand.Equals(w, StringComparison.OrdinalIgnoreCase) || (cand + ".gguf").Equals(w, StringComparison.OrdinalIgnoreCase)) return true;
            }
        }
        return false;
    }
}
