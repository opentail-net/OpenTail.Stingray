using OpenTail.Stingray.Core.Catalog;
using OpenTail.Stingray.Core.Net;

namespace OpenTail.Stingray.Engine.Scout;

public enum PreflightVerdict
{
    /// <summary>The upper-bound CPU estimate plus reserve fits this machine's memory.</summary>
    Allowed,
    /// <summary>A KNOWN estimate that does not fit.</summary>
    Blocked,
    /// <summary>No estimate could be made (a family that is not modelled, incomplete metadata): nothing is promised either way.</summary>
    Unknown,
    /// <summary>Not a GGUF this check understands, or the file could not be read; the loader reports that itself.</summary>
    NotApplicable,
}

public sealed record PreflightResult(PreflightVerdict Verdict, string Summary, long? EstimatedPeakBytes, long BudgetBytes, long ReserveBytes);

/// <summary>
/// The one memory check shared by <c>scout</c> and by the commands that load or install a model. It is a thin wrapper over
/// <see cref="ScoutAnalyzer"/>, the same code <c>scout</c> runs, so the two can never disagree about whether a model fits.
/// Deliberate policy: only a KNOWN estimate over budget blocks; an Unknown estimate warns and proceeds, so a model that loads today
/// (an MLA or RWKV family the estimator does not model yet) is not broken by the check. CPU runs only: GPU placement is not estimated.
/// </summary>
public static class LoadPreflight
{
    /// <summary>This machine's RAM as the budget, with the reserve scaled down on small machines (the smaller of 8 GiB and a quarter).</summary>
    public static (long Budget, long Reserve) MachineBudget(long? ramBytes = null)
    {
        long ram = ramBytes ?? HardwareProfile.Detect().RamBytes;
        return (ram, Math.Min(ScoutOptions.DefaultReserveBytes, ram / 4));
    }

    public static PreflightResult EvaluateIndex(ScoutInput input, int contextTokens, long? ramBytes = null)
    {
        var (budget, reserve) = MachineBudget(ramBytes);
        if (budget <= 0)
            return new(PreflightVerdict.NotApplicable, "this machine's RAM could not be detected", null, 0, 0);
        var report = ScoutAnalyzer.Analyze(input, new ScoutOptions("preflight", budget, reserve, null, contextTokens));
        long? peak = report.Resources.HostWorkingSet.Bytes;
        return report.Resources.ExecutionDecision switch
        {
            "allowed" => new(PreflightVerdict.Allowed, report.Resources.Reason, peak, budget, reserve),
            "blocked" when peak is not null => new(PreflightVerdict.Blocked, report.Resources.Reason, peak, budget, reserve),
            _ => new(PreflightVerdict.Unknown, report.Resources.Reason, null, budget, reserve),
        };
    }

    /// <summary>Reads only the index of a local GGUF (no weights) and evaluates it.</summary>
    public static PreflightResult EvaluateFile(string path, int contextTokens, long? ramBytes = null)
    {
        if (!path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) || !File.Exists(path))
            return new(PreflightVerdict.NotApplicable, "not a local GGUF file", null, 0, 0);
        try
        {
            using var model = GgufModel.Open(path);
            var input = new ScoutInput(Path.GetFileName(path), new FileInfo(path).Length, model.Header.Version, model.Header.TensorCount,
                model.Header.MetadataKvCount, model.Metadata, model.Tensors);
            return EvaluateIndex(input, contextTokens, ramBytes);
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or NotSupportedException or UnauthorizedAccessException or ArgumentException or OverflowException)
        {
            return new(PreflightVerdict.NotApplicable, "the file could not be read: " + ex.Message, null, 0, 0);
        }
    }

    /// <summary>
    /// Evaluates a catalog entry's main GGUF WITHOUT downloading it: reads its index at the pinned commit with Range requests. Returns null when the entry is
    /// not a GGUF, or the network is not allowed or reachable (then there is simply no pre-download opinion, and the post-download check still runs).
    /// </summary>
    public static async Task<PreflightResult?> EvaluateCatalogEntryAsync(CatalogEntry entry, int contextTokens, ExternalHttpClient http, CancellationToken ct, long? ramBytes = null)
    {
        var main = entry.MainFile;
        if (!main.RepoPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)) return null;
        try
        {
            // Every GGUF of the bundle: a split model's weights are spread over all shards, so reading only the first would understate the need.
            string[] shards = entry.Files.Where(f => f.Repo == main.Repo && f.Revision == main.Revision && f.RepoPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)).Select(f => f.RepoPath).ToArray();
            var idx = await RemoteGgufReader.ReadAsync(http, main.Repo, main.Revision, shards, RemoteGgufReader.DefaultMaxIndexBytes, ct).ConfigureAwait(false);
            if (idx.Outcome != RemoteIndexOutcome.Complete || idx.Index is null) return null;
            var i = idx.Index;
            return EvaluateIndex(new ScoutInput(main.FileName, idx.ShardSizes.Sum(), i.Header.Version, i.Header.TensorCount, i.Header.MetadataKvCount, i.Metadata, i.Tensors),
                contextTokens, ramBytes);
        }
        catch (ExternalAccessDeniedException) { return null; }
        catch (HttpRequestException) { return null; }
    }

    /// <summary>The decision a command acts on: whether to proceed, and what to tell the user. <paramref name="ignore"/> is the explicit override flag.</summary>
    public static bool ShouldProceed(PreflightResult r, bool ignore, out string? message)
    {
        message = null;
        switch (r.Verdict)
        {
            case PreflightVerdict.Blocked when ignore:
                message = $"warning: this model is estimated to need about {Gib(r.EstimatedPeakBytes)} but this machine has {Gib(r.BudgetBytes)} (reserve {Gib(r.ReserveBytes)}); continuing because the preflight was overridden.";
                return true;
            case PreflightVerdict.Blocked:
                message = $"This model is estimated to need about {Gib(r.EstimatedPeakBytes)} of RAM for a CPU run at this context, but this machine has {Gib(r.BudgetBytes)} " +
                          $"(keeping {Gib(r.ReserveBytes)} free). It would likely run out of memory. Use a smaller quantisation or model (stingray scout -r <repo> --quants), " +
                          "lower the context size, or pass --ignore-preflight to try anyway.";
                return false;
            case PreflightVerdict.Unknown:
                message = "note: memory use could not be estimated for this model family, so the fit is not checked.";
                return true;
            default:
                return true;
        }
    }

    internal static string Gib(long? bytes) => bytes is long b ? $"{b / 1073741824.0:F1} GiB" : "?";
}
