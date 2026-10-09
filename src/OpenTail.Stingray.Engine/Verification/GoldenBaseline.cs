using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace OpenTail.Stingray.Engine.Verification;

/// <summary>
/// A host-level verification and performance baseline recording the outcomes of all checked-in goldens
/// (docs/2-coverage/2026-10-08-golden-parity-and-admission-tooling-plan.md Phase 7).
/// </summary>
public sealed record GoldenBaselineFile
{
    public int Schema { get; init; } = 2;
    public string Host { get; init; } = Environment.MachineName;
    public string OsDescription { get; init; } = RuntimeInformation.OSDescription;
    public string ProcessArchitecture { get; init; } = RuntimeInformation.ProcessArchitecture.ToString();
    public int ProcessorCount { get; init; } = Environment.ProcessorCount;
    public string RuntimeDescription { get; init; } = RuntimeInformation.FrameworkDescription;
    public string TimestampUtc { get; init; } = DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture);
    public string? StingrayVersion { get; init; }
    public IReadOnlyList<GoldenBaselineEntry> Entries { get; init; } = [];

    public const int CurrentSchema = 2;

    public static GoldenBaselineFile Parse(string json)
    {
        var f = JsonSerializer.Deserialize(json, GoldenJsonContext.Default.GoldenBaselineFile)
                ?? throw new InvalidDataException("Baseline file is empty.");
        if (f.Schema < 1 || f.Schema > CurrentSchema)
            throw new InvalidDataException($"Baseline schema {f.Schema} is not supported (this build reads schema 1 and {CurrentSchema}).");
        return f;
    }

    public static GoldenBaselineFile Load(string path) => Parse(File.ReadAllText(path));

    public string ToJson() => JsonSerializer.Serialize(this, GoldenJsonContext.Default.GoldenBaselineFile);

    public void Save(string path) => File.WriteAllText(path, ToJson() + Environment.NewLine);
}

public sealed record GoldenBaselineEntry
{
    public string Architecture { get; init; } = "";
    public string GoldenFile { get; init; } = "";
    public string? GoldenSha256 { get; init; }
    public string ModelFile { get; init; } = "";
    public string? ModelSha256 { get; init; }
    public string PinStatus { get; init; } = "";
    public string Verdict { get; init; } = "";
    public bool Passed { get; init; }
    public int ComparedTokens { get; init; }
    public int MatchedTokens { get; init; }
    public int DecodeSteps { get; init; }
    public double? DecodeTokensPerSecond { get; init; }
    public double? PrefillTokensPerSecond { get; init; }
    public double ElapsedSeconds { get; init; }
    public float? StepwiseMaxAbsDiff { get; init; }
    public string? Detail { get; init; }
}

public enum BaselineDiffStatus
{
    Unchanged,
    Regression,
    Improvement,
    SpeedDrop,
    SpeedGain,
    NewEntry,
    MissingInCurrent,
    IdentityChanged
}

public sealed record GoldenBaselineDiff(
    string Architecture,
    string GoldenFile,
    string ModelFile,
    string? PriorVerdict,
    string CurrentVerdict,
    double? PriorSpeed,
    double? CurrentSpeed,
    BaselineDiffStatus Status,
    string Description
);

public static class GoldenBaselineComparator
{
    public static IReadOnlyList<GoldenBaselineDiff> Compare(
        GoldenBaselineFile baseline,
        GoldenBaselineFile current,
        double speedTolerancePct = 0.15)
    {
        ArgumentNullException.ThrowIfNull(baseline);
        ArgumentNullException.ThrowIfNull(current);

        var diffs = new List<GoldenBaselineDiff>();

        var baselineDuplicates = baseline.Entries
            .GroupBy(e => string.IsNullOrEmpty(e.GoldenFile) ? e.Architecture : e.GoldenFile, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault(g => g.Count() > 1);
        if (baselineDuplicates is not null)
            throw new InvalidDataException($"Prior baseline contains duplicate entry for '{baselineDuplicates.Key}'.");

        var currentDuplicates = current.Entries
            .GroupBy(e => string.IsNullOrEmpty(e.GoldenFile) ? e.Architecture : e.GoldenFile, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault(g => g.Count() > 1);
        if (currentDuplicates is not null)
            throw new InvalidDataException($"Current baseline contains duplicate entry for '{currentDuplicates.Key}'.");

        var baselineMap = baseline.Entries.ToDictionary(
            e => string.IsNullOrEmpty(e.GoldenFile) ? e.Architecture : e.GoldenFile,
            StringComparer.OrdinalIgnoreCase);
        var currentMap = current.Entries.ToDictionary(
            e => string.IsNullOrEmpty(e.GoldenFile) ? e.Architecture : e.GoldenFile,
            StringComparer.OrdinalIgnoreCase);

        bool envMatches = string.Equals(baseline.OsDescription, current.OsDescription, StringComparison.OrdinalIgnoreCase)
            && string.Equals(baseline.ProcessArchitecture, current.ProcessArchitecture, StringComparison.OrdinalIgnoreCase)
            && baseline.ProcessorCount == current.ProcessorCount;

        bool hasLegacySchema = baseline.Schema < 2 || current.Schema < 2;

        foreach (var (key, curr) in currentMap)
        {
            if (!baselineMap.TryGetValue(key, out var baseEntry))
            {
                diffs.Add(new GoldenBaselineDiff(
                    curr.Architecture,
                    curr.GoldenFile,
                    curr.ModelFile,
                    null,
                    curr.Verdict,
                    null,
                    curr.DecodeTokensPerSecond,
                    BaselineDiffStatus.NewEntry,
                    "New model in current run"));
                continue;
            }

            bool basePass = baseEntry.Passed;
            bool currPass = curr.Passed;
            BaselineDiffStatus status = BaselineDiffStatus.Unchanged;
            string desc = "Unchanged";

            bool modelHashChanged = !string.IsNullOrEmpty(baseEntry.ModelSha256)
                && !string.IsNullOrEmpty(curr.ModelSha256)
                && !string.Equals(baseEntry.ModelSha256, curr.ModelSha256, StringComparison.OrdinalIgnoreCase);

            bool goldenHashChanged = !string.IsNullOrEmpty(baseEntry.GoldenSha256)
                && !string.IsNullOrEmpty(curr.GoldenSha256)
                && !string.Equals(baseEntry.GoldenSha256, curr.GoldenSha256, StringComparison.OrdinalIgnoreCase);

            bool modelFileChanged = !string.IsNullOrEmpty(baseEntry.ModelFile)
                && !string.IsNullOrEmpty(curr.ModelFile)
                && !string.Equals(baseEntry.ModelFile, curr.ModelFile, StringComparison.OrdinalIgnoreCase);

            bool identityChanged = modelHashChanged || goldenHashChanged || modelFileChanged;

            if (basePass && !currPass)
            {
                status = BaselineDiffStatus.Regression;
                desc = $"Regression: was {baseEntry.Verdict}, now {curr.Verdict}"
                    + (modelHashChanged ? " (model hash changed)" : goldenHashChanged ? " (golden hash changed)" : modelFileChanged ? " (model file changed)" : "");
            }
            else if (!basePass && currPass)
            {
                status = BaselineDiffStatus.Improvement;
                desc = $"Improvement: was {baseEntry.Verdict}, now {curr.Verdict}"
                    + (modelHashChanged ? " (model hash changed)" : goldenHashChanged ? " (golden hash changed)" : modelFileChanged ? " (model file changed)" : "");
            }
            else if (identityChanged)
            {
                status = BaselineDiffStatus.IdentityChanged;
                if (modelHashChanged)
                {
                    string h1 = baseEntry.ModelSha256!.Length > 8 ? baseEntry.ModelSha256[..8] : baseEntry.ModelSha256;
                    string h2 = curr.ModelSha256!.Length > 8 ? curr.ModelSha256[..8] : curr.ModelSha256;
                    desc = $"Model hash changed ({h1} -> {h2}); speed not comparable";
                }
                else if (goldenHashChanged)
                {
                    string h1 = baseEntry.GoldenSha256!.Length > 8 ? baseEntry.GoldenSha256[..8] : baseEntry.GoldenSha256;
                    string h2 = curr.GoldenSha256!.Length > 8 ? curr.GoldenSha256[..8] : curr.GoldenSha256;
                    desc = $"Golden definition changed ({h1} -> {h2}); speed not comparable";
                }
                else
                {
                    desc = $"Model file changed ({baseEntry.ModelFile} -> {curr.ModelFile}); speed not comparable";
                }
            }
            else if (hasLegacySchema)
            {
                status = BaselineDiffStatus.Unchanged;
                desc = "Historical baseline (schema 1) uses setup-inclusive throughput; regenerate baseline for decode speed comparison";
            }
            else if (baseEntry.DecodeTokensPerSecond is { } bSpeed && curr.DecodeTokensPerSecond is { } cSpeed && bSpeed > 0 && cSpeed > 0)
            {
                double ratio = cSpeed / bSpeed;
                string envNote = !envMatches ? " (different environment/host)" : "";
                if (ratio < (1.0 - speedTolerancePct))
                {
                    status = BaselineDiffStatus.SpeedDrop;
                    desc = $"Speed drop: {bSpeed:F1} -> {cSpeed:F1} t/s ({((ratio - 1.0) * 100):F1}%){envNote}";
                }
                else if (ratio > (1.0 + speedTolerancePct))
                {
                    status = BaselineDiffStatus.SpeedGain;
                    desc = $"Speed gain: {bSpeed:F1} -> {cSpeed:F1} t/s (+{((ratio - 1.0) * 100):F1}%){envNote}";
                }
            }

            diffs.Add(new GoldenBaselineDiff(
                curr.Architecture,
                curr.GoldenFile,
                curr.ModelFile,
                baseEntry.Verdict,
                curr.Verdict,
                baseEntry.DecodeTokensPerSecond,
                curr.DecodeTokensPerSecond,
                status,
                desc));
        }

        foreach (var (key, baseEntry) in baselineMap)
        {
            if (!currentMap.ContainsKey(key))
            {
                diffs.Add(new GoldenBaselineDiff(
                    baseEntry.Architecture,
                    baseEntry.GoldenFile,
                    baseEntry.ModelFile,
                    baseEntry.Verdict,
                    "Missing",
                    baseEntry.DecodeTokensPerSecond,
                    null,
                    BaselineDiffStatus.MissingInCurrent,
                    "Model present in baseline but missing in current run"));
            }
        }

        return diffs;
    }
}

