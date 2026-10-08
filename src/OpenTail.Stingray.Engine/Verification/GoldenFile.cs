using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace OpenTail.Stingray.Engine.Verification;

/// <summary>
/// A recorded reference result for one model file (docs/2-coverage/2026-10-08-golden-parity-and-admission-tooling-plan.md).
/// Identity and evidence only: a bare file name, a hash, the reference engine's build and settings, token ids. It never holds a path to
/// a machine, so it can be committed. The prompt is stored as token ids, so the forward pass is checked in isolation from our tokenizer.
/// </summary>
public sealed record GoldenFile
{
    public int Schema { get; init; } = 1;
    public string Architecture { get; init; } = "";
    /// <summary>Provenance prose (checkpoint source, licence, caveats). Scanned for machine-specific text like everything else.</summary>
    public string? Notes { get; init; }
    public GoldenModel Model { get; init; } = new();
    public GoldenReference Reference { get; init; } = new();
    public IReadOnlyList<GoldenCase> Cases { get; init; } = [];
    /// <summary>Named overrides for the rare model whose receipt needs a non-default engine setting.</summary>
    public GoldenEngineSettings? EngineSettings { get; init; }
    /// <summary>
    /// Hyperparameter guards the old parity classes asserted by hand (e.g. <c>ropeDim</c> = 16, <c>numExperts</c> = 8, <c>hasFfnBias</c> = true): ModelHyperparams property name
    /// (case-insensitive) to the expected invariant string (<c>null</c> for null). They pin what an architecture's descriptor must resolve for this very file.
    /// </summary>
    public Dictionary<string, string>? ExpectedHyperparameters { get; init; }

    public const int CurrentSchema = 1;

    public static GoldenFile Parse(string json)
    {
        var g = JsonSerializer.Deserialize(json, GoldenJsonContext.Default.GoldenFile)
                ?? throw new InvalidDataException("Golden file is empty.");
        if (g.Schema != CurrentSchema)
            throw new InvalidDataException($"Golden schema {g.Schema} is not supported (this build reads schema {CurrentSchema}).");
        return g;
    }

    public static GoldenFile Load(string path) => Parse(File.ReadAllText(path));

    public string ToJson() => JsonSerializer.Serialize(this, GoldenJsonContext.Default.GoldenFile);

    /// <summary>Writes the golden, refusing when it would commit machine-specific text (see <see cref="FindMachineSpecificText"/>).</summary>
    public void Save(string path)
    {
        string json = ToJson();
        var problems = FindMachineSpecificText(json);
        if (problems.Count > 0)
            throw new InvalidOperationException("Golden contains machine-specific text and was not written: " + string.Join("; ", problems));
        File.WriteAllText(path, json + Environment.NewLine);
    }

    private static readonly Regex s_driveLetter = new(@"(?<![A-Za-z0-9])[A-Za-z]:(\\\\|\\|/)", RegexOptions.Compiled);
    private static readonly Regex s_unc = new(@"\\\\\\\\[A-Za-z0-9_.$-]+\\\\", RegexOptions.Compiled);

    /// <summary>
    /// Findings that would leak the author's machine into git: drive-letter paths, UNC paths, <c>/Users/</c> and <c>/home/</c>, and the
    /// current user or machine name. Empty means clean. Used by <see cref="Save"/>, by capture, and by a Fast test over every checked-in golden.
    /// </summary>
    public static IReadOnlyList<string> FindMachineSpecificText(string text)
    {
        var found = new List<string>();
        if (s_driveLetter.IsMatch(text)) found.Add("drive-letter path");
        if (s_unc.IsMatch(text)) found.Add("UNC path");
        if (text.Contains("/Users/", StringComparison.Ordinal) || text.Contains("/home/", StringComparison.Ordinal))
            found.Add("/Users/ or /home/ path");
        foreach (var (what, name) in new[] { ("user name", Environment.UserName), ("machine name", Environment.MachineName) })
            if (name.Length >= 3 && text.Contains(name, StringComparison.OrdinalIgnoreCase))
                found.Add(what);
        return found;
    }
}

public sealed record GoldenModel
{
    /// <summary>Bare file name (never a path).</summary>
    public string FileName { get; init; } = "";
    public long SizeBytes { get; init; }
    /// <summary>Lowercase hex SHA-256 of the verified file; null only for receipts migrated before hashes were captured.</summary>
    public string? Sha256 { get; init; }
    /// <summary>Hugging Face repo id (or other origin) the file was obtained from.</summary>
    public string? Source { get; init; }
}

public sealed record GoldenReference
{
    /// <summary><c>llama-server</c>, or <c>legacy-receipt</c> for arrays migrated from the old test classes.</summary>
    public string Engine { get; init; } = "";
    public string? Build { get; init; }
    /// <summary>Whitelisted server settings, e.g. <c>-ngl 0 -c 512 -t 4</c>; never a raw command line.</summary>
    public string? Settings { get; init; }
    public GoldenSampling Sampling { get; init; } = new();
    public string? CapturedUtc { get; init; }
}

public sealed record GoldenSampling
{
    public double Temperature { get; init; }
    public int TopK { get; init; } = 1;
    public int Seed { get; init; }
    public double RepeatPenalty { get; init; } = 1.0;
    public bool CachePrompt { get; init; }
}

public sealed record GoldenCase
{
    public string Name { get; init; } = "";
    public string? PromptText { get; init; }
    /// <summary>Authoritative prompt, including BOS when the model wants one.</summary>
    public int[] PromptTokens { get; init; } = [];
    /// <summary>How <see cref="PromptTokens"/> were produced (e.g. <c>llama-tokenize</c>), so a tokenizer can be checked against the oracle separately.</summary>
    public string? TokenizerOracle { get; init; }
    public int NPredict { get; init; }
    /// <summary>The reference continuation.</summary>
    public int[] Tokens { get; init; } = [];
    public string? Text { get; init; }
    /// <summary><c>free</c> (our greedy run is compared token for token until it diverges) or <c>teacherForced</c> (the reference tokens are fed and every position is compared).</summary>
    public string Mode { get; init; } = "free";
    /// <summary>
    /// The reference engine's own confidence per generated token: top-1 minus top-2 log-probability (nats). Where it is small the reference itself was nearly
    /// undecided, so a mismatch there is a near-tie, not evidence against us. Null for receipts migrated without it.
    /// </summary>
    public double[]? Margins { get; init; }
    /// <summary>Evidence requirement: at least this many positions must match where the reference itself was confident (margin at or above the confident threshold). 0 = none.</summary>
    public int MinConfident { get; init; }
}

public sealed record GoldenEngineSettings
{
    /// <summary>Run prefill with Q8 activations (DeepSeek2's receipt was taken that way).</summary>
    public bool Q8Prefill { get; init; }
}

[JsonSourceGenerationOptions(WriteIndented = true, PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(GoldenFile))]
internal partial class GoldenJsonContext : JsonSerializerContext
{
}

/// <summary>Applies <see cref="GoldenEngineSettings"/> to the process-wide kernel switches for the duration of a run, restoring them after.</summary>
public sealed class GoldenEngineSettingsScope : IDisposable
{
    private readonly bool _savedQ8Prefill = OpenTail.Stingray.Cpu.SimdKernels.Q8PrefillEnabled;

    public GoldenEngineSettingsScope(GoldenEngineSettings? settings)
    {
        if (settings is { Q8Prefill: true }) OpenTail.Stingray.Cpu.SimdKernels.Q8PrefillEnabled = true;
    }

    public void Dispose() => OpenTail.Stingray.Cpu.SimdKernels.Q8PrefillEnabled = _savedQ8Prefill;
}
