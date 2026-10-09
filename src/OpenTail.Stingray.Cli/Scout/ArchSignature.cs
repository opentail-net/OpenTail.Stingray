using System.Security.Cryptography;
using System.Text;

namespace OpenTail.Stingray.Cli.Scout;

// Structural signatures of ADMITTED architectures, used to rank "which architecture we can already run does this file most resemble".
//
// Design constraints (they exist so that someone else, with a different llama.cpp and different local files, gets the same answer):
//   * A signature is OUR scout's output, checked in as data. Nothing is read from llama.cpp or from the user's model folder at ranking time.
//   * The STRUCTURE is independent of quantization, model size, file name and tensor order: dtype and absolute shapes are not recorded; only
//     tensor name patterns, tensor rank, layer coverage, structural feature ids and the metadata key vocabulary.
//   * ORIGIN is kept apart from structure. Structure says what the model looks like; origin says which files demonstrated it. Matching reads
//     only the structure, so provenance can grow (more files, repo, hashes) without ever changing a ranking.
//   * Contributors add snapshots with scout --emit-signature and load them with --signatures <dir>, with no rebuild.
//   * Ranking is explainable counts of differences, not a probability. Structural identity is evidence for a relabel/variant, never admission.

/// <summary>One tensor-name pattern in a signature (layer index replaced by '*').</summary>
public sealed record SigPattern(string Pattern, int Rank, string Coverage);

/// <summary>What a model family looks like. Contains nothing about where the evidence came from.</summary>
public sealed record SigStructure(
    IReadOnlyList<SigPattern> Patterns,
    // Size-independent feature ids (see SignatureBuilder.IsStructuralFeature).
    IReadOnlyList<string> Features,
    // Metadata keys under the architecture prefix, prefix removed. Informational; does not change the match verdict or the structure id.
    IReadOnlyList<string> MetadataKeys,
    string? TokenizerModel);

/// <summary>A checkpoint that demonstrated a structure. Every field except the file name and scout build may be null (unknown, never guessed).</summary>
public sealed record SigOrigin(
    string FileName,
    long? FileBytes,
    // SHA-256 of the exact file, so the evidence is pinned to one conversion/quantization. Null when it was not computed.
    string? Sha256,
    // Hugging Face repo id and revision, only when the contributor supplied them.
    string? SourceRepo,
    string? SourceRevision,
    // The scout build that produced the structure (its version identifies the rules used).
    string ScoutBuild,
    // Where Sha256 came from: null when computed locally from the file's bytes, "published_by_huggingface" when it is the Hub's claim (not verified here).
    string? Sha256Source = null);

public sealed record ArchSignature(
    int SchemaVersion,
    // Registry descriptor id this structure belongs to (always an Admitted architecture when emitted).
    string ArchitectureId,
    string DeclaredArchitecture,
    // SHA-256 over the canonical patterns + features: identical structures share an id, so duplicates are detected rather than guessed at.
    string StructureId,
    SigStructure Structure,
    IReadOnlyList<SigOrigin> Origins)
{
    public const int CurrentSchemaVersion = 2;
}

public static class SignatureBuilder
{
    /// <summary>
    /// Feature ids that describe structure, not size. Excluded on purpose: head-count derived ids (attn.gqa/mha/mqa differ between sizes of one
    /// family), rope.partial / rope.scaling (context-extension and head-size dependent), quant.* and tokenizer.* (not architecture).
    /// </summary>
    public static bool IsStructuralFeature(string id) =>
        id.StartsWith("ffn.", StringComparison.Ordinal) || id.StartsWith("recurrent.", StringComparison.Ordinal) ||
        id.StartsWith("mtp.", StringComparison.Ordinal) || id.StartsWith("multimodal.", StringComparison.Ordinal) ||
        id is "rope.multi_axis" or "attn.fused_qkv" or "attn.separate_qkv" or "attn.qkv_layout_mixed" or "attn.qk_norm"
            or "attn.mla_low_rank" or "attn.sparse_indexer" or "attn.head_count_per_layer" or "attn.kv_heads_per_layer";

    /// <summary>Builds a signature from an analysed file. Returns null (with a reason) unless the file resolves to an Admitted architecture.</summary>
    public static ArchSignature? TryBuild(ScoutInput input, ScoutReport report, SigOrigin origin, out string refusal)
    {
        refusal = "";
        var arch = report.Architecture;
        if (arch is null || !arch.Resolved || arch.DescriptorId is null)
        {
            refusal = "the file does not resolve to a registered architecture";
            return null;
        }
        if (arch.Status != nameof(AdmissionStatus.Admitted))
        {
            refusal = $"architecture '{arch.DescriptorId}' is {arch.Status}, not Admitted; only admitted architectures may be reference signatures";
            return null;
        }

        string declared = arch.Declared ?? arch.DescriptorId;
        string prefix = declared + ".";
        var keys = input.Metadata.Keys.Where(k => k.StartsWith(prefix, StringComparison.Ordinal)).Select(k => k[prefix.Length..])
            .Order(StringComparer.Ordinal).ToArray();
        var tok = ArchitectureTriage.ClassifyTokenizer(input.Metadata);

        var structure = new SigStructure(
            Patterns(input.Tensors),
            report.Findings.Select(f => f.Id).Where(IsStructuralFeature).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray(),
            keys, tok.Shape == TokenizerShape.NoModel ? null : tok.Model);
        return new ArchSignature(ArchSignature.CurrentSchemaVersion, arch.DescriptorId, declared, StructureId(structure), structure, [origin]);
    }

    /// <summary>Hash of exactly what matching reads (patterns and features), in a canonical text form.</summary>
    public static string StructureId(SigStructure s)
    {
        var sb = new StringBuilder();
        foreach (var p in s.Patterns.OrderBy(p => p.Pattern, StringComparer.Ordinal))
            sb.Append("P|").Append(p.Pattern).Append('|').Append(p.Rank).Append('|').Append(p.Coverage).Append('\n');
        foreach (string f in s.Features.Order(StringComparer.Ordinal))
            sb.Append("F|").Append(f).Append('\n');
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(sb.ToString()))).ToLowerInvariant();
    }

    /// <summary>
    /// Adds the origins of <paramref name="added"/> to <paramref name="existing"/> when both describe the same structure of the same
    /// architecture; an origin already present (same file name and hash) is not duplicated. Returns null with a reason when they differ.
    /// </summary>
    public static ArchSignature? TryMerge(ArchSignature existing, ArchSignature added, out string refusal)
    {
        refusal = "";
        if (existing.ArchitectureId != added.ArchitectureId || existing.StructureId != added.StructureId)
        {
            refusal = existing.ArchitectureId != added.ArchitectureId
                ? $"existing signature is for '{existing.ArchitectureId}', new file is '{added.ArchitectureId}'"
                : "the structure differs from the existing signature (different structure_id); write it to a new file instead";
            return null;
        }
        var origins = existing.Origins.ToList();
        foreach (var o in added.Origins)
            if (!origins.Any(x => x.FileName == o.FileName && x.Sha256 == o.Sha256)) origins.Add(o);
        return existing with { Origins = origins.OrderBy(o => o.FileName, StringComparer.Ordinal).ToArray() };
    }

    public static IReadOnlyList<SigPattern> Patterns(IReadOnlyList<GgufTensorInfo> tensors)
    {
        var layersByStack = ScoutAnalyzer.LayersByStack(tensors);
        return tensors.Select(t => (t, n: ScoutAnalyzer.Normalize(t.Name)))
            .GroupBy(x => x.n.Pattern, StringComparer.Ordinal)
            .OrderBy(g => g.Key, StringComparer.Ordinal)
            .Select(g =>
            {
                var first = g.First();
                string coverage = first.n.Stack is null ? "unlayered"
                    : g.Select(x => x.n.Layer).Distinct().Count() == layersByStack[first.n.Stack].Count ? "all" : "subset";
                return new SigPattern(g.Key, first.t.NDimensions, coverage);
            }).ToArray();
    }
}

/// <summary>Where signatures come from: embedded snapshots plus an optional contributor directory.</summary>
public static class SignatureStore
{
    private const string ResourceSuffix = ".signature.json";

    /// <summary>Null when the signature is usable; otherwise why not. Shared by both sources so stale or hand-edited data is never half-trusted.</summary>
    public static string? Problem(ArchSignature? sig)
    {
        if (sig is null) return "empty signature";
        if (sig.SchemaVersion != ArchSignature.CurrentSchemaVersion) return $"unsupported schema {sig.SchemaVersion} (expected {ArchSignature.CurrentSchemaVersion}); regenerate it";
        if (sig.Structure is null || sig.Structure.Patterns is null || sig.Structure.Features is null) return "no structure";
        if (sig.Origins is null || sig.Origins.Count == 0) return "no origin: a signature must say which file demonstrated it";
        if (string.IsNullOrWhiteSpace(sig.ArchitectureId)) return "no architecture id";
        if (SignatureBuilder.StructureId(sig.Structure) != sig.StructureId) return "structure_id does not match the structure (edited by hand, or built by a different scout version)";
        return null;
    }

    /// <summary>The shipped set. A stale or damaged embedded signature is a build defect, so it fails loudly instead of skewing a ranking.</summary>
    public static IReadOnlyList<ArchSignature> LoadEmbedded()
    {
        var asm = typeof(SignatureStore).Assembly;
        var list = new List<ArchSignature>();
        foreach (string name in asm.GetManifestResourceNames().Where(n => n.EndsWith(ResourceSuffix, StringComparison.Ordinal)).Order(StringComparer.Ordinal))
        {
            using var s = asm.GetManifestResourceStream(name)!;
            var sig = JsonSerializer.Deserialize(s, ScoutJsonContext.Default.ArchSignature);
            if (Problem(sig) is { } why)
                throw new InvalidOperationException($"Embedded scout signature '{name}' is unusable: {why}.");
            list.Add(sig!);
        }
        return list;
    }

    /// <summary>Loads every <c>*.signature.json</c> in a directory; a file that does not parse or validate is reported, not silently skipped.</summary>
    public static IReadOnlyList<ArchSignature> LoadDirectory(string dir, List<string> problems)
    {
        var list = new List<ArchSignature>();
        foreach (string file in Directory.EnumerateFiles(dir, "*" + ResourceSuffix).Order(StringComparer.Ordinal))
        {
            try
            {
                var sig = JsonSerializer.Deserialize(File.ReadAllText(file), ScoutJsonContext.Default.ArchSignature);
                if (Problem(sig) is { } why) problems.Add($"{Path.GetFileName(file)}: {why}");
                else list.Add(sig!);
            }
            catch (JsonException ex) { problems.Add($"{Path.GetFileName(file)}: {ex.Message}"); }
        }
        return list;
    }
}
public static class SignatureMatcher
{
    private const int MaxDifferingLines = 12;

    /// <summary>
    /// Ranks snapshots against a file. One candidate per architecture (its closest structure; origins of equal structures are pooled). Order:
    /// fewest differences, then id; the count is the total of pattern/rank/coverage/feature differences, so "0" means structurally identical,
    /// not "equivalent". Snapshots whose architecture is no longer registered are dropped; status is read live from the registry.
    /// </summary>
    public static IReadOnlyList<ArchitectureCandidate> Rank(IReadOnlyList<GgufTensorInfo> tensors, IEnumerable<string> features,
        string? tokenizerModel, IEnumerable<ArchSignature> snapshots, int top = 3)
    {
        var mine = SignatureBuilder.Patterns(tensors).ToDictionary(p => p.Pattern, StringComparer.Ordinal);
        var myFeatures = features.Where(SignatureBuilder.IsStructuralFeature).ToHashSet(StringComparer.Ordinal);

        // Pool origins of identical structures first, so three files demonstrating one structure are one candidate with three origins.
        var pooled = snapshots.GroupBy(s => (s.ArchitectureId, s.StructureId))
            .Select(g => (First: g.First(), Origins: g.SelectMany(x => x.Origins).Select(OriginLabel).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray()));

        var best = new Dictionary<string, ArchitectureCandidate>(StringComparer.Ordinal);
        foreach (var (s, origins) in pooled)
        {
            var desc = ArchitectureRegistry.Find(s.ArchitectureId);
            if (desc is null) continue;

            var diffs = new List<string>();
            var theirs = s.Structure.Patterns.ToDictionary(p => p.Pattern, StringComparer.Ordinal);
            int shared = 0;
            foreach (var p in s.Structure.Patterns)
            {
                if (!mine.TryGetValue(p.Pattern, out var m)) diffs.Add($"missing tensor pattern {p.Pattern}");
                else if (m.Rank != p.Rank) diffs.Add($"{p.Pattern}: rank {m.Rank}, reference {p.Rank}");
                else if (m.Coverage != p.Coverage) diffs.Add($"{p.Pattern}: layer coverage {m.Coverage}, reference {p.Coverage}");
                else shared++;
            }
            diffs.AddRange(mine.Keys.Where(k => !theirs.ContainsKey(k)).Order(StringComparer.Ordinal).Select(k => $"extra tensor pattern {k}"));
            var refFeatures = s.Structure.Features.ToHashSet(StringComparer.Ordinal);
            diffs.AddRange(refFeatures.Except(myFeatures).Order(StringComparer.Ordinal).Select(f => $"feature absent here: {f}"));
            diffs.AddRange(myFeatures.Except(refFeatures).Order(StringComparer.Ordinal).Select(f => $"feature not in reference: {f}"));

            var matching = new List<string> { $"{shared} of {s.Structure.Patterns.Count} reference tensor patterns shared exactly (name, rank, layer coverage)" };
            var sharedFeatures = myFeatures.Intersect(refFeatures).Order(StringComparer.Ordinal).ToArray();
            if (sharedFeatures.Length > 0) matching.Add("shared features: " + string.Join(", ", sharedFeatures));
            if (tokenizerModel is not null && tokenizerModel == s.Structure.TokenizerModel) matching.Add($"same tokenizer model: {tokenizerModel}");

            var cand = new ArchitectureCandidate(desc.Id, desc.Status.ToString(), diffs.Count == 0 ? "identical_structure" : "differs",
                diffs.Count, s.StructureId, origins, matching,
                diffs.Take(MaxDifferingLines).Concat(diffs.Count > MaxDifferingLines ? [$"... and {diffs.Count - MaxDifferingLines} more"] : []).ToArray());
            if (!best.TryGetValue(desc.Id, out var cur) || Better(cand, cur)) best[desc.Id] = cand;
        }
        return best.Values.OrderBy(c => c.DifferenceCount).ThenBy(c => c.Id, StringComparer.Ordinal).Take(top).ToArray();
    }

    private static string OriginLabel(SigOrigin o) => o.Sha256 is { Length: >= 12 } h ? $"{o.FileName} (sha256 {h[..12]}...)" : o.FileName;

    private static bool Better(ArchitectureCandidate a, ArchitectureCandidate b) =>
        a.DifferenceCount != b.DifferenceCount ? a.DifferenceCount < b.DifferenceCount : string.CompareOrdinal(a.StructureId, b.StructureId) < 0;
}
