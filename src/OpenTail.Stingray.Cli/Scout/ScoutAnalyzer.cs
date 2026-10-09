using System.Globalization;
using System.Text;

namespace OpenTail.Stingray.Cli.Scout;

/// <summary>Everything scout needs from a checkpoint: the index and metadata only, never tensor values.</summary>
public sealed record ScoutInput(
    string FileName,
    long? FileBytes,
    uint GgufVersion,
    ulong HeaderTensorCount,
    ulong HeaderMetadataKvCount,
    IReadOnlyDictionary<string, object> Metadata,
    IReadOnlyList<GgufTensorInfo> Tensors);

public sealed record ScoutOptions(string BuildId, long? BudgetBytes = null, long ReserveBytes = ScoutOptions.DefaultReserveBytes)
{
    public const long DefaultReserveBytes = 8L << 30;
}

/// <summary>
/// Static, read-only analysis of a GGUF index. Pure function of <see cref="ScoutInput"/>: it constructs no forward pass and the tensor
/// source it hands the architecture probe throws if anything tries to read tensor values, so "scout never loads weights" is enforced
/// rather than promised.
/// </summary>
public static class ScoutAnalyzer
{
    private const int MaxMetadataValueChars = 120;
    private const int MaxEvidenceItems = 8;

    public static ScoutReport Analyze(ScoutInput input, ScoutOptions options)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(options);

        var blockers = new List<ScoutBlocker>();
        var findings = new List<ScoutFinding>();

        var tensors = SummarizeTensors(input.Tensors, blockers);
        var tokenizer = SummarizeTokenizer(input.Metadata);
        var metadata = new MetadataSummary(SummarizeMetadata(input.Metadata), tokenizer);

        var arch = ResolveArchitecture(input, blockers);
        CheckDTypes(input.Tensors, blockers);
        CheckTokenizer(tokenizer, input.Metadata, findings, blockers);
        findings.AddRange(ScoutFeatures.Evaluate(input.Metadata, input.Tensors));

        int shardCount = input.Tensors.Count == 0 ? 1 : input.Tensors.Max(t => t.ShardIndex) + 1;
        var artifact = new ArtifactInfo(input.FileName, input.FileBytes, input.GgufVersion, input.HeaderTensorCount,
            input.HeaderMetadataKvCount, shardCount, "not_computed");

        var resources = Preflight(input, tensors, options);

        // Deterministic order: confirmed before suspected, then by id.
        var orderedBlockers = blockers.OrderBy(b => b.Kind).ThenBy(b => b.Id, StringComparer.Ordinal).ToArray();
        var orderedFindings = findings.OrderBy(f => f.Id, StringComparer.Ordinal).ToArray();

        bool confirmed = orderedBlockers.Any(b => b.Kind == BlockerKind.Confirmed);
        var stages = new List<StageReceipt>
        {
            new("0_artifact", StageState.Passed, "GGUF header, metadata and tensor index read."),
            new("1_static_contract", confirmed ? StageState.Failed : StageState.Passed,
                confirmed ? "One or more confirmed blockers (see blockers)." : "No confirmed blockers from static checks. This is not admission."),
            new("2_feasibility", resources.ExecutionDecision == "allowed" ? StageState.Passed : StageState.Blocked, resources.Reason),
            new("3_smoke", StageState.NotRun, "Opt-in; not part of static scout."),
            new("4_internal_consistency", StageState.NotRun, "Opt-in; not part of static scout."),
            new("5_independent_reference", StageState.NotRun, "Use capture-golden / verify-goldens; scout does not run them."),
            new("6_admission_readiness", StageState.NotRun, "A human or coding agent decides admission from the evidence set."),
        };

        return new ScoutReport(
            ScoutReport.CurrentSchemaVersion, options.BuildId, artifact, metadata, tensors, arch,
            orderedFindings, orderedBlockers, resources, NextActions(input, arch, orderedBlockers), stages);
    }

    /// <summary>A report for a file that could not be opened as GGUF: artifact stage Failed, nothing else invented.</summary>
    public static ScoutReport Unreadable(string fileName, long? fileBytes, string detail, ScoutOptions options)
    {
        var blocker = new ScoutBlocker("scout.artifact.unreadable", BlockerKind.Confirmed, detail, []);
        return new ScoutReport(
            ScoutReport.CurrentSchemaVersion, options.BuildId,
            new ArtifactInfo(fileName, fileBytes, null, null, null, null, "not_computed"),
            null, null, null, [], [blocker],
            new ResourcePreflight(fileBytes, null, Unknown("file unreadable"), Unknown("file unreadable"), options.BudgetBytes,
                options.ReserveBytes, "not_assessed", "File could not be read."),
            [],
            [new StageReceipt("0_artifact", StageState.Failed, detail)]);
    }

    // ── tensors ────────────────────────────────────────────────────────────────

    private static readonly HashSet<string> s_layerTokens = new(StringComparer.Ordinal) { "blk", "layers", "layer", "block", "blocks" };

    /// <summary>Replaces the index segment after a layer token with '*'. Returns the pattern, the layer-stack prefix and the index.</summary>
    internal static (string Pattern, string? Stack, int Layer) Normalize(string name)
    {
        var parts = name.Split('.');
        for (int i = 0; i + 1 < parts.Length; i++)
        {
            if (s_layerTokens.Contains(parts[i]) && int.TryParse(parts[i + 1], NumberStyles.None, CultureInfo.InvariantCulture, out int layer))
            {
                string stack = string.Join('.', parts[..(i + 1)]);
                parts[i + 1] = "*";
                return (string.Join('.', parts), stack, layer);
            }
        }
        return (name, null, -1);
    }

    private static TensorSummary SummarizeTensors(IReadOnlyList<GgufTensorInfo> tensors, List<ScoutBlocker> blockers)
    {
        var sizes = new long[tensors.Count];
        var overflow = new List<string>();
        for (int i = 0; i < tensors.Count; i++)
        {
            try { sizes[i] = tensors[i].ByteSize; }
            catch (OverflowException) { overflow.Add(tensors[i].Name); }
        }
        if (overflow.Count > 0)
            blockers.Add(new ScoutBlocker("scout.tensor.size_overflow", BlockerKind.Confirmed,
                "Tensor dimensions overflow when computing byte size; the index is malformed.",
                overflow.Order(StringComparer.Ordinal).Take(MaxEvidenceItems).Select(n => new EvidenceItem("tensor", n, null)).ToArray()));

        var byDType = tensors.Select((t, i) => (t, size: sizes[i]))
            .GroupBy(x => x.t.DType.ToString())
            .OrderBy(g => g.Key, StringComparer.Ordinal)
            .Select(g => new DTypeBucket(g.Key, g.Count(), g.Sum(x => x.size)))
            .ToArray();

        var groups = tensors.Select(t => (t, norm: Normalize(t.Name)))
            .GroupBy(x => x.norm.Pattern, StringComparer.Ordinal)
            .OrderBy(g => g.Key, StringComparer.Ordinal)
            .ToArray();

        // Layer sets per layer stack (e.g. "blk" and a vision tower's "v.blk" are separate stacks).
        var layersByStack = new Dictionary<string, SortedSet<int>>(StringComparer.Ordinal);
        foreach (var (_, norm) in tensors.Select(t => (t, Normalize(t.Name))))
        {
            if (norm.Stack is null) continue;
            if (!layersByStack.TryGetValue(norm.Stack, out var set)) layersByStack[norm.Stack] = set = [];
            set.Add(norm.Layer);
        }

        var patterns = new List<TensorPattern>();
        var irregularities = new List<TensorIrregularity>();
        foreach (var g in groups)
        {
            var first = g.First().t;
            string? stack = g.First().norm.Stack;
            var shapes = g.Select(x => Shape(x.t)).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray();
            var dtypes = g.Select(x => x.t.DType.ToString()).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray();
            string? layers = null;
            if (stack is not null)
            {
                var present = new SortedSet<int>(g.Select(x => x.norm.Layer));
                layers = Ranges(present);
                var all = layersByStack[stack];
                if (present.Count != all.Count)
                    irregularities.Add(new TensorIrregularity(g.Key, "partial_layers",
                        $"present in {present.Count} of {all.Count} layers of '{stack}' (layers {layers}; stack spans {Ranges(all)})"));
            }
            if (dtypes.Length > 1)
                irregularities.Add(new TensorIrregularity(g.Key, "dtype_varies", "dtypes: " + string.Join(", ", dtypes)));
            if (shapes.Length > 1)
                irregularities.Add(new TensorIrregularity(g.Key, "shape_varies",
                    $"{shapes.Length} distinct shapes: " + string.Join(", ", shapes.Take(3)) + (shapes.Length > 3 ? ", ..." : "")));
            patterns.Add(new TensorPattern(g.Key, g.Count(), layers, dtypes, Shape(first), shapes.Length));
        }

        int? layerCount = layersByStack.TryGetValue("blk", out var blk) ? blk.Count : null;
        return new TensorSummary((ulong)tensors.Count, sizes.Sum(), layerCount, byDType, patterns,
            irregularities.OrderBy(x => x.Pattern, StringComparer.Ordinal).ThenBy(x => x.Kind, StringComparer.Ordinal).ToArray());
    }

    private static string Shape(GgufTensorInfo t) => "[" + string.Join(",", t.Dimensions.Take(t.NDimensions)) + "]";

    internal static string Ranges(IEnumerable<int> sorted)
    {
        var sb = new StringBuilder();
        int? start = null, prev = null;
        void Flush()
        {
            if (start is null) return;
            if (sb.Length > 0) sb.Append(',');
            sb.Append(start == prev ? start.Value.ToString(CultureInfo.InvariantCulture) : $"{start}-{prev}");
        }
        foreach (int v in sorted)
        {
            if (prev is int p && v == p + 1) { prev = v; continue; }
            Flush();
            start = prev = v;
        }
        Flush();
        return sb.ToString();
    }

    // ── metadata / tokenizer ───────────────────────────────────────────────────

    private static IReadOnlyList<MetadataEntry> SummarizeMetadata(IReadOnlyDictionary<string, object> metadata) =>
        metadata.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => Describe(kv.Key, kv.Value)).ToArray();

    private static MetadataEntry Describe(string key, object value)
    {
        switch (value)
        {
            case string s:
                return new MetadataEntry(key, "string", s.Length <= MaxMetadataValueChars ? s : s[..MaxMetadataValueChars] + $"... (length {s.Length})");
            case bool b:
                return new MetadataEntry(key, "bool", b ? "true" : "false");
            case Array a:
                return new MetadataEntry(key, "array", $"length {a.Length}");
            case float f:
                return new MetadataEntry(key, "float", f.ToString("R", CultureInfo.InvariantCulture));
            case double d:
                return new MetadataEntry(key, "float", d.ToString("R", CultureInfo.InvariantCulture));
            case IConvertible c:
                return new MetadataEntry(key, value.GetType().Name.ToLowerInvariant(), c.ToString(CultureInfo.InvariantCulture));
            case System.Collections.ICollection col:
                return new MetadataEntry(key, "array", $"length {col.Count}");
            default:
                return new MetadataEntry(key, value.GetType().Name.ToLowerInvariant(), "(unprintable)");
        }
    }

    private static long? AsLong(IReadOnlyDictionary<string, object> m, string key) =>
        m.TryGetValue(key, out var v) && v is IConvertible c && v is not string and not bool ? Convert.ToInt64(c, CultureInfo.InvariantCulture) : null;

    private static TokenizerSummary SummarizeTokenizer(IReadOnlyDictionary<string, object> m)
    {
        long? vocab = m.TryGetValue("tokenizer.ggml.tokens", out var t) && t is Array ta ? ta.Length
            : m.TryGetValue("tokenizer.ggml.tokens", out t) && t is System.Collections.ICollection tc ? tc.Count : null;
        return new TokenizerSummary(
            m.TryGetValue("tokenizer.ggml.model", out var model) ? Convert.ToString(model, CultureInfo.InvariantCulture) : null,
            m.ContainsKey("tokenizer.ggml.merges"),
            m.ContainsKey("tokenizer.ggml.scores"),
            m.ContainsKey("tokenizer.ggml.token_type"),
            m.ContainsKey("tokenizer.chat_template"),
            vocab,
            AsLong(m, "tokenizer.ggml.bos_token_id"),
            AsLong(m, "tokenizer.ggml.eos_token_id"),
            m.TryGetValue("tokenizer.ggml.add_bos_token", out var ab) && ab is bool b ? b : null);
    }

    private static void CheckTokenizer(TokenizerSummary tok, IReadOnlyDictionary<string, object> m, List<ScoutFinding> findings, List<ScoutBlocker> blockers)
    {
        // Shape classification is shared with `admit-arch` (ArchitectureTriage) so the two commands cannot disagree.
        var triage = ArchitectureTriage.ClassifyTokenizer(m);
        if (triage.Shape == TokenizerShape.NoModel)
        {
            blockers.Add(new ScoutBlocker("scout.tokenizer.no_model", BlockerKind.Suspected,
                "No tokenizer.ggml.model key: a tokenizer cannot be built from this file alone (may be a non-text model or a sidecar tokenizer).",
                [new EvidenceItem("metadata", "tokenizer.ggml.model", "(absent)")]));
            return;
        }

        var ev = new List<EvidenceItem>
        {
            new("metadata", "tokenizer.ggml.model", triage.Model),
            new("metadata", "tokenizer.ggml.merges", triage.HasMerges ? "present" : "absent"),
            new("metadata", "tokenizer.ggml.scores", triage.HasScores ? "present" : "absent"),
        };
        switch (triage.Shape)
        {
            case TokenizerShape.SpmScoresOnly:
                findings.Add(new ScoutFinding("tokenizer.spm_scores_only", "Scores-only SentencePiece shape (the minicpm/xverse/orion class).",
                    Certainty.Known, ev, "Handled by GgufTokenizer.SpmMergePiecesByScore for this exact shape; still compare prompt token ids against an oracle."));
                break;
            case TokenizerShape.Unigram:
                findings.Add(new ScoutFinding("tokenizer.unigram", "Unigram-LM vocabulary (t5 model type).", Certainty.Known, ev,
                    "Routed through UnigramTokenizer.FromGgufVocab."));
                break;
            case TokenizerShape.SpmNoMergesNoScores:
                blockers.Add(new ScoutBlocker("scout.tokenizer.llama_no_merges_no_scores", BlockerKind.Suspected,
                    "tokenizer.ggml.model=llama with neither merges nor scores: likely tokenizes to near-character fragments.", ev));
                break;
            case TokenizerShape.BpeNoMerges:
                blockers.Add(new ScoutBlocker("scout.tokenizer.bpe_no_merges", BlockerKind.Suspected,
                    "tokenizer.ggml.model=gpt2 (BPE) without tokenizer.ggml.merges.", ev));
                break;
        }

        if (!tok.HasChatTemplate)
            findings.Add(new ScoutFinding("tokenizer.no_chat_template", "No tokenizer.chat_template; chat use falls back to the architecture's FallbackChat layout.",
                Certainty.Known, [new EvidenceItem("metadata", "tokenizer.chat_template", "(absent)")], "A base model legitimately has none."));
    }
    // ── architecture / dtype ───────────────────────────────────────────────────

    private static ArchitectureResolution ResolveArchitecture(ScoutInput input, List<ScoutBlocker> blockers)
    {
        string? declared = input.Metadata.TryGetValue("general.architecture", out var a) ? Convert.ToString(a, CultureInfo.InvariantCulture) : null;
        if (string.IsNullOrEmpty(declared))
            blockers.Add(new ScoutBlocker("scout.arch.undeclared", BlockerKind.Suspected,
                "No general.architecture; only tensor-layout detectors can identify this file.",
                [new EvidenceItem("metadata", "general.architecture", "(absent)")]));

        ArchitectureDescriptor? d;
        try
        {
            var probe = new ArchitectureProbe { Architecture = declared, TensorSource = new IndexOnlyTensorSource(input.Metadata, input.Tensors), IsGguf = true };
            d = ArchitectureRegistry.TryResolve(probe, out var found) ? found : null;
        }
        catch (InvalidOperationException ex)
        {
            blockers.Add(new ScoutBlocker("scout.arch.probe_needs_values", BlockerKind.Suspected,
                "An architecture detector tried to read tensor values, which static scout forbids: " + ex.Message, []));
            d = null;
        }

        if (d is null)
        {
            if (!string.IsNullOrEmpty(declared))
                blockers.Add(new ScoutBlocker("scout.arch.not_registered", BlockerKind.Confirmed,
                    $"'{declared}' is not in the text-generation architecture registry; ModelCompatibility refuses it (this says nothing about audio/vision/diffusion pipelines that load the file by other routes).",
                    [new EvidenceItem("metadata", "general.architecture", declared)]));
            return new ArchitectureResolution(declared, false, null, null, null, null, null, null, []);
        }

        if (d.Status != AdmissionStatus.Admitted)
            blockers.Add(new ScoutBlocker("scout.arch.not_admitted", BlockerKind.Confirmed,
                $"Registered as {d.Status}: {d.RefusalReason}",
                [new EvidenceItem("descriptor", d.Id, d.EvidenceDoc)]));

        return new ArchitectureResolution(declared, true, d.Id, d.Status.ToString(), d.RefusalReason, d.EvidenceDoc,
            d.ForwardPassFamily.ToString(), d.SupportedBackends.ToString(), []);
    }

    /// <summary>Mirrors <see cref="ModelCompatibility.ValidateForTextGeneration"/>'s dtype rule so a Confirmed blocker means the engine would refuse.</summary>
    private static void CheckDTypes(IReadOnlyList<GgufTensorInfo> tensors, List<ScoutBlocker> blockers)
    {
        var gated = ExperimentalQuantGateRegistry.All;
        var bad = new List<GgufTensorInfo>();
        var gatedSeen = new HashSet<DType>();
        foreach (var t in tensors)
        {
            if (ModelCompatibility.IsSupportedWeightDType(t.DType)) continue;
            var gate = gated.FirstOrDefault(g => g.TensorTypePredicate(t.DType));
            if (gate is not null)
            {
                // One blocker per gated dtype. Enabled state reads the environment, so this is the one environment-dependent line in a report.
                if (gatedSeen.Add(t.DType))
                    blockers.Add(new ScoutBlocker($"scout.dtype.experimental.{t.DType}", gate.IsEnabled() ? BlockerKind.Suspected : BlockerKind.Confirmed,
                        gate.IsEnabled() ? $"{t.DType} runs only behind an experimental gate (currently enabled)." : gate.RefusalMessage,
                        [new EvidenceItem("tensor", t.Name, t.DType.ToString())]));
                continue;
            }
            bad.Add(t);
        }
        if (bad.Count == 0) return;
        blockers.Add(new ScoutBlocker("scout.dtype.unsupported", BlockerKind.Confirmed,
            $"{bad.Count} tensor(s) use storage types the portable text-generation path cannot execute: " +
            string.Join(", ", bad.Select(t => t.DType.ToString()).Distinct().Order(StringComparer.Ordinal)),
            bad.OrderBy(t => t.Name, StringComparer.Ordinal).Take(MaxEvidenceItems).Select(t => new EvidenceItem("tensor", t.Name, $"{t.DType} {Shape(t)}")).ToArray()));
    }

    // ── resources / next actions ───────────────────────────────────────────────

    private static SizeEstimate Unknown(string why) => new(Certainty.Unknown, null, why);

    private static ResourcePreflight Preflight(ScoutInput input, TensorSummary tensors, ScoutOptions options)
    {
        // File size and tensor bytes are facts. Host working set is NOT derivable from them (a memory-mapped file is not resident),
        // and no host-RAM estimator is wired yet, so it is Unknown and the execution decision can never be "allowed".
        var ws = Unknown("no host working-set estimator is wired into scout yet; file size is not peak RAM");
        var kv = Unknown("KV/cache sizing needs resolved hyperparameters and the requested context; not computed by static scout");
        string decision, reason;
        if (options.BudgetBytes is null)
        {
            decision = "not_assessed";
            reason = "No --budget given; execution feasibility not assessed.";
        }
        else
        {
            decision = "blocked";
            reason = "Working-set estimate is unknown, so no execution stage is allowed under the budget. Static inspection is unaffected.";
        }
        return new ResourcePreflight(input.FileBytes, tensors.TotalBytes, ws, kv, options.BudgetBytes,
            options.BudgetBytes is null ? null : options.ReserveBytes, decision, reason);
    }

    private static IReadOnlyList<NextAction> NextActions(ScoutInput input, ArchitectureResolution arch, IReadOnlyList<ScoutBlocker> blockers)
    {
        var actions = new List<NextAction>();
        void Add(string cmd, string why) => actions.Add(new NextAction(actions.Count + 1, cmd, why));
        string model = $"-m {input.FileName}";

        if (blockers.Any(b => b.Id == "scout.dtype.unsupported"))
            Add("(no command)", "Storage type is not executable by the portable path; a kernel/dequantizer is needed before any run. Check ModelCompatibility.IsSupportedWeightDType.");
        if (!arch.Resolved)
            Add($"admit-arch {model}", "Architecture is not registered; triage the tokenizer and run the bypassed forward pass (explicit, memory-bounded).");
        else if (arch.Status != nameof(AdmissionStatus.Admitted))
            Add($"admit-arch {model} --golden <golden.json>", $"Registered but {arch.Status}; the missing evidence is described in {arch.EvidenceDoc}.");
        else
            Add($"verify-goldens", "Architecture is admitted; confirm a hash-pinned golden exists and passes for this file.");
        Add($"hash {model}", "Compute and cache the SHA-256 so any golden or receipt is pinned to this exact file.");
        return actions;
    }
}

/// <summary>Tensor source for architecture probing that exposes the index only; any attempt to read values is a bug in scout's contract.</summary>
internal sealed unsafe class IndexOnlyTensorSource : IModelTensorSource
{
    private readonly Dictionary<string, GgufTensorInfo> _byName = new(StringComparer.Ordinal);

    public IndexOnlyTensorSource(IReadOnlyDictionary<string, object> metadata, IReadOnlyList<GgufTensorInfo> tensors)
    {
        Metadata = metadata;
        Tensors = tensors;
        foreach (var t in tensors) _byName.TryAdd(t.Name, t);
    }

    public IReadOnlyList<GgufTensorInfo> Tensors { get; }
    public IReadOnlyDictionary<string, object> Metadata { get; }
    public GgufTensorInfo? FindTensor(string name) => _byName.TryGetValue(name, out var t) ? t : null;
    public ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor) => throw new InvalidOperationException("static scout must not read tensor values");
    public byte* GetTensorDataPtr(GgufTensorInfo tensor) => throw new InvalidOperationException("static scout must not read tensor values");
}

/// <summary>Parses sizes like "64G", "512M", "1.5GiB" (base 1024).</summary>
public static class ScoutSize
{
    public static bool TryParse(string? text, out long bytes)
    {
        bytes = 0;
        if (string.IsNullOrWhiteSpace(text)) return false;
        var s = text.Trim().ToUpperInvariant();
        if (s.EndsWith("IB", StringComparison.Ordinal)) s = s[..^2];
        else if (s.EndsWith('B') && s.Length > 1 && !char.IsDigit(s[^2])) s = s[..^1];
        long mult = 1;
        if (s.Length > 0 && !char.IsDigit(s[^1]))
        {
            mult = s[^1] switch { 'K' => 1L << 10, 'M' => 1L << 20, 'G' => 1L << 30, 'T' => 1L << 40, _ => 0 };
            if (mult == 0) return false;
            s = s[..^1];
        }
        if (!double.TryParse(s, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out double v) || v < 0 || double.IsInfinity(v)) return false;
        double total = v * mult;
        if (total > long.MaxValue) return false;
        bytes = (long)total;
        return true;
    }
}
