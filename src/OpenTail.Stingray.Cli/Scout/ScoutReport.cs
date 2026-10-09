namespace OpenTail.Stingray.Cli.Scout;

// Schema for `stingray scout` (docs/3-product-and-runtime/2026-10-09-checkpoint-scout-and-ai-admission-plan.md).
// Rules this file exists to enforce:
//   * An unknown value is never encoded as 0/false/empty: it is null with a Certainty.Unknown (or an explicit state), so a
//     reader can tell "measured zero" from "not measured".
//   * No machine paths: only the bare file name is recorded.
//   * Nothing here is time- or machine-dependent except ScoutBuild, so two runs over one file on one build are byte-identical.
// Bump SchemaVersion on any incompatible change; additive fields do not need a bump.

/// <summary>How firmly a reported value is known.</summary>
public enum Certainty
{
    // <summary>Read directly from the file or computed exactly from what was read.</summary>
    Known,
    // <summary>Derived by a stated method; may be wrong by a stated margin.</summary>
    Estimated,
    // <summary>Suggested by evidence (names/shapes/metadata) but not established; never proof of semantics.</summary>
    Hypothesis,
    // <summary>Not determined. The value field is null.</summary>
    Unknown,
}

/// <summary>Outcome of one pipeline stage. <see cref="NotRun"/> and <see cref="Blocked"/> are never a pass.</summary>
public enum StageState { NotRun, Passed, Failed, Blocked }

/// <summary>A blocker is Confirmed only when the engine's own gate would refuse; otherwise it is a Suspicion.</summary>
public enum BlockerKind { Confirmed, Suspected }

public sealed record EvidenceItem(string Kind, string Name, string? Value);

public sealed record ScoutFinding(
    string Id,
    string Summary,
    Certainty Certainty,
    IReadOnlyList<EvidenceItem> Evidence,
    string Caveat);

public sealed record ScoutBlocker(
    string Id,
    BlockerKind Kind,
    string Summary,
    IReadOnlyList<EvidenceItem> Evidence);

public sealed record StageReceipt(string Stage, StageState State, string Detail);

public sealed record ArtifactInfo(
    string FileName,
    long? FileBytes,
    uint? GgufVersion,
    ulong? TensorCount,
    ulong? MetadataKeyCount,
    int? ShardCount,
    // "not_computed" (the default: scout never hashes a local file, it can take minutes) or "published_by_huggingface": the hash the Hub reports for a remote file.
    // That is the Hub's claim, not a hash anyone here computed from bytes; verify it after a download before relying on it.
    string Sha256State,
    // Where a remote file came from. Null for a local file.
    ArtifactSource? Source = null);

/// <summary>A file inspected on Hugging Face, pinned to an immutable commit. Every value is what the Hub reported at that commit.</summary>
public sealed record ArtifactSource(
    string Kind,
    string Repo,
    // Full commit SHA. All requests were pinned to it, so the metadata and the index bytes describe the same revision.
    string Revision,
    // File path(s) in the repo (several for a split model).
    IReadOnlyList<string> Paths,
    // Present only for a single-file model whose hash the Hub publishes; null otherwise (never invented, never one hash for many files).
    string? Sha256,
    string? Sha256Source,
    // null when not gated; otherwise "auto" or "manual".
    string? Gated,
    bool Private,
    string? License,
    // Repository download count over the last 30 days. Per repository, not per file, and not a measure of inference use.
    long? Downloads30Days,
    // What the Hub declares as the architecture. Compared with the file's own metadata; a disagreement is reported.
    string? HubArchitecture,
    // Bytes read to inspect the index (across all shards). The weights were not downloaded.
    long IndexBytesRead);

/// <summary>What leaving the machine cost for this report. No URLs, queries or tokens.</summary>
public sealed record NetworkUse(int Requests, IReadOnlyList<string> Hosts, long BytesReceived);

public sealed record MetadataEntry(string Key, string Type, string Value);

public sealed record TokenizerSummary(
    string? Model,
    bool HasMerges,
    bool HasScores,
    bool HasTokenTypes,
    bool HasChatTemplate,
    long? VocabSize,
    long? BosTokenId,
    long? EosTokenId,
    bool? AddBosToken);

public sealed record MetadataSummary(
    IReadOnlyList<MetadataEntry> Entries,
    TokenizerSummary Tokenizer);

public sealed record DTypeBucket(string DType, int Count, long Bytes);

public sealed record TensorPattern(
    // <summary>Tensor name with the layer index replaced by '*', e.g. <c>blk.*.attn_q.weight</c>.</summary>
    string Pattern,
    int Count,
    // <summary>Compact layer set, e.g. "0-39" or "0-2,4-39"; null for tensors outside any layer stack.</summary>
    string? Layers,
    IReadOnlyList<string> DTypes,
    string SampleShape,
    int DistinctShapes);

public sealed record TensorIrregularity(string Pattern, string Kind, string Detail);

public sealed record TensorSummary(
    ulong Count,
    long TotalBytes,
    int? LayerCount,
    IReadOnlyList<DTypeBucket> ByDType,
    IReadOnlyList<TensorPattern> Patterns,
    IReadOnlyList<TensorIrregularity> Irregularities);

public sealed record ArchitectureCandidate(
    string Id,
    // Live registry status of the candidate, not the status when the snapshot was taken.
    string Status,
    // "identical_structure" (zero differences) or "differs". Structure only: values, rope parameters and activation are not compared.
    string Kind,
    int DifferenceCount,
    // Id of the reference structure matched (SHA-256 of its canonical patterns + features).
    string StructureId,
    // Files that demonstrated that structure (name, and a hash prefix when known). Provenance only: it does not affect ranking.
    IReadOnlyList<string> ReferenceFiles,
    IReadOnlyList<string> Matching,
    IReadOnlyList<string> Differing);

public sealed record ArchitectureResolution(
    string? Declared,
    bool Resolved,
    string? DescriptorId,
    string? Status,
    string? RefusalReason,
    string? EvidenceDoc,
    string? ForwardPassFamily,
    string? SupportedBackends,
    // "computed" or "not_computed". Not computed means no reference signatures were supplied.
    string CandidatesState,
    // Closest admitted architectures by structural signature, fewest differences first: the nearest structural parent, then its siblings.
    IReadOnlyList<ArchitectureCandidate> Candidates);

public sealed record SizeEstimate(Certainty Certainty, long? Bytes, string Source);

public sealed record ResourcePreflight(
    long? FileBytes,
    long? TensorBytes,
    SizeEstimate HostWorkingSet,
    SizeEstimate KvCache,
    // Terms of the host working-set estimate (CPU run); a term with null bytes is why the total is Unknown.
    IReadOnlyList<WorkingSetComponent> WorkingSetComponents,
    // Context length the KV and scratch terms assume.
    int ContextTokens,
    long? BudgetBytes,
    long? ReserveBytes,
    // <summary>"allowed", "blocked" or "not_assessed". Unknown estimates are never "allowed".</summary>
    string ExecutionDecision,
    string Reason);

public sealed record NextAction(int Order, string Command, string Why);

public sealed record ScoutReport(
    int SchemaVersion,
    string ScoutBuild,
    ArtifactInfo Artifact,
    MetadataSummary? Metadata,
    TensorSummary? Tensors,
    ArchitectureResolution? Architecture,
    IReadOnlyList<ScoutFinding> Findings,
    IReadOnlyList<ScoutBlocker> Blockers,
    ResourcePreflight Resources,
    IReadOnlyList<NextAction> NextActions,
    IReadOnlyList<StageReceipt> Stages,
    // Set only when the network was used (remote scout).
    NetworkUse? Network = null)
{
    public const int CurrentSchemaVersion = 1;
}

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower, WriteIndented = true,
    UseStringEnumConverter = true, DefaultIgnoreCondition = JsonIgnoreCondition.Never)]
[JsonSerializable(typeof(ScoutReport))]
[JsonSerializable(typeof(ArchSignature))]
[JsonSerializable(typeof(QuantReport))]
internal partial class ScoutJsonContext : JsonSerializerContext;
