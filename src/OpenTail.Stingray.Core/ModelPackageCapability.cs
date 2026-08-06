namespace OpenTail.Stingray.Core;

/// <summary>Execution backends a package profile may advertise.</summary>
[Flags]
public enum ModelPackageBackends
{
    None = 0,
    Cpu = 1,
    Cuda = 2,
    Vulkan = 4,
}

/// <summary>Tokenizer asset families the loader can construct from.</summary>
public enum ModelPackageTokenizerFamily
{
    Unknown = 0,
    /// <summary>Hugging Face <c>tokenizer.json</c> (BPE).</summary>
    HuggingFaceJson = 1,
    /// <summary>SentencePiece <c>tokenizer.model</c> or <c>spiece.model</c>.</summary>
    SentencePiece = 2,
    /// <summary>Vocabulary embedded in GGUF metadata.</summary>
    GgufEmbedded = 3,
}

/// <summary>Why a package cannot be executed by a given profile.</summary>
public enum ModelPackageRejectionKind
{
    None = 0,
    MissingConfig,
    MissingWeights,
    MissingShard,
    MissingTokenizer,
    UnsupportedArchitecture,
    UnsupportedConfig,
    UnsupportedDtype,
    TensorMismatch,
    UnsupportedBackend,
    MalformedPackage,
}

/// <summary>
/// One reason a package was refused, naming the specific asset or setting at fault.
/// </summary>
/// <remarks>
/// The plan requires rejections to identify "the missing shard, unsupported config, tensor mismatch,
/// tokenizer family, unavailable backend, or memory requirement" — so the offending item is a
/// first-class field rather than prose inside a message. Callers can group or filter on it; a human
/// still gets <see cref="Detail"/>.
/// </remarks>
public sealed record ModelPackageRejection(
    ModelPackageRejectionKind Kind,
    string Subject,
    string Detail)
{
    public override string ToString() => $"{Kind} ({Subject}): {Detail}";
}

/// <summary>
/// A versioned statement of what one architecture profile can actually do with a package.
/// </summary>
/// <remarks>
/// <para>This exists so support can be answered <b>without constructing a forward pass</b>, which is
/// Phase 0's exit gate. It is deliberately a description of a profile plus the outcome of matching a
/// specific directory against it — not a promise about SafeTensors in general.</para>
///
/// <para><b>Never report a global "SafeTensors is supported".</b> Every claim is a row: architecture
/// profile, source dtypes, tokenizer family, backends, and the features explicitly advertised.
/// Anything absent from a row is unsupported, and <see cref="Exclusions"/> records the ones worth
/// saying out loud.</para>
/// </remarks>
public sealed record ModelPackageCapability(
    int SchemaVersion,
    string ProfileId,
    string Description,
    IReadOnlyList<string> ArchitectureIds,
    IReadOnlyList<string> SourceDtypes,
    ModelPackageTokenizerFamily TokenizerFamily,
    ModelPackageBackends Backends,
    bool SupportsBatching,
    bool SupportsSessions,
    bool SupportsSpeculation,
    bool SupportsAdapters,
    bool SupportsMultimodal,
    IReadOnlyList<string> Exclusions)
{
    /// <summary>Current schema version. Bump when a field's meaning changes, not when a row is added.</summary>
    public const int CurrentSchemaVersion = 1;

    /// <summary>
    /// The first profile: dense decoder-only Llama/Mistral, F16/BF16/F32, RMSNorm + RoPE + SiLU MLP,
    /// no projection bias, CPU only.
    /// </summary>
    /// <remarks>
    /// Tied output embeddings (<c>tie_word_embeddings: true</c>) are supported and aliased to
    /// <c>token_embd.weight</c> when <c>lm_head.weight</c> is omitted.
    /// </remarks>
    public static ModelPackageCapability DenseLlamaCpu { get; } = new(
        SchemaVersion: CurrentSchemaVersion,
        ProfileId: "dense-llama-cpu",
        Description: "Dense decoder-only Llama/Mistral, CPU, high-precision source weights.",
        ArchitectureIds: ["llama", "mistral"],
        SourceDtypes: ["F32", "F16", "BF16"],
        TokenizerFamily: ModelPackageTokenizerFamily.HuggingFaceJson,
        Backends: ModelPackageBackends.Cpu,
        SupportsBatching: false,
        SupportsSessions: false,
        SupportsSpeculation: false,
        SupportsAdapters: false,
        SupportsMultimodal: false,
        Exclusions:
        [
            "Attention or MLP projection bias.",
            "Non-SiLU activations.",
            "RoPE scaling of any kind.",
            "Quantized SafeTensors weights; use GGUF for block-quantized deployment.",
            "CUDA and Vulkan routes; each needs its own dtype, layout and transfer contract.",
        ]);

    /// <summary>All profiles OpenTail currently publishes.</summary>
    public static IReadOnlyList<ModelPackageCapability> All { get; } = [DenseLlamaCpu];
}

/// <summary>
/// The result of matching one package directory against one profile.
/// </summary>
public sealed record ModelPackageCapabilityReport(
    string PackagePath,
    string ProfileId,
    bool IsSupported,
    string? ArchitectureId,
    IReadOnlyList<string> SourceDtypes,
    ModelPackageTokenizerFamily TokenizerFamily,
    ModelPackageBackends AvailableBackends,
    long? EstimatedWeightBytes,
    long? EstimatedWorkingSetBytes,
    IReadOnlyList<ModelPackageRejection> Rejections)
{
    /// <summary>Single-line summary suitable for CLI output.</summary>
    public string ToSummary() => IsSupported
        ? $"{PackagePath}: SUPPORTED by profile '{ProfileId}' " +
          $"(arch {ArchitectureId}, dtypes {string.Join('/', SourceDtypes)}, " +
          $"tokenizer {TokenizerFamily}, backends {AvailableBackends}" +
          (EstimatedWeightBytes is { } b ? $", weights ~{b / (1024.0 * 1024.0):F1} MiB" : "") +
          (EstimatedWorkingSetBytes is { } w && w != EstimatedWeightBytes ? $", CPU working set ~{w / (1024.0 * 1024.0):F1} MiB" : "") + ")"
        : $"{PackagePath}: NOT SUPPORTED by profile '{ProfileId}' — " +
          string.Join("; ", Rejections.Select(r => r.ToString()));
}
