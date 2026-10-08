namespace OpenTail.Stingray.Core;

/// <summary>
/// Everything an architecture needs to turn the generic <see cref="Baseline"/> hyperparameters into its final,
/// architecture-correct form. Core-owned and deliberately narrow: no execution plan, backend or runtime types.
/// </summary>
public sealed class ModelArchitectureSemanticsContext
{
    public required IReadOnlyDictionary<string, object> Metadata { get; init; }
    public required IModelTensorSource TensorSource { get; init; }

    /// <summary>What the file literally declares in <c>general.architecture</c>, if anything.</summary>
    public string? DeclaredArchitecture { get; init; }

    /// <summary>The architecture the registry resolved (the descriptor id).</summary>
    public required string CanonicalArchitecture { get; init; }

    /// <summary>The key namespace the baseline was read from (declared for normal and relabelled files).</summary>
    public required string MetadataArchitecture { get; init; }

    /// <summary>Generic structural extraction, before any architecture-specific rule.</summary>
    public required ModelHyperparams Baseline { get; init; }

    // Typed readers over {MetadataArchitecture}.{suffix} so semantics code never repeats the namespace logic.
    public int Int(string suffix, int fallback = 0) => ModelHyperparams.GetInt(Metadata, $"{MetadataArchitecture}.{suffix}", fallback);
    public float Float(string suffix, float fallback = 0f) => ModelHyperparams.GetFloat(Metadata, $"{MetadataArchitecture}.{suffix}", fallback);
    public bool Bool(string suffix, bool fallback = false) => ModelHyperparams.GetBool(Metadata, $"{MetadataArchitecture}.{suffix}", fallback);
    public IReadOnlyList<int>? IntArray(string suffix) => ModelHyperparams.GetIntArray(Metadata, $"{MetadataArchitecture}.{suffix}");
    public IReadOnlyList<bool>? BoolArray(string suffix) => ModelHyperparams.GetBoolArray(Metadata, $"{MetadataArchitecture}.{suffix}");
    public bool HasKey(string suffix) => Metadata.ContainsKey($"{MetadataArchitecture}.{suffix}");
}
