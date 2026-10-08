namespace OpenTail.Stingray.Engine;

using OpenTail.Stingray.Core;

/// <summary>Outcome of <see cref="ArchitectureModelResolver.Resolve"/>.</summary>
/// <param name="Descriptor">Null when the registry does not recognise the file (admission refuses it later).</param>
/// <param name="Probe">Raw probe of what the file declares and contains.</param>
/// <param name="Hyperparams">Final hyperparameters: baseline plus the descriptor's semantics, if any.</param>
/// <param name="DeclaredArchitecture"><c>general.architecture</c> as written in the file, if present.</param>
/// <param name="CanonicalArchitecture">Descriptor id, else the declared architecture, else "unknown".</param>
/// <param name="MetadataArchitecture">Key namespace the baseline was read from.</param>
public sealed record ResolvedModelArchitecture(
    ArchitectureDescriptor? Descriptor,
    ArchitectureProbe Probe,
    ModelHyperparams Hyperparams,
    string? DeclaredArchitecture,
    string CanonicalArchitecture,
    string MetadataArchitecture);

/// <summary>
/// Owns the load-time ordering: raw probe, registry resolution, generic baseline, descriptor semantics.
/// Identity is always resolved before any architecture-specific rule is applied.
/// </summary>
public static class ArchitectureModelResolver
{
    /// <summary>Architecture-correct hyperparameters for a source; the replacement for calling the Core parser directly.</summary>
    public static ModelHyperparams ResolveHyperparams(IModelTensorSource tensorSource)
        => Resolve(tensorSource, null, tensorSource as GgufModel).Hyperparams;

    /// <summary>Metadata-only variant (no tensor inventory), for synthetic headers.</summary>
    public static ModelHyperparams ResolveHyperparams(IReadOnlyDictionary<string, object> metadata)
        => Resolve(new MetadataOnlyTensorSource(metadata), null).Hyperparams;

    public static ResolvedModelArchitecture Resolve(IModelTensorSource tensorSource, string? path, GgufModel? gguf = null)
    {
        ArgumentNullException.ThrowIfNull(tensorSource);
        string? declared = tensorSource.Metadata.TryGetValue("general.architecture", out var a) ? Convert.ToString(a) : null;

        var probe = new ArchitectureProbe
        {
            Path = path,
            Architecture = declared,
            TensorSource = tensorSource,
            IsGguf = gguf is not null,
            Gguf = gguf,
        };

        ArchitectureDescriptor? descriptor = ArchitectureRegistry.TryResolve(probe, out var found) ? found : null;
        string canonical = descriptor?.Id ?? declared ?? "unknown";
        // Same namespace the file's own keys use (declared, default "llama"): relabelled files keep their declared prefix.
        string metadataArch = declared ?? "llama";

        // A resolved descriptor owns its structural traits; an unregistered architecture (e.g. qwen3vlmoe) falls back to the by-name table.
        var baseline = ModelHyperparams.CreateBaseline(tensorSource.Metadata, tensorSource, metadataArch,
            traits: descriptor is null ? null : descriptor.Traits ?? OpenTail.Stingray.Core.ModelArchitectureTraits.None);
        if (descriptor is { UsesNeoxRope: true }
            && !(tensorSource.Metadata.TryGetValue($"{metadataArch}.rope.is_neox", out var neoxOverride) && neoxOverride is bool))
            baseline = baseline with { IsNeoxRope = true };
        var hp = descriptor?.ApplyModelSemantics is { } apply
            ? apply(new ModelArchitectureSemanticsContext
            {
                Metadata = tensorSource.Metadata,
                TensorSource = tensorSource,
                DeclaredArchitecture = declared,
                CanonicalArchitecture = canonical,
                MetadataArchitecture = metadataArch,
                Baseline = baseline,
            })
            : baseline;

        return new ResolvedModelArchitecture(descriptor, probe, hp, declared, canonical, metadataArch);
    }
}
