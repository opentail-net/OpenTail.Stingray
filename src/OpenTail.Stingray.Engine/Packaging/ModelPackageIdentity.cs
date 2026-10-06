using System.Collections.Immutable;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine.Packaging;

/// <summary>
/// Intrinsic content identity of a model package (§7 of plan).
/// Filesystem location paths are excluded from identity and do not participate in equality.
/// </summary>
public sealed record ModelPackageIdentity(
    string? ContentDigest,
    ModelFormat Format,
    ImmutableArray<ModelPackageComponentIdentity> Components,
    bool IsProvisional = false) : IEquatable<ModelPackageIdentity>
{
    /// <summary>
    /// Creates a provisional identity for a loose package whose full content digest is uncomputed.
    /// Path remains location metadata and is never silently promoted to semantic identity.
    /// </summary>
    public static ModelPackageIdentity CreateProvisional(
        ModelFormat format,
        ImmutableArray<ModelPackageComponentIdentity> components) =>
        new(ContentDigest: null, Format: format, Components: components, IsProvisional: true);

    public bool Equals(ModelPackageIdentity? other)
    {
        if (ReferenceEquals(this, other)) return true;
        if (other is null) return false;
        return ContentDigest == other.ContentDigest
            && Format == other.Format
            && IsProvisional == other.IsProvisional
            && (Components.IsDefault && other.Components.IsDefault
                || !Components.IsDefault && !other.Components.IsDefault && Components.SequenceEqual(other.Components));
    }

    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(ContentDigest);
        hash.Add(Format);
        hash.Add(IsProvisional);
        if (!Components.IsDefault)
        {
            foreach (var c in Components)
                hash.Add(c);
        }
        return hash.ToHashCode();
    }
}

/// <summary>
/// Identity of an individual constituent component within a model package.
/// </summary>
public sealed record ModelPackageComponentIdentity(
    string Role,
    string? Digest,
    string RelativeName);

/// <summary>
/// Concrete component descriptor with resolved local path and size.
/// </summary>
public sealed record ModelPackageComponent(
    string Role,
    string Path,
    long SizeBytes,
    string? Digest = null);

/// <summary>
/// Standard functional roles for model package components.
/// </summary>
public static class ModelPackageRoles
{
    public const string PrimaryWeights = "primary_weights";
    public const string VisionProjector = "vision_projector";
    public const string DraftModel = "draft_model";
    public const string Tokenizer = "tokenizer";
    public const string Config = "config";
    public const string ChatTemplate = "chat_template";
    public const string SidecarMetadata = "sidecar_metadata";
    public const string License = "license";
    public const string Other = "other";
}
