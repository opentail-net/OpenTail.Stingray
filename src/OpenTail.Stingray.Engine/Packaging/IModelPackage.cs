using System.Collections.Immutable;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine.Packaging;

/// <summary>
/// Logical component boundary for a runnable model (§1 &amp; §3 of plan).
/// Describes which external files/components together constitute one model (e.g. weights,
/// tokenizer, template, projector, draft model, configuration) without requiring a physical archive.
/// </summary>
public interface IModelPackage
{
    /// <summary>
    /// Content-based identity of the model package.
    /// Does not include filesystem path in equality.
    /// </summary>
    ModelPackageIdentity Identity { get; }

    /// <summary>
    /// Primary location or source directory/file used to open the package.
    /// Location metadata only; does not participate in package identity equality.
    /// </summary>
    string PrimaryPath { get; }

    /// <summary>
    /// Resolved constituent components that make up this package.
    /// </summary>
    ImmutableArray<ModelPackageComponent> Components { get; }

    /// <summary>
    /// Model format of the primary weights in this package.
    /// </summary>
    ModelFormat Format { get; }

    /// <summary>
    /// Optional advisory sidecar metadata (stingray.json), if discovered.
    /// Strictly advisory; never grants admission or overrides executable code.
    /// </summary>
    StingraySidecarMetadata? SidecarMetadata => null;
}
