using System.Collections.Immutable;
using System.IO;
using System.Text.Json;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine.Packaging;

/// <summary>
/// Logical package adapter for loose GGUF model files (§1 of plan).
/// Supports single GGUF weights, optional multimodal projectors (mmproj), optional draft models,
/// and optional advisory sidecar metadata (stingray.json).
/// </summary>
public sealed class LooseGgufModelPackage : IModelPackage
{
    public ModelPackageIdentity Identity { get; }
    public string PrimaryPath { get; }
    public ImmutableArray<ModelPackageComponent> Components { get; }
    public ModelFormat Format => ModelFormat.Gguf;
    public StingraySidecarMetadata? SidecarMetadata { get; }

    private LooseGgufModelPackage(
        ModelPackageIdentity identity,
        string primaryPath,
        ImmutableArray<ModelPackageComponent> components,
        StingraySidecarMetadata? sidecarMetadata)
    {
        Identity = identity;
        PrimaryPath = primaryPath;
        Components = components;
        SidecarMetadata = sidecarMetadata;
    }

    /// <summary>
    /// Opens a loose GGUF package from the primary model file path, resolving companion components.
    /// </summary>
    public static LooseGgufModelPackage Open(
        string ggufPath,
        string? mmprojPath = null,
        string? draftModelPath = null,
        string? contentDigest = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(ggufPath);
        if (!File.Exists(ggufPath))
            throw new FileNotFoundException($"Primary GGUF model file not found: '{ggufPath}'", ggufPath);

        var componentsBuilder = ImmutableArray.CreateBuilder<ModelPackageComponent>();
        var identBuilder = ImmutableArray.CreateBuilder<ModelPackageComponentIdentity>();

        var primaryFi = new FileInfo(ggufPath);
        componentsBuilder.Add(new ModelPackageComponent(
            Role: ModelPackageRoles.PrimaryWeights,
            Path: primaryFi.FullName,
            SizeBytes: primaryFi.Length,
            Digest: contentDigest));

        identBuilder.Add(new ModelPackageComponentIdentity(
            Role: ModelPackageRoles.PrimaryWeights,
            Digest: contentDigest,
            RelativeName: primaryFi.Name));

        // Companion vision projector
        if (!string.IsNullOrWhiteSpace(mmprojPath) && File.Exists(mmprojPath))
        {
            var mmFi = new FileInfo(mmprojPath);
            componentsBuilder.Add(new ModelPackageComponent(
                Role: ModelPackageRoles.VisionProjector,
                Path: mmFi.FullName,
                SizeBytes: mmFi.Length));
            identBuilder.Add(new ModelPackageComponentIdentity(
                Role: ModelPackageRoles.VisionProjector,
                Digest: null,
                RelativeName: mmFi.Name));
        }

        // Companion draft model
        if (!string.IsNullOrWhiteSpace(draftModelPath) && File.Exists(draftModelPath))
        {
            var draftFi = new FileInfo(draftModelPath);
            componentsBuilder.Add(new ModelPackageComponent(
                Role: ModelPackageRoles.DraftModel,
                Path: draftFi.FullName,
                SizeBytes: draftFi.Length));
            identBuilder.Add(new ModelPackageComponentIdentity(
                Role: ModelPackageRoles.DraftModel,
                Digest: null,
                RelativeName: draftFi.Name));
        }

        // Discover advisory sidecar metadata if present
        StingraySidecarMetadata? sidecar = null;
        string sidecarCandidate = Path.ChangeExtension(ggufPath, ".stingray.json");
        if (!File.Exists(sidecarCandidate))
        {
            string? dir = Path.GetDirectoryName(ggufPath);
            if (!string.IsNullOrEmpty(dir))
            {
                string dirCandidate = Path.Combine(dir, "stingray.json");
                if (File.Exists(dirCandidate))
                    sidecarCandidate = dirCandidate;
            }
        }

        if (File.Exists(sidecarCandidate))
        {
            try
            {
                string json = File.ReadAllText(sidecarCandidate);
                sidecar = JsonSerializer.Deserialize(json, ModelPackageJsonContext.Default.StingraySidecarMetadata);
                if (sidecar != null)
                {
                    var sidecarFi = new FileInfo(sidecarCandidate);
                    componentsBuilder.Add(new ModelPackageComponent(
                        Role: ModelPackageRoles.SidecarMetadata,
                        Path: sidecarFi.FullName,
                        SizeBytes: sidecarFi.Length));
                }
            }
            catch
            {
                // Sidecars are advisory; corrupt or unreadable sidecar does not prevent model execution
            }
        }

        var identity = contentDigest != null
            ? new ModelPackageIdentity(contentDigest, ModelFormat.Gguf, identBuilder.ToImmutable(), IsProvisional: false)
            : ModelPackageIdentity.CreateProvisional(ModelFormat.Gguf, identBuilder.ToImmutable());

        return new LooseGgufModelPackage(identity, primaryFi.FullName, componentsBuilder.ToImmutable(), sidecar);
    }
}
