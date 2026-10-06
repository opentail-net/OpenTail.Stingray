using System.Collections.Immutable;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine.Packaging;

/// <summary>
/// Logical package adapter for Ollama content-addressed packages (§2 &amp; §3 of plan).
/// Consumes Ollama-compatible manifests and blobs without requiring an Ollama daemon or installation.
/// </summary>
public sealed class OllamaModelPackage : IModelPackage
{
    public ModelPackageIdentity Identity { get; }
    public string PrimaryPath { get; }
    public ImmutableArray<ModelPackageComponent> Components { get; }
    public ModelFormat Format => ModelFormat.Gguf; // Ollama model blobs are GGUF
    public StingraySidecarMetadata? SidecarMetadata { get; }

    /// <summary>
    /// Path to the primary weight blob on disk.
    /// </summary>
    public string ModelBlobPath { get; }

    private OllamaModelPackage(
        ModelPackageIdentity identity,
        string primaryPath,
        string modelBlobPath,
        ImmutableArray<ModelPackageComponent> components,
        StingraySidecarMetadata? sidecarMetadata)
    {
        Identity = identity;
        PrimaryPath = primaryPath;
        ModelBlobPath = modelBlobPath;
        Components = components;
        SidecarMetadata = sidecarMetadata;
    }

    /// <summary>
    /// Opens an Ollama-compatible package from a manifest file and corresponding blobs directory.
    /// </summary>
    public static OllamaModelPackage Open(string manifestPath, string? blobsDirectory = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(manifestPath);
        if (!File.Exists(manifestPath))
            throw new FileNotFoundException($"Ollama manifest file not found: '{manifestPath}'", manifestPath);

        string manifestJson = File.ReadAllText(manifestPath);
        var manifest = JsonSerializer.Deserialize(manifestJson, ModelPackageJsonContext.Default.OllamaManifest);
        if (manifest == null || manifest.Layers == null || manifest.Layers.Count == 0)
            throw new InvalidOperationException($"Invalid or empty Ollama manifest at '{manifestPath}'");

        // Resolve blobs directory: explicitly supplied, or sibling "blobs" folder, or parent "blobs" folder
        string blobsDir = blobsDirectory ?? FindBlobsDirectory(manifestPath);

        var componentsBuilder = ImmutableArray.CreateBuilder<ModelPackageComponent>();
        var identBuilder = ImmutableArray.CreateBuilder<ModelPackageComponentIdentity>();
        string? primaryModelBlob = null;
        string? primaryModelDigest = null;

        foreach (var layer in manifest.Layers)
        {
            string role = MapMediaTypeToRole(layer.MediaType);
            string blobFileName = layer.Digest.Replace(':', '-');
            string blobPath = Path.Combine(blobsDir, blobFileName);

            if (File.Exists(blobPath))
            {
                var fi = new FileInfo(blobPath);
                componentsBuilder.Add(new ModelPackageComponent(
                    Role: role,
                    Path: fi.FullName,
                    SizeBytes: fi.Length,
                    Digest: layer.Digest));

                identBuilder.Add(new ModelPackageComponentIdentity(
                    Role: role,
                    Digest: layer.Digest,
                    RelativeName: blobFileName));

                if (role == ModelPackageRoles.PrimaryWeights && primaryModelBlob == null)
                {
                    primaryModelBlob = fi.FullName;
                    primaryModelDigest = layer.Digest;
                }
            }
        }

        if (primaryModelBlob == null)
            throw new FileNotFoundException($"Ollama package manifest has no resolved primary model weight blob in '{blobsDir}'");

        // Compute package identity: using primary model digest or manifest config digest
        string packageDigest = manifest.Config?.Digest ?? primaryModelDigest ?? "unknown";
        var identity = new ModelPackageIdentity(
            ContentDigest: packageDigest,
            Format: ModelFormat.Gguf,
            Components: identBuilder.ToImmutable(),
            IsProvisional: false);

        // Optional advisory sidecar next to manifest or blobs
        StingraySidecarMetadata? sidecar = null;
        string sidecarPath = Path.ChangeExtension(manifestPath, ".stingray.json");
        if (File.Exists(sidecarPath))
        {
            try
            {
                string json = File.ReadAllText(sidecarPath);
                sidecar = JsonSerializer.Deserialize(json, ModelPackageJsonContext.Default.StingraySidecarMetadata);
            }
            catch
            {
                // Advisory only
            }
        }

        return new OllamaModelPackage(
            identity,
            Path.GetFullPath(manifestPath),
            primaryModelBlob,
            componentsBuilder.ToImmutable(),
            sidecar);
    }

    private static string FindBlobsDirectory(string manifestPath)
    {
        string? dir = Path.GetDirectoryName(manifestPath);
        if (!string.IsNullOrEmpty(dir))
        {
            // Sibling "blobs" directory (e.g. models/manifests/... and models/blobs/...)
            string candidate = Path.Combine(dir, "blobs");
            if (Directory.Exists(candidate)) return candidate;

            var parent = Directory.GetParent(dir);
            while (parent != null)
            {
                string parentCandidate = Path.Combine(parent.FullName, "blobs");
                if (Directory.Exists(parentCandidate)) return parentCandidate;
                parent = parent.Parent;
            }
        }

        return Directory.GetCurrentDirectory();
    }

    private static string MapMediaTypeToRole(string mediaType) => mediaType switch
    {
        "application/vnd.ollama.image.model" => ModelPackageRoles.PrimaryWeights,
        "application/vnd.ollama.image.projector" => ModelPackageRoles.VisionProjector,
        "application/vnd.ollama.image.template" => ModelPackageRoles.ChatTemplate,
        "application/vnd.ollama.image.license" => ModelPackageRoles.License,
        "application/vnd.ollama.image.params" => ModelPackageRoles.Config,
        _ => ModelPackageRoles.Other,
    };
}

/// <summary>
/// Minimal Ollama manifest representation (§2 of plan).
/// </summary>
public sealed record OllamaManifest(
    [property: JsonPropertyName("schemaVersion")] int SchemaVersion,
    [property: JsonPropertyName("mediaType")] string? MediaType = null,
    [property: JsonPropertyName("config")] OllamaLayer? Config = null,
    [property: JsonPropertyName("layers")] IReadOnlyList<OllamaLayer>? Layers = null);

/// <summary>
/// Descriptor for a single layer/blob in an Ollama manifest.
/// </summary>
public sealed record OllamaLayer(
    [property: JsonPropertyName("mediaType")] string MediaType,
    [property: JsonPropertyName("digest")] string Digest,
    [property: JsonPropertyName("size")] long Size);
