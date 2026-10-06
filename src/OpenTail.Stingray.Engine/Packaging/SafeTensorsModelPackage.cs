using System.Collections.Immutable;
using System.IO;
using System.Text.Json;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine.Packaging;

/// <summary>
/// Logical package adapter for HuggingFace SafeTensors directories and files (§1 of plan).
/// Resolves primary weights, config.json, tokenizer components, and optional sidecars.
/// </summary>
public sealed class SafeTensorsModelPackage : IModelPackage
{
    public ModelPackageIdentity Identity { get; }
    public string PrimaryPath { get; }
    public ImmutableArray<ModelPackageComponent> Components { get; }
    public ModelFormat Format => ModelFormat.SafeTensors;
    public StingraySidecarMetadata? SidecarMetadata { get; }

    private SafeTensorsModelPackage(
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
    /// Opens a SafeTensors package from a model directory or primary weight file.
    /// </summary>
    public static SafeTensorsModelPackage Open(string packagePath, string? contentDigest = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(packagePath);
        string rootDir = Directory.Exists(packagePath)
            ? packagePath
            : (Path.GetDirectoryName(packagePath) ?? Directory.GetCurrentDirectory());

        if (!Directory.Exists(rootDir))
            throw new DirectoryNotFoundException($"SafeTensors package directory not found: '{rootDir}'");

        var componentsBuilder = ImmutableArray.CreateBuilder<ModelPackageComponent>();
        var identBuilder = ImmutableArray.CreateBuilder<ModelPackageComponentIdentity>();

        // Enumerate .safetensors files
        var weightFiles = Directory.EnumerateFiles(rootDir, "*.safetensors", SearchOption.TopDirectoryOnly)
            .OrderBy(f => f, StringComparer.Ordinal)
            .ToList();

        if (weightFiles.Count == 0 && File.Exists(packagePath) && packagePath.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            weightFiles.Add(packagePath);
        }

        foreach (var weightFile in weightFiles)
        {
            var fi = new FileInfo(weightFile);
            componentsBuilder.Add(new ModelPackageComponent(
                Role: ModelPackageRoles.PrimaryWeights,
                Path: fi.FullName,
                SizeBytes: fi.Length));
            identBuilder.Add(new ModelPackageComponentIdentity(
                Role: ModelPackageRoles.PrimaryWeights,
                Digest: null,
                RelativeName: fi.Name));
        }

        // Config file
        string configPath = Path.Combine(rootDir, "config.json");
        if (File.Exists(configPath))
        {
            var fi = new FileInfo(configPath);
            componentsBuilder.Add(new ModelPackageComponent(
                Role: ModelPackageRoles.Config,
                Path: fi.FullName,
                SizeBytes: fi.Length));
            identBuilder.Add(new ModelPackageComponentIdentity(
                Role: ModelPackageRoles.Config,
                Digest: null,
                RelativeName: fi.Name));
        }

        // Tokenizer files
        string tokenizerPath = Path.Combine(rootDir, "tokenizer.json");
        if (File.Exists(tokenizerPath))
        {
            var fi = new FileInfo(tokenizerPath);
            componentsBuilder.Add(new ModelPackageComponent(
                Role: ModelPackageRoles.Tokenizer,
                Path: fi.FullName,
                SizeBytes: fi.Length));
            identBuilder.Add(new ModelPackageComponentIdentity(
                Role: ModelPackageRoles.Tokenizer,
                Digest: null,
                RelativeName: fi.Name));
        }

        // Sidecar metadata
        StingraySidecarMetadata? sidecar = null;
        string sidecarPath = Path.Combine(rootDir, "stingray.json");
        if (File.Exists(sidecarPath))
        {
            try
            {
                string json = File.ReadAllText(sidecarPath);
                sidecar = JsonSerializer.Deserialize(json, ModelPackageJsonContext.Default.StingraySidecarMetadata);
                if (sidecar != null)
                {
                    var fi = new FileInfo(sidecarPath);
                    componentsBuilder.Add(new ModelPackageComponent(
                        Role: ModelPackageRoles.SidecarMetadata,
                        Path: fi.FullName,
                        SizeBytes: fi.Length));
                }
            }
            catch
            {
                // Sidecars are advisory
            }
        }

        var identity = contentDigest != null
            ? new ModelPackageIdentity(contentDigest, ModelFormat.SafeTensors, identBuilder.ToImmutable(), IsProvisional: false)
            : ModelPackageIdentity.CreateProvisional(ModelFormat.SafeTensors, identBuilder.ToImmutable());

        return new SafeTensorsModelPackage(identity, Path.GetFullPath(packagePath), componentsBuilder.ToImmutable(), sidecar);
    }
}
