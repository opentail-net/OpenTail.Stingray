using System.Collections.Immutable;
using System.IO;
using System.Text.Json;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine.Packaging;
using OpenTail.Stingray.Engine.Planning;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

public sealed class ModelPackageTests
{
    [Fact]
    public void ModelPackageIdentity_PathDoesNotParticipateInEquality()
    {
        var compId = new ModelPackageComponentIdentity(
            Role: ModelPackageRoles.PrimaryWeights,
            Digest: "sha256:1111222233334444555566667777888899990000aaaaabbbbbcccccdddddeeeee",
            RelativeName: "model.gguf");

        var id1 = new ModelPackageIdentity(
            ContentDigest: "sha256:rootdigest123",
            Format: ModelFormat.Gguf,
            Components: [compId],
            IsProvisional: false);

        var id2 = new ModelPackageIdentity(
            ContentDigest: "sha256:rootdigest123",
            Format: ModelFormat.Gguf,
            Components: [compId],
            IsProvisional: false);

        // Record equality must be true regardless of any hypothetical filesystem path
        Assert.Equal(id1, id2);
        Assert.Equal(id1.GetHashCode(), id2.GetHashCode());
    }

    [Fact]
    public void ModelPackageIdentity_ProvisionalIdentity_PreservesFlag()
    {
        var compId = new ModelPackageComponentIdentity(
            Role: ModelPackageRoles.PrimaryWeights,
            Digest: null,
            RelativeName: "loose.gguf");

        var provId = ModelPackageIdentity.CreateProvisional(ModelFormat.Gguf, [compId]);
        Assert.True(provId.IsProvisional);
        Assert.Null(provId.ContentDigest);
        Assert.Equal(ModelFormat.Gguf, provId.Format);
    }

    [Fact]
    public void LooseGgufModelPackage_ResolvesComponentsAndSidecar()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "stingray_pkg_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);
        try
        {
            string ggufPath = Path.Combine(tempDir, "test_model.gguf");
            File.WriteAllBytes(ggufPath, new byte[64]);

            string mmprojPath = Path.Combine(tempDir, "test_mmproj.gguf");
            File.WriteAllBytes(mmprojPath, new byte[32]);

            string sidecarPath = Path.Combine(tempDir, "test_model.stingray.json");
            var sidecar = new StingraySidecarMetadata(
                SchemaVersion: 1,
                ModelDigest: "sha256:mockdigest",
                VerificationProfileVersion: "1.0.0",
                Architecture: "llama",
                SemanticFamily: "dense",
                StateModel: "kv",
                Admission: new StingraySidecarAdmission("admitted", "verified in CI"),
                Capabilities: new StingraySidecarCapabilities(
                    VerifiedBackends: ["cpu", "vulkan"],
                    ContinuousBatching: true)
            );
            string sidecarJson = JsonSerializer.Serialize(sidecar, ModelPackageJsonContext.Default.StingraySidecarMetadata);
            File.WriteAllText(sidecarPath, sidecarJson);

            var pkg = LooseGgufModelPackage.Open(
                ggufPath,
                mmprojPath: mmprojPath,
                contentDigest: "sha256:mockdigest");

            Assert.Equal(ModelFormat.Gguf, pkg.Format);
            Assert.Equal(Path.GetFullPath(ggufPath), pkg.PrimaryPath);
            Assert.Equal("sha256:mockdigest", pkg.Identity.ContentDigest);
            Assert.False(pkg.Identity.IsProvisional);

            // Three components: primary weights, mmproj, and sidecar
            Assert.Equal(3, pkg.Components.Length);
            Assert.Contains(pkg.Components, c => c.Role == ModelPackageRoles.PrimaryWeights && c.Path == Path.GetFullPath(ggufPath));
            Assert.Contains(pkg.Components, c => c.Role == ModelPackageRoles.VisionProjector && c.Path == Path.GetFullPath(mmprojPath));
            Assert.Contains(pkg.Components, c => c.Role == ModelPackageRoles.SidecarMetadata && c.Path == Path.GetFullPath(sidecarPath));

            Assert.NotNull(pkg.SidecarMetadata);
            Assert.Equal("llama", pkg.SidecarMetadata.Architecture);
            Assert.Equal("admitted", pkg.SidecarMetadata.Admission.Status);
            Assert.True(pkg.SidecarMetadata.Capabilities.ContinuousBatching);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void SafeTensorsModelPackage_ResolvesComponents()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "stingray_st_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);
        try
        {
            string stPath = Path.Combine(tempDir, "model.safetensors");
            File.WriteAllBytes(stPath, new byte[128]);

            string configPath = Path.Combine(tempDir, "config.json");
            File.WriteAllText(configPath, "{\"architectures\":[\"Qwen2ForCausalLM\"]}");

            string tokPath = Path.Combine(tempDir, "tokenizer.json");
            File.WriteAllText(tokPath, "{\"version\":\"1.0\"}");

            var pkg = SafeTensorsModelPackage.Open(tempDir);
            Assert.Equal(ModelFormat.SafeTensors, pkg.Format);
            Assert.Equal(3, pkg.Components.Length);
            Assert.Contains(pkg.Components, c => c.Role == ModelPackageRoles.PrimaryWeights);
            Assert.Contains(pkg.Components, c => c.Role == ModelPackageRoles.Config);
            Assert.Contains(pkg.Components, c => c.Role == ModelPackageRoles.Tokenizer);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void OllamaModelPackage_ResolvesBlobsAndDigestWithoutOllamaDaemon()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), "stingray_ollama_test_" + Guid.NewGuid().ToString("N"));
        string blobsDir = Path.Combine(tempDir, "blobs");
        Directory.CreateDirectory(blobsDir);

        try
        {
            string modelDigest = "sha256:aaaa1111222233334444555566667777888899990000aaaabbbbccccddddeeee";
            string templateDigest = "sha256:bbbb1111222233334444555566667777888899990000aaaabbbbccccddddeeee";

            // Create real GGUF blob file using Ollama naming convention: sha256-<hex>
            string modelBlobFile = Path.Combine(blobsDir, modelDigest.Replace(':', '-'));
            CreateTestGgufBlob(modelBlobFile);

            string templateBlobFile = Path.Combine(blobsDir, templateDigest.Replace(':', '-'));
            File.WriteAllBytes(templateBlobFile, new byte[64]);

            // Manifest referencing these blobs
            var manifest = new OllamaManifest(
                SchemaVersion: 2,
                MediaType: "application/vnd.docker.distribution.manifest.v2+json",
                Config: new OllamaLayer("application/vnd.docker.container.image.v1+json", "sha256:configdigest", 123),
                Layers:
                [
                    new OllamaLayer("application/vnd.ollama.image.model", modelDigest, 256),
                    new OllamaLayer("application/vnd.ollama.image.template", templateDigest, 64)
                ]
            );

            string manifestPath = Path.Combine(tempDir, "manifest.json");
            File.WriteAllText(manifestPath, JsonSerializer.Serialize(manifest, ModelPackageJsonContext.Default.OllamaManifest));

            var pkg = OllamaModelPackage.Open(manifestPath, blobsDir);
            Assert.Equal(ModelFormat.Gguf, pkg.Format);
            Assert.Equal("sha256:configdigest", pkg.Identity.ContentDigest);
            Assert.False(pkg.Identity.IsProvisional);
            Assert.Equal(Path.GetFullPath(modelBlobFile), pkg.ModelBlobPath);
            Assert.Equal(Path.GetFullPath(modelBlobFile), pkg.PrimaryPath);
            Assert.Equal(Path.GetFullPath(manifestPath), pkg.ManifestPath);

            Assert.Equal(2, pkg.Components.Length);
            Assert.Contains(pkg.Components, c => c.Role == ModelPackageRoles.PrimaryWeights && c.Digest == modelDigest);
            Assert.Contains(pkg.Components, c => c.Role == ModelPackageRoles.ChatTemplate && c.Digest == templateDigest);

            // Verify Ollama package is directly consumable by ModelDescription and ExecutionPlanner
            var desc = ModelDescription.FromPackage(pkg);
            Assert.Equal("llama", desc.Semantics.Architecture);
            Assert.Equal(1, desc.PlanningFacts.NumLayers);

            var req = new ExecutionRequest
            {
                PinnedBackend = "cpu",
                PinnedGpuLayers = 0,
                PinnedContextSize = 512,
                NoGpuProbe = true
            };
            var plan = ExecutionPlanner.Plan(desc, req, BackendCapabilities.Detect(noGpuProbe: true));
            Assert.NotNull(plan);
            Assert.Equal(2, plan.SchemaVersion);
            Assert.Equal("cpu", plan.Backend);
            Assert.Equal(512, plan.ContextSize);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    private static void CreateTestGgufBlob(string path)
    {
        var metadata = new Dictionary<string, (GgufValueType, object)>
        {
            ["general.architecture"] = (GgufValueType.String, "llama"),
            ["llama.block_count"] = (GgufValueType.UInt32, 1u),
            ["llama.context_length"] = (GgufValueType.UInt32, 1024u),
            ["llama.embedding_length"] = (GgufValueType.UInt32, 64u),
            ["llama.feed_forward_length"] = (GgufValueType.UInt32, 128u),
            ["llama.attention.head_count"] = (GgufValueType.UInt32, 2u),
            ["llama.attention.head_count_kv"] = (GgufValueType.UInt32, 2u),
            ["llama.vocab_size"] = (GgufValueType.UInt32, 100u)
        };

        var tensors = new (string name, long[] dims, DType dtype, byte[] data)[]
        {
            ("token_embd.weight", [64, 100], DType.Float32, new byte[64 * 100 * 4]),
            ("output.weight", [64, 100], DType.Float32, new byte[64 * 100 * 4]),
            ("blk.0.attn_q.weight", [64, 64], DType.Float32, new byte[64 * 64 * 4])
        };

        using var fs = File.Create(path);
        using var writer = new System.IO.BinaryWriter(fs);

        // Header
        writer.Write(0x46554747u); // 'GGUF'
        writer.Write(3u); // Version
        writer.Write((ulong)tensors.Length);
        writer.Write((ulong)metadata.Count);

        // Metadata
        foreach (var (k, (type, val)) in metadata)
        {
            var keyBytes = System.Text.Encoding.UTF8.GetBytes(k);
            writer.Write((ulong)keyBytes.Length);
            writer.Write(keyBytes);
            writer.Write((uint)type);
            if (type == GgufValueType.UInt32)
                writer.Write((uint)val);
            else if (type == GgufValueType.String)
            {
                var sBytes = System.Text.Encoding.UTF8.GetBytes((string)val);
                writer.Write((ulong)sBytes.Length);
                writer.Write(sBytes);
            }
        }

        // Tensor infos
        ulong offset = 0;
        foreach (var (name, dims, dtype, data) in tensors)
        {
            var nBytes = System.Text.Encoding.UTF8.GetBytes(name);
            writer.Write((ulong)nBytes.Length);
            writer.Write(nBytes);
            writer.Write((uint)dims.Length);
            foreach (var d in dims) writer.Write((ulong)d);
            writer.Write((uint)dtype);
            writer.Write(offset);
            offset += (ulong)data.Length;
        }

        // Alignment padding (32)
        long current = fs.Position;
        long rem = current % 32;
        if (rem != 0)
        {
            fs.Write(new byte[32 - rem]);
        }

        // Tensor data
        foreach (var (_, _, _, data) in tensors)
        {
            fs.Write(data);
        }
    }
}
