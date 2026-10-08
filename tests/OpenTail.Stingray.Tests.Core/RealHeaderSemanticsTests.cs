using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Header-only checks (GGUF metadata + tensor inventory, no weight math) that descriptor-owned model semantics
/// resolve correctly on REAL checkpoints. Skips visibly when a checkpoint is not on disk; real-weight generation
/// receipts remain in the ForwardPass parity tests.
/// </summary>
public sealed class RealHeaderSemanticsTests
{
    private static string? Find(string name)
    {
        var dir = AppContext.BaseDirectory;
        for (int i = 0; i < 10 && dir is not null; i++)
        {
            foreach (var sub in new[] { "models", Path.Combine("models", "_models") })
            {
                var p = Path.Combine(dir, sub, name);
                if (File.Exists(p)) return p;
            }
            dir = Path.GetDirectoryName(dir);
        }
        return null;
    }

    private static ResolvedModelArchitecture Resolve(string file, out GgufModel gguf)
    {
        var path = Find(file);
        Assert.SkipWhen(path is null, $"{file} not present in models/");
        gguf = GgufModel.Open(path!);
        return ArchitectureModelResolver.Resolve(gguf, path, gguf);
    }

    [Fact]
    public void SmolLM3_RealHeader_NoRopeEveryFourthLayer()
    {
        var r = Resolve("SmolLM3-Q4_K_M.gguf", out var g);
        using (g)
        {
            Assert.Equal("smollm3", r.CanonicalArchitecture);
            Assert.Equal(4, r.Hyperparams.NoRopeLayerStep);
        }
    }

    [Fact]
    public void CommandR7B_RealHeader_Cohere2Semantics()
    {
        var r = Resolve("c4ai-command-r7b-12-2024-Q4_K_M.gguf", out var g);
        using (g)
        {
            Assert.Equal("cohere2", r.CanonicalArchitecture);
            var hp = r.Hyperparams;
            Assert.True(hp.UsesLayerNorm);
            Assert.True(hp.UseParallelResidual);
            Assert.True(hp.RopeOnlySwaLayers);
            Assert.True(hp.SlidingWindowSize > 0);
            Assert.NotNull(hp.IsSwaLayer);
            Assert.Equal(hp.NumLayers, hp.IsSwaLayer!.Count);
            // every 4th layer is global
            for (int i = 0; i < hp.NumLayers; i++)
                Assert.Equal(i % 4 != 3, hp.IsSwaLayer[i]);
            Assert.True(hp.LogitScale > 0f);
        }
    }

    [Theory]
    [InlineData("gemma-4-E4B-it-Q4_K_M.gguf")]
    [InlineData("gemma-4-12b-it-Q4_K_M.gguf")]
    public void Gemma4_RealHeader_PerLayerSemanticsPopulated(string file)
    {
        var r = Resolve(file, out var g);
        using (g)
        {
            Assert.Equal("gemma4", r.CanonicalArchitecture);
            var hp = r.Hyperparams;
            int n = hp.NumLayers;
            Assert.NotNull(hp.IsSwaLayer);
            Assert.Equal(n, hp.IsSwaLayer!.Count);
            Assert.Equal(n, hp.LayerHeadDim!.Count);
            Assert.Equal(n, hp.LayerRopeDim!.Count);
            Assert.Contains(hp.IsSwaLayer, b => b);
            Assert.Contains(hp.IsSwaLayer, b => !b);
            Assert.Equal(MathF.Sqrt(hp.EmbeddingDim), hp.EmbeddingScale);
            Assert.Equal(FfnActivation.GeluApprox, hp.FfnActivation);
            Assert.True(hp.SlidingWindowSize > 0);
            Assert.True(hp.RopeThetaSwa > 0f);
            if (hp.KvSourceLayer is { } src) Assert.Equal(n, src.Count);
            if (hp.LayerKvHeads is { } kv) Assert.All(kv, x => Assert.True(x > 0));
        }
    }

    /// <summary>
    /// Model.Load, ModelDescription.FromModel and the resolver must agree for the same real file
    /// (RuntimeInstance consumes Model.Hyperparams, so this also pins what the runtime sees).
    /// </summary>
    [Theory]
    [InlineData("SmolLM3-Q4_K_M.gguf")]
    [InlineData("c4ai-command-r7b-12-2024-Q4_K_M.gguf")]
    [InlineData("gemma-4-E4B-it-Q4_K_M.gguf")]
    public void ModelLoad_ModelDescription_AndResolver_Converge(string file)
    {
        var r = Resolve(file, out var g);
        using (g)
        {
            using var model = OpenTail.Stingray.Model.Load(Find(file)!);
            var desc = OpenTail.Stingray.Engine.Planning.ModelDescription.FromModel(model);
            var hp = model.Hyperparams;
            Assert.Equal(r.CanonicalArchitecture, model.Architecture);
            Assert.Equal(r.CanonicalArchitecture, desc.Semantics.Architecture);
            Assert.Equal(r.Hyperparams.NumLayers, hp.NumLayers);
            Assert.Equal(r.Hyperparams.NumLayers, desc.PlanningFacts.NumLayers);
            Assert.Equal(r.Hyperparams.NoRopeLayerStep, hp.NoRopeLayerStep);
            Assert.Equal(r.Hyperparams.SlidingWindowSize, hp.SlidingWindowSize);
            Assert.Equal(r.Hyperparams.UsesLayerNorm, hp.UsesLayerNorm);
            Assert.Equal(r.Hyperparams.EmbeddingScale, hp.EmbeddingScale);
            Assert.Equal(r.Hyperparams.IsSwaLayer?.ToArray(), hp.IsSwaLayer?.ToArray());
            Assert.Equal(r.Hyperparams.LayerHeadDim?.ToArray(), hp.LayerHeadDim?.ToArray());
        }
    }
}
