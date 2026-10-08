using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

public sealed unsafe class ArchitectureModelResolverTests
{
    private sealed class Source(Dictionary<string, object> metadata) : IModelTensorSource
    {
        public IReadOnlyList<GgufTensorInfo> Tensors => [];
        public IReadOnlyDictionary<string, object> Metadata { get; } = metadata;
        public GgufTensorInfo? FindTensor(string name) => null;
        public ReadOnlySpan<byte> GetTensorData(GgufTensorInfo tensor) => [];
        public byte* GetTensorDataPtr(GgufTensorInfo tensor) => null;
        public void Dispose() { }
    }

    private static Source Make(string arch, int layers, params (string Key, object Value)[] extra)
    {
        var m = new Dictionary<string, object>
        {
            ["general.architecture"] = arch,
            [$"{arch}.block_count"] = (uint)layers,
            [$"{arch}.embedding_length"] = 64u,
            [$"{arch}.attention.head_count"] = 4u,
        };
        foreach (var (k, v) in extra) m[k] = v;
        return new Source(m);
    }

    [Fact]
    public void Resolve_DeclaredArchitecture_UsesDeclaredNamespaceAndDescriptorIdentity()
    {
        var r = ArchitectureModelResolver.Resolve(Make("qwen3", 7), "m.gguf");
        Assert.NotNull(r.Descriptor);
        Assert.Equal("qwen3", r.DeclaredArchitecture);
        Assert.Equal(r.Descriptor!.Id, r.CanonicalArchitecture);
        Assert.Equal("qwen3", r.MetadataArchitecture);
        Assert.Equal(7, r.Hyperparams.NumLayers);
    }

    [Fact]
    public void Resolve_UnknownArchitecture_IsUnresolvedButStillProducesBaseline()
    {
        var r = ArchitectureModelResolver.Resolve(Make("totally_unknown_arch", 5), "m.gguf");
        Assert.Null(r.Descriptor);
        Assert.Equal("totally_unknown_arch", r.CanonicalArchitecture);
        Assert.Equal(5, r.Hyperparams.NumLayers);
    }

    [Fact]
    public void Resolve_RelabelledFile_KeepsDeclaredMetadataNamespace()
    {
        var r = ArchitectureModelResolver.Resolve(
            Make("llama", 9, ("general.name", "mistral-3-instruct-7b")), "m.gguf");
        Assert.Equal("llama", r.DeclaredArchitecture);
        Assert.Equal("llama", r.MetadataArchitecture);
        Assert.NotNull(r.Descriptor);
        Assert.Equal(9, r.Hyperparams.NumLayers); // read under llama.* despite canonical identity
    }

    [Fact]
    public void Smollm3_Semantics_NoRopeEveryFourthLayer_BaselineHasNone()
    {
        var src = Make("smollm3", 8);
        var r = ArchitectureModelResolver.Resolve(src, "m.gguf");
        Assert.Equal("smollm3", r.CanonicalArchitecture);
        Assert.Equal(4, r.Hyperparams.NoRopeLayerStep);
        Assert.Equal(0, ModelHyperparams.CreateBaseline(src.Metadata, src, "smollm3").NoRopeLayerStep);
    }

    [Fact]
    public void Cohere2_Semantics_SwaPatternLayerNormParallelResidualLogitScale()
    {
        var src = Make("cohere2", 8,
            ("cohere2.attention.sliding_window", 4096u),
            ("cohere2.logit_scale", 0.0625f));
        var r = ArchitectureModelResolver.Resolve(src, "m.gguf");
        Assert.Equal("cohere2", r.CanonicalArchitecture);
        var hp = r.Hyperparams;
        Assert.True(hp.UsesLayerNorm);
        Assert.True(hp.UseParallelResidual);
        Assert.True(hp.RopeOnlySwaLayers);
        Assert.Equal(4096, hp.SlidingWindowSize);
        Assert.Equal(0.0625f, hp.LogitScale);
        // period 4 (default), last layer of each block global
        Assert.Equal(new[] { true, true, true, false, true, true, true, false }, hp.IsSwaLayer!.ToArray());

        var baseline = ModelHyperparams.CreateBaseline(src.Metadata, src, "cohere2");
        Assert.False(baseline.UsesLayerNorm);
        Assert.False(baseline.UseParallelResidual);
        Assert.Null(baseline.IsSwaLayer);
    }

    [Fact]
    public void Llama4_Semantics_SigmoidGatingL2QkNormNoPEChunkedAttention()
    {
        var hp = ArchitectureModelResolver.Resolve(Make("llama4", 8), "m.gguf").Hyperparams;
        Assert.Equal(4, hp.NoRopeLayerStep);
        Assert.True(hp.UseSigmoidGating);
        Assert.True(hp.UseL2QkNorm);
        Assert.True(hp.HasQkNorm);
        Assert.Equal(8192, hp.AttentionChunkSize);
        Assert.Equal(0.1f, hp.AttnTempScale);
        // explicit sliding_window == 0 disables chunking
        var off = ArchitectureModelResolver.Resolve(Make("llama4", 8, ("llama4.attention.sliding_window", 0u)), "m.gguf").Hyperparams;
        Assert.Equal(0, off.AttentionChunkSize);
        Assert.Equal(0f, off.AttnTempScale);
    }

    [Theory]
    [InlineData("olmo")]
    [InlineData("falcon")]
    [InlineData("jais2")]
    [InlineData("starcoder")]
    [InlineData("phimoe")]
    [InlineData("hunyuan-moe")]
    [InlineData("nemotron_h")]
    public void SimpleSemantics_AppliedByDescriptor_NotBaseline(string arch)
    {
        var src = Make(arch, 4);
        var r = ArchitectureModelResolver.Resolve(src, "m.gguf");
        var b = ModelHyperparams.CreateBaseline(src.Metadata, src, arch);
        var hp = r.Hyperparams;
        switch (arch)
        {
            case "olmo": Assert.True(hp.UsesUnweightedNorm); Assert.False(b.UsesUnweightedNorm); break;
            case "falcon": Assert.True(hp.UseParallelResidual); break;
            case "jais2": Assert.True(hp.UsesReluSquared); Assert.False(b.UsesReluSquared); break;
            case "starcoder": Assert.Equal(1, hp.NoRopeLayerStep); Assert.Equal(0, b.NoRopeLayerStep); break;
            case "phimoe": Assert.False(hp.UsesLayerNorm); Assert.True(hp.NormalizeMoeTopKWeights); break;
            case "hunyuan-moe": Assert.True(hp.QkNormAfterRope); Assert.True(hp.NormalizeMoeTopKWeights); break;
            case "nemotron_h": Assert.True(hp.UsesReluSquared); Assert.Equal(1, hp.NoRopeLayerStep); break;
        }
    }

    [Fact]
    public void MiniCpm_Semantics_FormulaDefaults_GraniteUsesReciprocalLogitScale()
    {
        var m = ArchitectureModelResolver.Resolve(Make("minicpm", 16), "m.gguf").Hyperparams;
        Assert.Equal(12f, m.EmbeddingScale);
        Assert.Equal(1.4f / MathF.Sqrt(16), m.ResidualScale);
        Assert.Equal(256f / 64, m.LogitScale);
        Assert.True(m.ScaleRawEmbeddings);

        var g = ArchitectureModelResolver.Resolve(Make("granite", 4,
            ("granite.logit_scale", 8f), ("granite.attention.scale", 0.015625f), ("granite.embedding_scale", 12f)), "m.gguf").Hyperparams;
        Assert.Equal(1f / 8f, g.LogitScale);
        Assert.Equal(0.015625f, g.AttentionScaleOverride);
        Assert.Equal(12f, g.EmbeddingScale);
    }

    [Fact]
    public void Exaone4_Semantics_SwaOnlyFor64Layers()
    {
        var big = ArchitectureModelResolver.Resolve(Make("exaone4", 64), "m.gguf").Hyperparams;
        Assert.Equal(4096, big.SlidingWindowSize);
        Assert.True(big.RopeOnlySwaLayers);
        Assert.Equal(new[] { true, true, true, false }, big.IsSwaLayer!.Take(4).ToArray());
        var small = ArchitectureModelResolver.Resolve(Make("exaone4", 30), "m.gguf").Hyperparams;
        Assert.Null(small.IsSwaLayer);
        Assert.False(small.RopeOnlySwaLayers);
    }

    [Fact]
    public void Afmoe_Semantics_OutputGateRopeOnlySwaMupEmbedding()
    {
        var hp = ArchitectureModelResolver.Resolve(Make("afmoe", 8, ("afmoe.attention.sliding_window", 1024u)), "m.gguf").Hyperparams;
        Assert.True(hp.AttentionOutputGate);
        Assert.True(hp.RopeOnlySwaLayers);
        Assert.False(hp.InputEmbeddingRmsNorm);
        Assert.Equal(MathF.Sqrt(64), hp.EmbeddingScale);
        Assert.Equal(new[] { true, true, true, false }, hp.IsSwaLayer!.Take(4).ToArray());
    }

    [Fact]
    public void Baseline_GenericStructuralFacts_NeedNoRegistryOrArchitecturePolicy()
    {
        var src = Make("zz_unregistered", 12,
            ("zz_unregistered.context_length", 8192u),
            ("zz_unregistered.attention.head_count_kv", 2u),
            ("zz_unregistered.attention.key_length", 32u),
            ("zz_unregistered.vocab_size", 1000u));
        var hp = ModelHyperparams.CreateBaseline(src.Metadata, src, "zz_unregistered");
        Assert.Equal(12, hp.NumLayers);
        Assert.Equal(64, hp.EmbeddingDim);
        Assert.Equal(4, hp.NumHeads);
        Assert.Equal(2, hp.NumKvHeads);
        Assert.Equal(32, hp.HeadDim);
        Assert.Equal(8192, hp.ContextLength);
        Assert.Equal(1000, hp.VocabSize);
        // no architecture-specific policy leaks into the baseline
        Assert.Equal(0, hp.NoRopeLayerStep);
        Assert.False(hp.UsesLayerNorm);
        Assert.Null(hp.IsSwaLayer);
        Assert.Equal(1f, hp.EmbeddingScale);
    }

    [Fact]
    public void Resolve_SemanticsHookIsOptional_ForOrdinaryDescriptors()
    {
        var r = ArchitectureModelResolver.Resolve(Make("qwen3", 3), "m.gguf");
        Assert.Null(r.Descriptor!.ApplyModelSemantics);
        // the only difference from the baseline is the descriptor's declarative RoPE pairing
        Assert.True(r.Descriptor.UsesNeoxRope);
        Assert.Equal(ModelHyperparams.CreateBaseline(r.Probe.TensorSource.Metadata, r.Probe.TensorSource, "qwen3") with { IsNeoxRope = true }, r.Hyperparams);
    }
}
