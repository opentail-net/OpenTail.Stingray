using OpenTail.Stingray.Server;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Plan 6 C1 (docs/032 Phase 3): the model manager is told what a candidate is expected to put on an accelerator, so accelerator-aware admission can run in a real
/// server. These tests pin the rules of <see cref="ServiceCollectionExtensions.CreateAcceleratorEstimator"/> with the plan-based figure replaced; the figure
/// itself, against a real model and the real loaded runtime, is <c>AcceleratorResidencyTests</c> in Tests.Server.
/// </summary>
public sealed class AcceleratorEstimatorTests
{
    private const long Gib = 1L << 30;

    private static string PathOf(string name) => Path.Combine(Path.GetTempPath(), "opentail_estimator_" + name + ".gguf");

    [Fact]
    public void SingleModel_UsesTheEstimateOfItsOptions()
    {
        var opts = new OpenTailStingrayServerOptions { ModelPath = PathOf("single") };
        OpenTailStingrayServerOptions? seen = null;
        var estimator = ServiceCollectionExtensions.CreateAcceleratorEstimator(opts, o => { seen = o; return 3 * Gib; });

        Assert.Equal(3 * Gib, estimator(ModelId.Canonicalize(opts.ModelPath!)));
        Assert.Same(opts, seen);
    }

    [Fact]
    public void ACustomEngineFactory_IsNotEstimated_BecauseTheServerCannotKnowWhatItLoads()
    {
        var opts = new OpenTailStingrayServerOptions { ModelPath = PathOf("custom"), EngineFactory = _ => throw new InvalidOperationException("not loaded here") };
        var estimator = ServiceCollectionExtensions.CreateAcceleratorEstimator(opts, _ => throw new InvalidOperationException("must not be asked"));

        Assert.Equal(0, estimator(ModelId.Canonicalize(opts.ModelPath!)));
    }

    [Fact]
    public void NamedModels_EachGetTheEstimateOfTheirOwnEffectiveOptions()
    {
        var a = new NamedModelOptions { Alias = "a", ModelPath = PathOf("a"), Backend = ServerBackend.Vulkan, NGpuLayers = -1 };
        var b = new NamedModelOptions { Alias = "b", ModelPath = PathOf("b"), Backend = ServerBackend.Cpu };
        var opts = new OpenTailStingrayServerOptions { Models = [a, b] };
        var seenPaths = new List<(string? Path, ServerBackend Backend)>();
        var estimator = ServiceCollectionExtensions.CreateAcceleratorEstimator(opts, o =>
        {
            seenPaths.Add((o.ModelPath, o.Backend));
            return o.Backend == ServerBackend.Vulkan ? 2 * Gib : 0;
        });

        Assert.Equal(2 * Gib, estimator(ModelId.Canonicalize(a.ModelPath)));
        Assert.Equal(0, estimator(ModelId.Canonicalize(b.ModelPath)));
        Assert.Equal([(a.ModelPath, ServerBackend.Vulkan), (b.ModelPath, ServerBackend.Cpu)], seenPaths);   // the per-model overrides were applied
    }

    [Fact]
    public void AnUnknownModel_AndANamedModelWithItsOwnFactory_AreZero()
    {
        var withFactory = new NamedModelOptions { Alias = "factory", ModelPath = PathOf("factory"), EngineFactory = _ => throw new InvalidOperationException() };
        var opts = new OpenTailStingrayServerOptions { Models = [withFactory] };
        var estimator = ServiceCollectionExtensions.CreateAcceleratorEstimator(opts, _ => throw new InvalidOperationException("must not be asked"));

        Assert.Equal(0, estimator(ModelId.Canonicalize(PathOf("not-configured"))));
        Assert.Equal(0, estimator(ModelId.Canonicalize(withFactory.ModelPath)));
    }

    [Fact]
    public void EachAnswerIsComputedOnce()
    {
        var opts = new OpenTailStingrayServerOptions { ModelPath = PathOf("once") };
        int calls = 0;
        var estimator = ServiceCollectionExtensions.CreateAcceleratorEstimator(opts, _ => { calls++; return Gib; });
        var id = ModelId.Canonicalize(opts.ModelPath!);

        Assert.Equal(Gib, estimator(id));
        Assert.Equal(Gib, estimator(id));
        Assert.Equal(1, calls);
    }

    [Fact]
    public void ThePlanBasedEstimate_IsZeroWhenTheModelCannotBeRead_NotAnException()
    {
        var opts = new OpenTailStingrayServerOptions { ModelPath = PathOf("missing-" + Guid.NewGuid().ToString("N")) };

        Assert.Equal(0, InferenceEngineLoader.EstimateAcceleratorBytes(opts));
    }
}
