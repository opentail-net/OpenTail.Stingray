
namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// docs/032-multi-model-inference-runtime-plan.md Phase 7 — <see cref="IInferenceService"/> model
/// resolution/discovery. No real model/GGUF involved: multi-model entries use their own
/// per-model <see cref="NamedModelOptions.EngineFactory"/> (fakes), exactly like the existing
/// single-model <see cref="OpenTailStingrayServerOptions.EngineFactory"/> escape hatch tests do.
/// </summary>
public sealed class InferenceServiceTests
{
    [Fact]
    public void SingleModelMode_ResolveModel_IgnoresRequestedModel_AlwaysReturnsTheOneConfiguredModel()
    {
        var s = new ServiceCollection();
        s.AddOpenTailStingray(opts =>
        {
            opts.EngineFactory = _ => new LoadedEngine(new FakeInferenceEngine("x"), "qwen2", null);
            opts.ModelPath = "/models/only-one.gguf";
        });
        var sp = s.BuildServiceProvider();
        var service = sp.GetRequiredService<IInferenceService>();

        var expected = ModelId.Canonicalize("/models/only-one.gguf");
        Assert.Equal(expected, service.ResolveModel(null));
        Assert.Equal(expected, service.ResolveModel(""));
        Assert.Equal(expected, service.ResolveModel("gpt-4")); // whatever the client asks for — ignored
        Assert.Equal(expected, service.ResolveModel("only-one.gguf"));
    }

    [Fact]
    public void SingleModelMode_AvailableModelAliases_HasExactlyOneEntry()
    {
        var s = new ServiceCollection();
        s.AddOpenTailStingray(opts => opts.ModelPath = "/models/only-one.gguf");
        var sp = s.BuildServiceProvider();
        var service = sp.GetRequiredService<IInferenceService>();

        var alias = Assert.Single(service.AvailableModelAliases);
        Assert.Equal(ModelId.Canonicalize("/models/only-one.gguf").Value, alias);
    }

    [Fact]
    public async Task MultiModelMode_ResolveModel_MatchesAliasCaseInsensitively_AndRoutesAcquireToTheRightEngine()
    {
        var sidekickEngine = new FakeInferenceEngine("sidekick-engine");
        var reasonerEngine = new FakeInferenceEngine("reasoner-engine");

        var s = new ServiceCollection();
        s.AddOpenTailStingray(opts =>
        {
            opts.Models =
            [
                new NamedModelOptions
                {
                    Alias = "sidekick",
                    ModelPath = "/models/sidekick.gguf",
                    EngineFactory = _ => new LoadedEngine(sidekickEngine, "qwen2", null),
                },
                new NamedModelOptions
                {
                    Alias = "reasoner",
                    ModelPath = "/models/reasoner.gguf",
                    EngineFactory = _ => new LoadedEngine(reasonerEngine, "qwen2", null),
                },
            ];
        });
        var sp = s.BuildServiceProvider();
        var service = sp.GetRequiredService<IInferenceService>();

        var sidekickId = service.ResolveModel("SIDEKICK"); // case-insensitive
        var reasonerId = service.ResolveModel("reasoner");
        Assert.NotEqual(sidekickId, reasonerId);

        using var sidekickHandle = await service.AcquireAsync(sidekickId);
        using var reasonerHandle = await service.AcquireAsync(reasonerId);
        Assert.Same(sidekickEngine, sidekickHandle.Runtime.Loaded.Engine);
        Assert.Same(reasonerEngine, reasonerHandle.Runtime.Loaded.Engine);
    }

    [Fact]
    public void MultiModelMode_ResolveModel_NullOrEmpty_ReturnsFirstConfiguredModel()
    {
        var s = new ServiceCollection();
        s.AddOpenTailStingray(opts =>
        {
            opts.Models =
            [
                new NamedModelOptions
                {
                    Alias = "sidekick",
                    ModelPath = "/models/sidekick.gguf",
                    EngineFactory = _ => new LoadedEngine(new FakeInferenceEngine("a"), "qwen2", null),
                },
                new NamedModelOptions
                {
                    Alias = "reasoner",
                    ModelPath = "/models/reasoner.gguf",
                    EngineFactory = _ => new LoadedEngine(new FakeInferenceEngine("b"), "qwen2", null),
                },
            ];
        });
        var sp = s.BuildServiceProvider();
        var service = sp.GetRequiredService<IInferenceService>();

        Assert.Equal(service.ResolveModel("sidekick"), service.ResolveModel(null));
        Assert.Equal(service.ResolveModel("sidekick"), service.ResolveModel(""));
    }

    [Fact]
    public void MultiModelMode_ResolveModel_UnknownAlias_ThrowsModelNotFoundException()
    {
        var s = new ServiceCollection();
        s.AddOpenTailStingray(opts =>
        {
            opts.Models =
            [
                new NamedModelOptions
                {
                    Alias = "sidekick",
                    ModelPath = "/models/sidekick.gguf",
                    EngineFactory = _ => new LoadedEngine(new FakeInferenceEngine("a"), "qwen2", null),
                },
            ];
        });
        var sp = s.BuildServiceProvider();
        var service = sp.GetRequiredService<IInferenceService>();

        var ex = Assert.Throws<ModelNotFoundException>(() => service.ResolveModel("nonexistent"));
        Assert.Equal("nonexistent", ex.RequestedModel);
        Assert.Contains("sidekick", ex.AvailableAliases);
    }

    [Fact]
    public void MultiModelMode_AvailableModelAliases_ListsEveryConfiguredAlias_InOrder()
    {
        var s = new ServiceCollection();
        s.AddOpenTailStingray(opts =>
        {
            opts.Models =
            [
                new NamedModelOptions
                {
                    Alias = "sidekick",
                    ModelPath = "/models/sidekick.gguf",
                    EngineFactory = _ => new LoadedEngine(new FakeInferenceEngine("a"), "qwen2", null),
                },
                new NamedModelOptions
                {
                    Alias = "reasoner",
                    ModelPath = "/models/reasoner.gguf",
                    EngineFactory = _ => new LoadedEngine(new FakeInferenceEngine("b"), "qwen2", null),
                },
            ];
        });
        var sp = s.BuildServiceProvider();
        var service = sp.GetRequiredService<IInferenceService>();

        Assert.Equal(["sidekick", "reasoner"], service.AvailableModelAliases);
    }

    [Fact]
    public void MultiModelMode_DuplicateAlias_ThrowsAtFirstResolve()
    {
        var s = new ServiceCollection();
        s.AddOpenTailStingray(opts =>
        {
            opts.Models =
            [
                new NamedModelOptions
                {
                    Alias = "dup",
                    ModelPath = "/models/a.gguf",
                    EngineFactory = _ => new LoadedEngine(new FakeInferenceEngine("a"), "qwen2", null),
                },
                new NamedModelOptions
                {
                    Alias = "dup",
                    ModelPath = "/models/b.gguf",
                    EngineFactory = _ => new LoadedEngine(new FakeInferenceEngine("b"), "qwen2", null),
                },
            ];
        });
        var sp = s.BuildServiceProvider();

        Assert.Throws<InvalidOperationException>(() => sp.GetRequiredService<IInferenceService>());
    }

    [Fact]
    public async Task MultiModelMode_TwoAcquiresOfSameAlias_ShareOneRuntime()
    {
        var engine = new FakeInferenceEngine("shared");
        var s = new ServiceCollection();
        s.AddOpenTailStingray(opts =>
        {
            opts.Models =
            [
                new NamedModelOptions
                {
                    Alias = "sidekick",
                    ModelPath = "/models/sidekick.gguf",
                    EngineFactory = _ => new LoadedEngine(engine, "qwen2", null),
                },
            ];
        });
        var sp = s.BuildServiceProvider();
        var service = sp.GetRequiredService<IInferenceService>();
        var id = service.ResolveModel("sidekick");

        using var h1 = await service.AcquireAsync(id);
        using var h2 = await service.AcquireAsync(id);

        Assert.Same(h1.Runtime, h2.Runtime); // single-flight/shared residency, same as single-model mode
    }
}
