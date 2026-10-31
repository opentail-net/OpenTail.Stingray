using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Server;

/// <summary>
/// Plan 2 P4 side finding: the prefill chunk size (server option <c>PrefillChunkTokens</c> and <c>STINGRAY_PREFILL_CHUNK</c>) must reach the
/// continuous-batching engine the loader builds. Until 2026-10-31 <c>RuntimeInstance</c> never passed it, so the setting did nothing.
/// Uses a small real model (Qwen2.5-0.5B) because the wiring runs through the real planner and loader.
/// </summary>
public sealed class PrefillChunkWiringRealModelTests
{
    private const string ModelFile = "qwen2.5-0.5b-instruct-q4_k_m.gguf";

    [Fact]
    public void ServerOption_ReachesTheBatchingEngine()
    {
        string? model = FindModel();
        Assert.SkipUnless(model is not null, ModelFile + " is required for this test.");

        Assert.Equal(48, ChunkOfLoadedEngine(model!, optionValue: 48));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(0)]
    public void WithoutAnExplicitSize_TheEngineKeepsItsAutomaticChunk(int optionValue)
    {
        string? model = FindModel();
        Assert.SkipUnless(model is not null, ModelFile + " is required for this test.");

        int chunk = ChunkOfLoadedEngine(model!, optionValue);

        Assert.True(chunk > 0, $"the automatic chunk must stay enabled, got {chunk}");
        Assert.NotEqual(48, chunk);
    }

    [Fact]
    public void EnvironmentVariable_ReachesTheBatchingEngine_WhenTheOptionIsUnset()
    {
        string? model = FindModel();
        Assert.SkipUnless(model is not null, ModelFile + " is required for this test.");

        string? previous = Environment.GetEnvironmentVariable("STINGRAY_PREFILL_CHUNK");
        try
        {
            Environment.SetEnvironmentVariable("STINGRAY_PREFILL_CHUNK", "96");
            Assert.Equal(96, ChunkOfLoadedEngine(model!, optionValue: -1));
        }
        finally { Environment.SetEnvironmentVariable("STINGRAY_PREFILL_CHUNK", previous); }
    }

    private static int ChunkOfLoadedEngine(string modelPath, int optionValue)
    {
        var loaded = InferenceEngineLoader.Load(new OpenTailStingrayServerOptions
        {
            ModelPath = modelPath,
            Backend = ServerBackend.Cpu,
            NGpuLayers = 0,
            ContextSize = 512,
            MaxBatchSize = 2,   // continuous batching; the chunk size is only consulted there
            PrefillChunkTokens = optionValue,
        });
        try
        {
            var engine = Assert.IsType<ContinuousBatchingEngine>(Assert.IsType<OpenTail.Stingray.Engine.Runtime.RuntimeInstance>(loaded.Instance).Engine);
            return engine.PrefillChunkTokens;
        }
        finally { ((IDisposable?)loaded.Instance ?? loaded.Engine as IDisposable)?.Dispose(); }
    }

    private static string? FindModel()
    {
        foreach (string root in new[] { @"F:\_models", @"E:\_models", @"H:\_models" })
        {
            string candidate = Path.Combine(root, ModelFile);
            if (File.Exists(candidate)) return candidate;
        }
        string directory = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            string candidate = Path.Combine(directory, "models", ModelFile);
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(directory);
            if (parent is null) break;
            directory = parent.FullName;
        }
        return null;
    }
}
