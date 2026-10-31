using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Server;

/// <summary>
/// Plan 032 Phase 6, slice 5, against a real model: engines built by <c>InferenceEngineLoader</c> from options that carry a shared
/// <see cref="FairExecutionGate"/> take turns on it, and taking turns does not change what a model generates. The in-memory
/// engine tests (Tests.ForwardPass.Fast) pin the mechanism; this proves the options-to-engine wiring is real.
/// </summary>
public sealed class ExecutionTurnGateRealModelTests
{
    [Fact]
    public async Task TwoRealEnginesSharingAGate_TakeTurns_AndGenerateTheSameTextAsAnUngatedEngine()
    {
        string? modelPath = FindModel("SmolLM2-135M-Instruct-Q4_K_M.gguf");
        Assert.SkipUnless(modelPath is not null, "SmolLM2-135M-Instruct-Q4_K_M.gguf is required for this test.");

        var gate = new FairExecutionGate();
        using var gatedA = Load(modelPath!, gate);
        using var gatedB = Load(modelPath!, gate);
        using var plain = Load(modelPath!, gate: null);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 24 };
        const string prompt = "The capital of France is";

        string reference = await Generate(plain, prompt, sp);
        Assert.False(string.IsNullOrWhiteSpace(reference));

        // Both gated engines generate at the same time, so they genuinely compete for turns.
        var runA = Generate(gatedA, prompt, sp);
        var runB = Generate(gatedB, prompt, sp);
        string[] texts = await Task.WhenAll(runA, runB).WaitAsync(TimeSpan.FromMinutes(2));

        Assert.Equal(reference, texts[0]);
        Assert.Equal(reference, texts[1]);

        var stats = gate.GetStats();
        Assert.True(stats.Grants >= 2 * 20, $"only {stats.Grants} turns granted for two 24-token generations");
        Assert.Equal(0, stats.Waiting);
    }


    private static Owned Load(string modelPath, IExecutionTurnGate? gate) => new(
        InferenceEngineLoader.Load(new OpenTailStingrayServerOptions
        {
            ModelPath = modelPath,
            Backend = ServerBackend.Cpu,
            NGpuLayers = 0,
            ContextSize = 512,
            MaxBatchSize = 2,   // selects continuous batching, the only engine that takes turns
            ExecutionTurnGate = gate,
        }));

    private static async Task<string> Generate(Owned owned, string prompt, SamplingParams sp)
    {
        var engine = Assert.IsType<ContinuousBatchingEngine>(Assert.IsType<OpenTail.Stingray.Engine.Runtime.RuntimeInstance>(owned.Loaded.Instance).Engine);
        var text = new System.Text.StringBuilder();
        await foreach (var chunk in engine.GenerateChunksAsync(prompt, sp)) text.Append(chunk.Text);
        return text.ToString();
    }

    private sealed class Owned(LoadedEngine loaded) : IDisposable
    {
        public LoadedEngine Loaded { get; } = loaded;
        public void Dispose() => ((IDisposable?)Loaded.Instance ?? Loaded.Engine as IDisposable)?.Dispose();
    }

    private static string? FindModel(string fileName)
    {
        string directory = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            string candidate = Path.Combine(directory, "models", fileName);
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(directory);
            if (parent is null) break;
            directory = parent.FullName;
        }
        return null;
    }
}
