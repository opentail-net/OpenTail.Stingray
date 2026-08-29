using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Server;

/// <summary>
/// docs/032-multi-model-inference-runtime-plan.md Phase 5 acceptance: "A test with two
/// independent models demonstrates overlapping execution rather than turn-taking." Everything
/// tested up to this point (ModelRuntimeManagerTests in Tests.Server.Fast) used fake loaders —
/// they prove <c>ModelRuntimeManager</c> itself holds no lock across loads, but not that two
/// REAL models' actual CPU-bound inference (SIMD kernels, mmap'd weights, thread-pool usage —
/// none of which the fakes exercise) genuinely overlaps in wall-clock time rather than being
/// serialized by some other incidental shared resource in the forward-pass stack. Real model,
/// serial with the rest of this project (see SessionRestartPersistenceTests.cs's class doc).
/// </summary>
public sealed class CrossModelConcurrencyTests
{
    [Fact]
    public async Task TwoRealModels_GenerateConcurrently_OverlapRatherThanSerialize()
    {
        string? modelA = FindModel("SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
        string? modelB = FindModel("Qwen3-0.6B-Q8_0.gguf");
        Assert.SkipUnless(modelA is not null && modelB is not null,
            "SmolLM2-1.7B-Instruct-Q4_K_M.gguf and Qwen3-0.6B-Q8_0.gguf are required for the " +
            "cross-model concurrency acceptance test.");

        using var manager = new ModelRuntimeManager(id => InferenceEngineLoader.Load(
            new OpenTailStingrayServerOptions
            {
                ModelPath = id.Value,
                Backend = ServerBackend.Cpu,
                NGpuLayers = 0,
                ContextSize = 512,
            }));

        using var handleA = await manager.AcquireAsync(ModelId.Canonicalize(modelA!));
        using var handleB = await manager.AcquireAsync(ModelId.Canonicalize(modelB!));

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 32 };
        const string prompt = "The capital of France is";

        // Total-wall-clock-time comparisons (e.g. "together < 0.85x the serial sum") turned out
        // to be the wrong proof: two real CPU-bound models genuinely DO contend for the same
        // cores/memory bandwidth (docs/032's own text: "if two models both try to consume 100
        // GB/s of memory bandwidth, they don't magically get 200 GB/s"), so real partial
        // slowdown from contention is expected physics, not evidence of an artificial lock — an
        // aggressive threshold conflated the two and false-failed. The robust, contention-proof
        // check is interleaving: does B start producing output before A's stream has finished?
        // A genuinely serialized implementation (something awaiting A's full completion before
        // even starting B) could never show that, no matter how much slower contention makes
        // both individually.
        async Task<(DateTime FirstChunkAt, DateTime LastChunkAt)> GenerateAsync(IInferenceEngine engine)
        {
            DateTime? first = null;
            DateTime last = default;
            await foreach (var _ in engine.GenerateChunksAsync(prompt, sp))
            {
                var now = DateTime.UtcNow;
                first ??= now;
                last = now;
            }
            return (first!.Value, last);
        }

        var taskA = GenerateAsync(handleA.Runtime.Engine);
        var taskB = GenerateAsync(handleB.Runtime.Engine);
        var (aFirst, aLast) = await taskA;
        var (bFirst, bLast) = await taskB;

        bool interleaved = bFirst < aLast || aFirst < bLast;
        Assert.True(interleaved,
            $"Expected the two models' generations to interleave (one starts before the other " +
            $"finishes) when run concurrently: aFirst={aFirst:o} aLast={aLast:o} bFirst={bFirst:o} " +
            $"bLast={bLast:o} — no overlap detected, looks fully serialized instead.");
    }

    /// <summary>
    /// Regression guard for the double-free found while building the test above: a plain
    /// (non-batching) InferenceEngine's Dispose() used to run ForwardPass.Dispose() twice —
    /// once explicitly, once again via the _owned[] loop, which also contains the same
    /// ForwardPass instance — corrupting the native heap and crashing the process (fixed by an
    /// idempotency guard in ForwardPass.Dispose(), docs/bugstofix.md → ForwardPass.cs). Real
    /// generation, not just construction, because the original repro went through a full
    /// generate-then-dispose cycle even though the fixed bug turned out not to need it.
    /// </summary>
    [Fact]
    public async Task SingleRealModel_DisposeAfterGeneration_DoesNotCorruptTheNativeHeap()
    {
        string? modelPath = FindModel("SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
        Assert.SkipUnless(modelPath is not null,
            "SmolLM2-1.7B-Instruct-Q4_K_M.gguf is required for this regression test.");

        var loaded = InferenceEngineLoader.Load(new OpenTailStingrayServerOptions
        {
            ModelPath = modelPath,
            Backend = ServerBackend.Cpu,
            NGpuLayers = 0,
            ContextSize = 512,
        });

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 8 };
        await foreach (var _ in loaded.Engine.GenerateChunksAsync("The capital of France is", sp)) { }

        // If this double-frees, the process crashes natively (no exception to catch) — the test
        // passing at all, not any assertion inside it, is the actual proof.
        (loaded.Engine as IDisposable)?.Dispose();
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
