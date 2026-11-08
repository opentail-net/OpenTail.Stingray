using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Phase 2 acceptance (Qwen 3.8 plan): qwen4exp requests driven through <see cref="ContinuousBatchingEngine"/> must match
/// their isolated runs, under concurrency, staggered arrival and cancellation. Greedy decoding, real checkpoint only.
/// </summary>
public sealed class Qwen4ExpBatchingTests
{
    private static readonly string[] Prompts =
    {
        "The capital of France is",
        "The largest planet in the solar system is",
        "Once upon a time,",
        "def fibonacci(n):",
    };

    private const int MaxNewTokens = 6;

    private static string? FindModel() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(
        "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    [Fact]
    public async Task ConcurrentAndStaggeredRequests_MatchIsolatedRuns()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var model = GgufModel.Open(path!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var pass = new Qwen4ExpForwardPass(model, hp, tensors);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var adapter = new Qwen4ExpBatchedForwardPass(pass, hp, totalLayers);
        using var engine = new ContinuousBatchingEngine(adapter, tokenizer, "qwen4exp-test", maxBatchSize: 4);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = MaxNewTokens };
        using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(20));

        // 1. Isolated baselines, one request at a time.
        var baseline = new string[Prompts.Length];
        for (int i = 0; i < Prompts.Length; i++)
            baseline[i] = await CollectAsync(engine, Prompts[i], sp, cts.Token);

        // 2. All four concurrently.
        var concurrent = await Task.WhenAll(Prompts.Select(p => CollectAsync(engine, p, sp, cts.Token)));
        for (int i = 0; i < Prompts.Length; i++)
            Assert.Equal(baseline[i], concurrent[i]);

        // 3. Staggered arrivals: each request starts a little after the previous one.
        var staggered = new List<Task<string>>();
        foreach (var p in Prompts)
        {
            staggered.Add(CollectAsync(engine, p, sp, cts.Token));
            await Task.Delay(TimeSpan.FromMilliseconds(200), cts.Token);
        }
        var staggeredResults = await Task.WhenAll(staggered);
        for (int i = 0; i < Prompts.Length; i++)
            Assert.Equal(baseline[i], staggeredResults[i]);
    }

    [Fact]
    public async Task CancelledRequest_DoesNotChangeTheOthers()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var model = GgufModel.Open(path!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var pass = new Qwen4ExpForwardPass(model, hp, tensors);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var adapter = new Qwen4ExpBatchedForwardPass(pass, hp, totalLayers);
        using var engine = new ContinuousBatchingEngine(adapter, tokenizer, "qwen4exp-test", maxBatchSize: 4);

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = MaxNewTokens };
        using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(20));
        string expected = await CollectAsync(engine, Prompts[0], sp, cts.Token);

        using var cancelled = new CancellationTokenSource();
        var victim = CollectAsync(engine, Prompts[1], sp, cancelled.Token);
        var survivor = CollectAsync(engine, Prompts[0], sp, cts.Token);
        await Task.Delay(TimeSpan.FromMilliseconds(300), cts.Token);
        cancelled.Cancel();
        try { await victim; } catch (OperationCanceledException) { }

        Assert.Equal(expected, await survivor);
    }

    [Fact]
    public async Task ShortRequestFinishingFirst_LeavesLongerOnesIntact_AndReleasesMemory()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var model = GgufModel.Open(path!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var pass = new Qwen4ExpForwardPass(model, hp, tensors);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var adapter = new Qwen4ExpBatchedForwardPass(pass, hp, totalLayers);
        using var engine = new ContinuousBatchingEngine(adapter, tokenizer, "qwen4exp-test", maxBatchSize: 4);

        var shortSp = new SamplingParams { Temperature = 0f, MaxNewTokens = 2 };
        var longSp = new SamplingParams { Temperature = 0f, MaxNewTokens = 12 };
        using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(20));

        string longBaseline = await CollectAsync(engine, Prompts[2], longSp, cts.Token);
        string shortBaseline = await CollectAsync(engine, Prompts[0], shortSp, cts.Token);

        var longTask = CollectAsync(engine, Prompts[2], longSp, cts.Token);
        var shortTask = CollectAsync(engine, Prompts[0], shortSp, cts.Token);
        Assert.Equal(shortBaseline, await shortTask);
        Assert.Equal(longBaseline, await longTask);

        // Everything has finished: no committed KV should be left behind.
        Assert.Equal(0, engine.CommittedKvTokens);
    }

    private static async Task<string> CollectAsync(
        ContinuousBatchingEngine engine, string prompt, SamplingParams sp, CancellationToken ct)
    {
        var sb = new System.Text.StringBuilder();
        await foreach (var chunk in engine.GenerateAsync(prompt, sp, ct))
            sb.Append(chunk);
        return sb.ToString();
    }
}
