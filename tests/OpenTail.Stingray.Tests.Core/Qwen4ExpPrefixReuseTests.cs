using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Phase 3 acceptance (Qwen 3.8 plan): exact-prefix reuse for qwen4exp. A reused prefix must give the same output as a cold
/// run, and branching or A/B/A sequences must stay independent. Real checkpoint, greedy, skipped when absent.
/// </summary>
public sealed class Qwen4ExpPrefixReuseTests
{
    private const string SharedPrefix = "The capital of France is Paris. The capital of Germany is Berlin. The capital of";
    private const string BranchA = SharedPrefix + " Italy is";
    private const string BranchB = SharedPrefix + " Spain is";

    private static string? FindModel() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(
        Qwen4ExpTestSupport.TrunkFileName);

    [Fact]
    public async Task RepeatedPrompt_ReusesPrefix_AndMatchesColdRun()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var model = GgufModel.Open(path!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var pass = new Qwen4ExpForwardPass(model, hp, tensors);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 6 };
        using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(20));

        string cold = await ColdAsync(pass, tokenizer, BranchA, sp, cts.Token);

        using var engine = new ContinuousBatchingEngine(new Qwen4ExpBatchedForwardPass(pass, hp, totalLayers), tokenizer, "qwen4exp-test", maxBatchSize: 4);
        string first = await CollectAsync(engine, BranchA, sp, cts.Token);
        string second = await CollectAsync(engine, BranchA, sp, cts.Token);

        Assert.Equal(cold, first);
        Assert.Equal(cold, second);
        Assert.True(engine.PrefillTokensReused > 0, "The second identical prompt did not reuse any prefix.");
    }

    [Fact]
    public async Task Branches_StayIndependent_AndABA_MatchesColdRuns()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var model = GgufModel.Open(path!);
        int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
        using var pass = new Qwen4ExpForwardPass(model, hp, tensors);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 6 };
        using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(20));

        string coldA = await ColdAsync(pass, tokenizer, BranchA, sp, cts.Token);
        string coldB = await ColdAsync(pass, tokenizer, BranchB, sp, cts.Token);

        using var engine = new ContinuousBatchingEngine(new Qwen4ExpBatchedForwardPass(pass, hp, totalLayers), tokenizer, "qwen4exp-test", maxBatchSize: 4);
        string a1 = await CollectAsync(engine, BranchA, sp, cts.Token);
        string b1 = await CollectAsync(engine, BranchB, sp, cts.Token);
        string a2 = await CollectAsync(engine, BranchA, sp, cts.Token);

        Assert.Equal(coldA, a1);
        Assert.Equal(coldB, b1);
        Assert.Equal(coldA, a2);
    }

    private static async Task<string> ColdAsync(
        Qwen4ExpForwardPass pass, ITokenizer tokenizer, string prompt, SamplingParams sp, CancellationToken ct)
    {
        using var engine = new ContinuousBatchingEngine(
            new Qwen4ExpBatchedForwardPass(pass, pass.Hyperparameters, pass.NumLayers), tokenizer, "qwen4exp-cold", maxBatchSize: 4);
        return await CollectAsync(engine, prompt, sp, ct);
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
