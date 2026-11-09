using System.Collections.Immutable;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Phase 3 acceptance (Qwen 3.8 plan): a retained conversation. The second turn on a retained state must generate the
/// same tokens as a cold run over the whole conversation (prompt, first reply, second prompt). Real checkpoint, greedy.
/// </summary>
public sealed class Qwen4ExpRetainedTurnTests
{
    private const string Turn1 = "The capital of France is";
    private const string Turn2 = " Paris. The capital of Germany is";
    private const int MaxNewTokens = 6;

    private static string? FindModel() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(
        Qwen4ExpTestSupport.TrunkFileName);

    [Fact]
    public async Task SecondRetainedTurn_EqualsColdRunOfWholeConversation()
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
        using var engine = new ContinuousBatchingEngine(adapter, tokenizer, "qwen4exp-test", maxBatchSize: 2);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = MaxNewTokens };
        using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(20));

        using var state = new RetainedSequenceState();
        await DrainAsync(engine.GenerateRetainedChunksAsync(Turn1, sp, state, cts.Token));
        var outcome1 = state.LastTurn ?? throw new InvalidOperationException("turn 1 produced no outcome");
        int[] prompt1 = tokenizer.Encode(Turn1).ToArray();
        int[] reply1 = outcome1.GeneratedTokenIds.ToArray();
        Assert.Equal(prompt1.Length + reply1.Length, outcome1.MaterializedPosition);

        await DrainAsync(engine.GenerateRetainedChunksAsync(Turn2, sp, state, cts.Token));
        var outcome2 = state.LastTurn ?? throw new InvalidOperationException("turn 2 produced no outcome");
        int[] reply2 = outcome2.GeneratedTokenIds.ToArray();

        // Cold run over the same token sequence the retained cache holds: prompt 1, reply 1, prompt 2.
        int[] prompt2 = tokenizer.Encode(Turn2).ToArray();
        int[] conversation = prompt1.Concat(reply1).Concat(prompt2).ToArray();
        int[] coldReply2 = ColdGreedy(pass, conversation, MaxNewTokens, tokenizer);

        Assert.Equal(coldReply2, reply2);
    }

    [Fact]
    public async Task CancelledSecondTurn_RollsBack_AndRerunMatchesUninterrupted()
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
        using var engine = new ContinuousBatchingEngine(adapter, tokenizer, "qwen4exp-test", maxBatchSize: 2);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = MaxNewTokens };
        using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(20));

        // State A: turn 1, then turn 2 uninterrupted (reference).
        using var stateA = new RetainedSequenceState();
        await DrainAsync(engine.GenerateRetainedChunksAsync(Turn1, sp, stateA, cts.Token));
        await DrainAsync(engine.GenerateRetainedChunksAsync(Turn2, sp, stateA, cts.Token));
        int[] reference = stateA.LastTurn!.GeneratedTokenIds.ToArray();

        // State B: turn 1, then turn 2 cancelled after its first chunk.
        using var stateB = new RetainedSequenceState();
        await DrainAsync(engine.GenerateRetainedChunksAsync(Turn1, sp, stateB, cts.Token));
        int turn1Position = stateB.LastTurn!.MaterializedPosition;
        using (var cancelTurn2 = new CancellationTokenSource())
        {
            try
            {
                await foreach (var _ in engine.GenerateRetainedChunksAsync(Turn2, sp, stateB, cancelTurn2.Token))
                    cancelTurn2.Cancel();
            }
            catch (OperationCanceledException) { }
        }

        // The cancellation is only visible once the batcher has settled the state.
        Assert.False(stateB.IsInUse);

        // Re-running turn 2 must start from the turn-start boundary and reproduce the reference exactly.
        await DrainAsync(engine.GenerateRetainedChunksAsync(Turn2, sp, stateB, cts.Token));
        var rerun = stateB.LastTurn!;
        Assert.Equal(turn1Position, rerun.TurnStartPosition);
        Assert.Equal(reference, rerun.GeneratedTokenIds.ToArray());
    }

    /// <summary>Greedy generation from a fresh sequence over <paramref name="prompt"/>, stopping at end-of-generation.</summary>
    private static int[] ColdGreedy(Qwen4ExpForwardPass pass, int[] prompt, int steps, GgufTokenizer tokenizer)
    {
        var state = pass.CreateSequenceState();
        int pos = 0;
        ReadOnlySpan<float> logits = default;
        foreach (int t in prompt) logits = pass.Forward(t, pos++, state);

        var generated = new List<int>();
        for (int s = 0; s < steps; s++)
        {
            int best = 0;
            for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
            if (tokenizer.EogTokenIds.Contains(best)) break;
            generated.Add(best);
            logits = pass.Forward(best, pos++, state);
        }
        return generated.ToArray();
    }

    private static async Task DrainAsync(IAsyncEnumerable<GenerateChunk> chunks)
    {
        await foreach (var _ in chunks) { }
    }
}
