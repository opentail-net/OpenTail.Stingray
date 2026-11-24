namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Ordinary (non-retained) generation must never run a decode step at or past the context window (A29).
/// The canned forward pass records every decode position it is asked for; the window is set to 16 so the
/// boundary is reached within a few tokens. The fake's argmax is always 'X' and the single-byte tokenizer
/// maps each token to one character, so the output text is a run of 'X' whose length is the token count.
///
/// <para>Boundary rule (the off-by-one): entering a decode step at position P writes KV at P, so P must be
/// below the window. The token sampled at step P is materialised at P+1 on the next step, so it is kept only
/// when P+2 is at most the window. The first token comes from the prefill logits at position L (the prompt
/// length) and needs no KV write of its own.</para>
/// </summary>
public sealed class ContinuousBatchingDecodeBoundTests
{
    private const int Window = 16;

    [Fact]
    public async Task PromptOfTen_InWindowOfSixteen_DecodesOnlyUpToTheLastSlot_AndStopsWithLength()
    {
        // L = 10, window 16. t0 (prefill) sits at 10; step P writes KV at P and yields the next token.
        // Steps run at 10..15 (the last valid slot is 15), so t0..t5 are kept: 16 - 10 = 6 tokens.
        // The step at 15 yields t6, which would sit at 16, so it is discarded unlogged and unemitted.
        var (chunks, fwd) = await RunAsync(window: Window, promptLength: 10, maxNewTokens: 20);

        Assert.DoesNotContain(fwd.DecodePositions, p => p >= Window);
        Assert.Equal(Enumerable.Range(10, 6), fwd.DecodePositions);

        var stop = chunks.Single(c => c.Kind == GenerateChunkKind.Stop);
        Assert.Equal(6, stop.Stats!.OutputTokens);
        Assert.True(stop.TruncatedByMaxTokens, "a window stop must report the length reason");
        Assert.False(stop.TruncatedByResourceBudget);
        Assert.Equal(new string('X', 6), TextOf(chunks));
    }

    [Fact]
    public async Task PromptOfFifteen_InWindowOfSixteen_EmitsOneToken_AndOneDecodeStepAtTheLastSlot()
    {
        // L = 15: t0 sits at 15, the only valid slot. The single step at 15 materialises t0 and yields t1,
        // which would sit at 16 and is discarded. Expected: one token, one decode step, at 15.
        var (chunks, fwd) = await RunAsync(window: Window, promptLength: 15, maxNewTokens: 20);

        Assert.DoesNotContain(fwd.DecodePositions, p => p >= Window);
        Assert.Equal([15], fwd.DecodePositions);

        var stop = chunks.Single(c => c.Kind == GenerateChunkKind.Stop);
        Assert.Equal(1, stop.Stats!.OutputTokens);
        Assert.True(stop.TruncatedByMaxTokens);
        Assert.Equal("X", TextOf(chunks));
    }

    [Fact]
    public async Task PromptExactlyAtTheWindow_EmitsItsFirstToken_AndNeverDecodes()
    {
        // L = 16 = window: the first token comes from the prefill logits and needs no KV write, so it is
        // emitted, but no decode step may run (its position would be 16). Expected: one token, zero decode steps.
        var (chunks, fwd) = await RunAsync(window: Window, promptLength: Window, maxNewTokens: 20);

        Assert.Empty(fwd.DecodePositions);

        var stop = chunks.Single(c => c.Kind == GenerateChunkKind.Stop);
        Assert.Equal(1, stop.Stats!.OutputTokens);
        Assert.True(stop.TruncatedByMaxTokens);
        Assert.Equal("X", TextOf(chunks));
    }

    private static async Task<(List<GenerateChunk> Chunks, ContinuousBatchingConstraintTests.FakeBatchedForwardPass Fwd)> RunAsync(
        int window, int promptLength, int maxNewTokens)
    {
        var fwd = new ContinuousBatchingConstraintTests.FakeBatchedForwardPass(window);
        using var engine = new ContinuousBatchingEngine(
            fwd, new ContinuousBatchingConstraintTests.CharTokenizer(), "test", maxBatchSize: 1);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = maxNewTokens };
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));

        var chunks = new List<GenerateChunk>();
        await foreach (var chunk in engine.GenerateChunksAsync(new string('p', promptLength), sp, cts.Token))
            chunks.Add(chunk);
        return (chunks, fwd);
    }

    private static string TextOf(List<GenerateChunk> chunks) =>
        string.Concat(chunks.Where(c => c.Kind == GenerateChunkKind.Text).Select(c => c.Text));
}
