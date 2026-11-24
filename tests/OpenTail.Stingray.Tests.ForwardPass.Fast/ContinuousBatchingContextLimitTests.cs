namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Admission of a prompt against the context window (A28). A prompt longer than the window must be refused
/// with <see cref="ContextLimitExceededException"/> before any prefill work, and a prompt that exactly fills the
/// window must still be admitted. The canned forward pass and single-byte tokenizer come from
/// <see cref="ContinuousBatchingConstraintTests"/>; its forward pass reports <c>MaxSeqLen</c> 8192.
/// </summary>
public sealed class ContinuousBatchingContextLimitTests
{
    private const int Context = 8192;

    [Fact]
    public async Task PromptOneTokenLongerThanTheContext_IsRefusedAtAdmission_BeforeAnyPrefill()
    {
        var fwd = new ContinuousBatchingConstraintTests.FakeBatchedForwardPass();
        using var engine = new ContinuousBatchingEngine(
            fwd, new ContinuousBatchingConstraintTests.CharTokenizer(), "test", maxBatchSize: 1);
        string prompt = new('p', Context + 1);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 2 };

        var ex = await Assert.ThrowsAsync<ContextLimitExceededException>(() => DrainAsync(engine, prompt, sp));

        Assert.Equal(Context, ex.MaxContextTokens);
        Assert.Equal(Context + 1, ex.RequestedTokens);
        Assert.Equal(0, fwd.PrefilledTokens);
    }

    [Fact]
    public async Task PromptExactlyAtTheContext_IsAdmitted_AndGenerates()
    {
        var fwd = new ContinuousBatchingConstraintTests.FakeBatchedForwardPass();
        using var engine = new ContinuousBatchingEngine(
            fwd, new ContinuousBatchingConstraintTests.CharTokenizer(), "test", maxBatchSize: 1);
        string prompt = new('p', Context);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 2 };

        var chunks = new List<GenerateChunk>();
        await foreach (var chunk in engine.GenerateChunksAsync(prompt, sp))
            chunks.Add(chunk);

        Assert.Contains(chunks, c => c.Kind == GenerateChunkKind.Usage && c.PromptTokens == Context);
        Assert.Contains(chunks, c => c.Kind == GenerateChunkKind.Text);
    }

    private static async Task DrainAsync(ContinuousBatchingEngine engine, string prompt, SamplingParams sp)
    {
        await foreach (var chunk in engine.GenerateChunksAsync(prompt, sp))
            _ = chunk;
    }
}
