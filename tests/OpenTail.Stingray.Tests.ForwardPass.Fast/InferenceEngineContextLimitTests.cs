using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// The single-request <see cref="InferenceEngine"/> (used when a plan's batching is not continuous) against a context
/// window (A28 open follow-up, A29). A prompt longer than the window is refused with
/// <see cref="ContextLimitExceededException"/> before any prefill. A prompt of exactly the window runs its prefill, and
/// has no decode budget, so it ends at the window with no output tokens. Decode never goes past the window.
/// </summary>
public sealed class InferenceEngineContextLimitTests
{
    private const int Window = 8;
    private const int TokX = 4;

    [Fact]
    public async Task PromptOneTokenLongerThanTheWindow_IsRefusedBeforeAnyPrefill()
    {
        var tokenizer = new InferenceEngineChunkTests.ScriptedTokenizer { PromptTokens = Enumerable.Repeat(0, Window + 1).ToArray() };
        int prefills = 0;
        var fwd = new InferenceEngineChunkTests.ScriptedForwardPass([], tokenizer.VocabSize, Window) { OnPrefill = () => prefills++ };
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 4 };

        var ex = await Assert.ThrowsAsync<ContextLimitExceededException>(() => DrainAsync(engine, sp));

        Assert.Equal(Window, ex.MaxContextTokens);
        Assert.Equal(Window + 1, ex.RequestedTokens);
        Assert.Equal(0, prefills);
    }

    [Fact]
    public async Task PromptExactlyTheWindow_IsAdmitted_AndEndsAtTheWindowWithNoOutputTokens()
    {
        var tokenizer = new InferenceEngineChunkTests.ScriptedTokenizer { PromptTokens = Enumerable.Repeat(0, Window).ToArray() };
        int prefills = 0;
        var fwd = new InferenceEngineChunkTests.ScriptedForwardPass([TokX, TokX], tokenizer.VocabSize, Window) { OnPrefill = () => prefills++ };
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 4 };
        var chunks = new List<GenerateChunk>();

        await foreach (var chunk in engine.GenerateChunksAsync("seed", sp))
            chunks.Add(chunk);

        Assert.True(prefills >= 1, "a prompt of exactly the window must still be prefilled");
        Assert.Equal(Window, chunks.Single(c => c.Kind == GenerateChunkKind.Usage).PromptTokens);
        // No decode budget is left (the window is full), so no token is decoded. Stated as the single-request
        // behaviour: the continuous batcher emits one prefill token here, this engine emits none.
        Assert.DoesNotContain(chunks, c => c.Kind == GenerateChunkKind.Text);
        Assert.True(chunks.Single(c => c.Kind == GenerateChunkKind.Stop).TruncatedByMaxTokens,
            "a run that ends at the window reports the length reason");
    }

    [Fact]
    public async Task DecodeStopsAtTheWindow_WhenTheBudgetIsLargerThanTheRoom()
    {
        // Prompt 5, window 8: three positions remain, so at most three tokens are decoded even though 10 are requested.
        var tokenizer = new InferenceEngineChunkTests.ScriptedTokenizer { PromptTokens = Enumerable.Repeat(0, 5).ToArray() };
        var fwd = new InferenceEngineChunkTests.ScriptedForwardPass(Enumerable.Repeat(TokX, 10).ToArray(), tokenizer.VocabSize, Window);
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 10 };
        var chunks = new List<GenerateChunk>();

        await foreach (var chunk in engine.GenerateChunksAsync("seed", sp))
            chunks.Add(chunk);

        Assert.Equal(3, chunks.Count(c => c.Kind == GenerateChunkKind.Text));
        Assert.True(chunks.Single(c => c.Kind == GenerateChunkKind.Stop).TruncatedByMaxTokens);
    }

    [Fact]
    public void CheckPromptFitsContext_RefusesALongerPrompt_WithoutPrefill()
    {
        var tokenizer = new InferenceEngineChunkTests.ScriptedTokenizer { PromptTokens = Enumerable.Repeat(0, Window + 1).ToArray() };
        int prefills = 0;
        var fwd = new InferenceEngineChunkTests.ScriptedForwardPass([], tokenizer.VocabSize, Window) { OnPrefill = () => prefills++ };
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        var ex = Assert.Throws<ContextLimitExceededException>(() => engine.CheckPromptFitsContext("seed"));

        Assert.Equal(Window, ex.MaxContextTokens);
        Assert.Equal(0, prefills);
    }

    [Fact]
    public void CheckPromptFitsContext_AcceptsAPromptOfExactlyTheWindow()
    {
        var tokenizer = new InferenceEngineChunkTests.ScriptedTokenizer { PromptTokens = Enumerable.Repeat(0, Window).ToArray() };
        var fwd = new InferenceEngineChunkTests.ScriptedForwardPass([], tokenizer.VocabSize, Window);
        using var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1);

        engine.CheckPromptFitsContext("seed");
    }

    private static async Task DrainAsync(InferenceEngine engine, SamplingParams sp)
    {
        await foreach (var chunk in engine.GenerateChunksAsync("seed", sp))
            _ = chunk;
    }
}
