using OpenTail.Stingray;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Real-model check of the A29 context bound. Ordinary generation (no retained state) on a real GGUF, CPU backend,
/// with a 64-token context and a max_tokens far beyond the window must stop at the window, never run a decode step
/// at or past it, and throw nothing. Gated: runs only with STINGRAY_RUN_REAL_CONTEXT_TEST=1 and when the checkpoint
/// is found by <see cref="ModelLocator"/>; otherwise it is reported as skipped.
/// </summary>
public sealed class ContinuousBatchingRealContextTests
{
    private const string ModelFile = "qwen2.5-0.5b-instruct-q4_k_m.gguf";
    private const int ContextTokens = 64;
    private const int RequestedMaxTokens = 200;
    private const string Prompt = "Write a short story about a lighthouse keeper who finds a message in a bottle.";

    [Fact]
    public async Task OrdinaryGeneration_OnRealModel_StopsAtTheContextWindow_WithoutAnException()
    {
        if (Environment.GetEnvironmentVariable("STINGRAY_RUN_REAL_CONTEXT_TEST") != "1")
            Assert.Skip("set STINGRAY_RUN_REAL_CONTEXT_TEST=1 to run the real-model context-window check");
        string? path = ModelLocator.FindDeep(ModelFile);
        if (path is null) Assert.Skip($"{ModelFile} not found by ModelLocator (checkpoint missing).");

        using var model = Model.Load(new ModelParams(path) { Backend = "cpu" });
        using var ctx = (ModelContext)model.CreateContext(new ContextParams { ContextSize = ContextTokens });
        using var engine = ctx.CreateContinuousBatchingEngine(maxBatchSize: 1);

        // StopTokenIds replaces the end-of-generation set with an id the model never emits, so the window is the
        // only stop that can end the request: the test exercises the boundary rather than an early EOS.
        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = RequestedMaxTokens, StopTokenIds = [-1] };
        using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(5));
        var chunks = new List<GenerateChunk>();
        await foreach (var chunk in engine.GenerateChunksAsync(Prompt, sp, cts.Token))
            chunks.Add(chunk);

        int promptTokens = chunks.Single(c => c.Kind == GenerateChunkKind.Usage).PromptTokens;
        var stop = chunks.Single(c => c.Kind == GenerateChunkKind.Stop);
        int generated = stop.Stats!.OutputTokens;
        int windowAllowance = ContextTokens - promptTokens;
        Console.WriteLine($"A29 real-model: prompt={promptTokens} generated={generated} allowance={windowAllowance} lengthStop={stop.TruncatedByMaxTokens}");

        Assert.True(generated <= windowAllowance,
            $"generated {generated} tokens after a {promptTokens}-token prompt; the {ContextTokens}-token window allows {windowAllowance}");
        Assert.True(generated < RequestedMaxTokens, "the window, not max_tokens, must have ended the request");
        Assert.True(stop.TruncatedByMaxTokens, "a window stop must report the length reason");
        Assert.Equal(windowAllowance, generated);
    }
}
