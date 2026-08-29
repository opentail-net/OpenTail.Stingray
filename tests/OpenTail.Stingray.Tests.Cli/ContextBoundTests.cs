
namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// The active context bounds prompt + generated output together, not the prompt alone.
///
/// <para>This matters more than a UX nicety. <c>ForwardPass</c> sizes its attention-score scratch
/// (<c>numHeads * ctxLen</c> floats) and its RoPE tables (<c>ctxLen</c> positions) from the context
/// ceiling, but its <c>PagedKvCache</c> defaults to 8192 blocks — 131,072 positions — so the cache
/// keeps accepting appends long after the position has run off the end of those buffers. Attention
/// writes <c>scores[h * ctxLen + t]</c> and RoPE reads <c>ropeCos + pos * halfDim</c>, both
/// unchecked native accesses, so overrunning corrupts memory rather than failing.</para>
///
/// <para>Wiring <c>--ctx-size</c> through to the CPU <c>ForwardPass</c> made that reachable in
/// ordinary use: with <c>--ctx-size 512</c> and the default <c>--n-predict 512</c>, a prompt of
/// any length at all decodes past the ceiling unless EOS happens to stop it first. A prompt-only
/// check does not catch it. <c>ForwardCore</c> now throws on an out-of-range position as the
/// backstop; this clamp is what keeps ordinary "context full" from reaching that throw.</para>
/// </summary>
public sealed class ContextBoundTests
{
    [Fact]
    public void Clamp_LeavesRequestUnchanged_WhenPromptAndOutputFit()
    {
        var sp = new SamplingParams { MaxNewTokens = 100 };

        var result = RunCommand.ClampToRemainingContext(sp, promptTokens: 10, maxContextLength: 512);

        Assert.Equal(100, result.MaxNewTokens);
    }

    /// <summary>
    /// The exact case the prompt-only check misses: the prompt fits comfortably, but prompt +
    /// n-predict does not. Before the clamp this decoded to position 511 + 400 with scratch
    /// sized for 512.
    /// </summary>
    [Fact]
    public void Clamp_BoundsGeneration_WhenPromptFitsButOutputWouldOverrun()
    {
        var sp = new SamplingParams { MaxNewTokens = 512 };

        var result = RunCommand.ClampToRemainingContext(sp, promptTokens: 400, maxContextLength: 512);

        Assert.Equal(112, result.MaxNewTokens);
    }

    /// <summary>
    /// The default configuration is itself over-committed: --ctx-size N with the default
    /// --n-predict 512 cannot fit both once the prompt is non-empty. Pinned as a Theory so the
    /// invariant is prompt + output &lt;= context across sizes, not one arithmetic example.
    /// </summary>
    [Theory]
    [InlineData(512, 1, 512)]
    [InlineData(512, 256, 512)]
    [InlineData(2048, 100, 512)]
    [InlineData(128, 127, 512)]
    public void Clamp_KeepsPromptPlusOutputWithinContext(int ctx, int promptTokens, int requested)
    {
        var sp = new SamplingParams { MaxNewTokens = requested };

        var result = RunCommand.ClampToRemainingContext(sp, promptTokens, ctx);

        Assert.True(promptTokens + result.MaxNewTokens <= ctx,
            $"prompt {promptTokens} + output {result.MaxNewTokens} exceeds context {ctx}");
        // Clamp, never inflate: a request that already fits must be preserved exactly.
        Assert.True(result.MaxNewTokens <= requested);
    }
}
