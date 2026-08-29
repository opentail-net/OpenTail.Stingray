
namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// Verifies the behavioural helpers added for the llama.cpp on-ramp plan:
/// <list type="bullet">
///   <item><see cref="RunCommand.TryParseLogitBias"/> — parses TOKEN_ID(+/-BIAS) entries.</item>
///   <item><see cref="RunCommand.ProcessEscapeSequences"/> (via reflection) — handles \n \t \\ \r.</item>
/// </list>
/// These are pure-logic helpers; no model load, no forward pass.
/// </summary>
public sealed class LlamaCompatBehaviourTests
{
    // ── TryParseLogitBias ─────────────────────────────────────────────────────

    [Fact]
    public void ParseLogitBias_PositiveBias_Parsed()
    {
        bool ok = RunCommand.TryParseLogitBias(["1234+1.5"], out var map, out string? err);
        Assert.True(ok);
        Assert.Null(err);
        Assert.NotNull(map);
        Assert.Equal(1.5f, map[1234], precision: 4);
    }

    [Fact]
    public void ParseLogitBias_NegativeBias_Parsed()
    {
        bool ok = RunCommand.TryParseLogitBias(["5678-100"], out var map, out string? err);
        Assert.True(ok);
        Assert.Null(err);
        Assert.NotNull(map);
        Assert.Equal(-100f, map[5678], precision: 4);
    }

    [Fact]
    public void ParseLogitBias_MultipleEntries_AllPresent()
    {
        bool ok = RunCommand.TryParseLogitBias(["1+10", "2-50", "3+0"], out var map, out _);
        Assert.True(ok);
        Assert.NotNull(map);
        Assert.Equal(3, map.Count);
        Assert.Equal(10f,  map[1], precision: 4);
        Assert.Equal(-50f, map[2], precision: 4);
        Assert.Equal(0f,   map[3], precision: 4);
    }

    [Fact]
    public void ParseLogitBias_EmptyArray_ReturnsEmptyMap()
    {
        bool ok = RunCommand.TryParseLogitBias([], out var map, out _);
        Assert.True(ok);
        Assert.NotNull(map);
        Assert.Empty(map);
    }

    [Fact]
    public void ParseLogitBias_MissingSign_ReturnsError()
    {
        bool ok = RunCommand.TryParseLogitBias(["1234"], out _, out string? err);
        Assert.False(ok);
        Assert.NotNull(err);
        Assert.Contains("TOKEN_ID", err, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ParseLogitBias_InvalidTokenId_ReturnsError()
    {
        bool ok = RunCommand.TryParseLogitBias(["abc+1.0"], out _, out string? err);
        Assert.False(ok);
        Assert.NotNull(err);
        Assert.Contains("token id", err, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ParseLogitBias_InvalidBiasValue_ReturnsError()
    {
        bool ok = RunCommand.TryParseLogitBias(["123+notanumber"], out _, out string? err);
        Assert.False(ok);
        Assert.NotNull(err);
        Assert.Contains("bias", err, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ParseLogitBias_InvariantCulture_ParsesDecimalPoint()
    {
        // Must parse "." regardless of the OS locale.
        bool ok = RunCommand.TryParseLogitBias(["1+0.7"], out var map, out _);
        Assert.True(ok);
        Assert.Equal(0.7f, map![1], precision: 4);
    }

    // ── ProcessEscapeSequences ────────────────────────────────────────────────
    // Called via the internal static method (same assembly); we expose it as internal for tests.

    [Theory]
    [InlineData(@"\n",   "\n")]
    [InlineData(@"\t",   "\t")]
    [InlineData(@"\r",   "\r")]
    [InlineData(@"\\",   "\\")]
    [InlineData(@"hello\nworld", "hello\nworld")]
    [InlineData(@"\n\t\\", "\n\t\\")]
    public void ProcessEscapeSequences_KnownEscapes_AreReplaced(string input, string expected)
    {
        string result = RunCommand.ProcessEscapeSequences(input);
        Assert.Equal(expected, result);
    }

    [Fact]
    public void ProcessEscapeSequences_NoBackslash_ReturnsSameReference()
    {
        const string input = "hello world";
        string result = RunCommand.ProcessEscapeSequences(input);
        Assert.Equal(input, result);
        // Fast path returns same string object (no allocation)
        Assert.Same(input, result);
    }

    [Fact]
    public void ProcessEscapeSequences_UnknownEscape_LeftAsIs()
    {
        // \q is not a known escape; the backslash is preserved
        string result = RunCommand.ProcessEscapeSequences(@"\q");
        Assert.Equal(@"\q", result);
    }

    [Fact]
    public void ProcessEscapeSequences_TrailingBackslash_LeftAsIs()
    {
        string result = RunCommand.ProcessEscapeSequences(@"end\");
        Assert.Equal(@"end\", result);
    }

    // ── --repeat-last-n: the WINDOW, not the parsed integer ───────────────────
    //
    // These assert the history-buffer size the decode loop actually uses. The alias tests only
    // prove the flag binds; an earlier version bound any value and then hard-coded the buffer at
    // 64, so `--repeat-last-n 256` and `-1` both silently behaved as 64 with every test green.

    [Fact]
    public void PenaltyHistoryCap_Default_Is64()
        => Assert.Equal(64, RunCommand.ResolvePenaltyHistoryCap(64));

    [Fact]
    public void PenaltyHistoryCap_Zero_RetainsNoHistory()
        => Assert.Equal(0, RunCommand.ResolvePenaltyHistoryCap(0));

    [Fact]
    public void PenaltyHistoryCap_MinusOne_IsUnbounded()
        => Assert.Equal(int.MaxValue, RunCommand.ResolvePenaltyHistoryCap(-1));

    [Theory]
    [InlineData(1)]
    [InlineData(63)]
    [InlineData(128)]
    [InlineData(256)]
    [InlineData(4096)]
    public void PenaltyHistoryCap_ExplicitWindow_IsHonouredExactly(int n)
    {
        // The regression this pins: values above 64 must NOT collapse back to 64.
        Assert.Equal(n, RunCommand.ResolvePenaltyHistoryCap(n));
    }
}
