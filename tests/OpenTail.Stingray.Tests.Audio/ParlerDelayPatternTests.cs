
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Golden verification for <see cref="ParlerDelayPattern"/> against the real
/// `build_delay_pattern_mask`/`apply_delay_pattern_mask` from the already-local
/// `scratch-llamacpp-ref/parler-pkg/parler_tts-0.2.3/parler_tts/modeling_parler_tts.py`, run
/// directly via PyTorch (not reimplemented from the docstring alone -- the docstring's own
/// worked example is one of the three cases checked here, but cases 2 and 3 independently
/// exercise the real function with a non-empty prompt and a full apply-mask round trip).
/// </summary>
public sealed class ParlerDelayPatternTests
{
    private const int Bos = 1024;
    private const int Pad = 1025;
    private const int NumCodebooks = 4;
    private const int MaxLength = 8;

    [Fact]
    public void Build_NoPrompt_MatchesRealDocstringExample()
    {
        var input = new int[NumCodebooks][];
        for (int cb = 0; cb < NumCodebooks; cb++) input[cb] = [Bos];

        var (outIds, pattern) = ParlerDelayPattern.Build(input, Bos, Pad, MaxLength, NumCodebooks);

        int[][] expectedOutIds = [[Bos], [Bos], [Bos], [Bos]];
        int[][] expectedPattern =
        [
            [Bos, -1, -1, -1, -1, Pad, Pad, Pad],
            [Bos, Bos, -1, -1, -1, -1, Pad, Pad],
            [Bos, Bos, Bos, -1, -1, -1, -1, Pad],
            [Bos, Bos, Bos, Bos, -1, -1, -1, -1],
        ];

        Assert.Equal(expectedOutIds, outIds);
        Assert.Equal(expectedPattern, pattern);
    }

    [Fact]
    public void Build_WithRealPrompt_MatchesGoldenOutput()
    {
        int[][] prompt =
        [
            [Bos, 5, 6],
            [Bos, 7, 8],
            [Bos, 9, 10],
            [Bos, 11, 12],
        ];

        var (outIds, pattern) = ParlerDelayPattern.Build(prompt, Bos, Pad, MaxLength, NumCodebooks);

        int[][] expectedOutIds =
        [
            [Bos, 5, 6],
            [Bos, Bos, 7],
            [Bos, Bos, Bos],
            [Bos, Bos, Bos],
        ];
        int[][] expectedPattern =
        [
            [Bos, 5, 6, -1, -1, Pad, Pad, Pad],
            [Bos, Bos, 7, 8, -1, -1, Pad, Pad],
            [Bos, Bos, Bos, 9, 10, -1, -1, Pad],
            [Bos, Bos, Bos, Bos, 11, 12, -1, -1],
        ];

        Assert.Equal(expectedOutIds, outIds);
        Assert.Equal(expectedPattern, pattern);
    }

    [Fact]
    public void Apply_FullyGeneratedSequence_MatchesGoldenOutput()
    {
        int[][] pattern =
        [
            [Bos, -1, -1, -1, -1, Pad, Pad, Pad],
            [Bos, Bos, -1, -1, -1, -1, Pad, Pad],
            [Bos, Bos, Bos, -1, -1, -1, -1, Pad],
            [Bos, Bos, Bos, Bos, -1, -1, -1, -1],
        ];

        // Fill every -1 with a distinct dummy "generated" value, scanning codebook-major then
        // position-major -- matches the real oracle script's fill order exactly.
        var full = new int[NumCodebooks][];
        for (int cb = 0; cb < NumCodebooks; cb++) full[cb] = (int[])pattern[cb].Clone();
        int genVal = 100;
        for (int cb = 0; cb < NumCodebooks; cb++)
            for (int pos = 0; pos < MaxLength; pos++)
                if (full[cb][pos] == -1) full[cb][pos] = genVal++;

        var applied = ParlerDelayPattern.Apply(full, pattern);

        int[][] expected =
        [
            [Bos, 100, 101, 102, 103, Pad, Pad, Pad],
            [Bos, Bos, 104, 105, 106, 107, Pad, Pad],
            [Bos, Bos, Bos, 108, 109, 110, 111, Pad],
            [Bos, Bos, Bos, Bos, 112, 113, 114, 115],
        ];

        Assert.Equal(expected, applied);
    }
}
