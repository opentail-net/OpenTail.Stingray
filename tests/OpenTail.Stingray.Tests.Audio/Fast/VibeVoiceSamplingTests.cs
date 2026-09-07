using OpenTail.Stingray.Audio.VibeVoice;

namespace OpenTail.Stingray.Tests.Audio.Fast;

/// <summary>Structural tests for <see cref="VibeVoiceSampling"/>.</summary>
public sealed class VibeVoiceSamplingTests
{
    [Fact]
    public void ArgmaxToken_ReturnsHighestScoringIndex()
    {
        float[] logits = [0.1f, 5.0f, -2.0f, 3.0f];
        Assert.Equal(1, VibeVoiceSampling.ArgmaxToken(logits, compareBf16: false));
    }

    [Fact]
    public void ArgmaxToken_Bf16Rounding_TreatsNearlyEqualValuesAsTied_FirstWins()
    {
        // Values that differ only in low-precision mantissa bits should round to the same BF16
        // value, so the tie-break (first-encountered wins, via strict '>') keeps the earlier index.
        float a = 1.0f;
        float b = MathF.BitIncrement(MathF.BitIncrement(a)); // tiny epsilon above a, below BF16 resolution
        float[] logits = [a, b];
        Assert.Equal(0, VibeVoiceSampling.ArgmaxToken(logits, compareBf16: true));
    }

    [Fact]
    public void ApplyRepetitionPenalty_PenalizesSeenTokensOnce_UsesRealSignBranch()
    {
        float[] logits = [2.0f, -2.0f, 5.0f];
        VibeVoiceSampling.ApplyRepetitionPenalty(logits, promptIds: [0], generated: [1], penalty: 2.0f);

        // token 0: positive logit -> divide by penalty
        Assert.Equal(1.0f, logits[0], precision: 5);
        // token 1: negative logit -> multiply by penalty
        Assert.Equal(-4.0f, logits[1], precision: 5);
        // token 2: not seen -> unchanged
        Assert.Equal(5.0f, logits[2], precision: 5);
    }

    [Fact]
    public void ApplyRepetitionPenalty_UnitPenalty_NoOps()
    {
        float[] logits = [2.0f, -2.0f];
        float[] before = (float[])logits.Clone();
        VibeVoiceSampling.ApplyRepetitionPenalty(logits, [0, 1], [], penalty: 1.0f);
        Assert.Equal(before, logits);
    }
}
