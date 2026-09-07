using OpenTail.Stingray.Audio.HiggsAudio;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Pure-logic real-formula test for the delay-pattern helper ported from `codebooks.cpp`
/// (see `HiggsCodebooks`'s doc comment) -- no checkpoint needed.</summary>
public sealed class HiggsCodebooksTests
{
    [Fact]
    public void ApplyDelayPattern_ThenReverse_RoundTripsRawCodes()
    {
        const int rawFrames = 4, codebooks = 3;
        var raw = new int[rawFrames * codebooks];
        for (int i = 0; i < raw.Length; i++) raw[i] = i + 10;

        int delayedFrames = HiggsCodebooks.DelayedFrameCount(rawFrames, codebooks);
        Assert.Equal(rawFrames + codebooks - 1, delayedFrames);

        var delayed = HiggsCodebooks.ApplyDelayPattern(raw, rawFrames, codebooks);
        Assert.Equal(delayedFrames * codebooks, delayed.Length);

        // Codebook k's leading k frames are BOC padding.
        for (int codebook = 0; codebook < codebooks; codebook++)
            for (int frame = 0; frame < codebook; frame++)
                Assert.Equal(HiggsCodebooks.BocId, delayed[frame * codebooks + codebook]);

        // Codebook k's real codes start at delayed frame k.
        for (int codebook = 0; codebook < codebooks; codebook++)
            for (int frame = 0; frame < rawFrames; frame++)
                Assert.Equal(raw[frame * codebooks + codebook], delayed[(codebook + frame) * codebooks + codebook]);

        var roundTripped = HiggsCodebooks.ReverseDelayPattern(delayed, delayedFrames, codebooks);
        Assert.Equal(raw, roundTripped);
    }
}
