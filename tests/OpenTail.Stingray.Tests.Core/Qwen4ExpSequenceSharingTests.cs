using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Phase 3 memory (Qwen 3.8 plan): a qwen4exp sequence snapshot shares the QSA history rows with the live sequence and
/// copies only the list spines, so the attention history is not duplicated per snapshot. Synthetic, no checkpoint.
/// </summary>
public sealed class Qwen4ExpSequenceSharingTests
{
    private static Qwen4ExpSequenceState NewState() =>
        new(2, _ => false, gdnConvLength: 0, gdnStateLength: 0, pleHistoryLength: 0, pleHasher: null);

    [Fact]
    public void Clone_SharesRows_ButNotTheLists()
    {
        var source = NewState();
        source.QsaKeyCache[0].Add(new float[] { 1f, 2f });
        source.QsaKeyCache[0].Add(new float[] { 3f, 4f });

        var clone = source.Clone();

        Assert.NotSame(source.QsaKeyCache[0], clone.QsaKeyCache[0]);
        Assert.Same(source.QsaKeyCache[0][0], clone.QsaKeyCache[0][0]);
        Assert.Same(source.QsaKeyCache[0][1], clone.QsaKeyCache[0][1]);
    }

    [Fact]
    public void Divergence_DoesNotLeakBetweenSourceAndClone()
    {
        var source = NewState();
        source.QsaValCache[1].Add(new float[] { 5f });
        var clone = source.Clone();

        source.QsaValCache[1].Add(new float[] { 6f });
        clone.QsaValCache[1].Add(new float[] { 7f });

        Assert.Equal(2, source.QsaValCache[1].Count);
        Assert.Equal(2, clone.QsaValCache[1].Count);
        Assert.Equal(6f, source.QsaValCache[1][1][0]);
        Assert.Equal(7f, clone.QsaValCache[1][1][0]);
        Assert.Equal(5f, clone.QsaValCache[1][0][0]);
    }

    [Fact]
    public void ResetOfSource_DoesNotClearTheSnapshot()
    {
        var source = NewState();
        source.QsaRawIndexKCache[0].Add(new float[] { 8f });
        var snapshot = source.Snapshot();

        source.Reset();

        Assert.Empty(source.QsaRawIndexKCache[0]);
        Assert.Single(snapshot.QsaRawIndexKCache[0]);
        Assert.Equal(8f, snapshot.QsaRawIndexKCache[0][0][0]);
    }

    [Fact]
    public void Restore_ReturnsTheSequenceToTheSnapshotsHistory()
    {
        var source = NewState();
        source.QsaKeyCache[0].Add(new float[] { 1f });
        var snapshot = source.Snapshot();
        source.QsaKeyCache[0].Add(new float[] { 2f });
        source.QsaKeyCache[0].Add(new float[] { 3f });

        source.Restore(snapshot);

        Assert.Single(source.QsaKeyCache[0]);
        Assert.Equal(1f, source.QsaKeyCache[0][0][0]);
    }
}
