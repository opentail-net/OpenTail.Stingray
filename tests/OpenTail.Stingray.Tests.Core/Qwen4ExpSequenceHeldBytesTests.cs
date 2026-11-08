using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Job 3 part (b), Qwen 3.8 plan: a sequence state reports the bytes it holds beyond the rows it shares with its
/// snapshots. That is the GDN and PLE conv buffers it allocated, plus one reference per row in each QSA list. Synthetic.
/// </summary>
public sealed class Qwen4ExpSequenceHeldBytesTests
{
    [Fact]
    public void HeldBytes_CountsTheAllocatedBuffersPlusOneReferencePerRow()
    {
        // Layer 0 is recurrent (conv 3, state 4 floats, each at least one), layer 1 is attention. PLE history is 6 floats.
        var state = new Qwen4ExpSequenceState(
            2, l => l == 0, gdnConvLength: 3, gdnStateLength: 4, pleHistoryLength: 6, pleHasher: null);
        long allocatedFloats = state.PleConvHistory.Length;
        for (int l = 0; l < 2; l++)
            allocatedFloats += state.GdnConvState[l].Length + state.GdnState[l].Length;

        state.QsaKeyCache[1].Add(new float[3]);
        state.QsaKeyCache[1].Add(new float[3]);
        state.QsaValCache[1].Add(new float[2]);
        state.QsaRawIndexKCache[1].Add(new float[4]);
        state.QsaPooledKeys[1].Add(new float[5]);
        int rows = 5;

        Assert.Equal(allocatedFloats * sizeof(float) + rows * (long)IntPtr.Size, state.HeldBytes);
    }

    [Fact]
    public void HeldBytes_OfAnEmptyAttentionOnlyState_IsZero()
    {
        var state = new Qwen4ExpSequenceState(2, _ => false, gdnConvLength: 0, gdnStateLength: 0, pleHistoryLength: 0, pleHasher: null);

        Assert.Equal(0, state.HeldBytes);
    }
}
