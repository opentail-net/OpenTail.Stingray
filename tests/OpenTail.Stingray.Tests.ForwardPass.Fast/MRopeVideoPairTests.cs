using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Qwen-VL M-RoPE for a video pair (Qwen 3.8 plan, Phase 5.3 step c). The reference (tools/mtmd/mtmd.cpp, pinned
/// bed0a8566) gives a pair one image-token chunk: every token of the chunk has t = pos_0, x and y follow the grid, and
/// the text position advances by max(nx, ny) once per pair. These tests pin <see cref="MRopeImageLayout"/> to that rule.
/// </summary>
public sealed class MRopeVideoPairTests
{
    [Fact]
    public void PairChunk_SharesTheTemporalIndexAndFollowsTheGrid()
    {
        var layout = new MRopeImageLayout();
        layout.Add(startSlot: 3, nx: 2, ny: 2); // three text tokens, then one 2x2 pair chunk at slots 3..6

        Assert.Equal((3, 3, 3), layout.Position(3));
        Assert.Equal((3, 3, 4), layout.Position(4));
        Assert.Equal((3, 4, 3), layout.Position(5));
        Assert.Equal((3, 4, 4), layout.Position(6));
    }

    [Fact]
    public void TextAfterAPair_AdvancesByMaxOfTheGridOnce()
    {
        var layout = new MRopeImageLayout();
        layout.Add(startSlot: 3, nx: 3, ny: 2); // 6 slots, advances by max(3, 2) = 3

        Assert.Equal((6, 6, 6), layout.Position(9));   // 3 + 3
        Assert.Equal((7, 7, 7), layout.Position(10));
    }

    [Fact]
    public void TwoPairs_EachAdvancesOnce()
    {
        var layout = new MRopeImageLayout();
        layout.Add(startSlot: 0, nx: 2, ny: 2); // slots 0..3, advances 2
        layout.Add(startSlot: 4, nx: 2, ny: 2); // slots 4..7, starts at position 2

        Assert.Equal((2, 2, 2), layout.Position(4));
        Assert.Equal((2, 3, 2), layout.Position(6));
        Assert.Equal((4, 4, 4), layout.Position(8));   // text after both pairs: 4
    }
}
