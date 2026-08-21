using OpenTail.Stingray.Audio.Kokoro;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class KokoroAlignmentTests
{
    [Fact]
    public void ToPredDur_RoundsClampsAndAppliesSpeed()
    {
        float[] durationSums = [0.4f, 2.5f, 0.0f, 3.6f];
        int[] predDur = KokoroAlignment.ToPredDur(durationSums, speed: 1.0f);
        // round-half-to-even: 0.4->0, clamp(min=1)->1; 2.5->2 (banker's rounding); 0.0->0->1; 3.6->4
        Assert.Equal([1, 2, 1, 4], predDur);
    }

    [Fact]
    public void ToPredDur_DividesBySpeedBeforeRounding()
    {
        float[] durationSums = [4.0f];
        int[] predDur = KokoroAlignment.ToPredDur(durationSums, speed: 2.0f);
        Assert.Equal([2], predDur);
    }

    [Fact]
    public void BuildFrameToTokenMap_RepeatsEachTokenIndexByItsDuration()
    {
        int[] predDur = [2, 1, 3];
        int[] map = KokoroAlignment.BuildFrameToTokenMap(predDur);
        Assert.Equal([0, 0, 1, 2, 2, 2], map);
    }

    [Fact]
    public void Expand_GathersSourceColumnsPerFrameToTokenMap()
    {
        // 2 channels, 3 tokens, channel-first [2,3]: ch0=[10,20,30], ch1=[100,200,300].
        float[] source = [10, 20, 30, 100, 200, 300];
        int[] frameToToken = [0, 0, 1, 2];
        float[] expanded = KokoroAlignment.Expand(source, channels: 2, frameToToken);
        Assert.Equal([10, 10, 20, 30, 100, 100, 200, 300], expanded);
    }
}
