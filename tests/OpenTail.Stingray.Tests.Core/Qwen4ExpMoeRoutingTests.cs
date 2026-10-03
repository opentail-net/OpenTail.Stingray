using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

public sealed class Qwen4ExpMoeRoutingTests
{
    [Fact]
    public void Route_PicksTopKByProbability_RenormalisesToOne()
    {
        // experts 1 and 3 clearly dominate
        float[] logits = [0f, 5f, -2f, 4f];
        var idx = new int[2];
        var w = new float[2];
        Qwen4ExpMoeRouting.Route(logits, 2, 0f, idx, w);

        Assert.Equal([1, 3], idx);
        Assert.Equal(1f, w[0] + w[1], 1e-6f);
        // softmax ratio is preserved by renormalisation: w1/w3 = e^(5-4)
        Assert.Equal(MathF.E, w[0] / w[1], 1e-4f);
    }

    [Fact]
    public void Route_AppliesExpertWeightsScaleAfterNormalisation()
    {
        float[] logits = [1f, 1f, 1f, 1f];
        var idx = new int[2];
        var w = new float[2];
        Qwen4ExpMoeRouting.Route(logits, 2, 2.5f, idx, w);

        Assert.Equal([0, 1], idx);            // ties -> lower index
        Assert.Equal(1.25f, w[0], 1e-6f);     // 0.5 * 2.5
        Assert.Equal(1.25f, w[1], 1e-6f);
    }
}
