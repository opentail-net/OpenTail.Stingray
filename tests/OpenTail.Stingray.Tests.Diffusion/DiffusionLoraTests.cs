
namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class DiffusionLoraTests
{
    [Fact]
    public void LoraLayer_ComputeDelta_ComputesAccurateLowRankMatrix()
    {
        // inDim=2, outDim=2, rank=1, alpha=1.0
        // Down = [2.0, 3.0]
        // Up   = [4.0, 5.0]
        // Scale = 1.0 * (1.0 / 1) = 1.0
        // Delta = Up * Down = [ [4*2, 4*3], [5*2, 5*3] ] = [8.0, 12.0, 10.0, 15.0]
        var layer = new DiffusionLoraApplier.LoraLayer(
            targetName: "unet.attn1.to_q",
            downWeight: new float[] { 2.0f, 3.0f },
            upWeight: new float[] { 4.0f, 5.0f },
            inDim: 2,
            outDim: 2,
            rank: 1,
            alpha: 1.0f);

        var delta = layer.ComputeDelta(multiplier: 1.0f);

        Assert.Equal(4, delta.Length);
        Assert.Equal(8.0f, delta[0]);
        Assert.Equal(12.0f, delta[1]);
        Assert.Equal(10.0f, delta[2]);
        Assert.Equal(15.0f, delta[3]);
    }

    [Fact]
    public void LoraLayer_ApplyToWeights_ModifiesTargetInPlace()
    {
        var layer = new DiffusionLoraApplier.LoraLayer(
            targetName: "to_q",
            downWeight: new float[] { 1.0f, 1.0f },
            upWeight: new float[] { 2.0f, 2.0f },
            inDim: 2,
            outDim: 2,
            rank: 1,
            alpha: 1.0f);

        var weights = new Dictionary<string, float[]>
        {
            ["to_q.weight"] = new float[] { 10.0f, 10.0f, 10.0f, 10.0f }
        };

        int applied = DiffusionLoraApplier.ApplyToWeights(weights, new[] { layer }, multiplier: 0.5f);

        Assert.Equal(1, applied);
        // delta = 0.5 * 2 * 1 = 1.0 added to each element
        Assert.Equal(11.0f, weights["to_q.weight"][0]);
        Assert.Equal(11.0f, weights["to_q.weight"][1]);
        Assert.Equal(11.0f, weights["to_q.weight"][2]);
        Assert.Equal(11.0f, weights["to_q.weight"][3]);
    }
}
