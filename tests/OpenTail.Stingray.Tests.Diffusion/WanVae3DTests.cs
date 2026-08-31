namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class WanVae3DTests
{
    [Fact]
    public void WanVaeDecoder3D_Constants_MatchWanReference()
    {
        Assert.Equal(16, WanVaeDecoder3D.LatentChannels);
        Assert.Equal(4, WanVaeDecoder3D.TemporalScale);
        Assert.Equal(8, WanVaeDecoder3D.SpatialScale);
    }

    [Fact]
    public void Decode_ThrowsOnInvalidLatentSize()
    {
        using var decoder = new WanVaeDecoder3D();
        var badLatent = new float[10];
        Assert.Throws<ArgumentException>(() => decoder.Decode(badLatent, t: 1, latH: 16, latW: 16));
    }
}
