
namespace OpenTail.Stingray.Tests.Vision;

public sealed class FluxParamsTests
{
    [Fact]
    public void Defaults_MatchFlux1SchnellShape()
    {
        var p = new OpenTail.Stingray.Diffusion.FluxParams();
        Assert.Equal(3072, p.HiddenSize);
        Assert.Equal(24, p.NumHeads);
        Assert.Equal(128, p.HeadDim);
        Assert.Equal(19, p.DoubleBlocks);
        Assert.Equal(38, p.SingleBlocks);
        Assert.False(p.HasGuidanceIn);
    }

    [Fact]
    public void DerivedDims_TrackHiddenSize()
    {
        var p = new OpenTail.Stingray.Diffusion.FluxParams { HiddenSize = 1024, NumHeads = 8 };
        Assert.Equal(128, p.HeadDim);
        Assert.Equal(4096, p.TimeEmbDim);
        Assert.Equal(4096, p.VecEmbDim);
    }

    [Fact]
    public void FromMetadata_EmptyDictionary_FallsBackToFlux1Defaults()
    {
        var p = OpenTail.Stingray.Diffusion.FluxParams.FromMetadata(
            new Dictionary<string, object>());
        Assert.Equal(3072, p.HiddenSize);
        Assert.Equal(24, p.NumHeads);
        Assert.Equal(19, p.DoubleBlocks);
        Assert.Equal(38, p.SingleBlocks);
        Assert.False(p.HasGuidanceIn);
    }

    [Fact]
    public void FromMetadata_OverridesPresentKeys()
    {
        var meta = new Dictionary<string, object>
        {
            ["flux.hidden_size"] = 2048,
            ["flux.num_attention_heads"] = 16,
            ["flux.num_double_layers"] = 10,
            ["flux.num_single_layers"] = 20,
        };
        var p = OpenTail.Stingray.Diffusion.FluxParams.FromMetadata(meta);
        Assert.Equal(2048, p.HiddenSize);
        Assert.Equal(16, p.NumHeads);
        Assert.Equal(10, p.DoubleBlocks);
        Assert.Equal(20, p.SingleBlocks);
    }

    [Fact]
    public void FromMetadata_GuidanceKeyPresent_SetsHasGuidanceInTrue()
    {
        var meta = new Dictionary<string, object> { ["flux.guidance_embed"] = true };
        var p = OpenTail.Stingray.Diffusion.FluxParams.FromMetadata(meta);
        Assert.True(p.HasGuidanceIn);
    }

    [Fact]
    public void FromMetadata_WrongValueType_FallsBackToDefault()
    {
        // A non-int value for a key that expects int should be ignored, not throw.
        var meta = new Dictionary<string, object> { ["flux.hidden_size"] = "not-an-int" };
        var p = OpenTail.Stingray.Diffusion.FluxParams.FromMetadata(meta);
        Assert.Equal(3072, p.HiddenSize);
    }
}

public sealed class ZImageParamsTests
{
    [Fact]
    public void Defaults_MatchZImageTurboConfig()
    {
        var p = new OpenTail.Stingray.Diffusion.ZImageParams();
        Assert.Equal(3840, p.Dim);
        Assert.Equal(30, p.NHeads);
        Assert.Equal(30, p.NLayers);
        Assert.Equal([32, 48, 48], p.AxesDims);
        Assert.Equal([1536, 512, 512], p.AxesLens);
    }

    [Fact]
    public void DerivedDims_MatchExpectedZImageTurboValues()
    {
        var p = new OpenTail.Stingray.Diffusion.ZImageParams();
        Assert.Equal(128, p.HeadDim);      // 3840 / 30
        Assert.Equal(10240, p.FfnHidden);  // 3840 * 8/3, rounded
        Assert.Equal(64, p.PatchDim);      // 2*2*16
    }

    [Fact]
    public void QwenEncoderLayer_IsSecondToLastLayer()
    {
        var p = new OpenTail.Stingray.Diffusion.ZImageParams();
        Assert.Equal(34, p.QwenEncoderLayer); // 36 - 2
    }

    [Fact]
    public void Overrides_FlowThroughToDerivedProperties()
    {
        var p = new OpenTail.Stingray.Diffusion.ZImageParams { Dim = 512, NHeads = 8, QwenNumLayers = 10 };
        Assert.Equal(64, p.HeadDim);
        Assert.Equal(8, p.QwenEncoderLayer);
    }
}
