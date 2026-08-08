using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vision;

namespace OpenTail.Stingray.Tests.Vision;

public class VisionModelTests
{
    [Fact]
    public void Open_LoadsGemma4UvConfigAndTensors()
    {
        var path = VisionTestPaths.FindMmproj();
        Assert.SkipUnless(path is not null, "model fixture not present in this environment");

        using var m = VisionModel.Open(path);

        // config
        Assert.Equal(VisionModel.ProjectorTypeGemma4Uv, m.ProjectorType);
        Assert.True(m.HasVisionEncoder);
        Assert.Equal(16, m.ConfigPatchSize);
        Assert.Equal(3, m.NMerge);
        Assert.Equal(48, m.PatchSize);          // effective im2col patch (16 * 3)
        Assert.Equal(224, m.ImageSize);
        Assert.Equal(3840, m.EmbeddingLength);
        Assert.Equal(3840, m.ProjectionDim);
        Assert.Equal(40, m.MinImageTokens);
        Assert.Equal(280, m.MaxImageTokens);

        // tensor shapes (ne-order: fastest dim first)
        Assert.Equal(new long[] { 6912, 3840 }, m.PatchEmbdWeight.Dimensions);
        Assert.Equal(new long[] { 3840 }, m.PatchEmbdBias.Dimensions);
        Assert.Equal(new long[] { 6912 }, m.PatchNorm1W.Dimensions);
        Assert.Equal(new long[] { 3840 }, m.PatchNorm2W.Dimensions);
        Assert.Equal(new long[] { 3840 }, m.PatchNorm3W.Dimensions);
        Assert.Equal(new long[] { 3840, 1120, 2 }, m.PositionEmbd.Dimensions);
        Assert.Equal(new long[] { 3840, 3840 }, m.MmInputProjection.Dimensions);

        // dtypes: patch embed is F32, the mm projection is BF16
        Assert.Equal(DType.Float32, m.PatchEmbdWeight.DType);
        Assert.Equal(DType.BFloat16, m.MmInputProjection.DType);
    }

    [Fact]
    public void Open_RejectsTextModelAsMmproj()
    {
        var textPath = VisionTestPaths.FindTextModel();
        Assert.SkipUnless(textPath is not null, "model fixture not present in this environment");

        // The text GGUF is arch=gemma4, not a clip mmproj -> must be rejected clearly.
        Assert.ThrowsAny<NotSupportedException>(() => VisionModel.Open(textPath));
    }

    [Fact]
    public void Gemma4V_Open_ResolvesCompleteE4BViTInventory()
    {
        var path = VisionTestPaths.FindE4BMmproj();
        Assert.SkipUnless(path is not null,
            "gemma-4-E4B-it-mmproj.gguf is required for the E4B ViT inventory acceptance test.");

        using var model = Gemma4VVisionModel.Open(path!);

        Assert.Equal(224, model.ImageSize);
        Assert.Equal(16, model.PatchSize);
        Assert.Equal(768, model.EmbeddingLength);
        Assert.Equal(2560, model.ProjectionDim);
        Assert.Equal(3072, model.FeedForwardLength);
        Assert.Equal(16, model.BlockCount);
        Assert.Equal(12, model.HeadCount);
        Assert.Equal(16, model.Blocks.Length);
        Assert.Equal(new long[] { 16, 16, 3, 768 }, model.PatchEmbedding.Dimensions);
        Assert.Equal(new long[] { 768, 10240, 2 }, model.PositionEmbedding.Dimensions);
        Assert.Equal(new long[] { 768, 2560 }, model.InputProjection.Dimensions);
        Assert.Equal(new long[] { 768 }, model.Blocks[0].Ln1.Dimensions);
        Assert.Equal(new long[] { 768, 768 }, model.Blocks[15].AttnOut.Dimensions);
        Assert.Equal(new long[] { 3072, 768 }, model.Blocks[15].FfnDown.Dimensions);
    }
}
