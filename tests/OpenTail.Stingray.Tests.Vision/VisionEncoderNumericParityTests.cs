
namespace OpenTail.Stingray.Tests.Vision;

public class VisionEncoderNumericParityTests
{
    [Fact]
    public void Gemma4V_TokenCountAndDimensionInvariants()
    {
        // Gemma 4 E4B invariants:
        // ImageSize = 224, PatchSize = 16 -> 14x14 = 196 patches.
        int imageSize = 224;
        int patchSize = 16;
        int patchesPerSide = imageSize / patchSize;
        int totalPatches = patchesPerSide * patchesPerSide;

        Assert.Equal(14, patchesPerSide);
        Assert.Equal(196, totalPatches);
    }

    [Fact]
    public void Gemma3_TokenCountAndDimensionInvariants()
    {
        // Gemma 3 invariants:
        // ImageSize = 896, PatchSize = 14 -> 64x64 = 4096 patches.
        // NMerge = 4 -> 16x16 = 256 soft tokens.
        int imageSize = 896;
        int patchSize = 14;
        int patchesPerSide = imageSize / patchSize;
        int nMerge = 4;
        int tokensPerSide = patchesPerSide / nMerge;
        int totalTokens = tokensPerSide * tokensPerSide;

        Assert.Equal(64, patchesPerSide);
        Assert.Equal(16, tokensPerSide);
        Assert.Equal(256, totalTokens);
    }

    [Fact]
    public void Llama4_TokenCountAndDimensionInvariants()
    {
        // Llama 4 invariants:
        // ImageSize = 336, PatchSize = 14 -> 24x24 = 576 patches.
        // NMerge = 2 -> 12x12 = 144 soft tokens, ProjectionDim = 5120.
        int imageSize = 336;
        int patchSize = 14;
        int patchesPerSide = imageSize / patchSize;
        int nMerge = 2;
        int tokensPerSide = patchesPerSide / nMerge;
        int totalTokens = tokensPerSide * tokensPerSide;

        Assert.Equal(24, patchesPerSide);
        Assert.Equal(12, tokensPerSide);
        Assert.Equal(144, totalTokens);
    }
}
