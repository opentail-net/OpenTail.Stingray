
namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class WanTests
{
    [Fact]
    public void WanRoPE_Compute3DRoPE_ValidatesShapesAndFrequencies()
    {
        int numFrames = 4;
        int patchH = 8;
        int patchW = 8;
        int headDim = 128;

        var (cos, sin) = WanRoPE.Compute3DRoPE(numFrames, patchH, patchW, headDim);

        int totalTokens = numFrames * patchH * patchW;
        Assert.Equal(totalTokens * headDim, cos.Length);
        Assert.Equal(totalTokens * headDim, sin.Length);

        for (int i = 0; i < cos.Length; i++)
        {
            Assert.InRange(cos[i], -1.0001f, 1.0001f);
            Assert.InRange(sin[i], -1.0001f, 1.0001f);
        }
    }

    [Fact]
    public void Wan_PackLatents_ProducesExpectedConv3DChannelOrdering()
    {
        int numFrames = 1;
        int latH = 4;
        int latW = 4;
        int channels = 16;
        int totalElements = channels * numFrames * latH * latW;

        var original = new float[totalElements];
        // Mark channel 1, y=0, x=0
        original[(1 * numFrames + 0) * latH * latW + 0 * latW + 0] = 42.0f;

        var packed = WanModel.PackLatents(original, numFrames, latH, latW);
        // Pixel-major layout: slot = (dy*2+dx)*OutChannels + c
        // channel 1, dy=0, dx=0 -> slot = (0*2+0)*16 + 1 = 1
        Assert.Equal(42.0f, packed[1]);
    }

    [Fact]
    public void Wan_UnpackLatents_ProducesExpectedLinearChannelOrdering()
    {
        int numFrames = 1;
        int latH = 4;
        int latW = 4;
        int totalTokens = numFrames * (latH / 2) * (latW / 2);

        var packed = new float[totalTokens * 64];
        // Pixel-major layout: slot = (dy*2+dx)*OutChannels + c
        // Token (0, 0): c=3, dy=1, dx=0 -> slot = (1*2+0)*16 + 3 = 35
        packed[35] = 99.0f;

        var unpacked = WanModel.UnpackLatents(packed, numFrames, latH, latW);
        // Expected dst: c=3, y=1, x=0 -> ((3 * 1 + 0) * 4 + 1) * 4 + 0 = 3 * 16 + 4 = 52
        Assert.Equal(99.0f, unpacked[52]);
    }

    [Theory]
    [InlineData(1, 4, 4)]
    [InlineData(1, 16, 16)]
    [InlineData(3, 8, 8)]
    public void Wan_PackAndUnpackLatents_IsLosslessIdentity(int numFrames, int latH, int latW)
    {
        int channels = 16;
        int totalElements = channels * numFrames * latH * latW;

        var original = new float[totalElements];
        for (int i = 0; i < original.Length; i++)
            original[i] = i * 0.01f - 5.0f;

        var packed = WanModel.PackLatents(original, numFrames, latH, latW);
        int patchH = latH / 2;
        int patchW = latW / 2;
        int expectedTokens = numFrames * patchH * patchW;
        int expectedChannels = WanModel.InChannels; // 64
        Assert.Equal(expectedTokens * expectedChannels, packed.Length);

        var unpacked = WanModel.UnpackLatents(packed, numFrames, latH, latW);
        Assert.Equal(original.Length, unpacked.Length);

        for (int i = 0; i < original.Length; i++)
            Assert.Equal(original[i], unpacked[i], tolerance: 1e-6f);

        var repacked = WanModel.PackLatents(unpacked, numFrames, latH, latW);
        Assert.Equal(packed.Length, repacked.Length);
        for (int i = 0; i < packed.Length; i++)
            Assert.Equal(packed[i], repacked[i], tolerance: 1e-6f);
    }

    [Fact]
    public void Wan_FlowShift_ComputesExpectedSchedule()
    {
        float flowShift = 3.0f;
        float linearT = 0.5f;
        float shifted = (flowShift * linearT) / (1.0f + (flowShift - 1.0f) * linearT);
        Assert.Equal(0.75f, shifted, tolerance: 1e-6f);

        float flowShift14B = 5.0f;
        float shifted14B = (flowShift14B * linearT) / (1.0f + (flowShift14B - 1.0f) * linearT);
        Assert.Equal(2.5f / 3.0f, shifted14B, tolerance: 1e-6f);
    }

    [Fact]
    public void Wan_MultiFrame_LatentSliceExtraction_MatchesExpectedOffsets()
    {
        int numFrames = 3;
        int latH = 4, latW = 4, latC = 16;
        int singleFrameLen = latC * latH * latW;
        var videoLatent = new float[latC * numFrames * latH * latW];

        // Fill video latent with distinct markers per frame
        for (int c = 0; c < latC; c++)
        {
            for (int f = 0; f < numFrames; f++)
            {
                int baseOff = ((c * numFrames) + f) * (latH * latW);
                for (int p = 0; p < latH * latW; p++)
                    videoLatent[baseOff + p] = f + 1.0f;
            }
        }

        // Extract frame 1 (f=1)
        var frame1 = new float[singleFrameLen];
        int targetFrame = 1;
        for (int c = 0; c < latC; c++)
        {
            int srcOff = ((c * numFrames) + targetFrame) * (latH * latW);
            int dstOff = c * (latH * latW);
            Array.Copy(videoLatent, srcOff, frame1, dstOff, latH * latW);
        }

        for (int i = 0; i < frame1.Length; i++)
            Assert.Equal(2.0f, frame1[i]);
    }
}
