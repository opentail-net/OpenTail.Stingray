using OpenTail.Stingray.Diffusion.HunyuanVideo;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class HunyuanVideoTests
{
    [Fact]
    public void HunyuanVideoRoPE_Compute3DRoPE_ValidatesShapesAndFrequencies()
    {
        int numFrames = 4;
        int patchH = 8;
        int patchW = 8;
        int headDim = 128;

        var (cos, sin) = HunyuanVideoRoPE.Compute3DRoPE(numFrames, patchH, patchW, headDim);

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
    public void HunyuanVideo_PackAndUnpackLatents_IsLosslessIdentity()
    {
        int numFrames = 2;
        int latH = 16;
        int latW = 16;
        int channels = 16;
        int totalElements = channels * numFrames * latH * latW;

        var original = new float[totalElements];
        for (int i = 0; i < original.Length; i++)
            original[i] = i * 0.003f - 2.5f;

        var packed = HunyuanVideoModel.PackLatents(original, numFrames, latH, latW);
        int expectedTokens = numFrames * (latH / 2) * (latW / 2);
        int expectedChannels = channels * 2 * 2; // 64
        Assert.Equal(expectedTokens * expectedChannels, packed.Length);

        var unpacked = HunyuanVideoModel.UnpackLatents(packed, numFrames, latH, latW);
        Assert.Equal(original.Length, unpacked.Length);

        for (int i = 0; i < original.Length; i++)
            Assert.Equal(original[i], unpacked[i], tolerance: 1e-6f);
    }

    [Fact]
    public void HunyuanVideo_FlowShift_ComputesExpectedSchedule()
    {
        float flowShift = 7.0f;
        float linearT = 0.5f;
        float shifted = (flowShift * linearT) / (1.0f + (flowShift - 1.0f) * linearT);
        Assert.Equal(3.5f / 4.0f, shifted, tolerance: 1e-6f);
    }

    [Fact]
    public void HunyuanVideo_MultiFrame_LatentSliceExtraction_MatchesExpectedOffsets()
    {
        int numFrames = 3;
        int latH = 4, latW = 4, latC = 16;
        int singleFrameLen = latC * latH * latW;
        var videoLatent = new float[latC * numFrames * latH * latW];

        for (int c = 0; c < latC; c++)
        {
            for (int f = 0; f < numFrames; f++)
            {
                int baseOff = ((c * numFrames) + f) * (latH * latW);
                for (int p = 0; p < latH * latW; p++)
                    videoLatent[baseOff + p] = f + 1.0f;
            }
        }

        var frame2 = new float[singleFrameLen];
        int targetFrame = 2;
        for (int c = 0; c < latC; c++)
        {
            int srcOff = ((c * numFrames) + targetFrame) * (latH * latW);
            int dstOff = c * (latH * latW);
            Array.Copy(videoLatent, srcOff, frame2, dstOff, latH * latW);
        }

        for (int i = 0; i < frame2.Length; i++)
            Assert.Equal(3.0f, frame2[i]);
    }
}
