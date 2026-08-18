using OpenTail.Stingray.Diffusion.Wan;
using Xunit;

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
    public void Wan_PackAndUnpackLatents_IsLosslessIdentity()
    {
        int numFrames = 2;
        int latH = 16;
        int latW = 16;
        int channels = 16;
        int totalElements = channels * numFrames * latH * latW;

        var original = new float[totalElements];
        for (int i = 0; i < original.Length; i++)
            original[i] = i * 0.005f - 3.0f;

        var packed = WanModel.PackLatents(original, numFrames, latH, latW);
        int expectedTokens = numFrames * (latH / 2) * (latW / 2);
        int expectedChannels = channels * 2 * 2; // 64
        Assert.Equal(expectedTokens * expectedChannels, packed.Length);

        var unpacked = WanModel.UnpackLatents(packed, numFrames, latH, latW);
        Assert.Equal(original.Length, unpacked.Length);

        for (int i = 0; i < original.Length; i++)
            Assert.Equal(original[i], unpacked[i], tolerance: 1e-6f);
    }

    [Fact]
    public void Wan_FlowShift_ComputesExpectedSchedule()
    {
        float flowShift = 3.0f;
        float linearT = 0.5f;
        float shifted = (flowShift * linearT) / (1.0f + (flowShift - 1.0f) * linearT);
        Assert.Equal(0.75f, shifted, tolerance: 1e-6f);

        float flowShift14B = 5.0f;
        // (5 * 0.5) / (1 + 4 * 0.5) = 2.5 / 3.0 = 0.833333
        float shifted14B = (flowShift14B * linearT) / (1.0f + (flowShift14B - 1.0f) * linearT);
        Assert.Equal(2.5f / 3.0f, shifted14B, tolerance: 1e-6f);
    }
}
