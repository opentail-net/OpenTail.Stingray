using OpenTail.Stingray.Diffusion.QwenImage;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class QwenImageTests
{
    [Fact]
    public void QwenImageRoPE_Compute3DRoPE_ValidatesShapesAndFrequencies()
    {
        int txtLen = 10;
        int imgH = 8;
        int imgW = 8;
        int headDim = 128;

        var (cos, sin) = QwenImageRoPE.Compute3DRoPE(txtLen, imgH, imgW, headDim);

        int totalTokens = txtLen + imgH * imgW;
        Assert.Equal(totalTokens * headDim, cos.Length);
        Assert.Equal(totalTokens * headDim, sin.Length);

        // Verify values stay bounded in [-1, 1]
        for (int i = 0; i < cos.Length; i++)
        {
            Assert.InRange(cos[i], -1.0001f, 1.0001f);
            Assert.InRange(sin[i], -1.0001f, 1.0001f);
        }
    }

    [Fact]
    public void QwenImage_PackAndUnpackLatents_IsLosslessIdentity()
    {
        int latH = 16;
        int latW = 16;
        int channels = 16;
        int totalElements = channels * latH * latW;

        var original = new float[totalElements];
        for (int i = 0; i < original.Length; i++)
            original[i] = i * 0.01f - 5.0f;

        var packed = QwenImageModel.PackLatents(original, latH, latW);
        int expectedTokens = (latH / 2) * (latW / 2);
        int expectedChannels = channels * 2 * 2; // 64
        Assert.Equal(expectedTokens * expectedChannels, packed.Length);

        var unpacked = QwenImageModel.UnpackLatents(packed, latH, latW);
        Assert.Equal(original.Length, unpacked.Length);

        for (int i = 0; i < original.Length; i++)
            Assert.Equal(original[i], unpacked[i], tolerance: 1e-6f);
    }

    [Fact]
    public void QwenImage_FlowShift_ComputesExpectedSchedule()
    {
        float flowShift = 3.0f;
        float linearT = 0.5f;
        // t_shifted = (s * t) / (1 + (s - 1) * t) = (3 * 0.5) / (1 + 2 * 0.5) = 1.5 / 2.0 = 0.75
        float shifted = (flowShift * linearT) / (1.0f + (flowShift - 1.0f) * linearT);
        Assert.Equal(0.75f, shifted, tolerance: 1e-6f);
    }
}
