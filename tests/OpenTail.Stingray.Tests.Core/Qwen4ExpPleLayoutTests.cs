using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Regression for the PLE depthwise conv weight layout. GGUF stores blk.N.ple_conv1d.weight as
/// [channel][tap] (tap innermost, llama.cpp qwen4exp.cpp:1460 reads column k at offset k*nb[0] with
/// stride nb[1]). StepDilatedConv reads [tap][channel]. Found 2026-11-08 by diffing ple_conv_out-1
/// against the llama.cpp eval-callback dump (cosine 0.08 before the transpose, 0.99997 after).
/// </summary>
public sealed class Qwen4ExpPleLayoutTests
{
    [Fact]
    public void TransposeConvWeights_MapsChannelTapToTapChannel()
    {
        const int channels = 3;
        const int kernel = 4;
        var src = new float[channels * kernel];
        for (int c = 0; c < channels; c++)
            for (int k = 0; k < kernel; k++)
                src[c * kernel + k] = c * 10 + k; // unique value per (channel, tap)

        var dst = new float[channels * kernel];
        Qwen4ExpPle.TransposeConvWeights(src, channels, kernel, dst);

        for (int k = 0; k < kernel; k++)
            for (int c = 0; c < channels; c++)
                Assert.Equal(c * 10 + k, dst[k * channels + c]);
    }

    [Fact]
    public void StepDilatedConv_ReadsTapOfEachChannel_FromTransposedWeights()
    {
        // One channel, kernel 2, dilation 1, history of 2 slots: out = w0*x[t-1] + w1*x[t] (before SiLU).
        const int channels = 1;
        const int kernel = 2;
        var history = new float[] { 2f, 3f };
        var srcWeights = new float[] { 0.5f, 0.25f }; // file layout [channel][tap]
        var transposed = new float[kernel * channels];
        Qwen4ExpPle.TransposeConvWeights(srcWeights, channels, kernel, transposed);

        var convOut = new float[channels];
        Qwen4ExpPle.StepDilatedConv(history, transposed, convOut, channels, kernel, 1, 2);

        float pre = 0.5f * 2f + 0.25f * 3f; // 1.75
        float expected = pre / (1f + MathF.Exp(-pre)); // SiLU
        Assert.Equal(expected, convOut[0], 5);
    }
}
