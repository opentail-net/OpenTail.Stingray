
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Structural validation for VibeVoiceConvNeXtBlock with synthetic weights -- no real
/// VibeVoice ASR checkpoint has been golden-verified against yet (large download, deferred
/// separately; see docs/audio-review-progress.md). Confirms shapes, finiteness, and the real causal
/// padding arithmetic (`ExtraPaddingForConv1d`) match the reference's own formula for a range of
/// input lengths/kernels/strides.</summary>
public sealed class VibeVoiceConvNeXtBlockTests
{
    [Fact]
    public void Forward_SyntheticWeights_ProducesFiniteSameShapeOutput()
    {
        const int channels = 16;
        const int frames = 37;
        const int kernel = 7;
        const float eps = 1e-5f;

        var rng = new Random(42);
        float[][] input = MakeRandom(channels, frames, rng);
        var w = new OpenTail.Stingray.Audio.VibeVoice.VibeVoiceConvNeXtBlockWeights
        {
            NormWeight = MakeOnes(channels),
            MixerWeight = MakeRandom(channels, kernel, rng),
            MixerBias = null,
            Gamma = MakeOnes(channels),
            FfnNormWeight = MakeOnes(channels),
            FfnLinear1Weight = MakeRandomFlat(4 * channels * channels, rng),
            FfnLinear1Bias = null,
            FfnLinear2Weight = MakeRandomFlat(channels * 4 * channels, rng),
            FfnLinear2Bias = null,
            FfnGamma = MakeOnes(channels),
        };

        var output = OpenTail.Stingray.Audio.VibeVoice.VibeVoiceConvNeXtBlock.Forward(input, w, eps);

        Assert.Equal(channels, output.Length);
        foreach (var row in output)
        {
            Assert.Equal(frames, row.Length);
            foreach (var v in row) Assert.True(float.IsFinite(v));
        }
    }

    [Theory]
    [InlineData(37, 7, 1)]
    [InlineData(100, 3, 2)]
    [InlineData(64, 5, 4)]
    [InlineData(13, 11, 1)]
    public void CausalConv1d_ProducesExpectedFrameCountForVariousShapes(int frames, int kernel, int stride)
    {
        const int inCh = 4, outCh = 6;
        var rng = new Random(7);
        var input = MakeRandom(inCh, frames, rng);
        var weight = new float[outCh][][];
        for (int oc = 0; oc < outCh; oc++)
        {
            weight[oc] = new float[inCh][];
            for (int ic = 0; ic < inCh; ic++) weight[oc][ic] = MakeRandomFlat(kernel, rng);
        }

        var (output, outFrames) = OpenTail.Stingray.Audio.VibeVoice.VibeVoiceConvNeXtBlock.CausalConv1d(input, weight, null, stride);

        // Real Encodec/DAC-style causal conv should produce ceil(frames/stride) output frames.
        int expected = (int)Math.Ceiling((double)frames / stride);
        Assert.Equal(expected, outFrames);
        Assert.Equal(outCh, output.Length);
        foreach (var row in output)
        {
            Assert.Equal(outFrames, row.Length);
            foreach (var v in row) Assert.True(float.IsFinite(v));
        }
    }

    private static float[][] MakeRandom(int dim0, int dim1, Random rng)
    {
        var output = new float[dim0][];
        for (int i = 0; i < dim0; i++) output[i] = MakeRandomFlat(dim1, rng);
        return output;
    }

    private static float[] MakeRandomFlat(int count, Random rng)
    {
        var output = new float[count];
        for (int i = 0; i < count; i++) output[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        return output;
    }

    private static float[] MakeOnes(int count)
    {
        var output = new float[count];
        Array.Fill(output, 1f);
        return output;
    }
}
