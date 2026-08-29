
namespace OpenTail.Stingray.Tests.Audio;

public sealed class AudioResamplerTests
{
    [Fact]
    public void Resample_IdenticalRate_ReturnsExactCopy()
    {
        float[] input = [0.1f, -0.2f, 0.5f, -0.8f, 0.3f];
        float[] output = AudioResampler.Resample(input, 16000, 16000, channels: 1);
        Assert.Equal(input, output);
    }

    [Theory]
    [InlineData(48000, 16000, 1)]
    [InlineData(44100, 16000, 1)]
    [InlineData(16000, 24000, 1)]
    [InlineData(24000, 16000, 1)]
    [InlineData(48000, 16000, 2)]
    public void Resample_ProducesExpectedOutputLength(int inRate, int outRate, int channels)
    {
        int inFrames = inRate; // 1 second of audio
        float[] input = new float[inFrames * channels];
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = MathF.Sin(2.0f * MathF.PI * 440.0f * (i / channels) / inRate);
        }

        float[] output = AudioResampler.Resample(input, inRate, outRate, channels, ResampleQuality.Fast);
        int expectedFrames = (int)Math.Round((double)inFrames * outRate / inRate);
        Assert.Equal(expectedFrames * channels, output.Length);

        // Verify no NaNs or Infinities
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(float.IsNaN(output[i]), $"NaN at output index {i}");
            Assert.False(float.IsInfinity(output[i]), $"Infinity at output index {i}");
        }
    }

    [Fact]
    public void Resample_PreservesConstantDcSignal()
    {
        float[] input = new float[1600];
        Array.Fill(input, 0.75f);

        float[] output = AudioResampler.Resample(input, 16000, 24000, channels: 1, ResampleQuality.Balanced);
        
        // Interior samples (away from padding boundaries) should match DC level ~0.75
        for (int i = 50; i < output.Length - 50; i++)
        {
            Assert.InRange(output[i], 0.74f, 0.76f);
        }
    }

    [Fact]
    public void DownmixToMono_Stereo_AveragesChannels()
    {
        float[] stereo = [1.0f, 0.0f, 0.5f, 0.5f, -0.2f, 0.4f];
        float[] mono = AudioDownmixer.DownmixToMono(stereo, sourceChannels: 2);

        Assert.Equal(3, mono.Length);
        Assert.Equal(0.5f, mono[0], 4);
        Assert.Equal(0.5f, mono[1], 4);
        Assert.Equal(0.1f, mono[2], 4);
    }

    [Fact]
    public void DownmixToStereo_51Surround_FoldsChannels()
    {
        // 1 frame of 5.1: [L=1, R=0, C=0, LFE=0, Ls=0, Rs=0]
        float[] surround51 = [1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f];
        float[] stereo = AudioDownmixer.DownmixToStereo(surround51, sourceChannels: 6);

        Assert.Equal(2, stereo.Length);
        Assert.True(stereo[0] > stereo[1], "Left channel should dominate");
    }

    [Fact]
    public void WavWriter_TpdfDither_QuantizesWithNoiseShapingAndPreservesSilence()
    {
        // Silence input must produce exact zero PCM
        float[] silence = new float[100];
        byte[] wavBytes = WavWriter.ToWavBytes(silence, 16000, 1, DitherMode.Tpdf);
        Assert.NotEmpty(wavBytes);

        var (readSamples, sampleRate, channels) = WavReader.ReadWav(new MemoryStream(wavBytes));
        Assert.Equal(16000, sampleRate);
        Assert.Equal(1, channels);
        for (int i = 0; i < readSamples.Length; i++)
        {
            Assert.Equal(0.0f, readSamples[i]);
        }

        // Active sine audio should round-trip without clipping or NaNs
        float[] activeAudio = new float[1600];
        for (int i = 0; i < activeAudio.Length; i++)
        {
            activeAudio[i] = MathF.Sin(2.0f * MathF.PI * 440.0f * i / 16000.0f) * 0.5f;
        }

        byte[] activeWav = WavWriter.ToWavBytes(activeAudio, 16000, 1, DitherMode.Tpdf);
        var (roundTrip, _, _) = WavReader.ReadWav(new MemoryStream(activeWav));
        Assert.Equal(activeAudio.Length, roundTrip.Length);
        for (int i = 0; i < roundTrip.Length; i++)
        {
            Assert.InRange(roundTrip[i], -1.0f, 1.0f);
        }
    }
}
