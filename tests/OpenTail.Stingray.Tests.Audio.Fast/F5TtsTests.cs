using OpenTail.Stingray.Audio.F5TTS;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class F5TtsTests
{
    [Fact]
    public void F5MelExtractor_ExtractMel_Produces100ChannelMelSpectrogram()
    {
        var extractor = new F5MelExtractor();
        var pcm = new float[24000]; // 1.0 second of 24kHz audio
        for (int i = 0; i < pcm.Length; i++)
        {
            pcm[i] = 0.5f * MathF.Sin(2.0f * MathF.PI * 440.0f * i / 24000.0f);
        }

        float[] mel = extractor.ExtractMel(pcm);

        Assert.NotNull(mel);
        Assert.True(mel.Length >= F5MelExtractor.NumMels);
        Assert.Equal(0, mel.Length % F5MelExtractor.NumMels);

        for (int i = 0; i < mel.Length; i++)
        {
            Assert.False(float.IsNaN(mel[i]));
            Assert.False(float.IsInfinity(mel[i]));
        }
    }

    [Fact]
    public void F5TextEncoder_Encode_ProducesConvNeXtFeatures()
    {
        var encoder = new F5TextEncoder();
        string text = "Hello world from F5-TTS native ConvNeXt text encoder!";
        int targetFrames = 64;

        float[] features = encoder.Encode(text, targetFrames);

        Assert.NotNull(features);
        Assert.Equal(targetFrames * F5TextEncoder.TextDim, features.Length);

        for (int i = 0; i < features.Length; i++)
        {
            Assert.False(float.IsNaN(features[i]));
            Assert.False(float.IsInfinity(features[i]));
        }
    }

    // F5DiTModel_SolveFlowMatchingOde_SolvesTrajectory removed: F5DiTModel was ported to a
    // real, weight-driven static class (golden-verified against the real PyTorch reference --
    // see docs/audio-review-progress.md's F5-TTS DiT section) and no longer has the fake
    // instance-based procedural constructor/API this test exercised. Real coverage lives in
    // Tests.Audio/F5DiTModelTests.cs and Tests.Audio/F5TtsRealWeightsTests.cs (HeavyTestBase,
    // real GGUF/safetensors weights).

    [Fact]
    public void F5VocosVocoder_Synthesize_Produces24000HzAudio()
    {
        var vocoder = new F5VocosVocoder();
        int numFrames = 16;
        var melLatents = new float[numFrames * F5VocosVocoder.InChannels];
        for (int i = 0; i < melLatents.Length; i++) melLatents[i] = 0.1f * MathF.Sin(i * 0.2f);

        float[] audio = vocoder.Synthesize(melLatents, numFrames);

        Assert.NotNull(audio);
        Assert.Equal(numFrames * F5VocosVocoder.HopLength, audio.Length);

        for (int i = 0; i < audio.Length; i++)
        {
            Assert.False(float.IsNaN(audio[i]));
            Assert.False(float.IsInfinity(audio[i]));
            Assert.InRange(audio[i], -1.05f, 1.05f);
        }
    }

    [Fact]
    public void F5TtsPipeline_Generate_ProducesValidWav()
    {
        using var pipeline = new F5TtsPipeline();
        var req = new AudioGenerationRequest
        {
            Text = "Testing F5-TTS native flow-matching pipeline in OpenTail Stingray.",
            Speed = 1.0f
        };

        var result = pipeline.Generate(req);

        Assert.NotNull(result);
        Assert.Equal(24000, result.SampleRate);
        Assert.True(result.Samples.Length > 2400); // More than 0.1s
        Assert.True(result.Duration.TotalSeconds > 0.1);

        byte[] wav = result.ToWavBytes();
        Assert.True(wav.Length > 44);

        // Verify RIFF & WAVE headers
        Assert.Equal((byte)'R', wav[0]);
        Assert.Equal((byte)'I', wav[1]);
        Assert.Equal((byte)'F', wav[2]);
        Assert.Equal((byte)'F', wav[3]);
        Assert.Equal((byte)'W', wav[8]);
        Assert.Equal((byte)'A', wav[9]);
        Assert.Equal((byte)'V', wav[10]);
        Assert.Equal((byte)'E', wav[11]);
    }
}
