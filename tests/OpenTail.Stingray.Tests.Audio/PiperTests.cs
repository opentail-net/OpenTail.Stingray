using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Piper;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class PiperTests
{
    [Fact]
    public void PiperConfig_FromJson_ParsesExpectedMetadata()
    {
        string json = @"
        {
            ""audio"": { ""sample_rate"": 22050, ""quality"": ""medium"" },
            ""espeak"": { ""voice"": ""en-us"" },
            ""inference"": { ""noise_scale"": 0.667, ""length_scale"": 1.0, ""noise_w"": 0.8 },
            ""num_speakers"": 2,
            ""speaker_id_map"": { ""lessac"": 0, ""libritts"": 1 },
            ""phoneme_id_map"": {
                ""_"": [0],
                ""^"": [1],
                ""$"": [2],
                ""a"": [4]
            }
        }";

        var config = PiperConfig.FromJson(json);

        Assert.NotNull(config);
        Assert.Equal(22050, config.Audio.SampleRate);
        Assert.Equal("medium", config.Audio.Quality);
        Assert.Equal("en-us", config.Espeak?.Voice);
        Assert.Equal(0.667f, config.Inference.NoiseScale);
        Assert.Equal(2, config.NumSpeakers);
        Assert.Equal(0, config.SpeakerIdMap?["lessac"]);
        Assert.Equal(1, config.SpeakerIdMap?["libritts"]);
        Assert.Equal(4, config.PhonemeIdMap["a"][0]);
    }

    [Fact]
    public void PiperPhonemizer_Tokenize_InterspersesPadTokensCorrectly()
    {
        var phonemizer = new PiperPhonemizer();
        string text = "abc";

        // Without interspersing: [BOS, a, b, c, EOS]
        int[] raw = phonemizer.Tokenize(text, interspersePad: false);
        Assert.Equal(5, raw.Length);
        Assert.Equal(1, raw[0]); // BOS
        Assert.Equal(2, raw[^1]); // EOS

        // With interspersing: [0, BOS, 0, a, 0, b, 0, c, 0, EOS, 0]
        int[] interspersed = phonemizer.Tokenize(text, interspersePad: true);
        Assert.Equal(raw.Length * 2 + 1, interspersed.Length);
        Assert.Equal(0, interspersed[0]);
        Assert.Equal(1, interspersed[1]);
        Assert.Equal(0, interspersed[2]);
        Assert.Equal(0, interspersed[^1]);
    }

    [Fact]
    public void PiperModel_Forward_ProducesValidWaveform()
    {
        using var model = new PiperModel(sampleRate: 22050, hiddenDim: 64);
        int[] tokens = [0, 1, 0, 5, 0, 6, 0, 2, 0];

        float[] audio = model.Forward(
            tokens: tokens,
            speakerId: 0,
            noiseScale: 0.667f,
            lengthScale: 1.0f,
            noiseW: 0.8f);

        Assert.NotNull(audio);
        Assert.True(audio.Length > 0);

        for (int i = 0; i < audio.Length; i++)
        {
            Assert.False(float.IsNaN(audio[i]));
            Assert.False(float.IsInfinity(audio[i]));
            Assert.InRange(audio[i], -1.05f, 1.05f);
        }
    }

    [Fact]
    public void PiperPipeline_Generate_ProducesValid22050HzAudio()
    {
        using var pipeline = new PiperPipeline();
        var req = new AudioGenerationRequest
        {
            Text = "Hello from native Piper VITS speech synthesis in OpenTail Stingray!",
            Speed = 1.0f
        };

        var result = pipeline.Generate(req);

        Assert.NotNull(result);
        Assert.Equal(22050, result.SampleRate);
        Assert.True(result.Samples.Length > 2205); // More than 0.1s
        Assert.True(result.Duration.TotalSeconds > 0.1);

        byte[] wav = result.ToWavBytes();
        Assert.True(wav.Length > 44);

        // Verify RIFF and WAVE magic headers
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
