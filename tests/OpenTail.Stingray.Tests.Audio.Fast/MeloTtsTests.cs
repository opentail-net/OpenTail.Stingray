using OpenTail.Stingray.Audio.MeloTTS;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class MeloTtsTests
{
    [Fact]
    public void MeloPhonemizer_Phonemize_ProducesPhonesTonesLangIds()
    {
        var phonemizer = new MeloPhonemizer();
        string text = "Hello from MeloTTS!";

        var res = phonemizer.Phonemize(text, "EN-US");

        Assert.NotNull(res);
        Assert.True(res.Phones.Length > 5);
        Assert.Equal(res.Phones.Length, res.Tones.Length);
        Assert.Equal(res.Phones.Length, res.LangIds.Length);
        Assert.Equal(0, res.Phones[0]); // Pad
        Assert.Equal(0, res.LangIds[0]); // EN = 0
    }

    [Fact]
    public void MeloVoices_GetSpeakerId_MapsAccents()
    {
        Assert.Equal(0, MeloVoices.GetSpeakerId("EN-US"));
        Assert.Equal(1, MeloVoices.GetSpeakerId("EN-BR"));
        Assert.Equal(2, MeloVoices.GetSpeakerId("EN-INDIA"));
        Assert.Equal(10, MeloVoices.GetSpeakerId("ZH"));
        Assert.Equal(20, MeloVoices.GetSpeakerId("ES"));
    }

    [Fact]
    public void MeloBertEncoder_Encode_Produces512DimFeatures()
    {
        var encoder = new MeloBertEncoder();
        int[] phones = [0, 1, 0, 5, 0, 6, 0, 2, 0];
        int[] tones = [0, 0, 0, 1, 0, 2, 0, 0, 0];
        int[] langIds = [0, 0, 0, 0, 0, 0, 0, 0, 0];

        float[] features = encoder.Encode(phones, tones, langIds);

        Assert.NotNull(features);
        Assert.Equal(phones.Length * MeloBertEncoder.FeatureDim, features.Length);

        for (int i = 0; i < features.Length; i++)
        {
            Assert.False(float.IsNaN(features[i]));
            Assert.False(float.IsInfinity(features[i]));
        }
    }

    [Fact]
    public void MeloModel_Forward_ProducesValid44100HzAudio()
    {
        using var model = new MeloModel(sampleRate: 44100, hiddenDim: 64);
        int[] phones = [0, 1, 0, 5, 0, 6, 0, 2, 0];
        int[] tones = [0, 0, 0, 1, 0, 2, 0, 0, 0];
        int[] langIds = [0, 0, 0, 0, 0, 0, 0, 0, 0];
        var bertFeatures = new float[phones.Length * MeloBertEncoder.FeatureDim];

        float[] audio = model.Forward(
            phones: phones,
            tones: tones,
            langIds: langIds,
            bertFeatures: bertFeatures,
            speakerId: 0,
            speed: 1.0f);

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
    public void MeloPipeline_Generate_ProducesValid44100HzWav()
    {
        using var pipeline = new MeloPipeline();
        var req = new AudioGenerationRequest
        {
            Text = "Testing MeloTTS multilingual speech synthesis in OpenTail Stingray.",
            Voice = "EN-US",
            Speed = 1.0f
        };

        var result = pipeline.Generate(req);

        Assert.NotNull(result);
        Assert.Equal(44100, result.SampleRate);
        Assert.True(result.Samples.Length > 4410); // More than 0.1s
        Assert.True(result.Duration.TotalSeconds > 0.1);

        byte[] wav = result.ToWavBytes();
        Assert.True(wav.Length > 44);

        // Verify RIFF & WAVE magic headers
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
