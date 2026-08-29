
namespace OpenTail.Stingray.Tests.Audio;

public sealed class KokoroTests
{
    [Fact]
    public void KokoroPhonemizer_TextToPhonemes_ProducesExpectedPhonemes()
    {
        var phonemizer = new KokoroPhonemizer();
        string text = "Hello world! This is OpenTail Stingray.";
        string phonemes = phonemizer.TextToPhonemes(text);

        Assert.NotNull(phonemes);
        Assert.NotEmpty(phonemes);
        Assert.Contains("həlˈoʊ", phonemes);
        Assert.Contains("wˈɜːld", phonemes);
    }

    [Fact]
    public void KokoroPhonemizer_Tokenize_ProducesValidTokenIds()
    {
        var phonemizer = new KokoroPhonemizer();
        string phonemes = "həlˈoʊ wˈɜːld";
        int[] tokens = phonemizer.Tokenize(phonemes);

        Assert.NotNull(tokens);
        Assert.True(tokens.Length >= 4);
        Assert.Equal(0, tokens[0]); // Start token
        Assert.Equal(0, tokens[^1]); // End token
    }

    [Fact]
    public void KokoroVoices_GetVoiceStyle_Returns256DimNormalizedVector()
    {
        float[] style = KokoroVoices.GetVoiceStyle("af_heart");

        Assert.Equal(256, style.Length);

        float sumSq = 0f;
        for (int i = 0; i < style.Length; i++)
        {
            sumSq += style[i] * style[i];
            Assert.False(float.IsNaN(style[i]));
            Assert.False(float.IsInfinity(style[i]));
        }

        Assert.InRange(MathF.Sqrt(sumSq), 0.99f, 1.01f);
    }

    [Fact]
    public void KokoroModel_Forward_ProducesValidWaveform()
    {
        using var model = new KokoroModel(hiddenDim: 128, numLayers: 2);
        int[] tokens = [0, 15, 25, 35, 45, 0];
        float[] style = KokoroVoices.GetVoiceStyle("af_heart");

        float[] audio = model.Forward(tokens, style, speed: 1.0f);

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
    public void WavWriter_ProducesValidRiffWaveHeaderAndPcmBytes()
    {
        float[] samples = new float[2400]; // 0.1 second at 24kHz
        for (int i = 0; i < samples.Length; i++)
        {
            samples[i] = 0.5f * MathF.Sin(2.0f * MathF.PI * 440.0f * i / 24000.0f);
        }

        byte[] wavBytes = WavWriter.ToWavBytes(samples, sampleRate: 24000);

        Assert.NotNull(wavBytes);
        Assert.Equal(44 + samples.Length * 2, wavBytes.Length);

        // RIFF header
        Assert.Equal((byte)'R', wavBytes[0]);
        Assert.Equal((byte)'I', wavBytes[1]);
        Assert.Equal((byte)'F', wavBytes[2]);
        Assert.Equal((byte)'F', wavBytes[3]);

        // WAVE header
        Assert.Equal((byte)'W', wavBytes[8]);
        Assert.Equal((byte)'A', wavBytes[9]);
        Assert.Equal((byte)'V', wavBytes[10]);
        Assert.Equal((byte)'E', wavBytes[11]);
    }

    [Fact]
    public void KokoroPipeline_Generate_ProducesValidAudioResult()
    {
        using var pipeline = new KokoroPipeline();
        var req = new AudioGenerationRequest
        {
            Text = "Hello from OpenTail Stingray native TTS engine!",
            Voice = "af_heart",
            Speed = 1.0f
        };

        var result = pipeline.Generate(req);

        Assert.NotNull(result);
        Assert.Equal(24000, result.SampleRate);
        Assert.True(result.Samples.Length > 2400); // More than 0.1s
        Assert.True(result.Duration.TotalSeconds > 0.1);

        byte[] wavBytes = result.ToWavBytes();
        Assert.True(wavBytes.Length > 44);
    }
}
