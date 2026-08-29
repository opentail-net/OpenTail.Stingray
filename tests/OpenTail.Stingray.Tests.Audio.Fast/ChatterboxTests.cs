using OpenTail.Stingray.Audio.Chatterbox;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class ChatterboxTests
{
    [Fact]
    public void ChatterboxTokenizer_Encode_ProducesValidTokens()
    {
        var tokenizer = new ChatterboxTokenizer();
        string text = "Hello from Chatterbox!";

        int[] tokens = tokenizer.Encode(text);

        Assert.NotNull(tokens);
        Assert.True(tokens.Length >= 3);
        Assert.Equal(1, tokens[0]); // <s>
        Assert.Equal(2, tokens[^1]); // </s>
    }

    [Fact]
    public void ChatterboxVoices_GetSpeakerFeatures_Returns512DimVector()
    {
        float[] features = ChatterboxVoices.GetSpeakerFeatures("narrator");

        Assert.NotNull(features);
        Assert.Equal(ChatterboxVoices.FeatureDim, features.Length);

        for (int i = 0; i < features.Length; i++)
        {
            Assert.False(float.IsNaN(features[i]));
            Assert.False(float.IsInfinity(features[i]));
        }
    }

    [Fact]
    public void ChatterboxAcousticLm_GenerateSpeechTokens_ProducesFramedTokens()
    {
        using var lm = new ChatterboxAcousticLm();
        int[] textTokens = [1, 10, 15, 20, 2];
        float[] spk = ChatterboxVoices.GetSpeakerFeatures("resemble_default");

        var speechTokens = lm.GenerateSpeechTokens(textTokens, spk, temperature: 0.7f);

        Assert.NotNull(speechTokens);
        Assert.True(speechTokens.Count >= 3);
        Assert.Equal(ChatterboxAcousticLm.StartSpeechToken, speechTokens[0]);
        Assert.Equal(ChatterboxAcousticLm.StopSpeechToken, speechTokens[^1]);
    }

    [Fact]
    public void ChatterboxDecoder_Decode_ProducesValid24kHzAudio()
    {
        var decoder = new ChatterboxDecoder();
        int[] tokens = [ChatterboxAcousticLm.StartSpeechToken, 100, 105, 110, ChatterboxAcousticLm.StopSpeechToken];
        float[] spk = ChatterboxVoices.GetSpeakerFeatures("resemble_default");

        float[] audio = decoder.Decode(tokens, spk);

        Assert.NotNull(audio);
        Assert.Equal(3 * ChatterboxDecoder.HopLength, audio.Length);

        for (int i = 0; i < audio.Length; i++)
        {
            Assert.False(float.IsNaN(audio[i]));
            Assert.False(float.IsInfinity(audio[i]));
            Assert.InRange(audio[i], -1.05f, 1.05f);
        }
    }

    [Fact]
    public void ChatterboxPipeline_Generate_ProducesValidWav()
    {
        using var pipeline = new ChatterboxPipeline();
        var req = new AudioGenerationRequest
        {
            Text = "Testing Chatterbox-Turbo speech synthesis in OpenTail Stingray.",
            Voice = "narrator",
            Speed = 1.0f
        };

        var result = pipeline.Generate(req);

        Assert.NotNull(result);
        Assert.Equal(24000, result.SampleRate);
        Assert.True(result.Samples.Length > 2400); // More than 0.1s
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
