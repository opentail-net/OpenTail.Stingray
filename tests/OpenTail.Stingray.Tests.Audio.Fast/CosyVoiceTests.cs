using OpenTail.Stingray.Audio.CosyVoice;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class CosyVoiceTests
{
    [Fact]
    public void CosyVoiceTokenizer_EncodeAndDecode_PreservesSpecialEmotionTags()
    {
        var tokenizer = new CosyVoiceTokenizer();
        string text = "Hello world! [laughter] That was funny [sigh].";

        int[] tokens = tokenizer.Encode(text, addPromptBoundary: false);

        Assert.NotNull(tokens);
        Assert.NotEmpty(tokens);

        string decoded = tokenizer.Decode(tokens);
        Assert.Contains("Hello", decoded, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("[laughter]", decoded, StringComparison.Ordinal);
        Assert.Contains("[sigh]", decoded, StringComparison.Ordinal);
    }

    [Fact]
    public void CosyVoiceLlm_GenerateSpeechTokens_ProducesValidCodebookIndices()
    {
        using var llm = new CosyVoiceLlm();
        int[] promptText = [10, 20, 30];
        int[] promptSpeech = [100, 200, 300];
        int[] synthesisText = [40, 50, 60, 70];

        int[] speechTokens = llm.GenerateSpeechTokens(
            promptTextTokens: promptText,
            promptSpeechTokens: promptSpeech,
            synthesisTextTokens: synthesisText,
            maxTokens: 50,
            seed: 123);

        Assert.NotNull(speechTokens);
        Assert.NotEmpty(speechTokens);

        foreach (int token in speechTokens)
        {
            Assert.InRange(token, 0, llm.Config.SpeechTokenSize - 1);
        }
    }

    [Fact]
    public void CosyVoiceFlowDiT_SolveFlowMatchingOde_Generates80ChannelMelSpectrogram()
    {
        using var flowDiT = new CosyVoiceFlowDiT();
        int[] speechTokens = [12, 34, 56, 78];
        float[] spkEmbed = new float[80];
        Array.Fill(spkEmbed, 0.1f);

        float[] mel = flowDiT.SolveFlowMatchingOde(
            speechTokens: speechTokens,
            promptMel: [],
            speakerEmbedding: spkEmbed,
            odeSteps: 5,
            seed: 42);

        Assert.NotNull(mel);
        int expectedFrames = speechTokens.Length * flowDiT.Config.TokenMelRatio;
        int expectedElements = expectedFrames * flowDiT.Config.MelDim;
        Assert.Equal(expectedElements, mel.Length);

        for (int i = 0; i < mel.Length; i++)
        {
            Assert.False(float.IsNaN(mel[i]), $"NaN encountered at index {i}");
            Assert.False(float.IsInfinity(mel[i]), $"Infinity encountered at index {i}");
        }
    }

    [Fact]
    public void CosyVoiceHiFT_Synthesize_GeneratesValid24kHzPcmAudio()
    {
        using var hift = new CosyVoiceHiFT();
        int numFrames = 10;
        var mel = new float[numFrames * hift.Config.MelDim];
        for (int i = 0; i < mel.Length; i++)
        {
            mel[i] = 0.5f * MathF.Sin(i * 0.1f);
        }

        float[] pcm = hift.Synthesize(mel, numFrames);

        Assert.NotNull(pcm);
        Assert.Equal(numFrames * hift.Config.HopLength, pcm.Length);

        for (int i = 0; i < pcm.Length; i++)
        {
            Assert.False(float.IsNaN(pcm[i]), $"NaN at sample {i}");
            Assert.False(float.IsInfinity(pcm[i]), $"Infinity at sample {i}");
            Assert.InRange(pcm[i], -1.0f, 1.0f);
        }
    }

    [Fact]
    public void CosyVoicePipeline_Generate_EndToEndSynthesis()
    {
        using var pipeline = new CosyVoicePipeline();
        var request = new AudioGenerationRequest
        {
            Text = "CosyVoice 3 native synthesis in OpenTail.Stingray is working seamlessly!",
            Voice = "cosy_female_zh_en",
            Speed = 1.0f
        };

        var result = pipeline.Generate(request);

        Assert.NotNull(result);
        Assert.Equal(24000, result.SampleRate);
        Assert.NotNull(result.Samples);
        Assert.NotEmpty(result.Samples);
        Assert.True(result.Duration.TotalSeconds > 0.1);

        byte[] wavBytes = result.ToWavBytes();
        Assert.NotNull(wavBytes);
        Assert.True(wavBytes.Length > 44);
    }

    [Fact]
    public async Task CosyVoicePipeline_GenerateStreamAsync_YieldsStreamingChunks()
    {
        using var pipeline = new CosyVoicePipeline();
        var request = new AudioGenerationRequest
        {
            Text = "First streaming sentence. Second streaming sentence! And the final sentence.",
            Voice = "cosy_male_zh_en"
        };

        var chunks = new List<float[]>();
        await foreach (var chunk in pipeline.GenerateStreamAsync(request))
        {
            Assert.NotNull(chunk);
            Assert.NotEmpty(chunk);
            chunks.Add(chunk);
        }

        Assert.True(chunks.Count >= 2, $"Expected multiple streamed chunks, got {chunks.Count}");
    }
}
