
namespace OpenTail.Stingray.Tests.Audio;

public sealed class WhisperTests
{
    [Fact]
    public void WhisperMelExtractor_ExtractMel_Produces80ChannelLogMelSpectrogram()
    {
        var extractor = new WhisperMelExtractor(numMels: 80);
        float[] pcm = new float[16000]; // 1 second of 440Hz sine wave
        for (int i = 0; i < pcm.Length; i++)
        {
            pcm[i] = 0.5f * MathF.Sin(2.0f * MathF.PI * 440.0f * i / 16000.0f);
        }

        float[] mel = extractor.ExtractMel(pcm, padTo30Seconds: true);

        Assert.NotNull(mel);
        Assert.NotEmpty(mel);

        // Standard 30s Whisper input yields 3000 frames * 80 mel channels
        int numFrames = mel.Length / 80;
        Assert.True(numFrames >= 100);

        // Verify values are finite and normalized in reasonable range
        for (int i = 0; i < mel.Length; i++)
        {
            Assert.False(float.IsNaN(mel[i]), $"NaN found at mel index {i}");
            Assert.False(float.IsInfinity(mel[i]), $"Infinity found at mel index {i}");
            Assert.InRange(mel[i], -5.0f, 5.0f);
        }
    }

    [Fact]
    public void WhisperMelExtractor_128Channel_ProducesValidFilterbank()
    {
        var extractor = new WhisperMelExtractor(numMels: 128);
        Assert.Equal(128, extractor.NumMels);

        float[] pcm = new float[3200]; // 0.2s of audio
        float[] mel = extractor.ExtractMel(pcm, padTo30Seconds: false);

        Assert.NotNull(mel);
        int numFrames = mel.Length / 128;
        Assert.True(numFrames >= 1);
    }

    [Fact]
    public void WhisperTokenizer_SpecialTokensAndPromptBuilding_ProduceExpectedSequence()
    {
        var tokenizer = new WhisperTokenizer();

        // 1. Language tokens
        int enToken = tokenizer.GetLanguageToken("en");
        int esToken = tokenizer.GetLanguageToken("es");
        int zhToken = tokenizer.GetLanguageToken("zh");

        Assert.Equal(WhisperTokenizer.EnglishLanguageToken, enToken);
        Assert.NotEqual(enToken, esToken);
        Assert.NotEqual(enToken, zhToken);

        // 2. Initial prompt building for transcription with timestamps
        int[] promptTranscribe = tokenizer.BuildInitialPrompt("en", SpeechTask.Transcribe, enableTimestamps: true);
        Assert.Equal(3, promptTranscribe.Length);
        Assert.Equal(WhisperTokenizer.StartOfTranscript, promptTranscribe[0]);
        Assert.Equal(WhisperTokenizer.EnglishLanguageToken, promptTranscribe[1]);
        Assert.Equal(tokenizer.TranscribeToken, promptTranscribe[2]);

        // 3. Initial prompt building for translation without timestamps
        int[] promptTranslate = tokenizer.BuildInitialPrompt("fr", SpeechTask.Translate, enableTimestamps: false);
        Assert.Equal(4, promptTranslate.Length);
        Assert.Equal(WhisperTokenizer.StartOfTranscript, promptTranslate[0]);
        Assert.Equal(tokenizer.GetLanguageToken("fr"), promptTranslate[1]);
        Assert.Equal(tokenizer.TranslateToken, promptTranslate[2]);
        Assert.Equal(tokenizer.NoTimestampsToken, promptTranslate[3]);
    }

    [Fact]
    public void WhisperTokenizer_V3_HasCantoneseAndShiftedSpecialTokens()
    {
        var tokenizerV3 = WhisperTokenizer.CreateV3();
        Assert.True(tokenizerV3.IsV3);

        // Cantonese "yue" is 100th language token at 50358
        int yueToken = tokenizerV3.GetLanguageToken("yue");
        Assert.Equal(50358, yueToken);

        // Shifted special tokens in v3
        Assert.Equal(50359, tokenizerV3.TranslateToken);
        Assert.Equal(50360, tokenizerV3.TranscribeToken);
        Assert.Equal(50361, tokenizerV3.StartOfLmToken);
        Assert.Equal(50362, tokenizerV3.StartOfPrevToken);
        Assert.Equal(50363, tokenizerV3.NoSpeechToken);
        Assert.Equal(50364, tokenizerV3.NoTimestampsToken);
        Assert.Equal(50365, tokenizerV3.TimestampBegin);
        Assert.Equal(51865, tokenizerV3.TimestampEnd);

        // Build prompt for Cantonese
        int[] yuePrompt = tokenizerV3.BuildInitialPrompt("yue", SpeechTask.Transcribe, enableTimestamps: true);
        Assert.Equal(3, yuePrompt.Length);
        Assert.Equal(WhisperTokenizer.StartOfTranscript, yuePrompt[0]);
        Assert.Equal(50358, yuePrompt[1]);
        Assert.Equal(50360, yuePrompt[2]);
    }

    [Fact]
    public void WhisperConfig_LargeV3AndTurbo_PreserveArchitectureAndLayers()
    {
        var v3 = WhisperConfig.LargeV3;
        Assert.Equal(51866, v3.VocabSize);
        Assert.Equal(128, v3.NumMels);
        Assert.Equal(1280, v3.AudioState);
        Assert.Equal(32, v3.AudioLayer);
        Assert.Equal(32, v3.TextLayer);
        Assert.Equal(20, v3.TextHead);
        Assert.True(v3.IsV3);

        var turbo = WhisperConfig.LargeV3Turbo;
        Assert.Equal(51866, turbo.VocabSize);
        Assert.Equal(128, turbo.NumMels);
        Assert.Equal(1280, turbo.AudioState);
        Assert.Equal(32, turbo.AudioLayer);
        Assert.Equal(4, turbo.TextLayer); // 4 decoder layers in Turbo
        Assert.Equal(20, turbo.TextHead);
        Assert.True(turbo.IsV3);
    }

    [Fact]
    public void WhisperPipeline_CreateLargeV3AndTurbo_InstantiatesCorrectPipelines()
    {
        using var pipeV3 = WhisperPipeline.CreateLargeV3();
        Assert.Equal("OpenAI-Whisper-Large-V3", pipeV3.Architecture);
        Assert.Equal(128, pipeV3.Config.NumMels);
        Assert.Equal(32, pipeV3.Config.TextLayer);

        using var pipeTurbo = WhisperPipeline.CreateLargeV3Turbo();
        Assert.Equal("OpenAI-Whisper-Large-V3-Turbo", pipeTurbo.Architecture);
        Assert.Equal(128, pipeTurbo.Config.NumMels);
        Assert.Equal(4, pipeTurbo.Config.TextLayer);
    }

    [Fact]
    public void WhisperDecoder_Turbo_OperatesWith4Layers()
    {
        var config = WhisperConfig.LargeV3Turbo;
        var decoder = new WhisperDecoder(config);

        var cache = new WhisperKvCache(config.TextLayer, config.TextCtx, config.TextState);
        var audioState = new float[10 * config.AudioState];

        float[] logits = decoder.ForwardStep(
            tokenId: WhisperTokenizer.StartOfTranscript,
            position: 0,
            cache: cache,
            audioEncoderOutput: audioState,
            audioFrames: 10);

        Assert.NotNull(logits);
        Assert.Equal(config.VocabSize, logits.Length);
    }
}
