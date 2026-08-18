using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Whisper;
using Xunit;

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
        Assert.Equal(WhisperTokenizer.TranscribeToken, promptTranscribe[2]);

        // 3. Initial prompt building for translation without timestamps
        int[] promptTranslate = tokenizer.BuildInitialPrompt("fr", SpeechTask.Translate, enableTimestamps: false);
        Assert.Equal(4, promptTranslate.Length);
        Assert.Equal(WhisperTokenizer.StartOfTranscript, promptTranslate[0]);
        Assert.Equal(tokenizer.GetLanguageToken("fr"), promptTranslate[1]);
        Assert.Equal(WhisperTokenizer.TranslateToken, promptTranslate[2]);
        Assert.Equal(WhisperTokenizer.NoTimestampsToken, promptTranslate[3]);
    }

    [Fact]
    public void WhisperTokenizer_TimestampDecoding_ParsesSegmentsCorrectly()
    {
        var tokenizer = new WhisperTokenizer();

        // Construct mock sequence: <|startoftranscript|> <|0.00|> Hello <|2.00|> <|2.00|> World <|4.50|> <|endoftranscript|>
        int ts0 = WhisperTokenizer.TimestampBegin; // 0.00s
        int ts100 = WhisperTokenizer.TimestampBegin + 100; // 2.00s (100 * 0.02)
        int ts225 = WhisperTokenizer.TimestampBegin + 225; // 4.50s (225 * 0.02)

        int[] mockTokens =
        [
            WhisperTokenizer.StartOfTranscript,
            ts0,
            (int)'H', (int)'i',
            ts100,
            ts100,
            (int)'t', (int)'h', (int)'e', (int)'r', (int)'e',
            ts225,
            WhisperTokenizer.EndOfText
        ];

        var (text, segments) = tokenizer.DecodeSegments(mockTokens, TimeSpan.Zero);

        Assert.NotNull(text);
        Assert.Equal(2, segments.Count);

        Assert.Equal(TimeSpan.FromSeconds(0.0), segments[0].Start);
        Assert.Equal(TimeSpan.FromSeconds(2.0), segments[0].End);

        Assert.Equal(TimeSpan.FromSeconds(2.0), segments[1].Start);
        Assert.Equal(TimeSpan.FromSeconds(4.5), segments[1].End);
    }

    [Fact]
    public void WhisperEncoder_Forward_ProducesValidAudioRepresentations()
    {
        var config = WhisperConfig.Tiny;
        var encoder = new WhisperEncoder(config);

        int numFrames = 100;
        float[] mel = new float[config.NumMels * numFrames];
        for (int i = 0; i < mel.Length; i++) mel[i] = 0.1f;

        float[] hidden = encoder.Forward(mel, numFrames);

        Assert.NotNull(hidden);
        int expectedEncFrames = (numFrames + 1) / 2;
        Assert.Equal(expectedEncFrames * config.AudioState, hidden.Length);

        for (int i = 0; i < hidden.Length; i++)
        {
            Assert.False(float.IsNaN(hidden[i]));
            Assert.False(float.IsInfinity(hidden[i]));
        }
    }

    [Fact]
    public void WhisperDecoder_ForwardNextToken_ProducesVocabLogits()
    {
        var config = WhisperConfig.Tiny;
        var decoder = new WhisperDecoder(config);

        int encFrames = 50;
        float[] audioState = new float[encFrames * config.AudioState];
        int[] prompt = [WhisperTokenizer.StartOfTranscript, WhisperTokenizer.EnglishLanguageToken, WhisperTokenizer.TranscribeToken];

        float[] logits = decoder.ForwardNextToken(prompt, audioState, encFrames);

        Assert.NotNull(logits);
        Assert.Equal(config.VocabSize, logits.Length);

        for (int i = 0; i < logits.Length; i++)
        {
            Assert.False(float.IsNaN(logits[i]));
            Assert.False(float.IsInfinity(logits[i]));
        }
    }

    [Fact]
    public void WavReader_WavWriter_RoundtripsPcmAudioCorrectly()
    {
        float[] original = new float[1600]; // 0.1s @ 16kHz
        for (int i = 0; i < original.Length; i++)
        {
            original[i] = 0.4f * MathF.Sin(2.0f * MathF.PI * 440.0f * i / 16000.0f);
        }

        byte[] wavBytes = WavWriter.ToWavBytes(original, sampleRate: 16000);
        Assert.NotNull(wavBytes);

        using var ms = new MemoryStream(wavBytes);
        var (decodedSamples, sampleRate, channels) = WavReader.ReadWav(ms);

        Assert.Equal(16000, sampleRate);
        Assert.Equal(1, channels);
        Assert.Equal(original.Length, decodedSamples.Length);

        for (int i = 0; i < original.Length; i++)
        {
            Assert.InRange(decodedSamples[i], original[i] - 0.01f, original[i] + 0.01f);
        }
    }

    [Fact]
    public void WhisperPipeline_Transcribe_EndToEndExecutionSucceeds()
    {
        using var pipeline = new WhisperPipeline(WhisperConfig.Tiny);
        Assert.Equal("OpenAI-Whisper", pipeline.Architecture);
        Assert.Equal(16000, pipeline.SampleRate);

        float[] pcm = new float[16000]; // 1s synthetic audio
        for (int i = 0; i < pcm.Length; i++)
        {
            pcm[i] = 0.3f * MathF.Sin(2.0f * MathF.PI * 300.0f * i / 16000.0f);
        }

        var request = new SpeechToTextRequest
        {
            AudioSamples = pcm,
            SampleRate = 16000,
            Language = "en",
            Task = SpeechTask.Transcribe,
            EnableTimestamps = true,
            Temperature = 0.0f
        };

        var result = pipeline.Transcribe(request);

        Assert.NotNull(result);
        Assert.Equal("en", result.Language);
        Assert.Equal(TimeSpan.FromSeconds(1.0), result.Duration);
        Assert.NotNull(result.Segments);
    }
}
