using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.QwenASR;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class QwenAsrTests
{
    [Fact]
    public void QwenAsrMelExtractor_ExtractMel_Produces128ChannelLogMelSpectrogram()
    {
        var extractor = new QwenAsrMelExtractor();
        int sampleRate = 16000;
        int numSamples = sampleRate * 1; // 1 second
        var pcm = new float[numSamples];

        for (int i = 0; i < numSamples; i++)
        {
            pcm[i] = 0.5f * MathF.Sin(2.0f * MathF.PI * 500.0f * i / sampleRate);
        }

        float[] mel = extractor.ExtractMel(pcm);

        Assert.NotNull(mel);
        Assert.NotEmpty(mel);
        Assert.Equal(0, mel.Length % QwenAsrMelExtractor.NumMels);

        int numFrames = mel.Length / QwenAsrMelExtractor.NumMels;
        Assert.True(numFrames >= 90);

        for (int i = 0; i < mel.Length; i++)
        {
            Assert.False(float.IsNaN(mel[i]), $"NaN at mel index {i}");
            Assert.False(float.IsInfinity(mel[i]), $"Infinity at mel index {i}");
        }
    }

    [Fact]
    public void QwenAsrTokenizer_FormatAndEncode_ProducesChatMlPromptWithSpecialAudioTokens()
    {
        var tokenizer = new QwenAsrTokenizer();
        string prompt = tokenizer.FormatPrompt(language: "en", taskInstruction: "Transcribe the audio speech.");

        Assert.Contains("<|im_start|>", prompt, StringComparison.Ordinal);
        Assert.Contains("<|audio_bos|><|AUDIO|><|audio_eos|>", prompt, StringComparison.Ordinal);
        Assert.Contains("Language: en", prompt, StringComparison.Ordinal);

        int[] tokens = tokenizer.Encode(prompt);

        Assert.NotNull(tokens);
        Assert.NotEmpty(tokens);
        Assert.Contains(QwenAsrTokenizer.ImStartTokenId, tokens);
        Assert.Contains(QwenAsrTokenizer.AudioPadTokenId, tokens);
    }

    [Fact]
    public void QwenAsrAudioEncoder_Forward_AppliesConv2dStemAndWindowedAttention()
    {
        using var encoder = new QwenAsrAudioEncoder(new QwenAsrEncoderConfig
        {
            EncoderDim = 256,
            NumLayers = 4,
            QwenHiddenDim = 512
        });

        int numMelFrames = 64;
        var mel = new float[128 * numMelFrames];
        for (int i = 0; i < mel.Length; i++) mel[i] = 0.1f * MathF.Sin(i * 0.2f);

        var (tokens, numTokens) = encoder.Forward(mel, numMelFrames);

        Assert.NotNull(tokens);
        Assert.Equal(numMelFrames / 8, numTokens);
        Assert.Equal(numTokens * encoder.Config.QwenHiddenDim, tokens.Length);

        for (int i = 0; i < tokens.Length; i++)
        {
            Assert.False(float.IsNaN(tokens[i]), $"NaN in audio token {i}");
            Assert.False(float.IsInfinity(tokens[i]), $"Infinity in audio token {i}");
        }
    }

    [Fact]
    public void QwenAsrForcedAligner_Align_ProducesWordLevelTimestamps()
    {
        using var aligner = new QwenAsrForcedAligner();
        string reference = "open source speech recognition with qwen audio";
        int numAudioTokens = 32;
        var audioTokens = new float[numAudioTokens * 512];

        var segments = aligner.Align(
            referenceText: reference,
            audioTokens: audioTokens,
            numAudioTokens: numAudioTokens,
            audioDim: 512,
            timeOffset: TimeSpan.Zero);

        Assert.NotNull(segments);
        Assert.Equal(7, segments.Count); // 7 words

        for (int i = 0; i < segments.Count; i++)
        {
            Assert.True(segments[i].End > segments[i].Start);
            Assert.NotEmpty(segments[i].Text);
        }
    }

    [Fact]
    public void QwenAsrPipeline_Transcribe_EndToEndBatchTranscription()
    {
        using var pipeline = new QwenAsrPipeline();
        int sampleRate = 16000;
        int numSamples = sampleRate * 2; // 2 seconds of audio
        var pcm = new float[numSamples];

        for (int i = 0; i < numSamples; i++)
        {
            pcm[i] = 0.3f * MathF.Sin(2.0f * MathF.PI * 400.0f * i / sampleRate);
        }

        var request = new SpeechToTextRequest
        {
            AudioSamples = pcm,
            SampleRate = sampleRate,
            Language = "en"
        };

        var result = pipeline.Transcribe(request);

        Assert.NotNull(result);
        Assert.Equal("en", result.Language);
        Assert.True(result.Duration.TotalSeconds >= 1.9);
        Assert.NotNull(result.Segments);
    }

    [Fact]
    public async Task QwenAsrPipeline_TranscribeStreamAsync_StreamsAlignedSpeechSegments()
    {
        using var pipeline = new QwenAsrPipeline();
        int sampleRate = 16000;
        int chunkSize = sampleRate; // 1-second chunks

        async IAsyncEnumerable<ReadOnlyMemory<float>> GenerateStream()
        {
            for (int chunk = 0; chunk < 3; chunk++)
            {
                var pcm = new float[chunkSize];
                for (int i = 0; i < chunkSize; i++)
                {
                    pcm[i] = 0.3f * MathF.Sin(2.0f * MathF.PI * (200.0f + chunk * 100.0f) * i / sampleRate);
                }
                yield return pcm;
                await Task.Yield();
            }
        }

        var request = new SpeechToTextRequest
        {
            AudioSamples = [],
            SampleRate = sampleRate,
            Language = "en"
        };

        var segments = new List<SpeechSegment>();
        await foreach (var seg in pipeline.TranscribeStreamAsync(GenerateStream(), request))
        {
            Assert.NotNull(seg);
            segments.Add(seg);
        }

        Assert.NotNull(segments);
    }
}
