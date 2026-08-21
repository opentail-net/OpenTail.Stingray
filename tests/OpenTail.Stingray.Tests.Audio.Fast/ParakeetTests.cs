using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Parakeet;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class ParakeetTests
{
    [Fact]
    public void ParakeetMelExtractor_ExtractMel_Produces80ChannelLogMelSpectrogram()
    {
        var extractor = new ParakeetMelExtractor();
        int sampleRate = 16000;
        int durationSamples = sampleRate * 1; // 1 second of audio
        var pcm = new float[durationSamples];

        // Synthesize test audio (sine wave 440 Hz)
        for (int i = 0; i < durationSamples; i++)
        {
            pcm[i] = 0.5f * MathF.Sin(2.0f * MathF.PI * 440.0f * i / sampleRate);
        }

        float[] mel = extractor.ExtractMel(pcm);

        Assert.NotNull(mel);
        Assert.NotEmpty(mel);
        Assert.Equal(0, mel.Length % ParakeetMelExtractor.NumMels);

        int numFrames = mel.Length / ParakeetMelExtractor.NumMels;
        Assert.True(numFrames > 90); // ~100 frames for 1 second with 10ms hop

        for (int i = 0; i < mel.Length; i++)
        {
            Assert.False(float.IsNaN(mel[i]), $"NaN at mel index {i}");
            Assert.False(float.IsInfinity(mel[i]), $"Infinity at mel index {i}");
        }
    }

    [Fact]
    public void ParakeetTokenizer_EncodeAndDecode_PreservesTextAndHandlesSpecialTokens()
    {
        var tokenizer = new ParakeetTokenizer();
        string text = "the speech recognition model is fast and accurate";

        int[] tokens = tokenizer.Encode(text);

        Assert.NotNull(tokens);
        Assert.NotEmpty(tokens);

        foreach (int t in tokens)
        {
            Assert.InRange(t, 0, tokenizer.VocabSize - 1);
            Assert.NotEqual(ParakeetTokenizer.BlankTokenId, t);
        }

        string decoded = tokenizer.Decode(tokens);
        Assert.NotNull(decoded);
        Assert.NotEmpty(decoded);
        Assert.Equal(text, decoded, ignoreCase: true);
    }

    [Fact]
    public void ParakeetConformerEncoder_Forward_SubsamplesAndEncodesAcousticFeatures()
    {
        using var encoder = new ParakeetConformerEncoder(new ParakeetEncoderConfig
        {
            NumLayers = 4, // Fast test configuration
            HiddenDim = 256,
            SubsampleFactor = 8
        });

        int inNumFrames = 64;
        var melFrames = new float[inNumFrames * 80];
        for (int i = 0; i < melFrames.Length; i++)
        {
            melFrames[i] = MathF.Sin(i * 0.1f);
        }

        var (embeddings, outNumFrames) = encoder.Forward(melFrames, inNumFrames);

        Assert.NotNull(embeddings);
        Assert.Equal(inNumFrames / 8, outNumFrames);
        Assert.Equal(outNumFrames * encoder.Config.HiddenDim, embeddings.Length);

        for (int i = 0; i < embeddings.Length; i++)
        {
            Assert.False(float.IsNaN(embeddings[i]), $"NaN in embedding at {i}");
            Assert.False(float.IsInfinity(embeddings[i]), $"Infinity in embedding at {i}");
        }
    }

    [Fact]
    public void ParakeetCtcDecoder_DecodeGreedy_CollapsesBlanksAndProducesSegments()
    {
        var tokenizer = new ParakeetTokenizer();
        using var decoder = new ParakeetCtcDecoder(tokenizer);

        int numFrames = 16;
        int hiddenDim = 128;
        var embeddings = new float[numFrames * hiddenDim];

        for (int i = 0; i < embeddings.Length; i++)
        {
            embeddings[i] = 0.5f * MathF.Sin(i * 0.3f);
        }

        var (fullText, tokens, segments) = decoder.DecodeGreedy(
            embeddings: embeddings,
            numFrames: numFrames,
            hiddenDim: hiddenDim,
            timeOffset: TimeSpan.Zero);

        Assert.NotNull(fullText);
        Assert.NotNull(tokens);
        Assert.NotNull(segments);

        foreach (var seg in segments)
        {
            Assert.True(seg.End >= seg.Start);
            Assert.NotNull(seg.Text);
        }
    }

    [Fact]
    public void ParakeetPipeline_Transcribe_EndToEndBatchTranscription()
    {
        using var pipeline = new ParakeetPipeline();
        int sampleRate = 16000;
        int numSamples = sampleRate * 2; // 2 seconds of test audio
        var pcm = new float[numSamples];

        for (int i = 0; i < numSamples; i++)
        {
            pcm[i] = 0.4f * MathF.Sin(2.0f * MathF.PI * 300.0f * i / sampleRate);
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
    public async Task ParakeetPipeline_TranscribeStreamAsync_StreamsSpeechSegments()
    {
        using var pipeline = new ParakeetPipeline();
        int sampleRate = 16000;
        int chunkSize = sampleRate; // 1 second chunks

        async IAsyncEnumerable<ReadOnlyMemory<float>> GenerateChunks()
        {
            for (int chunk = 0; chunk < 3; chunk++)
            {
                var pcm = new float[chunkSize];
                for (int i = 0; i < chunkSize; i++)
                {
                    pcm[i] = 0.3f * MathF.Sin(2.0f * MathF.PI * (250.0f + chunk * 50.0f) * i / sampleRate);
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
        await foreach (var seg in pipeline.TranscribeStreamAsync(GenerateChunks(), request))
        {
            Assert.NotNull(seg);
            segments.Add(seg);
        }

        Assert.NotNull(segments);
    }
}
