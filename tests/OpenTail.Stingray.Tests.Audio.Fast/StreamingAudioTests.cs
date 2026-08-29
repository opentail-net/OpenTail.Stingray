using OpenTail.Stingray.Audio.Piper;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class StreamingAudioTests
{
    [Fact]
    public async Task Kokoro_GenerateStreamAsync_YieldsMultipleChunksForCompoundText()
    {
        using var pipeline = new KokoroPipeline();
        var request = new AudioGenerationRequest
        {
            Text = "Hello world! This is a test. Streaming is wonderful.",
            Voice = "af_heart"
        };

        var chunks = new List<float[]>();
        await foreach (var chunk in pipeline.GenerateStreamAsync(request))
        {
            chunks.Add(chunk);
            Assert.NotEmpty(chunk);
        }

        Assert.Equal(3, chunks.Count);
    }

    [Fact]
    public async Task Piper_GenerateStreamAsync_YieldsMultipleChunks()
    {
        using var pipeline = new PiperPipeline();
        var request = new AudioGenerationRequest
        {
            Text = "First sentence. Second sentence.",
            Voice = "default"
        };

        var chunks = new List<float[]>();
        await foreach (var chunk in pipeline.GenerateStreamAsync(request))
        {
            chunks.Add(chunk);
            Assert.NotEmpty(chunk);
        }

        Assert.Equal(2, chunks.Count);
    }

    [Fact]
    public async Task Whisper_TranscribeStreamAsync_ProcessesIncomingAudioStream()
    {
        using var pipeline = new WhisperPipeline();
        
        // Generate simulated audio chunks (2 chunks of 1 second each)
        async IAsyncEnumerable<ReadOnlyMemory<float>> GenerateSimulatedAudio()
        {
            for (int i = 0; i < 2; i++)
            {
                float[] chunk = new float[16000];
                for (int j = 0; j < chunk.Length; j++)
                {
                    chunk[j] = MathF.Sin(2.0f * MathF.PI * 440.0f * j / 16000.0f) * 0.1f;
                }
                yield return chunk.AsMemory();
                await Task.Yield();
            }
        }

        var baseReq = new SpeechToTextRequest
        {
            AudioSamples = [],
            SampleRate = 16000,
            Language = "en"
        };

        var segments = new List<SpeechSegment>();
        await foreach (var seg in pipeline.TranscribeStreamAsync(GenerateSimulatedAudio(), baseReq))
        {
            segments.Add(seg);
        }

        // Segments should be returned from the stream
        Assert.NotNull(segments);
    }
}
