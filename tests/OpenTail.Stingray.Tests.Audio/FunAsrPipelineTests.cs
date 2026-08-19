using System;
using System.Linq;
using System.Threading.Tasks;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.FunASR;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class FunAsrPipelineTests
{
    [Fact]
    public void FunAsr_Transcribe_ProducesValidTokensAndText()
    {
        using var pipeline = new FunAsrPipeline();
        Assert.Equal("Alibaba-FunASR-Nano", pipeline.Architecture);
        Assert.Equal(16000, pipeline.SampleRate);

        // Generate 2 seconds of 16kHz sine wave audio
        float[] audio = new float[16000 * 2];
        for (int i = 0; i < audio.Length; i++)
        {
            audio[i] = MathF.Sin(2.0f * MathF.PI * 440.0f * i / 16000.0f) * 0.5f;
        }

        var request = new SpeechToTextRequest
        {
            AudioSamples = audio,
            SampleRate = 16000,
            Language = "zh"
        };

        var result = pipeline.Transcribe(request);
        Assert.NotNull(result);
        Assert.Equal("zh", result.Language);
        Assert.True(result.Duration.TotalSeconds > 1.9);
        Assert.NotEmpty(result.Segments);
        Assert.False(string.IsNullOrWhiteSpace(result.Text));
    }

    [Fact]
    public async Task FunAsr_TranscribeStream_YieldsStreamingSegments()
    {
        using var pipeline = new FunAsrPipeline();

        async IAsyncEnumerable<ReadOnlyMemory<float>> GenerateAudioChunks()
        {
            for (int chunk = 0; chunk < 4; chunk++)
            {
                float[] samples = new float[16000]; // 1 second chunk
                for (int i = 0; i < samples.Length; i++)
                {
                    samples[i] = MathF.Sin(2.0f * MathF.PI * 220.0f * i / 16000.0f) * 0.3f;
                }
                yield return samples;
                await Task.Yield();
            }
        }

        var request = new SpeechToTextRequest
        {
            AudioSamples = [],
            SampleRate = 16000,
            Language = "zh"
        };

        int count = 0;
        await foreach (var segment in pipeline.TranscribeStreamAsync(GenerateAudioChunks(), request))
        {
            Assert.NotNull(segment);
            Assert.False(string.IsNullOrWhiteSpace(segment.Text));
            count++;
        }

        Assert.True(count >= 2, "Streaming should yield multiple segments");
    }
}
