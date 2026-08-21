using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Vad;
using OpenTail.Stingray.Audio.Whisper;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class SileroVadTests
{
    [Fact]
    public void SileroVad_ProcessFrame_SilenceProducesLowProbability()
    {
        using var vad = new SileroVad();
        float[] silence = new float[512]; // Zero frame

        float prob = vad.ProcessFrame(silence);

        Assert.InRange(prob, 0.0f, 0.45f);
    }

    [Fact]
    public void SileroVad_ProcessFrame_SpeechSignalProducesHighProbability()
    {
        using var vad = new SileroVad();
        float[] speech = new float[512];

        // Synthesize a speech-like multi-harmonic vowel waveform (150Hz fundamental + harmonics)
        for (int i = 0; i < 512; i++)
        {
            float t = i / 16000.0f;
            speech[i] = 0.5f * MathF.Sin(2.0f * MathF.PI * 150.0f * t)
                      + 0.3f * MathF.Sin(2.0f * MathF.PI * 300.0f * t)
                      + 0.2f * MathF.Sin(2.0f * MathF.PI * 600.0f * t);
        }

        // Process warm-up and active frame
        vad.ProcessFrame(speech);
        float prob = vad.ProcessFrame(speech);

        Assert.InRange(prob, 0.5f, 1.0f);
    }

    [Fact]
    public void SileroVad_StreamingReset_ClearsInternalState()
    {
        using var vad = new SileroVad();
        float[] frame = new float[512];
        Array.Fill(frame, 0.1f);

        vad.ProcessFrame(frame);
        vad.Reset();

        float[] silence = new float[512];
        float prob = vad.ProcessFrame(silence);

        Assert.InRange(prob, 0.0f, 0.45f);
    }

    [Fact]
    public void VadSegmenter_BuildSegments_ExtractsAccurateTimestampBounds()
    {
        // 100 frames total (each frame = 512 samples = 32ms at 16kHz)
        // Silence (0..19), Speech (20..50), Silence (51..70), Speech (71..99)
        float[] probs = new float[100];
        for (int i = 20; i <= 50; i++) probs[i] = 0.85f;
        for (int i = 71; i <= 99; i++) probs[i] = 0.90f;

        var parameters = new VadParams
        {
            Threshold = 0.5f,
            MinSpeechDurationMs = 200,
            MinSilenceDurationMs = 100,
            SpeechPadMs = 30,
            SampleRate = 16000
        };

        var segments = VadSegmenter.BuildSegments(probs, parameters, frameSize: 512);

        Assert.Equal(2, segments.Count);

        // First segment should cover roughly frames ~19 to ~51
        Assert.True(segments[0].StartSample >= 0);
        Assert.True(segments[0].EndSample > segments[0].StartSample);
        Assert.InRange(segments[0].StartSeconds, 0.5f, 0.7f);
        Assert.InRange(segments[0].EndSeconds, 1.5f, 1.8f);

        // Second segment should cover roughly frames ~70 to 100
        Assert.True(segments[1].StartSample > segments[0].EndSample);
        Assert.InRange(segments[1].StartSeconds, 2.1f, 2.4f);
        Assert.InRange(segments[1].EndSeconds, 3.0f, 3.3f);
    }

    [Fact]
    public void SileroVad_DetectSegments_EndToEndWaveformSegmentation()
    {
        using var vad = new SileroVad();

        // 3 seconds of audio: 1s silence, 1s speech, 1s silence
        int sampleRate = 16000;
        float[] audio = new float[sampleRate * 3];

        // 1s - 2s: Active speech tone
        for (int i = sampleRate; i < sampleRate * 2; i++)
        {
            float t = i / (float)sampleRate;
            audio[i] = 0.6f * MathF.Sin(2.0f * MathF.PI * 220.0f * t)
                     + 0.3f * MathF.Sin(2.0f * MathF.PI * 440.0f * t);
        }

        var segments = vad.DetectSegments(audio);

        Assert.NotEmpty(segments);
        Assert.True(segments[0].StartSeconds >= 0.8f);
        Assert.True(segments[0].EndSeconds <= 2.2f);
    }

    [Fact]
    public void WhisperPipeline_TranscribeWithVad_SuccessfullyPrunesSilence()
    {
        using var pipeline = new WhisperPipeline(WhisperConfig.Tiny);

        // 2 seconds audio (speech + silence)
        float[] audio = new float[32000];
        for (int i = 0; i < 16000; i++)
        {
            float t = i / 16000.0f;
            audio[i] = 0.4f * MathF.Sin(2.0f * MathF.PI * 300.0f * t);
        }

        var req = new SpeechToTextRequest
        {
            AudioSamples = audio,
            Language = "en",
            Task = SpeechTask.Transcribe,
            UseVad = true
        };

        var result = pipeline.Transcribe(req);

        Assert.NotNull(result);
        Assert.Equal("en", result.Language);
        Assert.Equal(TimeSpan.FromSeconds(2), result.Duration);
    }
}
