
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real end-to-end test for <see cref="OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralPipeline"/>
/// (perf-sweep Phase 1.4, docs/perf-sweep-plan.md) -- verifies the real, wired
/// <see cref="ISpeechToTextPipeline"/> implementation reproduces the same reference transcript
/// that <see cref="VoxtralGenerationLoopTests"/> already golden-verified by hand-driving
/// PrefillWithCache/Step directly. This is the actual usability check: before this pipeline
/// class existed, nothing outside test code could reach Voxtral's ASR at all.</summary>
public sealed class VoxtralPipelineEndToEndTests : HeavyTestBase
{
    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Transcribe_RealAudio_RealWeights_MatchesReferenceTranscript()
    {
        string? checkpointDir = FindRepoFile("models/_models/voxtral-mini-realtime/model.safetensors") is { } p
            ? Path.GetDirectoryName(p) : null;
        Assert.SkipUnless(checkpointDir != null, "voxtral-mini-realtime checkpoint not found");
        string? audioPath = FindRepoFile("examples/audio.cpp/assets/resources/a.wav");
        Assert.SkipUnless(audioPath != null, "reference a.wav not found");

        var (samples, sr, _) = OpenTail.Stingray.Audio.WavReader.ReadWav(audioPath!);
        if (sr != 16000) samples = OpenTail.Stingray.Audio.AudioResampler.Resample(samples, sr, 16000);

        using var pipeline = OpenTail.Stingray.Audio.VoxtralRealtime.VoxtralPipeline.Load(checkpointDir!);
        var result = pipeline.Transcribe(new OpenTail.Stingray.Audio.SpeechToTextRequest
        {
            AudioSamples = samples,
            SampleRate = 16000,
            Temperature = 0f,
        });

        const string referenceText = "This little work was finished in the year 1803, and intended for immediate publication.";
        Console.Error.WriteLine($"[VoxtralPipeline] text=\"{result.Text}\"");
        Console.Error.WriteLine($"[VoxtralPipeline] reference_text=\"{referenceText}\"");
        Assert.Equal(referenceText, result.Text);
        Assert.Single(result.Segments);
        Assert.Equal("Voxtral-Mini-4B-Realtime", pipeline.Architecture);
    }
}
