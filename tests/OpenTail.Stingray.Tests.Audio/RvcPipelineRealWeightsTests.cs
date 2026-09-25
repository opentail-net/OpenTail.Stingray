namespace OpenTail.Stingray.Tests.Audio;

/// <summary>End-to-end RVC voice conversion (<see cref="OpenTail.Stingray.Audio.Rvc.RvcPipeline"/>)
/// on the reference's own `a.wav` with the packaged v2 `default` voice. Asserts structure (voice
/// sample rate, output length = input duration, finite, non-silent); intelligibility and the
/// comparison against `audiocpp_cli --task vc --family rvc` are checked by a Whisper round trip on
/// the written WAV (`RVC_OUT`), see docs/audio-review-new-progress.md.</summary>
public sealed class RvcPipelineRealWeightsTests : HeavyTestBase
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
    public void Convert_RealAudio_V2DefaultVoice_ProducesVoiceRateAudioOfInputDuration()
    {
        string? checkpointPath = FindRepoFile("examples/audio.cpp/models/RVC-GGUF/rvc-f16.gguf");
        Assert.SkipUnless(checkpointPath != null, "rvc-f16.gguf not found");
        string? audioPath = Environment.GetEnvironmentVariable("RVC_IN") ?? FindRepoFile("examples/audio.cpp/assets/resources/a.wav");
        Assert.SkipUnless(audioPath != null, "reference a.wav not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(checkpointPath!);
        var source = new OpenTail.Stingray.Audio.Rvc.RvcPackedTensorSource(model);
        var hubert = new OpenTail.Stingray.Audio.Rvc.RvcHubertWeights(source);
        var rmvpe = new OpenTail.Stingray.Audio.Rvc.RvcRmvpeWeights(source);
        var synth = new OpenTail.Stingray.Audio.Rvc.RvcSynthesizerWeights(source, "voice_v2_default_checkpoint", sampleRate: 40000, v1: false, hasF0: true);

        var (samples, sr, channels) = WavReader.ReadWav(audioPath!);
        Assert.Equal(1, channels);
        if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var output = OpenTail.Stingray.Audio.Rvc.RvcPipeline.Convert(
            hubert, rmvpe, synth, retrieval: null, samples, new OpenTail.Stingray.Audio.Rvc.RvcInferenceOptions(), new Random(1234));
        sw.Stop();

        double inSeconds = samples.Length / 16000.0, outSeconds = output.Length / (double)synth.SampleRate;
        double rms = Math.Sqrt(output.Sum(v => (double)v * v) / output.Length);
        Console.WriteLine($"[RvcPipeline] in {inSeconds:F2}s -> out {outSeconds:F2}s @ {synth.SampleRate} Hz, rms {rms:F4}, convert {sw.Elapsed.TotalSeconds:F1}s");

        Assert.All(output, v => Assert.True(float.IsFinite(v)));
        Assert.InRange(outSeconds, inSeconds - 0.05, inSeconds + 0.05);
        Assert.True(rms > 1e-3, $"output is near-silent (rms {rms})");

        string? outPath = Environment.GetEnvironmentVariable("RVC_OUT");
        if (outPath is not null)
            new OpenTail.Stingray.Audio.AudioGenerationResult(output, synth.SampleRate).SaveWav(outPath);
    }
}
