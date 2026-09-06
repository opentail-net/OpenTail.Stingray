
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real forward-pass validation for RvcHubertEncoder against real weights and real
/// audio, cross-checked numerically against the real C++ reference (STINGRAY_RVC_TRACE=1 added to
/// examples/audio.cpp/src/models/rvc/hubert.cpp).
///
/// <para>Confirmed 2026-09-06: on the identical audio, our own hidden-state mean/std
/// (-0.0049/0.3319, 295 frames on a trimmed clip; -0.0049/0.3318, 297 frames on the full clip)
/// closely match the reference's (-0.0054/0.3400, 397 tokens on the full clip) -- strong evidence
/// the encoder math itself is correct. The reference's frame COUNT differs (397 vs 297) because
/// its native_pipeline pads audio with ~1s of silence on each side before HuBERT
/// (`audio_pad_duration_sec` default 1s: 397*320/16000=7.94s vs the raw clip's real 5.95s
/// duration, matching almost exactly) -- a pipeline-level preprocessing step, not implemented
/// here yet, NOT a bug in this encoder's own conv/transformer math.</para></summary>
public sealed class RvcHubertEncoderForwardTests : HeavyTestBase
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
    public void Forward_RealWeights_RealAudio_ProducesFiniteNonDegenerateOutput()
    {
        string? checkpointPath = FindRepoFile("examples/audio.cpp/models/RVC-GGUF/rvc-f16.gguf");
        Assert.SkipUnless(checkpointPath != null, "rvc-f16.gguf not found");
        string? audioPath = FindRepoFile("examples/audio.cpp/assets/resources/a.wav");
        Assert.SkipUnless(audioPath != null, "reference a.wav not found");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(checkpointPath!);
        var source = new OpenTail.Stingray.Audio.Rvc.RvcPackedTensorSource(model);
        var w = new OpenTail.Stingray.Audio.Rvc.RvcHubertWeights(source);

        var (samples, sr, _) = WavReader.ReadWav(audioPath!);
        if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);
        var clip = samples;

        var hidden = OpenTail.Stingray.Audio.Rvc.RvcHubertEncoder.Forward(w, clip);

        Assert.NotEmpty(hidden);
        Assert.Equal(OpenTail.Stingray.Audio.Rvc.RvcHubertWeights.HiddenDim, hidden[0].Length);

        double sum = 0, sumSq = 0;
        int n = 0;
        foreach (var row in hidden)
        {
            foreach (var v in row)
            {
                Assert.True(float.IsFinite(v), "non-finite value in hidden state");
                sum += v;
                sumSq += (double)v * v;
                n++;
            }
        }
        double mean = sum / n;
        double std = Math.Sqrt(Math.Max(0, sumSq / n - mean * mean));
        Console.Error.WriteLine($"[RvcHubertFwd] frames={hidden.Length} mean={mean:F4} std={std:F4}");
        Assert.InRange(std, 1e-3, 100.0);

        // Real regression guard: the reference's own hidden-state mean/std on this exact audio
        // (padding aside -- see class doc comment) is -0.0054/0.3400. Ours should stay close.
        Assert.InRange(mean, -0.05, 0.05);
        Assert.InRange(std, 0.25, 0.45);
    }
}
