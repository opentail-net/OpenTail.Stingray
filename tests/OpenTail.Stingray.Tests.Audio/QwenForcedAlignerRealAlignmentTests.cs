
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real end-to-end forced alignment test: real audio, real reference transcript, real
/// Qwen3-ForcedAligner-0.6B checkpoint, real forward pass (see QwenAsrForcedAligner.AlignReal's
/// doc comment). Reference audio/transcript pair taken directly from the real
/// examples/audio.cpp/tests/qwen3_forced_aligner/qwen3_forced_aligner_warm_bench_cases.json,
/// confirmed against the real C++ reference implementation.</summary>
public sealed class QwenForcedAlignerRealAlignmentTests : HeavyTestBase
{
    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void AlignReal_AWav_ProducesMonotonicWordTimestamps()
    {
        string? checkpointDir = FindRepoFile("models/qwen3-forcedaligner");
        Assert.SkipUnless(checkpointDir != null, "models/qwen3-forcedaligner not found");
        string? audioPath = FindRepoFile("examples/audio.cpp/assets/resources/a.wav");
        Assert.SkipUnless(audioPath != null, "reference a.wav not found");

        using var aligner = OpenTail.Stingray.Audio.QwenASR.QwenAsrForcedAligner.LoadReal(checkpointDir!);

        var (samples, sr, _) = WavReader.ReadWav(audioPath!);
        if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);

        const string referenceText = "This little work was finished in the year eighteen o three, and intended for immediate publication.";
        var segments = aligner.AlignReal(samples, referenceText, TimeSpan.Zero);

        Assert.NotEmpty(segments);
        Assert.Equal(referenceText.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length, segments.Count);

        var sb = new System.Text.StringBuilder("[ForcedAlignReal] ");
        foreach (var seg in segments)
            sb.Append($"{seg.Text}=[{seg.Start.TotalSeconds:F2},{seg.End.TotalSeconds:F2}] ");
        Console.Error.WriteLine(sb.ToString());

        for (int i = 0; i < segments.Count; i++)
        {
            Assert.True(segments[i].End >= segments[i].Start, $"Segment {i} ('{segments[i].Text}') has End < Start");
            if (i > 0)
                Assert.True(segments[i].Start >= segments[i - 1].Start, $"Segment {i} ('{segments[i].Text}') Start regressed vs previous word");
        }

        // Real expected words for this case (from the C++ reference's own warm-bench fixture):
        // "year", "eighteen", "o", "three" should fall within the audio's real duration.
        double audioDurationSec = samples.Length / 16000.0;
        foreach (var seg in segments)
        {
            Assert.True(seg.Start.TotalSeconds <= audioDurationSec + 0.5, $"'{seg.Text}' start {seg.Start.TotalSeconds:F2}s exceeds audio duration {audioDurationSec:F2}s");
        }
    }
}
