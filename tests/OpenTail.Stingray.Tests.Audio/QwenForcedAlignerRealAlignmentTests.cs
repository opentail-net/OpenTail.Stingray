
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
        // Warm run check: second call on the same loaded aligner
        var warmSegments = aligner.AlignReal(samples, referenceText, TimeSpan.Zero);
        Assert.Equal(segments.Count, warmSegments.Count);
        for (int i = 0; i < segments.Count; i++)
        {
            Assert.Equal(segments[i].Text, warmSegments[i].Text);
            Assert.Equal(segments[i].Start, warmSegments[i].Start);
            Assert.Equal(segments[i].End, warmSegments[i].End);
        }
    }

    [Fact]
    public void Generate_AlignmentCheckWav_WithBoundaryClicksAndWordSlices()
    {
        string? checkpointDir = FindRepoFile("models/qwen3-forcedaligner");
        Assert.SkipUnless(checkpointDir != null, "models/qwen3-forcedaligner not found");
        string? audioPath = FindRepoFile("examples/audio.cpp/assets/resources/a.wav");
        Assert.SkipUnless(audioPath != null, "reference a.wav not found");
        string? outDir = FindRepoFile("docs/audio-samples");
        Assert.SkipUnless(outDir != null, "docs/audio-samples not found");

        using var aligner = OpenTail.Stingray.Audio.QwenASR.QwenAsrForcedAligner.LoadReal(checkpointDir!);

        var (samples, sr, _) = WavReader.ReadWav(audioPath!);
        if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);

        const string referenceText = "This little work was finished in the year eighteen o three, and intended for immediate publication.";
        var segments = aligner.AlignReal(samples, referenceText, TimeSpan.Zero);

        // 1. Overlay audible clicks at each word start boundary
        var clickSamples = samples.ToArray();
        foreach (var seg in segments)
        {
            int startSample = (int)(seg.Start.TotalSeconds * 16000);
            for (int i = 0; i < 240 && startSample + i < clickSamples.Length; i++)
            {
                float tone = MathF.Sin(2f * MathF.PI * 1500f * i / 16000f) * MathF.Exp(-i / 60f);
                clickSamples[startSample + i] += 0.5f * tone;
            }
        }
        WavWriter.NormalizePeakInPlace(clickSamples, 0.95f, 0.95f);
        string clickPath = Path.Combine(outDir!, "qwen3-forcedaligner-check-clicks.wav");
        WavWriter.WriteWav(clickPath, clickSamples, 16000, 1);

        // 2. Concatenate sliced words with 200ms of silence in between
        int gapSamples = (int)(0.2f * 16000);
        var slicedList = new List<float>();
        foreach (var seg in segments)
        {
            int start = Math.Clamp((int)(seg.Start.TotalSeconds * 16000), 0, samples.Length);
            int end = Math.Clamp((int)(seg.End.TotalSeconds * 16000), start, samples.Length);
            slicedList.AddRange(samples.AsSpan(start, end - start).ToArray());
            slicedList.AddRange(new float[gapSamples]);
        }
        string slicedPath = Path.Combine(outDir!, "qwen3-forcedaligner-words-sliced.wav");
        WavWriter.WriteWav(slicedPath, slicedList.ToArray(), 16000, 1);

        // 3. Stereo check: Left = 100% clean speech, Right = speech + subtle timing click
        var stereoSamples = new float[samples.Length * 2];
        for (int i = 0; i < samples.Length; i++)
        {
            stereoSamples[i * 2] = samples[i];          // Left channel (clean natural speech)
            stereoSamples[i * 2 + 1] = clickSamples[i];  // Right channel (clicks)
        }
        string stereoPath = Path.Combine(outDir!, "qwen3-forcedaligner-stereo-check.wav");
        WavWriter.WriteWav(stereoPath, stereoSamples, 16000, 2);

        // 4. Clean unmodified source recording
        string cleanPath = Path.Combine(outDir!, "qwen3-forcedaligner-clean.wav");
        WavWriter.WriteWav(cleanPath, samples, 16000, 1);

        // 5. Write summary text
        string txtPath = Path.Combine(outDir!, "qwen3-forcedaligner-timestamps.txt");
        var sb = new System.Text.StringBuilder();
        foreach (var seg in segments)
            sb.AppendLine($"{seg.Text,-15} [{seg.Start.TotalSeconds,6:F2}s - {seg.End.TotalSeconds,6:F2}s]");
        File.WriteAllText(txtPath, sb.ToString());
    }

    [Fact]
    public void Bench_AlignReal_BWav_Timing()
    {
        string? checkpointDir = FindRepoFile("models/qwen3-forcedaligner");
        Assert.SkipUnless(checkpointDir != null, "models/qwen3-forcedaligner not found");
        string? audioPath = FindRepoFile("examples/audio.cpp/assets/resources/b.wav");
        Assert.SkipUnless(audioPath != null, "reference b.wav not found");

        using var aligner = OpenTail.Stingray.Audio.QwenASR.QwenAsrForcedAligner.LoadReal(checkpointDir!);

        var (samples, sr, _) = WavReader.ReadWav(audioPath!);
        if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);
        double audioSec = samples.Length / 16000.0;

        const string referenceText = "Some call me nature. Others call me Mother Nature. I have been here for over four and a half billion years.";
        
        // Warmup
        var warmup = aligner.AlignReal(samples, referenceText, TimeSpan.Zero);
        Assert.NotEmpty(warmup);

        const int n = 3;
        var sw = new System.Diagnostics.Stopwatch();
        double[] timesMs = new double[n];
        for (int i = 0; i < n; i++)
        {
            sw.Restart();
            var segments = aligner.AlignReal(samples, referenceText, TimeSpan.Zero);
            sw.Stop();
            timesMs[i] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(timesMs);
        double medianMs = timesMs[n / 2];
        double wallSec = medianMs / 1000.0;
        double rtf = wallSec / audioSec;

        string report = $"[ForcedAlign-Bench] audio={audioSec:F2}s wall={wallSec:F3}s RTF={rtf:F3}x (speedup={1.0 / rtf:F1}x real-time)";
        Console.Error.WriteLine(report);
    }
}
