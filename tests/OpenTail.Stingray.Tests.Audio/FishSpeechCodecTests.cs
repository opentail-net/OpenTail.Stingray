
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for <see cref="FishSpeechCodec"/> -- compares against
/// `scratch-llamacpp-ref/fish_speech_codec_golden.py`, which loads the real, already-local
/// `models/s2-pro-q4_k_m.gguf` weights directly via the `gguf` Python package and computes the
/// real decode math in numpy, transcribed from the real `fishaudio/fish-speech` GitHub repo.
/// </summary>
public sealed class FishSpeechCodecTests : HeavyTestBase
{
    private static string? FindModelPath(string fileName)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Decode_RealWeights_MatchesGoldenPcmOutput()
    {
        string? modelPath = FindModelPath("s2-pro-q4_k_m.gguf");
        Assert.SkipUnless(modelPath != null, "models/s2-pro-q4_k_m.gguf not found");

        string? codesPath = FindRepoFile("scratch-llamacpp-ref/fishspeech_codec_golden_codes.txt");
        string? pcmPath = FindRepoFile("scratch-llamacpp-ref/fishspeech_codec_golden_pcm.txt");
        Assert.SkipUnless(codesPath != null && pcmPath != null,
            "golden Fish Speech codec files not found (re-run scratch-llamacpp-ref/fish_speech_codec_golden.py)");

        var codeLines = File.ReadAllText(codesPath!).Trim().Split('\n');
        var semanticCodes = Array.ConvertAll(codeLines[0].Split(','), int.Parse);
        var residualCodes = new int[9][];
        for (int i = 0; i < 9; i++) residualCodes[i] = Array.ConvertAll(codeLines[i + 1].Split(','), int.Parse);

        var pcmLines = File.ReadAllText(pcmPath!).Split('\n');
        int goldenLen = int.Parse(pcmLines[0].Trim());
        var goldenParts = pcmLines[1].Trim().Split(',');
        Assert.Equal(goldenLen, goldenParts.Length);
        var golden = new float[goldenLen];
        for (int i = 0; i < goldenLen; i++) golden[i] = float.Parse(goldenParts[i]);

        using var weights = new FishSpeechCodecWeights(modelPath!);
        var pcm = FishSpeechCodec.Decode(weights, semanticCodes, residualCodes);

        Assert.Equal(goldenLen, pcm.Length);

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < goldenLen; i++)
        {
            dot += pcm[i] * golden[i];
            normA += pcm[i] * pcm[i];
            normB += golden[i] * golden[i];
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.99, $"cosine similarity {cosine} too low vs golden Fish Speech codec PCM output");
    }

    /// <summary>
    /// Isolated wall-clock benchmark for <see cref="FishSpeechCodec.Decode"/> alone (no slow-AR/
    /// fast-AR generation), 15 frames to match the real `s2.exe` CLI comparison run
    /// (`--codec-cpu -max-tokens 15`, measured 1223.7ms decode on this machine). Real codes aren't
    /// needed for a pure timing measurement -- random codes exercise the identical compute shape.
    /// </summary>
    [Fact]
    public void Decode_RealWeights_PerfBenchmark()
    {
        string? modelPath = FindModelPath("s2-pro-q4_k_m.gguf");
        Assert.SkipUnless(modelPath != null, "models/s2-pro-q4_k_m.gguf not found");

        using var weights = new FishSpeechCodecWeights(modelPath!);

        const int t = 15;
        var rnd = new Random(42);
        var semanticCodes = new int[t];
        for (int i = 0; i < t; i++) semanticCodes[i] = rnd.Next(0, 4096);
        var residualCodes = new int[9][];
        for (int cb = 0; cb < 9; cb++)
        {
            residualCodes[cb] = new int[t];
            for (int i = 0; i < t; i++) residualCodes[cb][i] = rnd.Next(0, 1024);
        }

        // Warmup (JIT).
        FishSpeechCodec.Decode(weights, semanticCodes, residualCodes);

        const int samples = 5;
        var msSamples = new double[samples];
        for (int s = 0; s < samples; s++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            FishSpeechCodec.Decode(weights, semanticCodes, residualCodes);
            sw.Stop();
            msSamples[s] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(msSamples);
        double median = msSamples[samples / 2];
        double mean = 0; foreach (var v in msSamples) mean += v; mean /= samples;
        string resultLine = $"samples_ms=[{string.Join(", ", Array.ConvertAll(msSamples, v => v.ToString("F1")))}] mean_ms={mean:F2} median_ms={median:F2}";
        Console.WriteLine(resultLine);
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "fishspeech_codec_bench_result.txt"), resultLine + Environment.NewLine);
    }
}
