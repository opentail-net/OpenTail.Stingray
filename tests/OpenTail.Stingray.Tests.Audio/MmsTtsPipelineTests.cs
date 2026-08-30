using OpenTail.Stingray.Audio.MmsTts;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real numeric golden verification for the full MMS-TTS port, against
/// `scratch-llamacpp-ref/mms_tts_golden.py`'s real HuggingFace `transformers.VitsModel` output
/// (deterministic input "hello world", real captured noise draws so the stochastic stages are
/// tested with the SAME noise as the reference -- isolating "is the math right" from "does the
/// RNG match").</summary>
public sealed class MmsTtsPipelineTests : HeavyTestBase
{
    private const string ModelDir = "models/mms-tts-eng";

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

    private static string? FindRepoDir(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static float[] ReadCsv(string path) =>
        Array.ConvertAll(File.ReadAllText(path).Trim().Split(','), float.Parse);

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb));
    }

    private static (MmsTtsWeights Weights, int[] Ids, float[] EncoderHidden, float[] Mu, float[] Logs) LoadCommon()
    {
        string modelDir = FindRepoDir(ModelDir)!;
        var config = MmsTtsConfig.Load(Path.Combine(modelDir, "config.json"));
        var weights = new MmsTtsWeights(Path.Combine(modelDir, "model.safetensors"), config);
        var ids = Array.ConvertAll(File.ReadAllText(FindRepoFile("scratch-llamacpp-ref/mms_tts_golden_input_ids.txt")!).Trim().Split(','), int.Parse);
        var (encoderHidden, mu, logs) = MmsTtsTextEncoder.Forward(weights, ids);
        return (weights, ids, encoderHidden, mu, logs);
    }

    [Fact]
    public void DurationPredictor_RealWeights_MatchesGoldenLogw()
    {
        Assert.SkipUnless(FindRepoDir(ModelDir) != null, "models/mms-tts-eng not found");
        string? logwPath = FindRepoFile("scratch-llamacpp-ref/mms_tts_golden_logw.txt");
        string? noisePath = FindRepoFile("scratch-llamacpp-ref/mms_tts_golden_randn_12_randn.txt");
        Assert.SkipUnless(logwPath != null && noisePath != null, "golden MMS-TTS SDP files not found (re-run scratch-llamacpp-ref/mms_tts_golden.py)");

        var (weights, ids, encoderHidden, _, _) = LoadCommon();
        float[] noise = ReadCsv(noisePath!); // [1,2,T] raw N(0,1), channel-first -- matches Predict's expected [2*T] layout
        Assert.Equal(2 * ids.Length, noise.Length);

        float[] logw = MmsTtsDurationPredictor.Predict(weights, encoderHidden, ids.Length, noise, noiseScaleW: 0.8f);
        float[] goldenLogw = ReadCsv(logwPath!);

        Assert.Equal(goldenLogw.Length, logw.Length);
        foreach (float v in logw)
        {
            Assert.False(float.IsNaN(v), "logw must not contain NaN");
            Assert.False(float.IsInfinity(v), "logw must not contain Infinity");
        }
        double cosine = Cosine(logw, goldenLogw);
        Assert.True(cosine > 0.99, $"logw cosine {cosine} too low vs golden. (logw={string.Join(",", logw)}, golden={string.Join(",", goldenLogw)})");
    }

    [Fact]
    public void Flow_RealWeights_MatchesGoldenOutput()
    {
        Assert.SkipUnless(FindRepoDir(ModelDir) != null, "models/mms-tts-eng not found");
        string? zpPath = FindRepoFile("scratch-llamacpp-ref/mms_tts_golden_flow_input_zp.txt");
        string? outPath = FindRepoFile("scratch-llamacpp-ref/mms_tts_golden_flow_output.txt");
        Assert.SkipUnless(zpPath != null && outPath != null, "golden MMS-TTS flow files not found");

        string modelDir = FindRepoDir(ModelDir)!;
        var config = MmsTtsConfig.Load(Path.Combine(modelDir, "config.json"));
        var weights = new MmsTtsWeights(Path.Combine(modelDir, "model.safetensors"), config);

        float[] zp = ReadCsv(zpPath!);
        int tFrames = zp.Length / weights.HiddenDim;

        float[] flowOut = MmsTtsFlow.Reverse(weights, zp, tFrames);
        float[] golden = ReadCsv(outPath!);

        Assert.Equal(golden.Length, flowOut.Length);
        double cosine = Cosine(flowOut, golden);
        Assert.True(cosine > 0.99, $"flow output cosine {cosine} too low vs golden");
    }

    [Fact]
    public void Waveform_RealWeights_MatchesGoldenOutput()
    {
        Assert.SkipUnless(FindRepoDir(ModelDir) != null, "models/mms-tts-eng not found");
        string? outPath = FindRepoFile("scratch-llamacpp-ref/mms_tts_golden_waveform.txt");
        string? zpPath = FindRepoFile("scratch-llamacpp-ref/mms_tts_golden_flow_input_zp.txt");
        Assert.SkipUnless(outPath != null && zpPath != null, "golden MMS-TTS waveform files not found");

        string modelDir = FindRepoDir(ModelDir)!;
        var config = MmsTtsConfig.Load(Path.Combine(modelDir, "config.json"));
        var weights = new MmsTtsWeights(Path.Combine(modelDir, "model.safetensors"), config);

        float[] zp = ReadCsv(zpPath!);
        int tFrames = zp.Length / weights.HiddenDim;
        float[] flowOut = MmsTtsFlow.Reverse(weights, zp, tFrames);
        float[] waveform = MmsTtsHifiGanDecoder.Forward(weights, flowOut, tFrames);

        float[] golden = ReadCsv(outPath!);
        Assert.Equal(golden.Length, waveform.Length);
        double cosine = Cosine(waveform, golden);
        Assert.True(cosine > 0.99, $"waveform cosine {cosine} too low vs golden");
    }

    /// <summary>End-to-end smoke test using the real pipeline's own RNG (not golden noise) -- checks
    /// real, non-degenerate audio comes out and saves a listenable sample.</summary>
    [Fact]
    public void Generate_RealWeights_ProducesNonDegenerateAudio()
    {
        string? modelDir = FindRepoDir(ModelDir);
        Assert.SkipUnless(modelDir != null, "models/mms-tts-eng not found");

        using var pipeline = MmsTtsPipeline.Load(modelDir!);
        var wav = pipeline.Generate("Hello, I will make some lunch, darling!", seed: 42);

        Assert.True(wav.Length > 0, "MMS-TTS produced empty audio");
        float peak = 0f;
        foreach (var s in wav)
        {
            Assert.False(float.IsNaN(s), "waveform must not contain NaN");
            peak = Math.Max(peak, Math.Abs(s));
        }
        Assert.True(peak > 0.01f, $"waveform looks silent/degenerate, peak={peak}");

        string? outDir = FindRepoDir("docs/audio-samples");
        if (outDir != null)
        {
            string path = Path.Combine(outDir, "mms-tts-first-real-clip.wav");
            new AudioGenerationResult(wav, pipeline.DefaultSampleRate).SaveWav(path);
            Console.WriteLine($"Saved {path}: {wav.Length} samples ({wav.Length / (float)pipeline.DefaultSampleRate:F2}s)");
        }
    }
}
