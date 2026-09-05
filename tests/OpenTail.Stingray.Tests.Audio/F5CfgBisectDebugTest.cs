
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: isolates whether F5-TTS's new batched CFG path
/// (<see cref="F5DiTModel.ForwardVelocityBatch2"/>, added in commit ce6a914) or the pre-existing
/// single-stream path (<see cref="F5DiTModel.ForwardVelocity"/>) is responsible for the
/// static/garbage output observed after that commit. <see cref="F5FlowMatchingOde.Solve"/> already
/// branches internally: cfgStrength below 1e-5 forces the OLD unbatched path for every ODE step,
/// while cfgStrength >= 1e-5 always uses the NEW batched path. Comparing RMS/peak stats between the
/// two, with everything else identical (same seed, same weights, same tokens), pins the bug to one
/// code path without touching any production file.</summary>
public sealed class F5CfgBisectDebugTest : HeavyTestBase
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p) || Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static (float rms, float peak, bool anyNaN) Stats(float[] x)
    {
        double sumsq = 0;
        float peak = 0;
        bool anyNaN = false;
        foreach (var v in x)
        {
            if (float.IsNaN(v) || float.IsInfinity(v)) anyNaN = true;
            sumsq += (double)v * v;
            float a = MathF.Abs(v);
            if (a > peak) peak = a;
        }
        return ((float)Math.Sqrt(sumsq / x.Length), peak, anyNaN);
    }

    [Fact]
    public void Bisect_CfgStrength_0_vs_1()
    {
        string? modelPath = FindRepoFile("models/f5tts_base.safetensors");
        Assert.SkipUnless(modelPath != null, "F5TTS safetensors model not found");
        string? vocabPath = FindRepoFile("models/f5tts_vocab.txt");
        Assert.SkipUnless(vocabPath != null, "F5TTS vocab not found");

        var weights = new OpenTail.Stingray.Audio.F5TTS.F5TtsWeights(modelPath!);
        var tokenizer = new OpenTail.Stingray.Audio.F5TTS.F5Tokenizer(vocabPath!);

        const string prompt = "Hello, I will make some lunch, darling!";
        int[] tokens = tokenizer.Encode(prompt);
        int totalFrames = 256; // fixed, deterministic frame count for this diagnostic
        float[] condMel = new float[totalFrames * OpenTail.Stingray.Audio.F5TTS.F5MelExtractor.NumMels]; // no reference audio, zero-cond

        var melCfg1 = OpenTail.Stingray.Audio.F5TTS.F5FlowMatchingOde.Solve(
            weights, condMel, tokens, totalFrames, steps: 16, cfgStrength: 1.0f, swaySamplingCoef: -1.0f, seed: 42);
        var melCfg0 = OpenTail.Stingray.Audio.F5TTS.F5FlowMatchingOde.Solve(
            weights, condMel, tokens, totalFrames, steps: 16, cfgStrength: 0.0f, swaySamplingCoef: -1.0f, seed: 42);

        var (rms1, peak1, nan1) = Stats(melCfg1);
        var (rms0, peak0, nan0) = Stats(melCfg0);

        string msg = $"[F5-CFG-Bisect] cfgStrength=1.0 (NEW batched path): rms={rms1:F4} peak={peak1:F4} anyNaN={nan1}\n" +
                     $"[F5-CFG-Bisect] cfgStrength=0.0 (OLD single-stream path): rms={rms0:F4} peak={peak0:F4} anyNaN={nan0}";
        Console.Error.WriteLine(msg);
        string? docsDir = FindRepoFile("docs");
        if (docsDir != null)
            File.AppendAllText(Path.Combine(docsDir, "tts-benchmark-log.txt"), msg + "\n\n");
    }
}
