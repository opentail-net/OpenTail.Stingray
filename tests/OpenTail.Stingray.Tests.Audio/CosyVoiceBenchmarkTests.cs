
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Perf hotloops for CosyVoice2/CosyVoice3's individual real-weight stages, mirroring
/// <see cref="ChatterboxVocoderBenchmarkTests"/>'s shape: isolate each stage's own cost with
/// synthetic-but-realistically-scaled input instead of chaining the full pipeline, so iterating
/// on one stage's performance doesn't require the others to be wired up or correct. Correctness
/// of the synthetic input's *content* doesn't matter here -- see the matching *Tests.cs files
/// for real correctness checks. Timings are logged to stderr and appended to a temp diag log for
/// before/after comparison across optimization passes.
/// </summary>
public sealed class CosyVoiceBenchmarkTests : HeavyTestBase
{
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

    private static void LogTiming(string label, System.Diagnostics.Stopwatch sw)
    {
        Console.Error.WriteLine($"[CosyVoiceBenchmark] {label}: {sw.ElapsedMilliseconds}ms");
        try
        {
            File.AppendAllText(
                Path.Combine(Path.GetTempPath(), "stingray-cosyvoice-diag.log"),
                $"[CosyVoiceBenchmark] {label}: {sw.ElapsedMilliseconds}ms{Environment.NewLine}");
        }
        catch { /* best-effort */ }
    }

    [Fact]
    public void HiftVocoder_Benchmark_SyntheticMelAtRealisticScale()
    {
        string? path = FindRepoFile("models/cosyvoice2_hift.safetensors");
        if (path is null) return;

        using var w = new CosyVoiceHiftWeights(path);

        // ~250 mel frames matches a real few-second sentence's generated tail, same scale used
        // by ChatterboxVocoderBenchmarkTests for the analogous S3Gen vocoder.
        const int melFrames = 250;
        var mel = new float[80 * melFrames];
        var rng = new Random(42);
        for (int i = 0; i < mel.Length; i++) mel[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var wav = CosyVoiceHiftVocoder.Generate(w, mel, melFrames, rng);
        sw.Stop();

        LogTiming($"HiftVocoder.Generate {melFrames} mel frames -> {wav.Length} samples", sw);
        Assert.NotEmpty(wav);
    }

    [Fact]
    public void FlowEncoder_Benchmark_RealisticTokenCounts()
    {
        string? path = FindRepoFile("models/cosyvoice2_flow.safetensors");
        if (path is null) return;

        using var w = new CosyVoiceFlowWeights(path);

        // A real few-second sentence: a handful of prompt tokens plus ~100-150 generated speech
        // tokens (roughly 12.5 tokens/sec of audio at CosyVoice2's speech-token rate).
        var rng = new Random(7);
        var promptTokens = new int[8];
        for (int i = 0; i < promptTokens.Length; i++) promptTokens[i] = rng.Next(0, w.SpeechVocabSize);
        var speechTokens = new int[120];
        for (int i = 0; i < speechTokens.Length; i++) speechTokens[i] = rng.Next(0, w.SpeechVocabSize);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var (mu, totalFrames) = CosyVoiceFlowEncoder.Forward(w, promptTokens, speechTokens);
        sw.Stop();

        LogTiming($"FlowEncoder.Forward {promptTokens.Length + speechTokens.Length} tokens -> {totalFrames} frames", sw);
        Assert.NotEmpty(mu);
    }

    private static CosyVoiceLlmTensorSource OpenLlm(string path) => new(
        path,
        numLayers: 24, hiddenDim: 896, numHeads: 14, numKvHeads: 2, headDim: 64,
        ffDim: 4864, vocabSize: 151936, ropeTheta: 1_000_000f, rmsNormEps: 1e-6f);

    [Fact]
    public void LlmTensorSource_Benchmark_PrefillRealisticPromptLength()
    {
        string? path = FindRepoFile("models/cosyvoice2_llm.safetensors");
        if (path is null) return;

        using var source = OpenLlm(path);
        var hp = ModelHyperparams.FromGgufMetadata(source.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(source, backend, hp);

        // A realistic short-sentence text-prompt token count.
        var rng = new Random(7);
        var prompt = new int[24];
        for (int i = 0; i < prompt.Length; i++) prompt[i] = rng.Next(0, 100000);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var logits = fwd.Prefill(prompt).ToArray();
        sw.Stop();

        LogTiming($"CosyVoiceLlm.Prefill {prompt.Length} tokens -> {logits.Length} logits", sw);
        Assert.NotEmpty(logits);
    }

    [Fact]
    public void CosyVoice3DiTModel_Benchmark_RunBackbone_RealisticFrameCount()
    {
        string? path = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        if (path is null) return;

        using var model = GgufModel.Open(path);
        var w = new CosyVoice3DiTWeights(model);

        // ~250 mel frames matches a real few-second sentence, same scale as the HiFT benchmark.
        const int numFrames = 250;
        var h = new float[numFrames * CosyVoice3DiTWeights.HiddenDim];
        var rng = new Random(11);
        for (int i = 0; i < h.Length; i++) h[i] = (float)(rng.NextDouble() * 0.02 - 0.01);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var velocity = CosyVoice3DiTModel.RunBackbone(w, h, timestep: 0.5f, numFrames);
        sw.Stop();

        LogTiming($"CosyVoice3DiTModel.RunBackbone {numFrames} frames, {w.NumLayers} layers -> {velocity.Length} vals", sw);
        Assert.NotEmpty(velocity);
    }

    /// <summary>
    /// Real perf pass (CLAUDE.md rule 7) on this session's new CosyVoice2 CFM decoder: measures
    /// the full real 10-step Euler ODE solve at a realistic frame count (~128 frames, matching
    /// a real few-second sentence's generated speech-token count after the flow encoder's 2x
    /// upsample). A handful of runs, not one, per rule 7's own methodology.
    /// </summary>
    [Fact]
    public void CosyVoiceCfmDecoder_Benchmark_RealisticFrameCount()
    {
        string? path = FindRepoFile("models/cosyvoice2_flow.safetensors");
        if (path is null) return;

        using var flow = new CosyVoiceFlowWeights(path);
        var w = new CosyVoiceCfmDecoderWeights(flow);

        const int totalFrames = 128;
        var rng = new Random(13);
        var mu = new float[80 * totalFrames];
        for (int i = 0; i < mu.Length; i++) mu[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
        var spkEmbed = new float[80];
        for (int i = 0; i < spkEmbed.Length; i++) spkEmbed[i] = (float)(rng.NextDouble() * 0.2 - 0.1);

        const int runs = 3;
        var times = new long[runs];
        for (int r = 0; r < runs; r++)
        {
            var runRng = new Random(100 + r);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var mel = CosyVoiceCfmDecoder.Generate(w, mu, spkEmbed, totalFrames, runRng, nSteps: 10);
            sw.Stop();
            times[r] = sw.ElapsedMilliseconds;
            Assert.NotEmpty(mel);
        }

        double avg = 0;
        foreach (var t in times) avg += t;
        avg /= runs;
        Console.Error.WriteLine($"[CosyVoiceBenchmark] CosyVoiceCfmDecoder.Generate {totalFrames} frames, 10 Euler steps -> runs: [{string.Join(", ", times)}]ms, avg {avg:F0}ms");
        try
        {
            File.AppendAllText(
                Path.Combine(Path.GetTempPath(), "stingray-cosyvoice-diag.log"),
                $"[CosyVoiceBenchmark] CosyVoiceCfmDecoder.Generate {totalFrames} frames, 10 Euler steps -> runs: [{string.Join(", ", times)}]ms, avg {avg:F0}ms{Environment.NewLine}");
        }
        catch { /* best-effort */ }
    }
}
