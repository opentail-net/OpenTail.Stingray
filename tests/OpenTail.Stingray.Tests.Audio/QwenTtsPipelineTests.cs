
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real end-to-end smoke test for <see cref="QwenTtsPipeline"/> -- text -&gt; real Talker semantic
/// codes -&gt; real Code Predictor acoustic depth-expansion (using the real
/// <see cref="OpenTail.Stingray.Engine.ForwardPass.LastHidden"/> bridge this session added) -&gt;
/// real, independently golden-verified codec decode chain -&gt; waveform. First time all of
/// QwenTTS's real components (Talker, Code Predictor, and every codec stage) run chained
/// together in one call. Matches this project's established first-pass bar for a from-scratch
/// end-to-end pipeline (finite + non-silent RMS, not a fabricated numeric oracle).
/// </summary>
public sealed class QwenTtsPipelineTests : HeavyTestBase
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

    [Fact]
    public void Generate_RealWeights_ProducesFiniteNonSilentWaveform()
    {
        string? talkerPath = FindRepoFile("models/qwen-talker-0.6b-base-Q8_0.gguf");
        string? codecPath = FindRepoFile("models/qwen-tokenizer-12hz-Q8_0.gguf");
        Assert.SkipUnless(talkerPath != null, "models/qwen-talker-0.6b-base-Q8_0.gguf not found");
        Assert.SkipUnless(codecPath != null, "models/qwen-tokenizer-12hz-Q8_0.gguf not found");

        using var pipeline = QwenTtsPipeline.Load(talkerPath!, codecPath!);
        Assert.Equal(24000, pipeline.SampleRate);

        var wav = pipeline.Generate("Hello there.", maxFrames: 6);

        Assert.True(wav.Length > 0, "pipeline produced zero samples");

        double sumSq = 0;
        foreach (var s in wav)
        {
            Assert.False(float.IsNaN(s) || float.IsInfinity(s), "waveform contains NaN/Inf");
            sumSq += (double)s * s;
        }
        double rms = System.Math.Sqrt(sumSq / wav.Length);
        Assert.True(rms > 1e-6, $"waveform looks silent/degenerate: rms={rms}");
    }

    // TEMP bisection harness for the golden-verification investigation (docs/audio-review-
    // progress.md's QwenTTS entries) -- set STINGRAY_QWENTTS_GOLDEN_DUMP and STINGRAY_QWENTTS_BISECT_LAYERS
    // to dump the N-layer talker trunk's hidden state for comparison against the real PyTorch
    // reference. Not a real correctness assertion; TODO revert/remove once the bug is found.
    [Fact]
    public void Bisect_TalkerLayers()
    {
        string? talkerPath = FindRepoFile("models/qwen-talker-0.6b-base-Q8_0.gguf");
        string? codecPath = FindRepoFile("models/qwen-tokenizer-12hz-Q8_0.gguf");
        Assert.SkipUnless(talkerPath != null, "models/qwen-talker-0.6b-base-Q8_0.gguf not found");
        Assert.SkipUnless(codecPath != null, "models/qwen-tokenizer-12hz-Q8_0.gguf not found");
        string? nLayersEnv = System.Environment.GetEnvironmentVariable("STINGRAY_QWENTTS_BISECT_LAYERS");
        Assert.SkipUnless(nLayersEnv != null, "STINGRAY_QWENTTS_BISECT_LAYERS not set");
        int nLayers = int.Parse(nLayersEnv!);

        using var pipeline = QwenTtsPipeline.Load(talkerPath!, codecPath!);
        _ = pipeline.Generate("Hello there", talkerNumLayers: nLayers, maxFrames: 1);
    }

    // TEMP: single-token (T=1) bisection -- eliminates all multi-position attention/causal-mask/
    // GQA-across-keys complexity, isolating Q/K/V proj + QK-norm + RoPE-at-position-0 + O-proj +
    // FFN. Set STINGRAY_QWENTTS_GOLDEN_DUMP and STINGRAY_QWENTTS_BISECT_LAYERS. TODO revert/remove.
    [Fact]
    public void Bisect_SingleTokenLayer()
    {
        string? talkerPath = FindRepoFile("models/qwen-talker-0.6b-base-Q8_0.gguf");
        Assert.SkipUnless(talkerPath != null, "models/qwen-talker-0.6b-base-Q8_0.gguf not found");
        string? nLayersEnv = Environment.GetEnvironmentVariable("STINGRAY_QWENTTS_BISECT_LAYERS");
        Assert.SkipUnless(nLayersEnv != null, "STINGRAY_QWENTTS_BISECT_LAYERS not set");
        int nLayers = int.Parse(nLayersEnv!);
        string? dumpDir = Environment.GetEnvironmentVariable("STINGRAY_QWENTTS_GOLDEN_DUMP");
        Assert.SkipUnless(dumpDir != null, "STINGRAY_QWENTTS_GOLDEN_DUMP not set");

        using var model = GgufModel.Open(talkerPath!);
        using var source = new QwenTtsTalkerTensorSource(model, nLayers);

        const int hiddenDim = QwenTtsTalkerPromptBuilder.TalkerHiddenDim;
        var rng = new Random(7);
        var embed = new float[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) embed[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        source.SetPromptEmbedding(embed, 1);

        var hp = ModelHyperparams.FromGgufMetadata(source.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(source, backend, hp);
        _ = fwd.Prefill([0]);
        var hidden = fwd.LastHidden.ToArray();

        Directory.CreateDirectory(dumpDir!);
        File.WriteAllText(Path.Combine(dumpDir!, "t1_embed.csv"), string.Join(",", embed));
        File.WriteAllText(Path.Combine(dumpDir!, "t1_hidden.csv"), string.Join(",", hidden));
    }

    // TEMP: T=2 bisection (simplest multi-position case) -- exercises causal masking between
    // position 0/1 and RoPE at position 1, the two things T=1 could not test. TODO revert/remove.
    [Fact]
    public void Bisect_TwoTokenLayer()
    {
        string? talkerPath = FindRepoFile("models/qwen-talker-0.6b-base-Q8_0.gguf");
        Assert.SkipUnless(talkerPath != null, "models/qwen-talker-0.6b-base-Q8_0.gguf not found");
        string? nLayersEnv = Environment.GetEnvironmentVariable("STINGRAY_QWENTTS_BISECT_LAYERS");
        Assert.SkipUnless(nLayersEnv != null, "STINGRAY_QWENTTS_BISECT_LAYERS not set");
        int nLayers = int.Parse(nLayersEnv!);
        string? dumpDir = Environment.GetEnvironmentVariable("STINGRAY_QWENTTS_GOLDEN_DUMP");
        Assert.SkipUnless(dumpDir != null, "STINGRAY_QWENTTS_GOLDEN_DUMP not set");

        using var model = GgufModel.Open(talkerPath!);
        using var source = new QwenTtsTalkerTensorSource(model, nLayers);

        const int hiddenDim = QwenTtsTalkerPromptBuilder.TalkerHiddenDim;
        var rng = new Random(7);
        var embed = new float[2 * hiddenDim];
        for (int i = 0; i < embed.Length; i++) embed[i] = (float)(rng.NextDouble() * 0.1 - 0.05);

        // Matches QwenTtsPipeline.GenerateFrames' real pattern: Prefill everything except the
        // last row (via a fresh single-row SetPromptEmbedding + Prefill([0])), then a separate
        // Forward call for the last row -- raw multi-row Prefill()+LastHidden is a KNOWN,
        // already-documented no-op (LastHidden reads back all-zero), not what production code
        // actually does or what this bisection should be testing. Must SetPromptEmbedding at
        // least once BEFORE constructing ForwardPass -- its constructor resolves
        // "token_embd.weight" immediately and throws "Missing tensor" if it isn't set yet.
        var row0 = embed[..hiddenDim];
        source.SetPromptEmbedding(row0, 1);

        var hp = ModelHyperparams.FromGgufMetadata(source.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(source, backend, hp);
        _ = fwd.Prefill([0]);
        var row1 = embed[hiddenDim..];
        source.SetPromptEmbedding(row1, 1);
        var logits = fwd.Forward(0, 1);
        _ = logits.Length; // touch it so the call isn't seen as dead
        var hidden = fwd.LastHidden.ToArray();

        Directory.CreateDirectory(dumpDir!);
        File.WriteAllText(Path.Combine(dumpDir!, "t2_embed.csv"), string.Join(",", embed));
        File.WriteAllText(Path.Combine(dumpDir!, "t2_hidden.csv"), string.Join(",", hidden));
    }
}
