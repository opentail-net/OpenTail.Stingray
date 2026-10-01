
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for <see cref="FishSpeechFastAr"/> -- compares against
/// `scratch-llamacpp-ref/fish_speech_fastar_golden.py`, which fetches the real 4-layer
/// `audio_decoder.*` weights (~800MB) directly from the real `fishaudio/s2-pro` safetensors via
/// byte-range HTTP requests (no full 9.1GB download) and computes the real math in numpy,
/// transcribed from `fish_speech/models/text2semantic/llama.py`. Deterministic input: hidden
/// state = all 0.1, prefix codebook values = [7, 42, 99].
/// </summary>
public sealed class FishSpeechFastArTests : HeavyTestBase
{
    private static string? FindModelPath(string fileName)
    {
        if (Path.IsPathRooted(fileName)) return File.Exists(fileName) ? fileName : null;
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var pModels = Path.Combine(dir, "models", "_models", fileName);
            if (File.Exists(pModels)) return pModels;
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
    public void Forward_Q4KM_vs_FullPrecision_CharacterizesQuantizationLoss()
    {
        string? modelPath = FindModelPath("s2-pro-q4_k_m.gguf");
        Assert.SkipUnless(modelPath != null, "models/s2-pro-q4_k_m.gguf not found");

        string? inputPath = FindRepoFile("scratch-llamacpp-ref/fishspeech_fastar_golden_input.txt");
        string? logitsPath = FindRepoFile("scratch-llamacpp-ref/fishspeech_fastar_golden_logits.txt");
        Assert.SkipUnless(inputPath != null && logitsPath != null,
            "golden fast-AR files not found (re-run scratch-llamacpp-ref/fish_speech_fastar_golden.py)");

        var inputLines = File.ReadAllText(inputPath!).Trim().Split('\n');
        var hiddenCsv = inputLines[0].Split(',');
        var hidden = new float[hiddenCsv.Length];
        for (int i = 0; i < hidden.Length; i++) hidden[i] = float.Parse(hiddenCsv[i]);

        var prefixCsv = inputLines[1].Split(',');
        var prefix = new int[prefixCsv.Length];
        for (int i = 0; i < prefix.Length; i++) prefix[i] = int.Parse(prefixCsv[i]);

        var logitsLines = File.ReadAllText(logitsPath!).Trim().Split('\n');
        int codebookSize = int.Parse(logitsLines[0]);
        var goldenParts = logitsLines[1].Split(',');
        Assert.Equal(codebookSize, goldenParts.Length);
        var golden = new float[codebookSize];
        for (int i = 0; i < codebookSize; i++) golden[i] = float.Parse(goldenParts[i]);

        using var weights = new FishSpeechWeights(modelPath!);
        Assert.Equal(hidden.Length, weights.EmbeddingDim);

        var logits = FishSpeechFastAr.Forward(weights, hidden, prefix);
        Assert.Equal(codebookSize, logits.Length);

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < codebookSize; i++)
        {
            dot += logits[i] * golden[i];
            normA += logits[i] * logits[i];
            normB += golden[i] * golden[i];
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        // CHARACTERIZATION, not a correctness gate (docs/1-correctness/08 Phase 2.2). Original BF16 weights vs Q4_K_M:
        // the pure float reference on the Q4_K_M-dequantized weights already gives cosine 0.489 (argmax flips, top-10 overlap
        // 8/10); this C# path measured 0.4406. Correctness of the Q4 path is asserted by
        // Forward_Q4KMWeights_MatchesDequantizedPathOracle. This only guards against the loss growing far beyond that.
        Console.WriteLine($"[FishFastAR Q4_K_M vs original bf16] cosine={cosine:R}");
        Assert.InRange(cosine, 0.30, 0.70);
    }

    /// <summary>Same golden comparison but against the Q8_0 (near-lossless) quant instead of Q4_K_M -- tests whether the Q4_K_M mismatch found earlier this fire is quantization-compounding (cosine should improve substantially) or a real code bug (cosine would stay low).</summary>
    [Fact]
    public void Forward_Q8_0Weights_MatchesGoldenOracle()
    {
        string? modelPath = FindModelPath("s2-pro-q8_0.gguf");
        Assert.SkipUnless(modelPath != null, "models/s2-pro-q8_0.gguf not found");

        string? inputPath = FindRepoFile("scratch-llamacpp-ref/fishspeech_fastar_golden_input.txt");
        string? logitsPath = FindRepoFile("scratch-llamacpp-ref/fishspeech_fastar_golden_logits.txt");
        Assert.SkipUnless(inputPath != null && logitsPath != null,
            "golden fast-AR files not found (re-run scratch-llamacpp-ref/fish_speech_fastar_golden.py)");

        var inputLines = File.ReadAllText(inputPath!).Trim().Split('\n');
        var hiddenCsv = inputLines[0].Split(',');
        var hidden = new float[hiddenCsv.Length];
        for (int i = 0; i < hidden.Length; i++) hidden[i] = float.Parse(hiddenCsv[i]);

        var prefixCsv = inputLines[1].Split(',');
        var prefix = new int[prefixCsv.Length];
        for (int i = 0; i < prefix.Length; i++) prefix[i] = int.Parse(prefixCsv[i]);

        var logitsLines = File.ReadAllText(logitsPath!).Trim().Split('\n');
        int codebookSize = int.Parse(logitsLines[0]);
        var goldenParts = logitsLines[1].Split(',');
        var golden = new float[codebookSize];
        for (int i = 0; i < codebookSize; i++) golden[i] = float.Parse(goldenParts[i]);

        using var weights = new FishSpeechWeights(modelPath!);
        var logits = FishSpeechFastAr.Forward(weights, hidden, prefix);

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < codebookSize; i++)
        {
            dot += logits[i] * golden[i];
            normA += logits[i] * logits[i];
            normB += golden[i] * golden[i];
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));

        File.WriteAllText(Path.Combine(Path.GetTempPath(), "fishspeech_fastar_q8_cosine.txt"), $"cosine={cosine}\n");
        Assert.True(cosine > 0.99, $"cosine similarity {cosine} too low vs golden fast-AR logits (Q8_0)");
    }

    /// <summary>
    /// Direct equivalence check for the perf-optimized <see cref="FishSpeechFastAr.LayerStep"/>
    /// (fused MatVecDual gate/up, TensorPrimitives.Dot attention) against the untouched, golden-
    /// verified <see cref="FishSpeechFastAr.Forward"/> path -- the golden oracle only exercises
    /// `Forward`/`Layer`, never `ForwardStep`/`LayerStep`, so this is the only correctness check
    /// for the code actually touched by this perf pass. Per the doc comment on `ForwardStep`,
    /// sequential ForwardStep calls over a prefix must be mathematically equivalent to one
    /// Forward call over the same prefix.
    /// </summary>
    [Fact]
    public void ForwardStep_MatchesForward_ForSamePrefix()
    {
        string? modelPath = FindModelPath("s2-pro-q8_0.gguf");
        Assert.SkipUnless(modelPath != null, "models/s2-pro-q8_0.gguf not found");

        string? inputPath = FindRepoFile("scratch-llamacpp-ref/fishspeech_fastar_golden_input.txt");
        Assert.SkipUnless(inputPath != null, "golden fast-AR input file not found");

        var inputLines = File.ReadAllText(inputPath!).Trim().Split('\n');
        var hiddenCsv = inputLines[0].Split(',');
        var hidden = new float[hiddenCsv.Length];
        for (int i = 0; i < hidden.Length; i++) hidden[i] = float.Parse(hiddenCsv[i]);

        var prefixCsv = inputLines[1].Split(',');
        var prefix = new int[prefixCsv.Length];
        for (int i = 0; i < prefix.Length; i++) prefix[i] = int.Parse(prefixCsv[i]);

        using var weights = new FishSpeechWeights(modelPath!);
        var expected = FishSpeechFastAr.Forward(weights, hidden, prefix);

        var cache = new FishSpeechFastArCache(weights);
        var stepLogits = FishSpeechFastAr.ForwardStep(weights, cache, hidden);
        foreach (var t in prefix)
            stepLogits = FishSpeechFastAr.ForwardStep(weights, cache, FishSpeechFastAr.EmbedFastToken(weights, t));

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            dot += stepLogits[i] * expected[i];
            normA += stepLogits[i] * stepLogits[i];
            normB += expected[i] * expected[i];
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.999, $"ForwardStep vs Forward cosine similarity {cosine} too low -- LayerStep perf changes may have broken math");
    }

    // ---- Dequantized-path oracles (docs/1-correctness/08, Phase 2) -------------------------------------------------
    // fish_speech_fastar_golden.py with FASTAR_GGUF=<gguf> runs the SAME float32 reference math on the GGUF's tensors
    // dequantized to float (tags _q4path / _q8path). Comparing the C# path (GGUF -> F32 -> Q8_0 requant -> inference)
    // to it separates "is our math right" from "how far is this quantization from the original weights".

    private sealed record FastArMetrics(double Cosine, double MaxAbs, double RelL2, bool Top1Agrees, int Top10Overlap);

    private static FastArMetrics RunFastArAgainst(string modelFile, string goldenTag)
    {
        string? modelPath = FindModelPath(modelFile);
        Assert.SkipUnless(modelPath != null, $"models/{modelFile} not found");
        string? inputPath = FindRepoFile($"scratch-llamacpp-ref/fishspeech_fastar{goldenTag}_golden_input.txt");
        string? logitsPath = FindRepoFile($"scratch-llamacpp-ref/fishspeech_fastar{goldenTag}_golden_logits.txt");
        Assert.SkipUnless(inputPath != null && logitsPath != null,
            $"golden fast-AR files for '{goldenTag}' not found (FASTAR_GGUF=... FASTAR_TAG={goldenTag} python scratch-llamacpp-ref/fish_speech_fastar_golden.py)");

        var inputLines = File.ReadAllText(inputPath!).Trim().Split('\n');
        var hidden = Array.ConvertAll(inputLines[0].Split(','), float.Parse);
        var prefix = Array.ConvertAll(inputLines[1].Split(','), int.Parse);
        var logitsLines = File.ReadAllText(logitsPath!).Trim().Split('\n');
        int n = int.Parse(logitsLines[0]);
        var golden = Array.ConvertAll(logitsLines[1].Split(','), float.Parse);
        Assert.Equal(n, golden.Length);

        using var weights = new FishSpeechWeights(modelPath!);
        var logits = FishSpeechFastAr.Forward(weights, hidden, prefix);
        Assert.Equal(n, logits.Length);

        double dot = 0, na = 0, nb = 0, nd = 0, maxAbs = 0;
        for (int i = 0; i < n; i++)
        {
            dot += logits[i] * golden[i]; na += logits[i] * logits[i]; nb += golden[i] * golden[i];
            double e = logits[i] - golden[i]; nd += e * e; maxAbs = Math.Max(maxAbs, Math.Abs(e));
        }
        static int[] Top(float[] v, int k) => Enumerable.Range(0, v.Length).OrderByDescending(i => v[i]).Take(k).ToArray();
        var tg = Top(golden, 10); var tl = Top(logits, 10);
        var m = new FastArMetrics(dot / (Math.Sqrt(na) * Math.Sqrt(nb)), maxAbs, Math.Sqrt(nd / nb), tg[0] == tl[0], tg.Intersect(tl).Count());
        Console.WriteLine($"[FishFastAR {modelFile} vs '{goldenTag}'] cosine={m.Cosine:R} maxAbs={m.MaxAbs:G4} relL2={m.RelL2:G4} top1Agrees={m.Top1Agrees} top10Overlap={m.Top10Overlap}");
        return m;
    }

    [Fact]
    public void Forward_Q8_0Weights_MatchesDequantizedPathOracle()
    {
        var m = RunFastArAgainst("s2-pro-q8_0.gguf", "_q8path");
        // Measured 2026-10-01: cosine 0.99892, relL2 5.6%, top-1 agrees, top-10 overlap 9/10.
        Assert.True(m.Cosine > 0.995, $"cosine {m.Cosine} vs the Q8_0-dequantized float reference");
        Assert.True(m.Top1Agrees, "top-1 token differs from the Q8_0-dequantized float reference");
    }

    [Fact]
    public void Forward_Q4KMWeights_MatchesDequantizedPathOracle()
    {
        var m = RunFastArAgainst("s2-pro-q4_k_m.gguf", "_q4path");
        // Measured 2026-10-01: cosine 0.99836, relL2 5.7%, top-1 agrees, top-10 overlap 10/10. This is the valid Q4 correctness
        // claim: same dequantized weights, float reference vs C# (Q8_0 requant + inference). Distance from the ORIGINAL
        // weights is quantization loss and is characterized separately below.
        Assert.True(m.Cosine > 0.995, $"cosine {m.Cosine} vs the Q4_K_M-dequantized float reference");
        Assert.True(m.Top1Agrees, "top-1 token differs from the Q4_K_M-dequantized float reference");
        Assert.True(m.Top10Overlap >= 9, $"top-10 overlap {m.Top10Overlap} < 9 vs the Q4_K_M-dequantized float reference");
    }

    /// <summary>
    /// Mixed-precision recipe from docs/1-correctness/08 (Q4_K_M with Fast-AR <c>wo</c> and <c>w3</c> taken from Q8_0, built with
    /// <c>stingray gguf-transplant</c>). Opt-in: set <c>STINGRAY_S2_MIXED_GGUF</c> to the file (it is not in the repo). Checks both that the C#
    /// path is correct for the mixed weights (vs the float reference on the mixed file's dequantized tensors) and that the recipe recovers
    /// most of the distance to the ORIGINAL weights that pure Q4_K_M loses (cosine 0.44 -> ~0.97).
    /// </summary>
    [Fact]
    public void Forward_Q4KMWithQ8WoW3_MatchesDequantizedPathOracle_AndRecoversOriginal()
    {
        string? mixed = Environment.GetEnvironmentVariable("STINGRAY_S2_MIXED_GGUF");
        Assert.SkipUnless(!string.IsNullOrEmpty(mixed) && File.Exists(mixed),
            "set STINGRAY_S2_MIXED_GGUF to a gguf-transplant output (Q4_K_M + Q8_0 fast-AR wo/w3)");

        var path = RunFastArAgainst(mixed!, "_mixedpath");
        Assert.True(path.Cosine > 0.995, $"cosine {path.Cosine} vs the mixed file's dequantized float reference");
        Assert.True(path.Top1Agrees, "top-1 differs from the mixed file's dequantized float reference");

        var original = RunFastArAgainst(mixed!, "");
        Assert.True(original.Cosine > 0.90, $"cosine {original.Cosine} vs the ORIGINAL weights; the recipe should recover ~0.97 (pure Q4_K_M: 0.44)");
        Assert.True(original.Top1Agrees, "top-1 differs from the original weights");
    }
}
