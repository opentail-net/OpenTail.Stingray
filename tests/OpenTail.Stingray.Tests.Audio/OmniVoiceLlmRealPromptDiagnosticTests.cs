namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Follow-up diagnostic to <see cref="OmniVoiceLlmWiringDiagnosticTests"/>: that test
/// found the LLM's argmax output suspiciously convergent across different ARBITRARY token ids,
/// but left open the real possibility that low-numbered ids (1-5, 100-500) are just
/// near-untrained special/byte tokens producing unremarkable predictions for any healthy LLM.
/// This test uses OmniVoice's REAL tokenizer (`k2-fsa/OmniVoice`'s real `tokenizer.json`,
/// downloaded directly from HF for this checkpoint since the `audiocpp`-packed GGUF variant was
/// not used) to encode real, different natural-language sentences and checks whether the
/// argmax/logits still converge -- a real bug would still show convergence here; a real
/// input-was-meaningless artifact would not.</summary>
public sealed class OmniVoiceLlmRealPromptDiagnosticTests : HeavyTestBase
{
    private static string? FindRepoDir(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Forward_RealDifferentSentences_ProducesVariedContextSensitiveLogits()
    {
        string? modelDir = FindRepoDir("models/_models/omnivoice");
        Assert.SkipUnless(modelDir != null
            && File.Exists(Path.Combine(modelDir, "model.safetensors"))
            && File.Exists(Path.Combine(modelDir, "tokenizer.json")),
            "omnivoice model.safetensors/tokenizer.json not found");

        var result = OpenTail.Stingray.Core.HuggingFaceTokenizerSource.Load(modelDir!);
        Assert.True(result.IsUsable, $"tokenizer load failed: {string.Join("; ", result.Rejections)}");
        var tokenizer = OpenTail.Stingray.Core.GgufTokenizer.FromSource(result.Source!);

        var promptA = tokenizer.Encode("The quick brown fox jumps over the lazy dog.").ToArray();
        var promptB = tokenizer.Encode("Quantum computers use qubits instead of classical bits.").ToArray();
        Console.Error.WriteLine($"[OmniVoiceRealPrompt] promptA tokens: {string.Join(",", promptA)}");
        Console.Error.WriteLine($"[OmniVoiceRealPrompt] promptB tokens: {string.Join(",", promptB)}");
        Assert.NotEmpty(promptA);
        Assert.NotEmpty(promptB);

        using var source = new OpenTail.Stingray.Audio.OmniVoice.OmniVoiceLlmTensorSource(
            Path.Combine(modelDir!, "model.safetensors"),
            numLayers: 28, hiddenDim: 1024, numHeads: 16, numKvHeads: 8, headDim: 128, ffDim: 3072,
            vocabSize: 151676, ropeTheta: 1_000_000f, rmsNormEps: 1e-6f);

        var hp = OpenTail.Stingray.Core.ModelHyperparams.FromGgufMetadata(source.Metadata);
        using var backend = new OpenTail.Stingray.Cpu.CpuBackend();
        using var fwd = new OpenTail.Stingray.Engine.ForwardPass(source, backend, hp);

        var logitsA = fwd.Prefill(promptA).ToArray();
        var logitsB = fwd.Prefill(promptB).ToArray();

        int argmaxA = ArgMax(logitsA);
        int argmaxB = ArgMax(logitsB);
        Console.Error.WriteLine($"[OmniVoiceRealPrompt] argmaxA={argmaxA} ({tokenizer.Decode([argmaxA])}) argmaxB={argmaxB} ({tokenizer.Decode([argmaxB])})");

        double sumSqDiff = 0;
        for (int i = 0; i < logitsA.Length; i++) { double d = logitsA[i] - logitsB[i]; sumSqDiff += d * d; }
        double rmsDiff = Math.Sqrt(sumSqDiff / logitsA.Length);
        Console.Error.WriteLine($"[OmniVoiceRealPrompt] logits rmsDiff between real prompts={rmsDiff:F6}");

        Assert.True(rmsDiff > 1e-4, "logits are identical for two different REAL sentences -- confirms a real cross-cutting wiring bug, not a meaningless-input artifact");
    }

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    /// <summary>Targeted hypothesis check: is the dominant token's (85473) embedding row an
    /// outlier by norm compared to a random sample of other rows? A real, common symptom of a
    /// mis-scaled/duplicated tensor during weight loading.</summary>
    [Fact]
    public unsafe void TokenEmbeddingRow85473_NormComparedToRandomSample()
    {
        string? modelDir = FindRepoDir("models/_models/omnivoice");
        Assert.SkipUnless(modelDir != null && File.Exists(Path.Combine(modelDir, "model.safetensors")),
            "omnivoice model.safetensors not found");

        using var source = new OpenTail.Stingray.Audio.OmniVoice.OmniVoiceLlmTensorSource(
            Path.Combine(modelDir!, "model.safetensors"),
            numLayers: 28, hiddenDim: 1024, numHeads: 16, numKvHeads: 8, headDim: 128, ffDim: 3072,
            vocabSize: 151676, ropeTheta: 1_000_000f, rmsNormEps: 1e-6f);

        var tensorInfo = source.FindTensor("token_embd.weight");
        Assert.NotNull(tensorInfo);
        int hiddenDim = checked((int)tensorInfo!.Value.Dimensions[0]);
        int vocabSize = checked((int)tensorInfo.Value.Dimensions[1]);

        var data = source.GetTensorData(tensorInfo.Value);
        var allFloats = new float[data.Length / sizeof(float)];
        System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(data).CopyTo(allFloats);

        float RowNorm(int row)
        {
            double sumSq = 0;
            long baseIdx = (long)row * hiddenDim;
            for (int d = 0; d < hiddenDim; d++) { float v = allFloats[baseIdx + d]; sumSq += (double)v * v; }
            return (float)Math.Sqrt(sumSq);
        }

        float dominantNorm = RowNorm(85473);

        var rng = new Random(7);
        var sampleNorms = new List<float>();
        for (int i = 0; i < 200; i++) sampleNorms.Add(RowNorm(rng.Next(vocabSize)));
        sampleNorms.Sort();
        float median = sampleNorms[sampleNorms.Count / 2];
        float max = sampleNorms[^1];

        Console.Error.WriteLine($"[OmniVoiceEmbeddingNorm] row85473_norm={dominantNorm:F4} sample_median={median:F4} sample_max={max:F4}");
    }

    /// <summary>Layer-by-layer bisection (real next step per the doc's own entry): captures each
    /// layer's hidden-state tap for two different real prompts and reports the per-layer RMS
    /// difference between them, to find WHERE (if anywhere) the representations collapse toward
    /// each other -- the same technique already used successfully for the Qwen3 Forced Aligner
    /// bug this session.</summary>
    [Fact]
    public void PerLayerHiddenStateDivergence_BetweenTwoRealPrompts()
    {
        string? modelDir = FindRepoDir("models/_models/omnivoice");
        Assert.SkipUnless(modelDir != null
            && File.Exists(Path.Combine(modelDir, "model.safetensors"))
            && File.Exists(Path.Combine(modelDir, "tokenizer.json")),
            "omnivoice model.safetensors/tokenizer.json not found");

        var result = OpenTail.Stingray.Core.HuggingFaceTokenizerSource.Load(modelDir!);
        Assert.True(result.IsUsable, $"tokenizer load failed: {string.Join("; ", result.Rejections)}");
        var tokenizer = OpenTail.Stingray.Core.GgufTokenizer.FromSource(result.Source!);

        var promptA = tokenizer.Encode("The quick brown fox jumps over the lazy dog.").ToArray();
        var promptB = tokenizer.Encode("Quantum computers use qubits instead of classical bits.").ToArray();

        const int numLayers = 28;
        const int hiddenDim = 1024;

        using var sourceA = new OpenTail.Stingray.Audio.OmniVoice.OmniVoiceLlmTensorSource(
            Path.Combine(modelDir!, "model.safetensors"),
            numLayers: numLayers, hiddenDim: hiddenDim, numHeads: 16, numKvHeads: 8, headDim: 128, ffDim: 3072,
            vocabSize: 151676, ropeTheta: 1_000_000f, rmsNormEps: 1e-6f);
        var hpA = OpenTail.Stingray.Core.ModelHyperparams.FromGgufMetadata(sourceA.Metadata);
        using var backendA = new OpenTail.Stingray.Cpu.CpuBackend();
        using var fwdA = new OpenTail.Stingray.Engine.ForwardPass(sourceA, backendA, hpA);

        using var sourceB = new OpenTail.Stingray.Audio.OmniVoice.OmniVoiceLlmTensorSource(
            Path.Combine(modelDir!, "model.safetensors"),
            numLayers: numLayers, hiddenDim: hiddenDim, numHeads: 16, numKvHeads: 8, headDim: 128, ffDim: 3072,
            vocabSize: 151676, ropeTheta: 1_000_000f, rmsNormEps: 1e-6f);
        var hpB = OpenTail.Stingray.Core.ModelHyperparams.FromGgufMetadata(sourceB.Metadata);
        using var backendB = new OpenTail.Stingray.Cpu.CpuBackend();
        using var fwdB = new OpenTail.Stingray.Engine.ForwardPass(sourceB, backendB, hpB);

        Assert.True(fwdA.SupportsHiddenTaps, "ForwardPass does not support hidden-state taps -- cannot bisect.");

        var layerIds = Enumerable.Range(0, numLayers).ToArray();
        fwdA.EnableHiddenTaps(layerIds);
        fwdB.EnableHiddenTaps(layerIds);

        fwdA.Prefill(promptA);
        fwdB.Prefill(promptB);

        // fwd.LastHidden is the POST-final-norm, pre-lm-head vector -- compare its cosine
        // similarity between the two prompts directly. Near-1.0 here (despite the pre-norm
        // hidden states clearly differing per the per-layer bisection below) would pinpoint the
        // final RMSNorm step itself as producing near-identical directions regardless of input.
        var postNormA = fwdA.LastHidden.ToArray();
        var postNormB = fwdB.LastHidden.ToArray();
        double dot = 0, normA2 = 0, normB2 = 0;
        for (int d = 0; d < postNormA.Length; d++)
        {
            dot += (double)postNormA[d] * postNormB[d];
            normA2 += (double)postNormA[d] * postNormA[d];
            normB2 += (double)postNormB[d] * postNormB[d];
        }
        double cosineSim = dot / (Math.Sqrt(normA2) * Math.Sqrt(normB2));
        Console.Error.WriteLine($"[OmniVoicePostNormCosine] postNormA_rms={Math.Sqrt(normA2 / postNormA.Length):F4} postNormB_rms={Math.Sqrt(normB2 / postNormB.Length):F4} cosineSim={cosineSim:F6}");

        var tapsA = fwdA.HiddenTapsAt(promptA.Length - 1).ToArray();
        var tapsB = fwdB.HiddenTapsAt(promptB.Length - 1).ToArray();
        Assert.Equal(numLayers * hiddenDim, tapsA.Length);
        Assert.Equal(numLayers * hiddenDim, tapsB.Length);

        for (int layer = 0; layer < numLayers; layer++)
        {
            double sumSqA = 0, sumSqB = 0, sumSqDiff = 0;
            int baseIdx = layer * hiddenDim;
            for (int d = 0; d < hiddenDim; d++)
            {
                float a = tapsA[baseIdx + d], b = tapsB[baseIdx + d];
                sumSqA += (double)a * a;
                sumSqB += (double)b * b;
                sumSqDiff += (double)(a - b) * (a - b);
            }
            float rmsA = (float)Math.Sqrt(sumSqA / hiddenDim);
            float rmsB = (float)Math.Sqrt(sumSqB / hiddenDim);
            float rmsDiff = (float)Math.Sqrt(sumSqDiff / hiddenDim);
            Console.Error.WriteLine($"[OmniVoiceLayerBisect] layer={layer,2} rmsA={rmsA:F4} rmsB={rmsB:F4} rmsDiff={rmsDiff:F4} relDiff={rmsDiff / MathF.Max(rmsA, rmsB):F4}");
        }

        // Real, cheap check proposed by the doc's own next-step: does the layer-27 (final)
        // pre-norm hidden state have a small number of outlier dimensions dominating its RMS?
        int lastBase = (numLayers - 1) * hiddenDim;
        var dimsA = new (int Dim, float AbsVal)[hiddenDim];
        for (int d = 0; d < hiddenDim; d++) dimsA[d] = (d, MathF.Abs(tapsA[lastBase + d]));
        Array.Sort(dimsA, (x, y) => y.AbsVal.CompareTo(x.AbsVal));
        Console.Error.WriteLine("[OmniVoiceOutlierDims] top-10 |value| dims of prompt A's final pre-norm hidden state:");
        for (int i = 0; i < 10; i++)
            Console.Error.WriteLine($"    dim={dimsA[i].Dim} |value|={dimsA[i].AbsVal:F4}");
        float median = dimsA[hiddenDim / 2].AbsVal;
        Console.Error.WriteLine($"[OmniVoiceOutlierDims] median |value|={median:F4}, top1/median ratio={dimsA[0].AbsVal / median:F2}");
    }
}
