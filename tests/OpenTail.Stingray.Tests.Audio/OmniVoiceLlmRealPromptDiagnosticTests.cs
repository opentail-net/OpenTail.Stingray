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
}
