
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Diagnostic (not a golden test): checks whether the general "present a non-native
/// checkpoint as synthetic qwen3 GGUF metadata" bridging technique -- shared by
/// <see cref="OpenTail.Stingray.Audio.OmniVoice.OmniVoiceLlmTensorSource"/>,
/// <c>FunAsrNanoLlmTensorSource</c>/<c>FunAsrNanoLlmGgufTensorSource</c>, and
/// <c>QwenAsrLlmSafetensorsTensorSource</c> -- has the SAME degenerate-output bug found in
/// Fun-ASR-Nano's decode path (see docs/audio-review-progress.md), using a real downloaded
/// checkpoint with a DIFFERENT precision (float32 safetensors, not Q8_0/BF16 GGUF-packed) to check
/// whether the bug is precision-specific or a genuine cross-cutting wiring issue. No tokenizer is
/// needed for this check -- arbitrary small token ids are enough to see whether ForwardPass
/// produces varied, context-sensitive logits or a fixed degenerate token regardless of input.
/// </summary>
public sealed class OmniVoiceLlmWiringDiagnosticTests : HeavyTestBase
{
    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Forward_ArbitraryTokens_ProducesVariedContextSensitiveLogits()
    {
        string? modelPath = FindRepoFile("models/_models/omnivoice/model.safetensors");
        Assert.SkipUnless(modelPath != null, "omnivoice model.safetensors not found");

        using var source = new OpenTail.Stingray.Audio.OmniVoice.OmniVoiceLlmTensorSource(
            modelPath!, numLayers: 28, hiddenDim: 1024, numHeads: 16, numKvHeads: 8, headDim: 128, ffDim: 3072, vocabSize: 151676, ropeTheta: 1_000_000f, rmsNormEps: 1e-6f);

        var hp = OpenTail.Stingray.Core.ModelHyperparams.FromGgufMetadata(source.Metadata);
        using var backend = new OpenTail.Stingray.Cpu.CpuBackend();
        using var fwd = new OpenTail.Stingray.Engine.ForwardPass(source, backend, hp);

        // Two DIFFERENT short prompts -- if the wiring is healthy, they should produce DIFFERENT
        // next-token predictions (or at least different logits distributions). If broken the same
        // way as Fun-ASR-Nano, both will collapse to the identical degenerate token.
        var promptA = new[] { 1, 2, 3, 4, 5 };
        var promptB = new[] { 100, 200, 300, 400, 500 };

        var logitsA = fwd.Prefill(promptA).ToArray();
        var logitsB = fwd.Prefill(promptB).ToArray();

        int argmaxA = ArgMax(logitsA);
        int argmaxB = ArgMax(logitsB);
        Console.Error.WriteLine($"[OmniVoiceLlmWiringDiagnostic] argmaxA={argmaxA} argmaxB={argmaxB}");

        double sumSqDiff = 0;
        for (int i = 0; i < logitsA.Length; i++) { double d = logitsA[i] - logitsB[i]; sumSqDiff += d * d; }
        double rmsDiff = Math.Sqrt(sumSqDiff / logitsA.Length);
        Console.Error.WriteLine($"[OmniVoiceLlmWiringDiagnostic] logits rmsDiff between prompts={rmsDiff:F6}");

        // Also check within-prompt: does feeding a NEW token at the next position change the
        // output at all, or does it stay fixed (the Fun-ASR-Nano symptom)?
        var afterStep1 = fwd.Forward(argmaxA, promptA.Length).ToArray();
        var afterStep2 = fwd.Forward(999, promptA.Length + 1).ToArray();
        int argmaxStep1 = ArgMax(afterStep1);
        int argmaxStep2 = ArgMax(afterStep2);
        Console.Error.WriteLine($"[OmniVoiceLlmWiringDiagnostic] argmaxStep1={argmaxStep1} argmaxStep2(fed token 999)={argmaxStep2}");

        Assert.True(rmsDiff > 1e-4, "logits are identical for two different prompts -- likely the same degenerate wiring bug as Fun-ASR-Nano");
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
