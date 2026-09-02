using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.AceStep.Text;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.AceStep;

/// <summary>
/// First real-weight smoke test for ACE-Step's text-encoding step: real
/// `Qwen/Qwen3-Embedding-0.6B-GGUF` (Q8_0) weights through <see cref="AceStepQwen3TextEncoder"/>
/// (which reuses the existing `Engine.ForwardPass`/`EnableHiddenTaps` machinery, not a new
/// transformer -- see docs/064-acestep-implementation-plan.md). Non-degeneracy receipt (finite,
/// non-trivial, shape-correct), not yet a numeric golden-parity test against a real HF
/// `transformers` `Qwen3Model` reference run.
///
/// <para>Uses the Q8_0 quant specifically, not f16 -- see
/// <see cref="AceStepQwen3TextEncoder"/>'s doc comment for the real NaN bug found in the f16
/// quant's layer-27 output on this exact real prompt during this test's development, and why Q8_0
/// avoids it.</para>
/// </summary>
public sealed class AceStepQwen3TextEncoderTests
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
    public void Encode_RealWeights_ProducesNonDegenerateHiddenStates()
    {
        string? ggufPath = FindRepoFile("models/qwen3-embedding-0.6b/qwen3-embedding-0.6b-q8_0.gguf");
        Assert.SkipUnless(ggufPath != null, "models/qwen3-embedding-0.6b/qwen3-embedding-0.6b-q8_0.gguf not found");

        using var encoder = new AceStepQwen3TextEncoder(ggufPath!);

        // Real SFT_GEN_PROMPT template, transcribed from the real diffusers ACE-Step pipeline
        // (see docs/064-acestep-implementation-plan.md's "Corrections and confirmations").
        string prompt =
            "# Instruction\nFill the audio semantic mask based on the given conditions:\n\n" +
            "# Caption\nA cinematic orchestral soundtrack with deep drums\n\n" +
            "# Metas\n- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: 30 seconds\n<|endoftext|>\n";

        var hidden = encoder.Encode(prompt);

        Assert.True(hidden.Length > 0, "encoder produced zero token positions");
        foreach (var row in hidden)
        {
            Assert.Equal(1024, row.Length); // real Qwen3-Embedding-0.6B hidden_size
            foreach (var v in row)
                Assert.True(float.IsFinite(v), "hidden state contains NaN/Inf -- degenerate output");
        }

        double sumSq = 0;
        int count = 0;
        foreach (var row in hidden)
            foreach (var v in row) { sumSq += (double)v * v; count++; }
        double rms = Math.Sqrt(sumSq / count);
        Assert.True(rms > 1e-3, $"hidden state RMS ({rms}) is near-zero -- likely a wiring bug");

        // Real RMSNorm output should NOT be degenerate-uniform across positions (a common
        // wiring bug: taking the wrong tap or a constant embedding would produce identical rows).
        bool anyRowDiffers = false;
        for (int i = 1; i < hidden.Length; i++)
        {
            double diff = 0;
            for (int d = 0; d < hidden[0].Length; d++) diff += Math.Abs(hidden[i][d] - hidden[0][d]);
            if (diff > 1e-3) { anyRowDiffers = true; break; }
        }
        Assert.True(anyRowDiffers || hidden.Length == 1, "all token positions produced identical hidden states -- likely a wiring bug");
    }
}
