using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Phase 0.2 diagnostic (docs/2-coverage/2026-11-07-coverage-completion-programme-plan.md): runs the
/// five-token prompt "The capital of France is" through Qwen4ExpForwardPass and dumps l_last_N and
/// result_output at the last position, for comparison with the llama.cpp eval-callback dump.
/// Opt-in: skipped unless QWEN_STINGRAY_DUMP_DIR names an existing directory. Not a parity assertion.
/// </summary>
public sealed class Qwen4ExpRefDumpTests
{
    // Token IDs as printed by llama-eval-callback for "The capital of France is" (BOS-less, 5 tokens).
    private static readonly int[] PromptTokens = { 760, 6511, 314, 9338, 369 };

    private static string? FindModel() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(
        "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    [Fact]
    public void PromptPass_DumpsLastPositionResidualsAndLogits()
    {
        string? outDir = Environment.GetEnvironmentVariable("QWEN_STINGRAY_DUMP_DIR");
        Assert.SkipUnless(outDir is not null && Directory.Exists(outDir),
            "Set QWEN_STINGRAY_DUMP_DIR to an existing directory to run the Qwen dump.");
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        int last = PromptTokens.Length - 1;
        Environment.SetEnvironmentVariable("STINGRAY_Q4E_DUMP_DIR", outDir);
        Environment.SetEnvironmentVariable("STINGRAY_Q4E_DUMP_POS", last.ToString());
        try
        {
            using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
            int totalLayers = (int)Convert.ToInt64(model.Metadata["qwen4exp.block_count"]);
            var hp = Qwen4ExpHyperparams.FromMetadata(model.Metadata, totalLayers);
            var tensors = Qwen4ExpTensorSet.Load(model, hp, totalLayers);
            using var forwardPass = new Qwen4ExpForwardPass(model, hp, tensors);

            ReadOnlySpan<float> logits = default;
            for (int pos = 0; pos <= last; pos++)
            {
                logits = forwardPass.Forward(PromptTokens[pos], pos);
            }

            Assert.False(float.IsNaN(logits[0]) || float.IsInfinity(logits[0]));
            Assert.True(File.Exists(Path.Combine(outDir!, "result_output.bin")));
            Assert.True(File.Exists(Path.Combine(outDir!, $"l_last_{totalLayers - 2}.bin")));
        }
        finally
        {
            Environment.SetEnvironmentVariable("STINGRAY_Q4E_DUMP_DIR", null);
            Environment.SetEnvironmentVariable("STINGRAY_Q4E_DUMP_POS", null);
        }
    }
}
