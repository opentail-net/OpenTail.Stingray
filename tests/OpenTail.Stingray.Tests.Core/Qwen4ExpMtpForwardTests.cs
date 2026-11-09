using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Phase 4 forward: one MTP draft step on the real trunk and head. Checks shape, finiteness and determinism; the per-step
/// comparison with the reference is a separate check (programme plan, Phase 4). Real checkpoint and head; skipped when absent.
/// </summary>
public sealed class Qwen4ExpMtpForwardTests
{
    private static readonly int[] Prompt = { 760, 6511, 314, 9338, 369 }; // "The capital of France is"

    private static string? FindTrunk() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(
        Qwen4ExpTestSupport.TrunkFileName);

    private static string? FindMtp() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(
        "mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf");

    [Fact]
    public void DraftStep_IsFiniteAndDeterministic()
    {
        string? trunkPath = FindTrunk();
        string? mtpPath = FindMtp();
        Assert.SkipUnless(trunkPath is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");
        Assert.SkipUnless(mtpPath is not null, "Qwen3.8 MTP shared Q4_K_M head is required (1.9 GB).");

        using var trunk = GgufModel.Open(trunkPath!);
        int totalLayers = (int)Convert.ToInt64(trunk.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(trunk.Metadata, totalLayers);
        var tensors = Qwen4ExpTensorSet.Load(trunk, hp, totalLayers);
        using var pass = new Qwen4ExpForwardPass(trunk, hp, tensors);

        using var mtp = GgufModel.Open(mtpPath!);
        var head = Qwen4ExpMtpHead.Load(mtp, tensors, blockIndex: totalLayers, embedDim: hp.EmbedDim, hc: hp.HyperConnectionCount);
        pass.AttachMtpHead(head);

        var state = pass.CreateSequenceState();
        ReadOnlySpan<float> trunkLogits = default;
        for (int pos = 0; pos < Prompt.Length; pos++) trunkLogits = pass.Forward(Prompt[pos], pos, state);
        int draft = 0;
        for (int i = 1; i < trunkLogits.Length; i++) if (trunkLogits[i] > trunkLogits[draft]) draft = i;

        var hNext = pass.LastHcResidual.ToArray();
        int hcDim = hp.HyperConnectionCount * hp.EmbedDim;
        Assert.Equal(hcDim, hNext.Length);

        var hOut = new float[hcDim];
        var logits1 = pass.ForwardMtp(draft, hNext, Prompt.Length, state, hOut).ToArray();
        Assert.Equal(trunk_vocab(tensors), logits1.Length);

        // Hand-off to the reference driver: the same residual, token and position, and Stingray's logits.
        string? dumpDir = Environment.GetEnvironmentVariable("QWEN_MTP_DUMP_DIR");
        if (dumpDir is not null && Directory.Exists(dumpDir))
        {
            var bytes = new byte[hNext.Length * 4];
            Buffer.BlockCopy(hNext, 0, bytes, 0, bytes.Length);
            File.WriteAllBytes(Path.Combine(dumpDir, "mtp_h_in.bin"), bytes);
            File.WriteAllText(Path.Combine(dumpDir, "mtp_meta.txt"), $"{draft} {Prompt.Length} {hNext.Length} {logits1.Length}");
            var lb = new byte[logits1.Length * 4];
            Buffer.BlockCopy(logits1, 0, lb, 0, lb.Length);
            File.WriteAllBytes(Path.Combine(dumpDir, "mtp_logits_stingray.bin"), lb);
            var hb = new byte[hOut.Length * 4];
            Buffer.BlockCopy(hOut, 0, hb, 0, hb.Length);
            File.WriteAllBytes(Path.Combine(dumpDir, "mtp_hout_stingray.bin"), hb);
        }
        Assert.All(logits1, v => Assert.True(float.IsFinite(v)));
        Assert.All(hOut, v => Assert.True(float.IsFinite(v)));

        // The head's own cache advanced, so a repeat at the same position is not required to match; determinism is
        // checked on a fresh sequence instead.
        var fresh = pass.CreateSequenceState();
        for (int pos = 0; pos < Prompt.Length; pos++) pass.Forward(Prompt[pos], pos, fresh);
        var h2 = new float[hcDim];
        var logits2 = pass.ForwardMtp(draft, pass.LastHcResidual, Prompt.Length, fresh, h2).ToArray();
        Assert.Equal(logits1, logits2);
        Assert.Equal(hOut, h2);
    }

    private static int trunk_vocab(Qwen4ExpTensorSet tensors) => (int)tensors.Output.Info.Dimensions[1];
}
