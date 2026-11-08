using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Phase 4 loading: the qwen4exp MTP head loads from its own GGUF against the trunk, with the reference tensor shapes.
/// Real files only; skipped visibly when either file is absent.
/// </summary>
public sealed class Qwen4ExpMtpLoadTests
{
    private static string? FindTrunk() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(
        "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf");

    private static string? FindMtp() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(
        "mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf");

    [Fact]
    public void Head_LoadsFromMtpFile_WithReferenceShapes()
    {
        string? trunkPath = FindTrunk();
        string? mtpPath = FindMtp();
        Assert.SkipUnless(trunkPath is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");
        Assert.SkipUnless(mtpPath is not null, "Qwen3.8 MTP shared Q4_K_M head is required (1.9 GB).");

        using var trunk = GgufModel.Open(trunkPath!);
        int totalLayers = (int)Convert.ToInt64(trunk.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(trunk.Metadata, totalLayers);
        var trunkTensors = Qwen4ExpTensorSet.Load(trunk, hp, totalLayers);

        using var mtp = GgufModel.Open(mtpPath!);
        var head = Qwen4ExpMtpHead.Load(mtp, trunkTensors, blockIndex: totalLayers, embedDim: hp.EmbedDim, hc: hp.HyperConnectionCount);

        Assert.Equal(totalLayers, head.BlockIndex);
        Assert.False(head.Layer.IsRecurrent);
        Assert.NotNull(head.Layer.AttnQ);
        Assert.NotNull(head.Layer.AttnK);
        Assert.NotNull(head.Layer.AttnOut);
        Assert.NotNull(head.Layer.FfnDownExps);
        Assert.NotNull(head.Layer.HcAttnDown);
        Assert.Equal(new long[] { 2 * hp.EmbedDim, hp.EmbedDim }, head.EhProj.Info.Dimensions.Select(d => (long)d).ToArray());
        Assert.Equal(trunkTensors.TokEmbd.Name, head.TokEmbd.Name);
    }

    [Fact]
    public void TrunkFileWithoutNextnTensors_IsRejected()
    {
        string? trunkPath = FindTrunk();
        Assert.SkipUnless(trunkPath is not null, "Qwen3.8-Flash-Next UD-IQ1_S checkpoint is required (paged, 72.5 GB).");

        using var trunk = GgufModel.Open(trunkPath!);
        int totalLayers = (int)Convert.ToInt64(trunk.Metadata["qwen4exp.block_count"]);
        var hp = Qwen4ExpHyperparams.FromMetadata(trunk.Metadata, totalLayers);
        var trunkTensors = Qwen4ExpTensorSet.Load(trunk, hp, totalLayers);

        Assert.Throws<InvalidOperationException>(() =>
            Qwen4ExpMtpHead.Load(trunk, trunkTensors, blockIndex: totalLayers, embedDim: hp.EmbedDim, hc: hp.HyperConnectionCount));
    }
}
