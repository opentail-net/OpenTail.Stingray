namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// First real-weight load attempt for glm5next (GLM-5.3-Flash). Bypasses the engine's admission
/// gate deliberately (glm5next is NotAdmitted) by constructing Glm5NextForwardPass directly.
///
/// <para>Found 2026-11-07: the real checkpoint (unsloth/GLM-5.3-Flash-GGUF, UD-IQ1_S, 87 GB)
/// declares <c>general.architecture = "glm5-next"</c> (hyphenated) -- not "glm5next", the id this
/// descriptor was registered under. ArchitectureRegistry.Find is an exact string match with no
/// normalization, so without an alias this real checkpoint would never resolve to this
/// architecture at all (it would hit the generic "architecture not supported" message, not even
/// this file's specific RefusalReason). Added "glm5-next" as an Alias; this test confirms the
/// lookup now succeeds for the real checkpoint's real metadata.</para>
/// </summary>
public sealed class Glm5NextRealWeightsSmokeTests
{
    private static string? FindModel() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(
        "GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf");

    [Fact]
    public void RealCheckpoint_ArchitectureString_ResolvesViaAlias()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "GLM-5.3-Flash UD-IQ1_S checkpoint is required (paged, 87 GB).");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        string arch = (string)model.Metadata["general.architecture"];
        Assert.Equal("glm5-next", arch);

        var descriptor = OpenTail.Stingray.Engine.ArchitectureRegistry.Find(arch);
        Assert.NotNull(descriptor);
        Assert.Equal("glm5next", descriptor!.Id);
    }

    [Fact]
    public void Construction_OnRealCheckpoint_DoesNotThrowOnLoad()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "GLM-5.3-Flash UD-IQ1_S checkpoint is required (paged, 87 GB).");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        var metadata = model.Metadata;
        const string arch = "glm5-next";
        int numLayerAll = (int)Convert.ToInt64(metadata[$"{arch}.block_count"]);
        int embedDim = (int)Convert.ToInt64(metadata[$"{arch}.embedding_length"]);
        int numHeads = (int)Convert.ToInt64(metadata[$"{arch}.attention.head_count"]);
        int headDim = (int)Convert.ToInt64(metadata[$"{arch}.attention.key_length"]);
        int numExperts = (int)Convert.ToInt64(metadata[$"{arch}.expert_count"]);
        int numExpertsUsed = (int)Convert.ToInt64(metadata[$"{arch}.expert_used_count"]);

        var hp = OpenTail.Stingray.Engine.Glm5NextHyperparams.FromGgufMetadata(
            metadata, arch, numLayerAll, embedDim, numHeads, headDim, numExperts, numExpertsUsed);

        using var forwardPass = new OpenTail.Stingray.Engine.Glm5NextForwardPass(model, hp);

        var tokenizer = OpenTail.Stingray.Core.GgufTokenizer.FromGgufModel(model);
        var ids = new List<int>();
        if (tokenizer.AddBosToken) ids.Add(tokenizer.BosTokenId);
        ids.AddRange(tokenizer.Encode("The capital of France is"));

        var logits = forwardPass.Prefill(ids.ToArray());
        Assert.True(logits.Length > 0);
        Assert.All(logits.ToArray(), l => Assert.True(float.IsFinite(l)));
    }
}
