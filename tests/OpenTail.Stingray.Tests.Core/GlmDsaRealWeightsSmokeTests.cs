namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Metadata-only real-checkpoint check for glm-dsa (GLM-5.2) -- deliberately NOT a construction
/// test like <see cref="DeepSeek4RealWeightsSmokeTests"/> or <see
/// cref="DeepSeek41RealWeightsSmokeTests"/>, because <see cref="OpenTail.Stingray.Engine.
/// GlmDsaForwardPass"/>'s constructor calls <c>GlmDsaTensorSet.Load</c> immediately, with no
/// metadata-only guard before it, and that Load requires every layer's attention/FFN tensors to be
/// present in the GGUF's tensor-info table. Only shard 1 of the real checkpoint
/// (unsloth/GLM-5.2-GGUF, UD-IQ1_S, ~9 MB) is on this machine -- renamed to drop the
/// <c>-00001-of-00006.gguf</c> suffix so <see cref="OpenTail.Stingray.Core.GgufModel.Open"/> treats
/// it as standalone. A split GGUF's tensor-info table is assembled shard-by-shard as each shard is
/// parsed (see GgufModel.Open's "later shards ... add their own tensors" comment), so this one tiny
/// shard only carries tensor info for whatever is physically in the first ~9 MB -- nowhere near the
/// full 79-layer tensor set. Attempting construction here would throw "Missing required glm-dsa
/// tensor", which is an artifact of the file being incomplete, not a finding about glm-dsa's
/// correctness -- so it is not asserted. What IS real and verifiable from this shard: the GGUF
/// key/value metadata is present regardless of which shard, so the hyperparam values the port
/// expects to see can be checked against this real checkpoint's real header.
///
/// <para>Found 2026-11-07: every value in GlmDsaAlpha.cs's own "ARCHITECTURE SPEC" header comment
/// matches this real checkpoint exactly -- n_head=64, q_lora_rank=2048, kv_lora_rank=512,
/// indexer_n_head=32, indexer_head_size=128, indexer_top_k=2048. The port's reading of these
/// metadata keys was correct from the start; the gap is tensor loading/forward-pass verification,
/// not hyperparameter parsing.</para>
/// </summary>
public sealed class GlmDsaRealWeightsSmokeTests
{
    private static string? FindModel() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(
        "GLM-5.2-UD-IQ1_S.gguf");

    [Fact]
    public void RealCheckpoint_Metadata_MatchesPortedHyperparamSpec()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "GLM-5.2 UD-IQ1_S metadata shard is required.");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        var metadata = model.Metadata;
        const string arch = "glm-dsa";
        Assert.Equal(arch, (string)metadata["general.architecture"]);

        var descriptor = OpenTail.Stingray.Engine.ArchitectureRegistry.Find(arch);
        Assert.NotNull(descriptor);
        Assert.Equal("glm-dsa", descriptor!.Id);

        int numLayerAll = (int)Convert.ToInt64(metadata[$"{arch}.block_count"]);
        int embedDim = (int)Convert.ToInt64(metadata[$"{arch}.embedding_length"]);
        int numHeads = (int)Convert.ToInt64(metadata[$"{arch}.attention.head_count"]);
        int qLoraRank = (int)Convert.ToInt64(metadata[$"{arch}.attention.q_lora_rank"]);
        int kvLoraRank = (int)Convert.ToInt64(metadata[$"{arch}.attention.kv_lora_rank"]);
        int indexerHeads = (int)Convert.ToInt64(metadata[$"{arch}.attention.indexer.head_count"]);
        int indexerHeadSize = (int)Convert.ToInt64(metadata[$"{arch}.attention.indexer.key_length"]);
        int indexerTopK = (int)Convert.ToInt64(metadata[$"{arch}.attention.indexer.top_k"]);

        Assert.Equal(64, numHeads);
        Assert.Equal(2048, qLoraRank);
        Assert.Equal(512, kvLoraRank);
        Assert.Equal(32, indexerHeads);
        Assert.Equal(128, indexerHeadSize);
        Assert.Equal(2048, indexerTopK);
        Assert.True(numLayerAll > 0 && embedDim > 0);
    }
}
