namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// First real-weight load attempt for deepseek4 (DeepSeek-V4 Flash), per docs/2-coverage/
/// 2026-10-03-deepseek-v4-review-plan.md. Bypasses the engine's admission gate deliberately
/// (deepseek4 is NotAdmitted) by constructing DeepSeek4ForwardPass directly, the same way
/// DeepSeek4AlphaTests does with synthetic tensors -- here with the real GGUF instead.
///
/// <para>Found 2026-11-07: this real checkpoint (unsloth/DeepSeek-V4-Flash-GGUF, UD-IQ1_S, 86 GB)
/// declares <c>rope.scaling.type=yarn, factor=16</c>, which this alpha forward pass does not
/// implement anywhere (plain RoPE only) and, until this same day, did not even read or guard
/// against -- it would have silently run with wrong attention on every layer. Added
/// DeepSeek4Hyperparams.RopeScalingFactor and a constructor refusal (matching deepseek41's
/// existing identical guard). This test confirms that refusal actually fires against the real
/// checkpoint's real metadata, not just a synthetic dictionary -- so for this specific checkpoint,
/// going further (prefill/decode) is correctly blocked rather than attempted.</para>
///
/// <para>This checkpoint's compress_ratios are also non-trivial (observed: 0, 0, 4, 128, ...),
/// which would have been the first real exercise of the CSA/HCA code paths (previously only
/// smoke-tested with ratio 0) -- not reached here because construction refuses first on YaRN.</para>
/// </summary>
public sealed class DeepSeek4RealWeightsSmokeTests
{
    private static string? FindModel() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(
        "DeepSeek-V4-Flash-UD-IQ1_S-00001-of-00003.gguf");

    [Fact]
    public void Construction_OnRealCheckpoint_RefusesYarnRopeScaling()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "DeepSeek-V4-Flash UD-IQ1_S checkpoint is required (paged, 86 GB).");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        var metadata = model.Metadata;
        const string arch = "deepseek4";
        int numLayerAll = (int)Convert.ToInt64(metadata[$"{arch}.block_count"]);
        int embedDim = (int)Convert.ToInt64(metadata[$"{arch}.embedding_length"]);
        int numHeads = (int)Convert.ToInt64(metadata[$"{arch}.attention.head_count"]);
        int headDim = (int)Convert.ToInt64(metadata[$"{arch}.attention.key_length"]);
        int ropeDim = (int)Convert.ToInt64(metadata[$"{arch}.rope.dimension_count"]);
        int numExperts = (int)Convert.ToInt64(metadata[$"{arch}.expert_count"]);
        int numExpertsUsed = (int)Convert.ToInt64(metadata[$"{arch}.expert_used_count"]);

        var hp = OpenTail.Stingray.Engine.DeepSeek4Hyperparams.FromGgufMetadata(
            metadata, arch, numLayerAll, embedDim, numHeads, headDim, ropeDim, numExperts, numExpertsUsed);

        Assert.Equal(16f, hp.RopeScalingFactor);

        var ex = Assert.Throws<NotSupportedException>(() => new OpenTail.Stingray.Engine.DeepSeek4ForwardPass(model, hp));
        Assert.Contains("YaRN", ex.Message);
    }
}
