namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// First real-weight metadata check for deepseek41 (DeepSeek-V4.1 Flash), mirroring
/// <see cref="DeepSeek4RealWeightsSmokeTests"/>. Only shard 1 of the real checkpoint
/// (unsloth/DeepSeek-V4.1-Flash-1017-GGUF, UD-IQ1_S) is on this machine -- renamed to drop the
/// <c>-00001-of-00005.gguf</c> suffix so <see cref="OpenTail.Stingray.Core.GgufModel.Open"/> treats
/// it as a standalone file instead of chasing the four sibling shards (which hold only tensor
/// data, not metadata, and are not present: the full checkpoint is ~335 GB, far beyond what fits
/// here). This is sound specifically because <see cref="OpenTail.Stingray.Engine.
/// DeepSeek41ForwardPass"/>'s constructor runs its compress-ratio and YaRN refusal checks purely
/// from <see cref="OpenTail.Stingray.Engine.DeepSeek41Hyperparams"/> (itself built only from GGUF
/// key/value metadata) before calling <c>DeepSeek41TensorSet.Load</c> -- no tensor bytes are read
/// before the throw, so a metadata-only shard is enough to exercise it for real.
///
/// <para>Found 2026-11-07: the real checkpoint declares <c>general.architecture = "deepseek41"</c>
/// directly (no alias needed), <c>rope.scaling.factor = 16</c> (matches the "official config: 16"
/// comment already in <c>DeepSeek41Alpha.cs</c>), and <c>attention.compress_ratios</c> starting
/// <c>[0, 0, 2, 2, ...]</c> -- so both refusal conditions fire on the first real checkpoint, same
/// shape as deepseek4's YaRN finding.</para>
/// </summary>
public sealed class DeepSeek41RealWeightsSmokeTests
{
    private static string? FindModel() => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(
        "DeepSeek-V4.1-Flash-1017-UD-IQ1_S.gguf");

    [Fact]
    public void Construction_OnRealCheckpoint_RefusesCompressedAttention()
    {
        string? path = FindModel();
        Assert.SkipUnless(path is not null, "DeepSeek-V4.1-Flash UD-IQ1_S metadata shard is required.");

        using var model = OpenTail.Stingray.Core.GgufModel.Open(path!);
        var metadata = model.Metadata;
        Assert.Equal("deepseek41", (string)metadata["general.architecture"]);

        var hp = OpenTail.Stingray.Engine.DeepSeek41Hyperparams.FromGgufMetadata(metadata);

        Assert.Equal(16f, hp.RopeScalingFactor);
        Assert.True(hp.CompressRatios.Count > 2 && hp.CompressRatios[2] > 0,
            "expected the real checkpoint's compress_ratios to include a nonzero entry (compressed attention), as seen 2026-11-07");

        // Compress ratio > 0 is checked before YaRN in the constructor, so that is the refusal
        // actually observed here, not the YaRN one -- both are real and both are present, but
        // only the first-checked one is reachable in a single construction attempt.
        var ex = Assert.Throws<NotSupportedException>(() => new OpenTail.Stingray.Engine.DeepSeek41ForwardPass(model, hp));
        Assert.Contains("compress ratio", ex.Message);
    }
}
