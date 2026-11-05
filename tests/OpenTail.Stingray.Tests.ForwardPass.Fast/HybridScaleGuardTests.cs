namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// The Vulkan hybrid (<c>-g N</c>) applies the embedding, residual, logit and attention scale fields (Granite, Gemma 3) but no logit soft-capping
/// (Gemma 2), so a model that sets a cap is refused with a reason instead of running with wrong numbers. Pure function of the hyperparameters, so
/// no model file is needed; the scale fields are proved against the CPU pass on real weights in <c>HybridScaleParityTests</c>.
/// </summary>
public sealed class HybridScaleGuardTests
{
    private static ModelHyperparams Plain() => new();

    [Fact]
    public void PlainHyperparameters_AreNotRefused()
        => Assert.Null(GpuForwardPass.UnappliedSoftcapReason(Plain()));

    [Fact]
    public void ScaleFieldsAlone_AreNotRefused_TheHybridAppliesThem()
        => Assert.Null(GpuForwardPass.UnappliedSoftcapReason(Plain() with
        {
            EmbeddingScale = 12f, ResidualScale = 0.22f, LogitScale = 0.125f, AttentionScaleOverride = 0.015625f,
        }));

    [Theory]
    [InlineData("final")]
    [InlineData("attention")]
    public void AnySoftcap_IsRefused_WithAReasonThatNamesIt(string which)
    {
        var hp = which == "final" ? Plain() with { FinalLogitSoftcap = 30f } : Plain() with { AttnLogitSoftcap = 50f };

        string? reason = GpuForwardPass.UnappliedSoftcapReason(hp);

        Assert.NotNull(reason);
        Assert.Contains("soft-capping", reason);
    }
}
