namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// The Vulkan hybrid (<c>-g N</c>) applies none of the model scale hyperparameters, so a model that sets one runs with wrong numbers and no error
/// (measured 2026-10-31: Granite 3.3-2B answers with garbage, gemma-3-4b disagrees with the CPU pass). The guard refuses them with a reason.
/// Pure function of the hyperparameters, so no model file is needed.
/// </summary>
public sealed class HybridScaleGuardTests
{
    private static ModelHyperparams Plain() => new();

    [Fact]
    public void PlainHyperparameters_AreNotRefused()
        => Assert.Null(GpuForwardPass.UnappliedScaleReason(Plain()));

    [Theory]
    [InlineData("embedding")]
    [InlineData("residual")]
    [InlineData("logit")]
    [InlineData("attention")]
    public void AnyUnappliedScale_IsRefused_WithAReasonThatNamesScaling(string which)
    {
        var hp = which switch
        {
            "embedding" => Plain() with { EmbeddingScale = 12f },
            "residual" => Plain() with { ResidualScale = 0.22f },
            "logit" => Plain() with { LogitScale = 0.125f },
            _ => Plain() with { AttentionScaleOverride = 0.015625f },
        };

        string? reason = GpuForwardPass.UnappliedScaleReason(hp);

        Assert.NotNull(reason);
        Assert.Contains("scaling", reason);
    }

    [Fact]
    public void GraniteStyleScales_AllTogether_AreRefused()
        => Assert.NotNull(GpuForwardPass.UnappliedScaleReason(Plain() with
        {
            EmbeddingScale = 12f, ResidualScale = 0.22f, LogitScale = 0.125f, AttentionScaleOverride = 0.015625f,
        }));
}
