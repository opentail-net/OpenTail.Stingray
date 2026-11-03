using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Pipeline;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// The Vulkan hybrid pass must size its lazily filled routed-expert cache from the planner's budget and the real upload
/// size of an expert, not default to every expert on every GPU layer.
/// </summary>
public sealed class VulkanMoeExpertCacheSizingTests : IDisposable
{
    private const uint F32 = 0, Q8_0 = 8, Q4_K = 12, Q6_K = 14;
    private const int ExpertDim = 256, EmbDim = 256; // both multiples of the 256-wide K-quant block
    private const long MiB = 1024 * 1024;

    private readonly List<string> _tempFiles = [];

    public void Dispose()
    {
        foreach (var f in _tempFiles)
            try { File.Delete(f); } catch { }
    }

    private GgufModel Open(params SyntheticGguf.T[] tensors)
    {
        var path = Path.Combine(Path.GetTempPath(), $"moe-sizing-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, SyntheticGguf.Build(3, [SyntheticGguf.KvStr("general.architecture", "qwen3moe")], tensors, dataBytes: 1 << 20));
        _tempFiles.Add(path);
        return GgufModel.Open(path);
    }

    private static SyntheticGguf.T Exps(int layer, string role, uint type) => new($"blk.{layer}.{role}.weight", [EmbDim, ExpertDim, 2], type);

    [Fact]
    public void ExpertFootprint_RawForQ4KAndQ6K_F32ExpansionForEveryOtherDtype()
    {
        Assert.Equal(4L * (256 / 256) * 144, ExpertSlotManager.ExpertFootprintBytes(DType.Q4_K, 4, 256));
        Assert.Equal(4L * (256 / 256) * 210, ExpertSlotManager.ExpertFootprintBytes(DType.Q6_K, 4, 256));
        Assert.Equal(4L * 256 * 4, ExpertSlotManager.ExpertFootprintBytes(DType.Q8_0, 4, 256));
        Assert.Equal(4L * 256 * 4, ExpertSlotManager.ExpertFootprintBytes(DType.Q5_K, 4, 256));
        Assert.Equal(4L * 256 * 4, ExpertSlotManager.ExpertFootprintBytes(DType.Float32, 4, 256));
    }

    [Fact]
    public void PerExpertGpuBytes_SumsRoles_AndPricesEachRoleAtItsLargestLayer()
    {
        // gate: Q4_K on layer 0 but Q8_0 (F32 on the GPU) on layer 1; up and down stay Q4_K.
        using var model = Open(
            Exps(0, "ffn_gate_exps", Q4_K), Exps(0, "ffn_up_exps", Q4_K), Exps(0, "ffn_down_exps", Q4_K),
            Exps(1, "ffn_gate_exps", Q8_0), Exps(1, "ffn_up_exps", Q4_K), Exps(1, "ffn_down_exps", Q4_K));

        long q4k = ExpertSlotManager.ExpertFootprintBytes(DType.Q4_K, ExpertDim, EmbDim);
        long f32 = ExpertSlotManager.ExpertFootprintBytes(DType.Q8_0, ExpertDim, EmbDim);

        Assert.Equal(3 * q4k, ExpertSlotManager.PerExpertGpuBytes(model, 1, ExpertDim, EmbDim));
        Assert.Equal(f32 + 2 * q4k, ExpertSlotManager.PerExpertGpuBytes(model, 2, ExpertDim, EmbDim));
    }

    [Fact]
    public void PerExpertGpuBytes_IsZeroWhenARoleIsMissing()
    {
        using var model = Open(Exps(0, "ffn_gate_exps", Q6_K), Exps(0, "ffn_up_exps", Q6_K));
        Assert.Equal(0, ExpertSlotManager.PerExpertGpuBytes(model, 1, ExpertDim, EmbDim));
    }

    private static LayerPlacement Placement(long budget, long routed) =>
        new(GpuLayers: 4, CpuLayers: 0, GpuWeightBytes: 0, GpuKvBytes: 0, RecommendedCtxSize: 2048,
            ExpertCacheBudgetBytes: budget, MoeRoutedExpertBytes: routed);

    [Fact]
    public void Capacity_IsTheBudgetDividedByPerExpertBytes()
    {
        // 4 GPU layers x 8 experts = 32 total; 10 MiB budget at 1 MiB per expert = 10 slots.
        int slots = HybridForwardPass.ResolveExpertSlotCapacity(-1, Placement(10 * MiB, 40 * MiB), MiB, 4, 8, 2, out var sizing);

        Assert.Equal(10, slots);
        Assert.Equal(10, sizing!.Value.Slots);
    }

    [Fact]
    public void Capacity_NeverExceedsTheTotalExpertCount()
    {
        int slots = HybridForwardPass.ResolveExpertSlotCapacity(-1, Placement(500 * MiB, 40 * MiB), MiB, 4, 8, 2, out _);
        Assert.Equal(32, slots);
    }

    [Fact]
    public void Capacity_ExplicitRequestWinsOverTheBudget_CappedAtTheTotal()
    {
        Assert.Equal(6, HybridForwardPass.ResolveExpertSlotCapacity(6, Placement(10 * MiB, 40 * MiB), MiB, 4, 8, 2, out var sizing));
        Assert.Null(sizing);
        Assert.Equal(32, HybridForwardPass.ResolveExpertSlotCapacity(999, Placement(10 * MiB, 40 * MiB), MiB, 4, 8, 2, out _));
    }

    [Fact]
    public void Capacity_PlacementWithoutMoeSizing_KeepsEveryExpertCacheable()
    {
        int slots = HybridForwardPass.ResolveExpertSlotCapacity(-1, Placement(0, 0), MiB, 4, 8, 2, out var sizing);

        Assert.Equal(32, slots);
        Assert.Null(sizing);
    }

    [Fact]
    public void Capacity_PlannedButNoBudgetLeft_ClampsToOneSlotAndSaysSo()
    {
        int slots = HybridForwardPass.ResolveExpertSlotCapacity(-1, Placement(0, 40 * MiB), MiB, 4, 8, 2, out var sizing);

        Assert.Equal(1, slots);
        Assert.Equal(MoeCacheSizingStatus.BudgetExhausted, sizing!.Value.Status);
    }

    [Fact]
    public void Capacity_UnknownExpertSize_FallsBackToTheTotalRatherThanGuessing()
    {
        int slots = HybridForwardPass.ResolveExpertSlotCapacity(-1, Placement(10 * MiB, 40 * MiB), 0, 4, 8, 2, out var sizing);

        Assert.Equal(32, slots);
        Assert.Equal(MoeCacheSizingStatus.UnknownExpertSize, sizing!.Value.Status);
    }
}
