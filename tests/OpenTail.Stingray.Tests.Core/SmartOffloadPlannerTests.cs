using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

public sealed class SmartOffloadPlannerTests
{
    [Fact]
    public void HardwareCapabilities_ProbesCpuFeaturesInstantly()
    {
        var topology = HardwareCapabilities.Current;

        Assert.NotNull(topology);
        Assert.NotNull(topology.Cpu);
        Assert.True(topology.Cpu.LogicalCores > 0);
        Assert.True(topology.Cpu.PhysicalCores > 0);
        Assert.True(topology.Cpu.PhysicalCores <= topology.Cpu.LogicalCores);
    }

    [Fact]
    public void SmartOffloadPlanner_FullVram_ProducesDiscreteGpuFull()
    {
        var topology = new HardwareTopology
        {
            Cpu = new CpuProfile
            {
                LogicalCores = 16,
                PhysicalCores = 12,
                HasAvx2 = true,
                HasAvx512 = true
            },
            Gpus =
            [
                new GpuProfile
                {
                    Index = 0,
                    Name = "NVIDIA RTX 4090",
                    Backend = "cuda",
                    DeviceType = GpuDeviceType.Discrete,
                    TotalMemoryBytes = 24L * 1024 * 1024 * 1024,
                    AvailableMemoryBytes = 22L * 1024 * 1024 * 1024
                }
            ]
        };

        long modelSize = 4L * 1024 * 1024 * 1024; // 4GB model
        int layers = 32;

        var plan = SmartOffloadPlanner.Plan(modelSize, layers, contextTokens: 2048, topology);

        Assert.Equal(ExecutionStrategy.DiscreteGpuFull, plan.Strategy);
        Assert.Equal(32, plan.GpuLayersToOffload);
        Assert.Equal("cuda", plan.RecommendedBackend);
        Assert.Equal(12, plan.RecommendedThreads);
        Assert.Single(plan.DeviceAllocations);
        Assert.Equal(32, plan.DeviceAllocations[0].LayerCount);
    }

    [Fact]
    public void SmartOffloadPlanner_MultiGpu_PoolsVramAndSplitsLayersProportionally()
    {
        var topology = new HardwareTopology
        {
            Cpu = new CpuProfile
            {
                LogicalCores = 16,
                PhysicalCores = 12,
                HasAvx2 = true
            },
            Gpus =
            [
                new GpuProfile
                {
                    Index = 0,
                    Name = "NVIDIA RTX 3060",
                    Backend = "cuda",
                    DeviceType = GpuDeviceType.Discrete,
                    TotalMemoryBytes = 12L * 1024 * 1024 * 1024,
                    AvailableMemoryBytes = 10L * 1024 * 1024 * 1024 // 10GB usable
                },
                new GpuProfile
                {
                    Index = 1,
                    Name = "AMD Radeon RX 6600",
                    Backend = "vulkan",
                    DeviceType = GpuDeviceType.Discrete,
                    TotalMemoryBytes = 8L * 1024 * 1024 * 1024,
                    AvailableMemoryBytes = 6L * 1024 * 1024 * 1024 // 6GB usable
                }
            ]
        };

        long modelSize = 14L * 1024 * 1024 * 1024; // 14GB model (cannot fit on either card alone!)
        int layers = 32;

        var plan = SmartOffloadPlanner.Plan(modelSize, layers, contextTokens: 2048, topology);

        Assert.Equal(ExecutionStrategy.MultiGpuPooled, plan.Strategy);
        Assert.Equal(32, plan.GpuLayersToOffload); // 100% of layers offloaded across the 2 pooled GPUs!
        Assert.Equal(2, plan.DeviceAllocations.Count);

        var gpu0 = plan.DeviceAllocations[0];
        var gpu1 = plan.DeviceAllocations[1];

        Assert.Equal(0, gpu0.DeviceIndex);
        Assert.Equal("cuda", gpu0.Backend);
        Assert.True(gpu0.LayerCount > gpu1.LayerCount); // GPU 0 has more VRAM, gets more layers!
        Assert.Equal(0, gpu0.StartLayer);

        Assert.Equal(1, gpu1.DeviceIndex);
        Assert.Equal("vulkan", gpu1.Backend);
        Assert.Equal(gpu0.EndLayer + 1, gpu1.StartLayer); // Continuous layer range!
        Assert.Equal(31, gpu1.EndLayer);
        Assert.Equal(32, gpu0.LayerCount + gpu1.LayerCount);
    }

    [Fact]
    public void SmartOffloadPlanner_TightVram_ProducesPartialOffload()
    {
        var topology = new HardwareTopology
        {
            Cpu = new CpuProfile
            {
                LogicalCores = 8,
                PhysicalCores = 6,
                HasAvx2 = true
            },
            Gpus =
            [
                new GpuProfile
                {
                    Index = 0,
                    Name = "NVIDIA RTX 3060",
                    Backend = "cuda",
                    DeviceType = GpuDeviceType.Discrete,
                    TotalMemoryBytes = 6L * 1024 * 1024 * 1024,
                    AvailableMemoryBytes = 4L * 1024 * 1024 * 1024 // 4GB free
                }
            ]
        };

        long modelSize = 8L * 1024 * 1024 * 1024; // 8GB model
        int layers = 32;

        var plan = SmartOffloadPlanner.Plan(modelSize, layers, contextTokens: 2048, topology);

        Assert.Equal(ExecutionStrategy.DiscreteGpuPartial, plan.Strategy);
        Assert.InRange(plan.GpuLayersToOffload, 1, 31);
        Assert.Equal("cuda", plan.RecommendedBackend);
    }

    [Fact]
    public void SmartOffloadPlanner_IntegratedGpu_ProducesHybridPlan()
    {
        var topology = new HardwareTopology
        {
            Cpu = new CpuProfile
            {
                LogicalCores = 16,
                PhysicalCores = 8,
                HasAvx2 = true
            },
            Gpus =
            [
                new GpuProfile
                {
                    Index = 0,
                    Name = "AMD Radeon Vega Graphics",
                    Backend = "vulkan",
                    DeviceType = GpuDeviceType.Integrated,
                    TotalMemoryBytes = 2L * 1024 * 1024 * 1024,
                    AvailableMemoryBytes = 1500L * 1024 * 1024
                }
            ]
        };

        long modelSize = 2L * 1024 * 1024 * 1024;
        int layers = 24;

        var plan = SmartOffloadPlanner.Plan(modelSize, layers, contextTokens: 2048, topology);

        Assert.Equal(ExecutionStrategy.IntegratedGpuHybrid, plan.Strategy);
        Assert.Equal(0, plan.GpuLayersToOffload); // Memory bound GEMM on CPU to avoid bus contention
        Assert.Equal("vulkan", plan.RecommendedBackend);
        Assert.Contains("APU/iGPU Hybrid", plan.Description);
    }

    [Fact]
    public void SmartOffloadPlanner_NoGpu_ProducesCpuSimdPlan()
    {
        var topology = new HardwareTopology
        {
            Cpu = new CpuProfile
            {
                LogicalCores = 8,
                PhysicalCores = 4,
                HasAvx2 = true,
                HasAvx512 = false
            },
            Gpus = []
        };

        var plan = SmartOffloadPlanner.Plan(modelSizeBytes: 1024 * 1024 * 500, totalLayers: 16, contextTokens: 2048, topology);

        Assert.Equal(ExecutionStrategy.CpuSimdOnly, plan.Strategy);
        Assert.Equal(0, plan.GpuLayersToOffload);
        Assert.Equal("cpu", plan.RecommendedBackend);
        Assert.Equal(4, plan.RecommendedThreads);
    }
}
