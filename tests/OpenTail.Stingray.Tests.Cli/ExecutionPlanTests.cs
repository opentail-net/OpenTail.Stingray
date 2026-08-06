using OpenTail.Stingray.Cli;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Cli;

public sealed class ExecutionPlanTests
{
    [Fact]
    public void AutoPlanInputs_UsesContextPin_NotGenerationLimit_AndHonoursDeviceNone()
    {
        var inputs = RunCommand.ResolveAutoPlanInputs(new RunCommand.Settings
        {
            Device = "none",
            NGpuLayers = -1,
            CtxSize = 4096,
            NPredict = 37,
            Backend = "cuda"
        });

        Assert.Equal("cpu", inputs.Backend);
        Assert.Equal(0, inputs.GpuLayers);
        Assert.Equal(4096, inputs.ContextSize);
    }

    [Fact]
    public void AutoPlanInputs_LeavesUnsetContextForGoalResolution()
    {
        var inputs = RunCommand.ResolveAutoPlanInputs(new RunCommand.Settings
        {
            Backend = "vulkan",
            NGpuLayers = null,
            CtxSize = 0,
            NPredict = 512
        });

        Assert.Equal("vulkan", inputs.Backend);
        Assert.Null(inputs.GpuLayers);
        Assert.Null(inputs.ContextSize);
    }

    [Fact]
    public void CompactSummary_FormatsCleanly()
    {
        var plan = new ExecutionPlan(
            SchemaVersion: 1,
            ModelPath: "models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
            Goal: "balanced",
            Backend: "vulkan",
            GpuLayers: 24,
            TotalLayers: 24,
            ContextSize: 8192,
            KvDtype: "fp16",
            EstimatedVramMb: 1124.5,
            EstimatedRamMb: 0.0,
            Decisions: new List<ExecutionPlanDecisionDetail>
            {
                new("BACKEND", "vulkan", "Vulkan compute device detected.", "auto_planner")
            },
            Warnings: new List<string>()
        );

        string summary = plan.CompactSummary();

        Assert.Contains("SmolLM2-1.7B-Instruct-Q4_K_M.gguf", summary);
        Assert.Contains("ctx 8192", summary);
        Assert.Contains("full VULKAN GPU weights (24/24 layers)", summary);
        Assert.Contains("KV: fp16", summary);
        Assert.Contains("est. VRAM: 1124.5 MiB", summary);
    }
}
