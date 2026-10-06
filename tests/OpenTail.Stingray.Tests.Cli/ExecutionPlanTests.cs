
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

    [Fact]
    public void StaticPlanCommand_ProducesSchemaV2Plan()
    {
        string path = Path.Combine(Path.GetTempPath(), $"opentail-schema2-plan-{Guid.NewGuid():N}.gguf");
        try
        {
            using (var stream = File.Create(path))
            using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: false))
            {
                writer.Write(0x46554747u);
                writer.Write(3u);
                writer.Write(0UL);
                writer.Write(6UL);

                WriteKv(writer, "general.architecture", GgufValueType.String, "llama");
                WriteKv(writer, "llama.block_count", GgufValueType.UInt32, 16u);
                WriteKv(writer, "llama.context_length", GgufValueType.UInt32, 2048u);
                WriteKv(writer, "llama.embedding_length", GgufValueType.UInt32, 128u);
                WriteKv(writer, "llama.attention.head_count", GgufValueType.UInt32, 4u);
                WriteKv(writer, "llama.feed_forward_length", GgufValueType.UInt32, 256u);
            }

            var settings = new StaticPlanCommand.Settings { ModelPath = path };
            var profile = new StaticPlanProfile();
            var config = StaticPlanConfiguration.Resolve(settings, profile, _ => null);
            using var model = GgufModel.Open(path);
            var report = StaticPlanReport.Create(path, model, config, StaticPlanRuntimeFacts.Detect(noGpuProbe: true), includePlacement: false);

            Assert.NotNull(report.ExecutionPlan);
            Assert.Equal(2, report.ExecutionPlan.SchemaVersion);
            Assert.Equal("llama", report.ExecutionPlan.Provenance?.TargetArchitecture);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    private static void WriteKv(BinaryWriter writer, string key, GgufValueType type, object value)
    {
        byte[] keyBytes = System.Text.Encoding.UTF8.GetBytes(key);
        writer.Write((ulong)keyBytes.Length);
        writer.Write(keyBytes);
        writer.Write((uint)type);
        if (type == GgufValueType.String)
        {
            byte[] strBytes = System.Text.Encoding.UTF8.GetBytes((string)value);
            writer.Write((ulong)strBytes.Length);
            writer.Write(strBytes);
        }
        else if (type == GgufValueType.UInt32)
        {
            writer.Write((uint)value);
        }
    }
}
