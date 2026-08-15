using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Server;

namespace OpenTail.Stingray.Tests.Server;

/// <summary>
/// docs/032-multi-model-inference-runtime-plan.md Phase 3, per-runtime accelerator (GPU)
/// residency tracking. <c>ModelRuntimeManagerTests</c> (Tests.Server.Fast) proves
/// <c>ModelRuntime.IsAcceleratorResident</c>/<c>AcceleratorResidentBytesEstimate</c> against fake
/// <c>LoadedEngine</c>s with a hand-set <c>RuntimeResolution</c> — this proves the real thing:
/// that <c>InferenceEngineLoader</c>'s actual Vulkan dispatch branch produces a
/// <c>RuntimeResolution</c> that reports residency correctly, end to end, against real hardware
/// and a real model — not just that the plumbing forwards a value someone else set.
/// </summary>
public sealed class AcceleratorResidencyTests
{
    [Fact]
    public async Task RealVulkanModel_ReportsAcceleratorResident_WithPositiveByteEstimate()
    {
        string? modelPath = FindModel("Qwen3-0.6B-Q8_0.gguf");
        Assert.SkipUnless(modelPath is not null,
            "Qwen3-0.6B-Q8_0.gguf is required for this accelerator-residency test.");

        // Same construction-failure guard VulkanInitTests.CreateBackendOrSkip uses, checked
        // BEFORE going anywhere near ModelRuntimeManager's async load machinery — simpler and
        // more certain than trying to make Assert.Skip's special exception survive a round trip
        // through RunLoad's catch-all -> TaskCompletionSource.SetException -> re-await.
        SkipUnlessVulkanAvailable();

        using var manager = new ModelRuntimeManager(id => InferenceEngineLoader.Load(
            new OpenTailStingrayServerOptions
            {
                ModelPath = id.Value,
                Backend = ServerBackend.Vulkan,
                NGpuLayers = -1,
                ContextSize = 512,
            }));

        using var handle = await manager.AcquireAsync(ModelId.Canonicalize(modelPath!));

        Assert.True(handle.Runtime.IsAcceleratorResident);
        Assert.True(handle.Runtime.AcceleratorResidentBytesEstimate > 0);
        Assert.Equal("vulkan", handle.Runtime.Loaded.RuntimeResolution?.Backend);

        var stats = manager.Snapshot();
        var entry = Assert.Single(stats);
        Assert.True(entry.IsAcceleratorResident);
        Assert.Equal(handle.Runtime.AcceleratorResidentBytesEstimate, entry.AcceleratorResidentBytesEstimate);
        Assert.Equal(entry.AcceleratorResidentBytesEstimate, manager.GetStats().EstimatedAcceleratorResidentBytes);
    }

    [Fact]
    public async Task RealCpuModel_ReportsNotAcceleratorResident()
    {
        string? modelPath = FindModel("Qwen3-0.6B-Q8_0.gguf");
        Assert.SkipUnless(modelPath is not null,
            "Qwen3-0.6B-Q8_0.gguf is required for this accelerator-residency test.");

        using var manager = new ModelRuntimeManager(id => InferenceEngineLoader.Load(
            new OpenTailStingrayServerOptions
            {
                ModelPath = id.Value,
                Backend = ServerBackend.Cpu,
                NGpuLayers = 0,
                ContextSize = 512,
            }));

        using var handle = await manager.AcquireAsync(ModelId.Canonicalize(modelPath!));

        Assert.False(handle.Runtime.IsAcceleratorResident);
        Assert.Equal(0, handle.Runtime.AcceleratorResidentBytesEstimate);
        Assert.Equal("cpu", handle.Runtime.Loaded.RuntimeResolution?.Backend);
    }

    private static void SkipUnlessVulkanAvailable()
    {
        try
        {
            using var probe = new OpenTail.Stingray.Vulkan.VulkanBackend();
        }
        catch (Exception ex)
        {
            Assert.Skip($"Vulkan device could not be created in this environment: {ex.Message}");
        }
    }

    private static string? FindModel(string fileName)
    {
        string directory = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            string candidate = Path.Combine(directory, "models", fileName);
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(directory);
            if (parent is null) break;
            directory = parent.FullName;
        }
        return null;
    }
}
