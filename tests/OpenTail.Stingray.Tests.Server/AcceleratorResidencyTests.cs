using OpenTail.Stingray.Engine;

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

    [Fact]
    public async Task RealVulkanModel_PartialOffload_AcceleratorEstimate_IsSmallerThanFullFileSize()
    {
        // docs/032 Phase 3 Slice 8 follow-up: a hybrid/partial-offload model previously reported
        // the whole file size as its accelerator-resident estimate (a documented overestimate).
        // With GpuWeightBytesExact threaded through from TierPlanner's own placement, only a few
        // GPU-resident layers should report a meaningfully smaller figure than the full file —
        // proving the precise value is actually being used, not silently falling back.
        string? modelPath = FindModel("Qwen3-0.6B-Q8_0.gguf"); // 28 layers total (qwen3.block_count)
        Assert.SkipUnless(modelPath is not null,
            "Qwen3-0.6B-Q8_0.gguf is required for this accelerator-residency test.");
        SkipUnlessVulkanAvailable();

        using var manager = new ModelRuntimeManager(id => InferenceEngineLoader.Load(
            new OpenTailStingrayServerOptions
            {
                ModelPath = id.Value,
                Backend = ServerBackend.Vulkan,
                NGpuLayers = 4, // well short of 28 — forces the genuinely-partial hybrid path
                ContextSize = 512,
            }));

        using var handle = await manager.AcquireAsync(ModelId.Canonicalize(modelPath!));

        long fileSize = new FileInfo(modelPath!).Length;
        Assert.True(handle.Runtime.IsAcceleratorResident);
        Assert.NotNull(handle.Runtime.Loaded.GpuWeightBytesExact);
        long estimate = handle.Runtime.AcceleratorResidentBytesEstimate;
        Assert.True(estimate > 0, $"expected a positive accelerator byte estimate, got {estimate}");
        Assert.True(estimate < fileSize,
            $"partial offload (4/28 layers) should report meaningfully less than the full file " +
            $"size ({fileSize:N0} bytes), got {estimate:N0} bytes — looks like the fix isn't wired up.");
    }

    // ── Full end-to-end admission through the real HostResourceBudget ───────

    [Fact]
    public async Task RealVulkanModel_ResourceBudgetEnabled_ReasonableAcceleratorEstimate_LoadsNormally()
    {
        string? modelPath = FindModel("Qwen3-0.6B-Q8_0.gguf");
        Assert.SkipUnless(modelPath is not null,
            "Qwen3-0.6B-Q8_0.gguf is required for this accelerator-residency test.");
        SkipUnlessVulkanAvailable();

        // HostResourceBudget needs `manager` itself (for Snapshot()-based resident-byte totals),
        // so ResourceBudget is set as a second step after construction — same two-step pattern
        // the production DI wiring uses (ServiceCollectionExtensions.cs).
        using var manager = new ModelRuntimeManager(
            id => InferenceEngineLoader.Load(new OpenTailStingrayServerOptions
            {
                ModelPath = id.Value,
                Backend = ServerBackend.Vulkan,
                NGpuLayers = -1,
                ContextSize = 512,
            }),
            // A real, modest estimate (the file's own size) — proves resource admission being
            // ON doesn't accidentally block an ordinary, easily-fittable real load.
            estimateAcceleratorBytes: id => new FileInfo(id.Value).Length);
        manager.ResourceBudget = new HostResourceBudget(manager);

        using var handle = await manager.AcquireAsync(ModelId.Canonicalize(modelPath!));

        Assert.True(handle.Runtime.IsAcceleratorResident);
        Assert.Equal(0, manager.GetStats().AdmissionRejects);
    }

    [Fact]
    public async Task RealVulkanModel_ResourceBudgetEnabled_AbsurdAcceleratorEstimate_ThrowsInsufficientAcceleratorMemory()
    {
        string? modelPath = FindModel("Qwen3-0.6B-Q8_0.gguf");
        Assert.SkipUnless(modelPath is not null,
            "Qwen3-0.6B-Q8_0.gguf is required for this accelerator-residency test.");
        SkipUnlessVulkanAvailable();

        using var manager = new ModelRuntimeManager(
            id => InferenceEngineLoader.Load(new OpenTailStingrayServerOptions
            {
                ModelPath = id.Value,
                Backend = ServerBackend.Vulkan,
                NGpuLayers = -1,
                ContextSize = 512,
            }),
            // No real GPU has this much VRAM — the real HostResourceBudget below must reject
            // this before InferenceEngineLoader.Load ever runs (no wasted real GPU load attempt).
            estimateAcceleratorBytes: _ => long.MaxValue / 4);
        manager.ResourceBudget = new HostResourceBudget(manager);

        var ex = await Assert.ThrowsAsync<InsufficientResourcesException>(
            async () => await manager.AcquireAsync(ModelId.Canonicalize(modelPath!)));

        Assert.Equal(ResourceAdmission.InsufficientAcceleratorMemory, ex.Reason);
        Assert.Equal(long.MaxValue / 4, ex.CandidateAcceleratorBytes);
        Assert.Empty(manager.Snapshot()); // rejected before any load started — nothing resident
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
