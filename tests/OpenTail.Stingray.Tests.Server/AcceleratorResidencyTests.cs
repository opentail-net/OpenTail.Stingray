using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Server;

/// <summary>
/// docs/3-product-and-runtime/032-multi-model-inference-runtime-plan.md Phase 3, per-runtime accelerator (GPU)
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

    /// <summary>
    /// Plan 6 C1: the production wiring hands the manager a candidate accelerator estimate BEFORE it loads, and that figure is what the loaded runtime records
    /// afterwards (both are the weights the execution plan places on the GPU). Before this, <c>AddOpenTailStingray</c> built the manager without an estimator,
    /// so every candidate counted as 0 and accelerator-aware admission could never run in a real server. The unit tests (Tests.Server.Fast) pass a fake estimator;
    /// this goes through the real DI wiring, the real planner and a real model.
    /// </summary>
    [Fact]
    public async Task RealVulkanModel_ProductionWiring_PassesTheCandidateEstimate_ThatEqualsTheLoadedResidentFigure()
    {
        string? modelPath = FindModelAnywhere("Qwen3-0.6B-Q8_0.gguf");
        Assert.SkipUnless(modelPath is not null, "Qwen3-0.6B-Q8_0.gguf is required for this test.");
        SkipUnlessVulkanAvailable();

        var services = new ServiceCollection();
        services.AddOpenTailStingray(o =>
        {
            o.ModelPath = modelPath;
            o.Backend = ServerBackend.Vulkan;
            o.NGpuLayers = -1;
            o.ContextSize = 512;
        });
        using var provider = services.BuildServiceProvider();
        var manager = (ModelRuntimeManager)provider.GetRequiredService<IModelRuntimeManager>();
        var budget = new RecordingBudget();
        manager.ResourceBudget = budget;

        var clock = System.Diagnostics.Stopwatch.StartNew();
        using var handle = await manager.AcquireAsync(ModelId.Canonicalize(modelPath!));
        TimeSpan took = clock.Elapsed;

        long candidate = Assert.Single(budget.CandidateAcceleratorBytes);
        Assert.True(candidate > 0, "the production wiring passed 0 accelerator bytes for a GPU-resident model");
        Assert.Equal(handle.Runtime.AcceleratorResidentBytesEstimate, candidate);
        Assert.True(took > TimeSpan.FromSeconds(0.2), $"the load took {took.TotalMilliseconds:F0} ms; a real model takes longer");
        Console.Error.WriteLine($"[AcceleratorResidencyTests] candidate {candidate:N0} bytes == resident {handle.Runtime.AcceleratorResidentBytesEstimate:N0} bytes; load {took.TotalSeconds:F1} s");
    }

    /// <summary>Always admits, and records what the manager asked it to judge.</summary>
    private sealed class RecordingBudget : IResourceBudget
    {
        public List<long> CandidateAcceleratorBytes { get; } = [];
        public ResourceSnapshot GetCurrent() => new(0, 0, null, 0, 0);
        public ResourceAdmission EstimateAdmission(long candidateModelBytes, long candidateAcceleratorBytes = 0)
        {
            CandidateAcceleratorBytes.Add(candidateAcceleratorBytes);
            return ResourceAdmission.Allowed;
        }
    }

    /// <summary><c>models/</c> or <c>models/_models/</c> above the test, else a <c>_models</c> folder at the root of any fixed drive.</summary>
    private static string? FindModelAnywhere(string fileName)
    {
        for (var dir = new DirectoryInfo(Directory.GetCurrentDirectory()); dir is not null; dir = dir.Parent)
            foreach (string candidate in new[] { Path.Combine(dir.FullName, "models", fileName), Path.Combine(dir.FullName, "models", "_models", fileName) })
                if (File.Exists(candidate)) return candidate;
        foreach (var drive in DriveInfo.GetDrives().Where(d => d.DriveType == DriveType.Fixed && d.IsReady))
        {
            string candidate = Path.Combine(drive.RootDirectory.FullName, "_models", fileName);
            if (File.Exists(candidate)) return candidate;
        }
        return null;
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

    private static string? FindModel(string fileName) => FindModelAnywhere(fileName);   // one lookup for every test here: models/, models/_models/, then a _models folder on each fixed drive
}
