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
    public async Task RealVulkanModel_PartialOffload_PlansFewerAcceleratorBytesThanFullOffload()
    {
        // docs/032 Phase 3 Slice 8 follow-up: a hybrid/partial-offload model previously reported
        // the whole file size as its accelerator-resident estimate (a documented overestimate).
        // With GpuWeightBytesExact threaded through from TierPlanner's own placement, only a few
        // GPU-resident layers should report a meaningfully smaller figure than the full file —
        // proving the precise value is actually being used, not silently falling back.
        // A14 (2026-11-24): the file-size premise is wrong for a tied output. The planner counts token_embd once as F32 (622 MB for this
        // model), so 4 of 28 layers plan about 707 MB, above the 639 MB file, and the live Vulkan backend holds the same (measured with the
        // B1 live figure: planned 706,659,840 vs live 706,668,036 bytes at 4 layers; 1,208,624,640 vs 1,208,645,380 at 28). So this compares
        // partial offload with full offload of the same model instead: fewer GPU layers must plan fewer accelerator bytes.
        string? modelPath = FindModel("Qwen3-0.6B-Q8_0.gguf"); // 28 layers total (qwen3.block_count)
        Assert.SkipUnless(modelPath is not null,
            "Qwen3-0.6B-Q8_0.gguf is required for this accelerator-residency test.");
        SkipUnlessVulkanAvailable();

        // well short of 28 — forces the genuinely-partial hybrid path
        long partial = await LoadAcceleratorEstimateAsync(modelPath!, nGpuLayers: 4);
        long full = await LoadAcceleratorEstimateAsync(modelPath!, nGpuLayers: 28);

        Assert.True(partial > 0, $"expected a positive accelerator byte estimate for 4/28 layers, got {partial}");
        Assert.True(full > 0, $"expected a positive accelerator byte estimate for 28/28 layers, got {full}");
        Assert.True(partial < full,
            $"partial offload (4/28 layers) should plan fewer accelerator bytes than full offload (28/28), " +
            $"got {partial:N0} vs {full:N0} bytes.");
    }

    /// <summary>Loads the model on the Vulkan backend with <paramref name="nGpuLayers"/> GPU layers; returns the accelerator estimate the loaded runtime records.</summary>
    private static async Task<long> LoadAcceleratorEstimateAsync(string modelPath, int nGpuLayers)
    {
        using var manager = new ModelRuntimeManager(id => InferenceEngineLoader.Load(
            new OpenTailStingrayServerOptions
            {
                ModelPath = id.Value,
                Backend = ServerBackend.Vulkan,
                NGpuLayers = nGpuLayers,
                ContextSize = 512,
            }));

        using var handle = await manager.AcquireAsync(ModelId.Canonicalize(modelPath));

        Assert.True(handle.Runtime.IsAcceleratorResident);
        Assert.NotNull(handle.Runtime.Loaded.GpuWeightBytesExact);
        return handle.Runtime.AcceleratorResidentBytesEstimate;
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
    /// this goes through the real DI wiring, the real planner and a real model. The equality proves the wiring and that both numbers come from the same plan (the dense
    /// and the hybrid-GDN full-offload branch both pass), NOT that the figure matches device memory: both are the planner's weight figure, which is a lower bound
    /// (plan 6 F2 holds the measurement).
    /// </summary>
    [Theory]
    [InlineData("Qwen3-0.6B-Q8_0.gguf")]        // dense: the forward pass the figure was first checked on
    [InlineData("Qwen3.5-0.8B-Q8_0.gguf")]      // hybrid Gated-DeltaNet (qwen35): full offload runs HybridGdnForwardPass, whose placement is not TierPlanner-driven
    public async Task RealVulkanModel_ProductionWiring_PassesTheCandidateEstimate_ThatEqualsTheLoadedResidentFigure(string modelFile)
    {
        string? modelPath = FindModelAnywhere(modelFile);
        Assert.SkipUnless(modelPath is not null, $"{modelFile} is required for this test.");
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

    /// <summary>Resolved through ModelLocator (repo models folders first, then the model drives).</summary>
    private static string? FindModelAnywhere(string fileName) =>
        OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(fileName);

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
