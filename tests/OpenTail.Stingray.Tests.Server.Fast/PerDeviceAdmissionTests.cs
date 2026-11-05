namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>Plan 6 G2: accelerator capacity and resident bytes are kept per backend and device, not as one Vulkan-only global sum.</summary>
public sealed class PerDeviceAdmissionTests
{
    private static readonly AcceleratorTarget Vk = new("vulkan", -1);
    private static readonly AcceleratorTarget Vk1 = new("vulkan", 1);
    private static readonly AcceleratorTarget Cuda = new("cuda", 0);

    private static LoadedEngine Load(string id, string backend, int? device = null) =>
        new(new FakeInferenceEngine(id), "qwen2", null, RuntimeResolution: new ServerRuntimeResolution(backend, "fake", "gguf", 512, DeviceIndex: device));

    private static ModelRuntimeManager Manager(Dictionary<string, (string Backend, int? Device)> placement, Func<ModelId, long>? bytes = null) =>
        new(id => Load(id.Value, placement[id.Value].Backend, placement[id.Value].Device), estimateBytes: bytes ?? (_ => 1000));

    private static HostResourceBudget Budget(ModelRuntimeManager m, Dictionary<AcceleratorTarget, long?> capacity) =>
        new(m, 1.0, t => capacity.GetValueOrDefault(t));

    [Fact]
    public void From_NormalisesBackendAndDevice()
    {
        Assert.Equal(new AcceleratorTarget("cuda", 0), AcceleratorTarget.From("cuda", 3));   // one CUDA device per process
        Assert.Equal(new AcceleratorTarget("vulkan", -1), AcceleratorTarget.From("vulkan", null));
        Assert.Equal(new AcceleratorTarget("vulkan", 2), AcceleratorTarget.From("vulkan", 2));
        Assert.Null(AcceleratorTarget.From("cpu", 0));
        Assert.Null(AcceleratorTarget.From(null, null));
    }

    [Fact]
    public async Task RuntimeAccelerator_ComesFromTheResolutionsBackendAndDevice()
    {
        using var m = Manager(new() { ["a"] = ("vulkan", 1), ["b"] = ("cpu", null), ["c"] = ("cuda", 0) });
        using var a = await m.AcquireAsync(ModelId.Canonicalize("a"));
        using var b = await m.AcquireAsync(ModelId.Canonicalize("b"));
        using var c = await m.AcquireAsync(ModelId.Canonicalize("c"));
        Assert.Equal(Vk1, a.Runtime.Accelerator);
        Assert.Null(b.Runtime.Accelerator);
        Assert.Equal(Cuda, c.Runtime.Accelerator);
    }

    [Fact]
    public async Task AVulkanResident_DoesNotUseUpACudaCandidatesCapacity()
    {
        using var m = Manager(new() { ["gpuA"] = ("vulkan", null) });
        using var h = await m.AcquireAsync(ModelId.Canonicalize("gpuA"));   // 1000 bytes resident on vulkan
        var budget = Budget(m, new() { [Vk] = 1200, [Cuda] = 1200 });

        Assert.Equal(ResourceAdmission.InsufficientAcceleratorMemory, budget.EstimateAdmission(0, 500, Vk));
        Assert.Equal(ResourceAdmission.Allowed, budget.EstimateAdmission(0, 500, Cuda));
    }

    [Fact]
    public async Task TwoVulkanDevices_AreCountedSeparately()
    {
        using var m = Manager(new() { ["onZero"] = ("vulkan", 0), ["onOne"] = ("vulkan", 1) });
        using var h = await m.AcquireAsync(ModelId.Canonicalize("onZero"));
        var zero = new AcceleratorTarget("vulkan", 0);
        var budget = Budget(m, new() { [zero] = 1200, [Vk1] = 1200 });

        Assert.Equal(ResourceAdmission.InsufficientAcceleratorMemory, budget.EstimateAdmission(0, 500, zero));
        Assert.Equal(ResourceAdmission.Allowed, budget.EstimateAdmission(0, 500, Vk1));
    }

    [Fact]
    public void UnknownCapacityForTheTarget_Admits()
    {
        using var m = Manager([]);
        var budget = Budget(m, new() { [Cuda] = null });
        Assert.Equal(ResourceAdmission.Allowed, budget.EstimateAdmission(0, long.MaxValue / 4, Cuda));
    }

    [Fact]
    public async Task Snapshot_ListsEachResidentTargetWithItsOwnCapacityAndBytes()
    {
        using var m = Manager(new() { ["a"] = ("vulkan", null), ["b"] = ("cuda", null) });
        using var a = await m.AcquireAsync(ModelId.Canonicalize("a"));
        using var b = await m.AcquireAsync(ModelId.Canonicalize("b"));
        var snap = Budget(m, new() { [Vk] = 5000, [Cuda] = 9000 }).GetCurrent();

        Assert.Contains(snap.Accelerators!, x => x.Target == Vk && x.CapacityBytes == 5000 && x.ResidentBytes == 1000);
        Assert.Contains(snap.Accelerators!, x => x.Target == Cuda && x.CapacityBytes == 9000 && x.ResidentBytes == 1000);
    }

    [Fact]
    public async Task Manager_PassesTheCandidateTargetToAdmission_AndTheErrorNamesIt()
    {
        using var resident = Manager(new() { ["gpuA"] = ("vulkan", null) });
        using var h = await resident.AcquireAsync(ModelId.Canonicalize("gpuA"));
        var budget = Budget(resident, new() { [Vk] = 1200 });
        using var m = new ModelRuntimeManager(
            id => Load(id.Value, "vulkan"), estimateBytes: _ => 1000,
            estimateAcceleratorBytes: _ => 500, estimateAcceleratorTarget: _ => Vk)
        { ResourceBudget = new HostOnlyIgnoringBudget(budget) };

        var ex = await Assert.ThrowsAsync<InsufficientResourcesException>(async () => await m.AcquireAsync(ModelId.Canonicalize("candidate")));
        Assert.Equal(Vk, ex.CandidateAcceleratorTarget);
        Assert.Contains("on vulkan", ex.Message);
    }

    /// <summary>Admission against the resident manager's snapshot (host bytes ignored) for the manager under test.</summary>
    private sealed class HostOnlyIgnoringBudget(HostResourceBudget inner) : IResourceBudget
    {
        public ResourceSnapshot GetCurrent() => inner.GetCurrent();
        public ResourceAdmission EstimateAdmission(long m, long a = 0) => inner.EstimateAdmission(m, a);
        public ResourceAdmission EstimateAdmission(long m, long a, AcceleratorTarget? t) => inner.EstimateAdmission(0, a, t);
    }

    [Fact]
    public void Estimators_ShareOnePlanPerModel_AndCarryTheTarget()
    {
        int calls = 0;
        var opts = new OpenTailStingrayServerOptions { ModelPath = "x.gguf" };
        var (bytes, target) = ServiceCollectionExtensions.CreateAcceleratorEstimators(opts, _ => { calls++; return (4096, Cuda); });
        var id = ModelId.Canonicalize("x.gguf");

        Assert.Equal(4096, bytes(id));
        Assert.Equal(Cuda, target(id));
        Assert.Equal(1, calls);
    }
}
