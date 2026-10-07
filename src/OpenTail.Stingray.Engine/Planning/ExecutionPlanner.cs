#nullable enable

using System.Collections.Immutable;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine.Planning;

/// <summary>
/// Authoritative planner orchestrating candidate evaluation, hardware budgeting,
/// and pass selection into an immutable, inspectable <see cref="ExecutionPlan"/>.
/// Guarantees that runtime execution never encounters ambiguous or rediscovering policies.
/// </summary>
public static class ExecutionPlanner
{
    public static ExecutionPlan Plan(
        ModelDescription model,
        ExecutionRequest request,
        BackendCapabilities capabilities)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(capabilities);

        var decisions = ImmutableArray.CreateBuilder<ExecutionPlanDecisionDetail>();
        var warnings = ImmutableArray.CreateBuilder<string>();

        string resolvedGoal = NormalizeGoal(request.Goal);

        // 1. Candidate Evaluation & Backend Selection
        var (selectedBackend, backendDeviceName) = SelectBackend(
            model, request, capabilities, decisions, warnings);

        // 2. Context Size Selection
        int ctxSize = ResolveContextSize(
            model.PlanningFacts, resolvedGoal, request.PinnedContextSize, decisions, warnings);

        // 3. KV DType Selection
        DType kvDtype = ResolveKvDtype(
            resolvedGoal, request.PinnedKvDtype, decisions, warnings);

        // 4. Placement & Layer Offload via TierPlanner
        int? requestedGpuLayers = request.PinnedGpuLayers;
        if (requestedGpuLayers == -1)
        {
            requestedGpuLayers = model.PlanningFacts.NumLayers;
        }

        if (!requestedGpuLayers.HasValue && selectedBackend == ForwardPassBackend.Cpu)
        {
            requestedGpuLayers = 0;
        }

        var hardwareProfile = capabilities.HardwareProfile;
        if (selectedBackend == ForwardPassBackend.Cpu)
        {
            hardwareProfile = hardwareProfile with { VramBytes = 0 };
        }

        var placementResult = TierPlanner.Plan(
            model.PlanningFacts,
            hardwareProfile,
            turboQuant: request.TurboQuant,
            tqBits: request.TurboQuantBits,
            requestedCtxSize: ctxSize,
            tqFp32Window: request.TurboQuantFp32Window,
            kvDtype: kvDtype,
            pinGpuLayers: requestedGpuLayers);

        int resolvedGpuLayers = placementResult.GpuLayers;
        if (request.PinnedGpuLayers.HasValue)
        {
            decisions.Add(new("GPU_LAYERS", resolvedGpuLayers.ToString(),
                "User explicitly pinned GPU layer count.", "request_pin"));
        }
        else
        {
            decisions.Add(new("GPU_LAYERS", resolvedGpuLayers.ToString(),
                resolvedGpuLayers >= model.PlanningFacts.NumLayers
                    ? "Full GPU weight offload selected."
                    : $"Offloaded {resolvedGpuLayers}/{model.PlanningFacts.NumLayers} layers based on VRAM budget.",
                "auto_planner"));
        }

        // 5. Select ForwardPassKind via ForwardPassSelection
        string? unsupportedBackend = null;
        if (!string.IsNullOrEmpty(request.PinnedBackend) &&
            !string.Equals(request.PinnedBackend, "auto", StringComparison.OrdinalIgnoreCase))
        {
            string b = request.PinnedBackend.Trim().ToLowerInvariant();
            if (b is not ("cpu" or "cuda" or "vulkan"))
            {
                unsupportedBackend = request.PinnedBackend;
            }
        }

        var passReq = model.CreateForwardPassRequest(
            backend: selectedBackend,
            gpuLayers: resolvedGpuLayers,
            plannedGpuLayers: resolvedGpuLayers,
            cudaAvailable: capabilities.CudaAvailable,
            turboQuant: request.TurboQuant,
            turboQuantMode: request.TurboQuantMode,
            hasDraftModel: !string.IsNullOrEmpty(request.DraftModelPath),
            isContinuousBatching: request.BatchingMode == BatchingMode.Continuous,
            targetContextLength: ctxSize,
            allowUnverifiedArchitecture: request.AllowUnverifiedArchitecture,
            unsupportedBackendName: unsupportedBackend,
            hasDSparkModel: !string.IsNullOrEmpty(request.DSparkModelPath));

        var passDecision = ForwardPassSelection.Select(passReq);
        if (passDecision.IsRefused)
        {
            throw new NotSupportedException(passDecision.Refusal);
        }

        ForwardPassKind forwardPassKind = passDecision.Kind!.Value;
        decisions.Add(new("FORWARD_PASS", forwardPassKind.ToString(),
            passDecision.Notice ?? "Selected optimal forward pass.", "forward_pass_selection"));

        // 6. Build Sub-Plans
        int threadCount = request.ThreadCount > 0 ? request.ThreadCount : capabilities.RecommendedThreadCount;
        var backendPlan = new BackendPlan(
            Backend: selectedBackend,
            DeviceName: backendDeviceName,
            DeviceIndex: selectedBackend == ForwardPassBackend.Cpu ? 0 : request.DeviceIndex,
            CudaAvailable: capabilities.CudaAvailable,
            VulkanAvailable: capabilities.VulkanAvailable,
            ThreadCount: threadCount);

        var placementPlan = new PlacementPlan(
            GpuLayers: placementResult.GpuLayers,
            CpuLayers: placementResult.CpuLayers,
            TotalLayers: placementResult.GpuLayers + placementResult.CpuLayers,
            GpuWeightBytes: placementResult.GpuWeightBytes,
            CpuWeightBytes: placementResult.CpuWeightBytes,
            ExpertCacheBudgetBytes: placementResult.ExpertCacheBudgetBytes,
            MoeRoutedExpertBytes: placementResult.MoeRoutedExpertBytes,
            FixedWeightsOnCpu: model.PlanningFacts.ShouldKeepFixedWeightsOnCpu);

        int headDim = model.PlanningFacts.HeadDim > 0
            ? model.PlanningFacts.HeadDim
            : (model.PlanningFacts.EmbeddingDim > 0 && model.PlanningFacts.NumHeads > 0
                ? model.PlanningFacts.EmbeddingDim / model.PlanningFacts.NumHeads
                : 0);

        TqQuantizer resolvedTqQuantizer = TqQuantizer.LloydMax;
        if (request.TurboQuant)
        {
            string tqMode = (request.TurboQuantMode ?? "auto").Trim().ToLowerInvariant();
            string? blockedReason = TqSupport.KVarNBlockedReason(
                headDim,
                request.SnapKvEnabled,
                onGpu: placementResult.GpuLayers > 0,
                isVulkan: selectedBackend == ForwardPassBackend.Vulkan,
                cudaAvailable: capabilities.CudaAvailable,
                isMoE: model.PlanningFacts.IsMoE,
                window: request.TurboQuantFp32Window);

            if (tqMode == "kvarn")
            {
                if (blockedReason != null)
                {
                    throw new NotSupportedException($"TurboQuant KVarN is not supported: {blockedReason}");
                }
                resolvedTqQuantizer = TqQuantizer.KVarN;
            }
            else if (tqMode == "lloydmax")
            {
                resolvedTqQuantizer = TqQuantizer.LloydMax;
            }
            else // auto
            {
                resolvedTqQuantizer = blockedReason == null ? TqQuantizer.KVarN : TqQuantizer.LloydMax;
            }
        }

        float effectiveRopeTheta = model.PlanningFacts.RopeTheta;
        if (request.RopeFrequencyBase is { } rfb and > 0)
        {
            effectiveRopeTheta = rfb;
        }
        if (request.RopeFrequencyScale is { } rfs and > 0)
        {
            effectiveRopeTheta /= rfs;
        }

        var statePlan = new StatePlan(
            StateModel: model.Capabilities.StateModel,
            ContextLength: ctxSize,
            KvDType: kvDtype,
            TurboQuant: request.TurboQuant,
            TurboQuantMode: request.TurboQuantMode ?? "none",
            TurboQuantBits: request.TurboQuantBits,
            TurboQuantFp32Window: request.TurboQuantFp32Window,
            SnapKvEnabled: request.SnapKvEnabled,
            SnapKvBudget: request.SnapKvBudget,
            TqQuantizer: resolvedTqQuantizer,
            FlashAttention: request.FlashAttention,
            HeadDim: headDim,
            RopeFrequencyBase: request.RopeFrequencyBase,
            RopeFrequencyScale: request.RopeFrequencyScale,
            EffectiveRopeTheta: effectiveRopeTheta);

        var batchingPlan = new BatchingPlan(
            Mode: request.BatchingMode,
            MaxBatchSize: request.MaxBatchSize,
            MaxConcurrentSessions: request.MaxBatchSize,
            EnableSessions: request.EnableSessions);

        var speculationPlan = ResolveSpeculation(
            model, request, placementResult, hardwareProfile, forwardPassKind, decisions, warnings);

        var moePlan = new MoePlan(
            IsMoE: model.PlanningFacts.IsMoE,
            NumExperts: model.PlanningFacts.NumExperts,
            NumActiveExperts: model.PlanningFacts.NumActiveExperts,
            CpuMoe: request.CpuMoe,
            GpuMoePrefill: request.GpuMoePrefill,
            WarmPin: request.MoeWarmPin,
            WarmPinAfter: request.MoeWarmPinAfter,
            PredictPrefetch: request.MoePredictPrefetch,
            ExpertStatsPath: request.ExpertStatsPath);

        var modalityPlan = new ModalityPlan(
            SupportsVision: model.Capabilities.SupportsVision,
            SupportsEmbeddingInput: model.Capabilities.SupportsEmbeddingInput,
            MmprojPath: request.MmprojPath);

        double estVramMb = (placementResult.GpuWeightBytes + placementResult.GpuKvBytes) / (1024.0 * 1024.0);
        double estRamMb = placementResult.CpuWeightBytes / (1024.0 * 1024.0);

        var memoryPlan = new MemoryPlan(
            EstimatedVramMb: estVramMb,
            EstimatedRamMb: estRamMb,
            ScratchBytes: model.PlanningFacts.ScratchBytes);

        string primaryModelPath = request.ModelPath
            ?? model.PrimaryPath
            ?? model.Identity.Components.FirstOrDefault()?.RelativeName
            ?? "model";

        var provenance = new PlanProvenance(
            CreatedAtUtc: DateTime.UtcNow.ToString("o"),
            PlannerVersion: "2.0.0",
            PrimaryModelPath: primaryModelPath,
            TargetArchitecture: model.Semantics.Architecture,
            Goal: resolvedGoal);

        // 7. Construct ExecutionPlan v2
        var plan = ExecutionPlan.CreateV2(
            packageIdentity: model.Identity,
            forwardPassKind: forwardPassKind,
            backendPlan: backendPlan,
            placement: placementPlan,
            state: statePlan,
            batching: batchingPlan,
            speculation: speculationPlan,
            modality: modalityPlan,
            memory: memoryPlan,
            provenance: provenance,
            decisions: decisions.ToImmutable(),
            warnings: warnings.ToImmutable(),
            modelFormat: model.Semantics.Format,
            isExecutable: passDecision.Kind.HasValue,
            moe: moePlan);

        // 8. Validate Plan Invariants
        ExecutionPlanValidator.Validate(plan);

        return plan;
    }

    /// <summary>
    /// Resolves speculation, including the DSpark draft-head placement (GPU/CPU/off). This is the ONLY place
    /// DSpark placement is decided; <c>AttachDSpark</c>/the CLI runner just obey the recorded decision.
    /// </summary>
    private static SpeculationPlan ResolveSpeculation(
        ModelDescription model,
        ExecutionRequest request,
        LayerPlacement targetPlacement,
        HardwareProfile hardware,
        ForwardPassKind passKind,
        ImmutableArray<ExecutionPlanDecisionDetail>.Builder decisions,
        ImmutableArray<string>.Builder warnings)
    {
        if (string.IsNullOrWhiteSpace(request.DSparkModelPath))
        {
            return new SpeculationPlan(request.SpeculationMode, request.DraftModelPath, null);
        }

        if (model.Semantics.Format != ModelFormat.Gguf)
            throw new NotSupportedException("DSpark is not supported for SafeTensors packages.");

        string stPath = request.DSparkModelPath;
        if (Directory.Exists(stPath)) stPath = Path.Combine(stPath, "model.safetensors");
        if (!File.Exists(stPath))
            throw new FileNotFoundException($"DSpark model not found: {stPath}");
        string cfgPath = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(stPath))!, "config.json");
        if (!File.Exists(cfgPath))
            throw new FileNotFoundException($"DSpark config.json not found next to the safetensors: {cfgPath}");

        var cfg = DSparkConfig.FromJsonFile(cfgPath);
        var facts = model.PlanningFacts;
        if (cfg.VocabSize != facts.VocabSize || cfg.NumTargetLayers != facts.NumLayers
            || cfg.HiddenSize != facts.EmbeddingDim)
            throw new InvalidOperationException(
                $"DSpark head/target mismatch — head expects vocab {cfg.VocabSize}, " +
                $"{cfg.NumTargetLayers} target layers, hidden {cfg.HiddenSize}; target has " +
                $"vocab {facts.VocabSize}, {facts.NumLayers} layers, hidden {facts.EmbeddingDim}.");

        DSparkPlacement userPlace;
        try
        {
            userPlace = DSparkPlacementPlanner.ResolvePlacement(request.DSparkPlace);
        }
        catch (ArgumentException ex)
        {
            throw new NotSupportedException(ex.Message, ex);
        }

        long headGpu = CudaDSparkDraftModel.EstimateGpuResidentBytes(cfg);
        long headCpu = DSparkDraftModel.EstimateResidentBytes(cfg);
        long tapBytes = (long)targetPlacement.RecommendedCtxSize * cfg.TapDim * sizeof(float);
        var decision = DSparkPlacementPlanner.Plan(hardware, targetPlacement, headGpu, headCpu, userPlace, tapBytes);

        // A GPU draft needs the target's CudaBackend (shared stream orders tap producer and draft consumer).
        // Gpu -> Cpu -> Off: re-plan in Auto over a GPU-less profile so the RAM budget is actually checked.
        if (decision.Placement == DSparkPlacement.Gpu && passKind != ForwardPassKind.CudaDense)
        {
            decision = DSparkPlacementPlanner.Plan(
                hardware with { VramBytes = 0 }, targetPlacement, headGpu, headCpu, DSparkPlacement.Auto, tapBytes);
            warnings.Add("A GPU DSpark draft requires a full-CUDA target; re-planned for CPU — " + decision.Reason);
        }

        decisions.Add(new("DSPARK_PLACEMENT", decision.Placement.ToString(), decision.Reason,
            userPlace == DSparkPlacement.Auto ? "auto_planner" : "request_pin"));

        bool enabled = decision.Placement != DSparkPlacement.Off;
        return new SpeculationPlan(
            Mode: enabled ? SpeculationMode.DSpark : request.SpeculationMode,
            DraftModelPath: request.DraftModelPath,
            DSparkModelPath: request.DSparkModelPath,
            DSparkEnabled: enabled,
            DSparkPlacement: decision.Placement,
            DSparkPlacementReason: decision.Reason,
            DSparkHeadBytesGpu: headGpu,
            DSparkHeadBytesCpu: headCpu,
            DSparkTapBytes: tapBytes);
    }

    private static (ForwardPassBackend Backend, string DeviceName) SelectBackend(
        ModelDescription model,
        ExecutionRequest request,
        BackendCapabilities capabilities,
        ImmutableArray<ExecutionPlanDecisionDetail>.Builder decisions,
        ImmutableArray<string>.Builder warnings)
    {
        var descriptor = ArchitectureRegistry.Find(model.Semantics.Architecture);
        bool archSupportsCuda = descriptor is null || (descriptor.SupportedBackends & SupportedBackends.Cuda) != 0;
        bool archSupportsVulkan = descriptor is null || (descriptor.SupportedBackends & SupportedBackends.Vulkan) != 0;

        if (!string.IsNullOrEmpty(request.PinnedBackend) &&
            !string.Equals(request.PinnedBackend, "auto", StringComparison.OrdinalIgnoreCase))
        {
            string pinned = request.PinnedBackend.Trim().ToLowerInvariant();
            if (pinned == "cuda" && !archSupportsCuda)
            {
                throw new NotSupportedException($"Architecture '{model.Semantics.Architecture}' does not support CUDA backend: {descriptor?.BackendLimitation ?? "CUDA unsupported."}");
            }
            if (pinned == "vulkan" && !archSupportsVulkan)
            {
                throw new NotSupportedException($"Architecture '{model.Semantics.Architecture}' does not support Vulkan backend: {descriptor?.BackendLimitation ?? "Vulkan unsupported."}");
            }

            decisions.Add(new("BACKEND", pinned, "User explicitly pinned backend.", "request_pin"));

            return pinned switch
            {
                "cuda" => (ForwardPassBackend.Cuda, capabilities.CudaDeviceName ?? "CUDA Device 0"),
                "vulkan" => (ForwardPassBackend.Vulkan, capabilities.VulkanDeviceName ?? "Vulkan Device 0"),
                _ => (ForwardPassBackend.Cpu, "CPU Host")
            };
        }

        // Candidate Evaluation
        if (model.Semantics.Format == ModelFormat.SafeTensors)
        {
            decisions.Add(new("BACKEND", "cpu", "SafeTensors packages currently execute on CPU.", "auto_planner"));
            return (ForwardPassBackend.Cpu, "CPU Host");
        }

        if (capabilities.CudaAvailable && CanUseCuda(model, descriptor))
        {
            decisions.Add(new("BACKEND", "cuda", "CUDA compute device detected and supported by architecture.", "auto_planner"));
            return (ForwardPassBackend.Cuda, capabilities.CudaDeviceName ?? "CUDA Device 0");
        }

        if (capabilities.VulkanAvailable && CanUseVulkan(model, descriptor))
        {
            decisions.Add(new("BACKEND", "vulkan", "Vulkan compute device detected and supported by architecture.", "auto_planner"));
            return (ForwardPassBackend.Vulkan, capabilities.VulkanDeviceName ?? "Vulkan Device 0");
        }

        decisions.Add(new("BACKEND", "cpu", "No accelerator available or architecture requires CPU execution.", "auto_planner"));
        return (ForwardPassBackend.Cpu, "CPU Host");
    }

    private static bool CanUseCuda(ModelDescription model, ArchitectureDescriptor? descriptor)
    {
        if (descriptor is not null && (descriptor.SupportedBackends & SupportedBackends.Cuda) == 0)
            return false;
        // RWKV, SafeTensors, and GPT-OSS currently lack CUDA kernels
        if (model.Semantics.Family is ForwardPassFamily.Rwkv or ForwardPassFamily.GptOss)
            return false;
        return true;
    }

    private static bool CanUseVulkan(ModelDescription model, ArchitectureDescriptor? descriptor)
    {
        if (descriptor is not null && (descriptor.SupportedBackends & SupportedBackends.Vulkan) == 0)
            return false;
        return true;
    }

    private static string NormalizeGoal(string goal)
    {
        return goal.ToLowerInvariant() switch
        {
            "quality" => "quality",
            "throughput" => "throughput",
            "long-context" => "long-context",
            "low-memory" => "low-memory",
            _ => "balanced"
        };
    }

    private static int ResolveContextSize(
        ModelPlanningFacts hp,
        string goal,
        int? pinContextSize,
        ImmutableArray<ExecutionPlanDecisionDetail>.Builder decisions,
        ImmutableArray<string>.Builder warnings)
    {
        if (pinContextSize.HasValue)
        {
            int pinned = pinContextSize.Value;
            if (pinned > hp.ContextLength)
            {
                warnings.Add($"Requested context size ({pinned}) exceeds model training context length ({hp.ContextLength}).");
            }
            decisions.Add(new("CONTEXT_SIZE", pinned.ToString(), "User explicitly pinned context size.", "request_pin"));
            return pinned;
        }

        int ctx = goal switch
        {
            "long-context" => hp.ContextLength,
            "low-memory" => Math.Min(2048, hp.ContextLength),
            "throughput" => Math.Min(4096, hp.ContextLength),
            "quality" => Math.Min(8192, hp.ContextLength),
            _ => Math.Min(8192, hp.ContextLength)
        };

        decisions.Add(new("CONTEXT_SIZE", ctx.ToString(), $"Context size selected based on '{goal}' goal.", "auto_planner"));
        return ctx;
    }

    private static DType ResolveKvDtype(
        string goal,
        string? pinKvDtype,
        ImmutableArray<ExecutionPlanDecisionDetail>.Builder decisions,
        ImmutableArray<string>.Builder warnings)
    {
        if (!string.IsNullOrEmpty(pinKvDtype))
        {
            string pinned = pinKvDtype.Trim().ToLowerInvariant();
            decisions.Add(new("KV_DTYPE", pinned, "User explicitly pinned KV cache datatype.", "request_pin"));
            return pinned switch
            {
                "q8_0" or "q8" => DType.Q8_0,
                "bfloat16" or "bf16" => DType.BFloat16,
                _ => DType.Float32
            };
        }

        DType selected = goal switch
        {
            "low-memory" => DType.Q8_0,
            "quality" => DType.Float32,
            _ => DType.Float32
        };

        decisions.Add(new("KV_DTYPE", selected.ToString().ToLowerInvariant(),
            $"KV cache datatype selected based on '{goal}' goal.", "auto_planner"));
        return selected;
    }
}
