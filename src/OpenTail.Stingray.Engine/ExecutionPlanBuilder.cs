using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Converts user intent (goal + optional explicit pins) and hardware capabilities into a validated <see cref="ExecutionPlan"/>.
/// </summary>
public static class ExecutionPlanBuilder
{
    public static ExecutionPlan Build(
        string modelPath,
        string goal = "balanced",
        string? pinBackend = null,
        int? pinGpuLayers = null,
        int? pinContextSize = null,
        string? pinKvDtype = null,
        bool noGpuProbe = false)
    {
        string resolvedGoal = NormalizeGoal(goal);
        var decisions = new List<ExecutionPlanDecisionDetail>();
        var warnings = new List<string>();

        using var model = GgufModel.Open(modelPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var hardware = HardwareProfile.Detect(noGpuProbe ? null : TryCreateVulkan());

        // 1. Backend Selection
        string backend = ResolveBackend(hardware, pinBackend, decisions);

        // 2. Context Size Selection
        int ctxSize = ResolveContextSize(hp, resolvedGoal, pinContextSize, decisions, warnings);

        // 3. KV DType Selection
        DType kvDtype = ResolveKvDtype(resolvedGoal, pinKvDtype, decisions, warnings);

        // 4. Placement & Layer Offload via TierPlanner
        int? requestedGpuLayers = pinGpuLayers;
        if (requestedGpuLayers == -1) requestedGpuLayers = hp.NumLayers;
        // A CPU backend pin is meaningful even when the caller did not specify -g: do not
        // let the GPU capacity visible to TierPlanner turn that explicit request into an
        // offload plan. An explicit -g remains a conflicting caller choice and is preserved
        // for the frontend to diagnose.
        if (!requestedGpuLayers.HasValue && backend == "cpu") requestedGpuLayers = 0;

        var placement = TierPlanner.Plan(
            model, hp, hardware,
            requestedCtxSize: ctxSize,
            kvDtype: kvDtype,
            pinGpuLayers: requestedGpuLayers);

        int resolvedGpuLayers = placement.GpuLayers;
        if (pinGpuLayers.HasValue)
        {
            decisions.Add(new("GPU_LAYERS", resolvedGpuLayers.ToString(), "User explicitly pinned GPU layer count.", "cli_pin"));
        }
        else
        {
            decisions.Add(new("GPU_LAYERS", resolvedGpuLayers.ToString(),
                resolvedGpuLayers >= hp.NumLayers
                    ? "Full GPU weight offload selected."
                    : $"Offloaded {resolvedGpuLayers}/{hp.NumLayers} layers based on VRAM budget.", "auto_planner"));
        }

        double estVramMb = (placement.GpuWeightBytes + placement.GpuKvBytes) / (1024.0 * 1024.0);
        double estRamMb = placement.CpuWeightBytes / (1024.0 * 1024.0);

        if (hardware.VramBytes > 0 && estVramMb > (hardware.VramBytes / (1024.0 * 1024.0)))
        {
            warnings.Add($"Estimated VRAM usage ({estVramMb:F0} MiB) exceeds reported VRAM capacity ({hardware.VramBytes / (1024.0 * 1024.0):F0} MiB).");
        }

        return new ExecutionPlan(
            SchemaVersion: 1,
            ModelPath: modelPath,
            Goal: resolvedGoal,
            Backend: backend,
            GpuLayers: resolvedGpuLayers,
            TotalLayers: hp.NumLayers,
            ContextSize: ctxSize,
            KvDtype: kvDtype.ToString().ToLowerInvariant(),
            EstimatedVramMb: estVramMb,
            EstimatedRamMb: estRamMb,
            Decisions: decisions,
            Warnings: warnings,
            ModelFormat: ModelFormat.Gguf
        );
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

    private static string ResolveBackend(HardwareProfile hardware, string? pinBackend, List<ExecutionPlanDecisionDetail> decisions)
    {
        if (!string.IsNullOrEmpty(pinBackend) && !string.Equals(pinBackend, "auto", StringComparison.OrdinalIgnoreCase))
        {
            string pinned = pinBackend.ToLowerInvariant();
            decisions.Add(new("BACKEND", pinned, "User explicitly pinned backend.", "cli_pin"));
            return pinned;
        }

        if (hardware.VramBytes > 0)
        {
            decisions.Add(new("BACKEND", "vulkan", "Vulkan GPU compute device detected with VRAM capacity.", "auto_planner"));
            return "vulkan";
        }

        decisions.Add(new("BACKEND", "cpu", "No compatible GPU backend detected; defaulting to SIMD CPU execution.", "auto_planner"));
        return "cpu";
    }

    private static int ResolveContextSize(ModelHyperparams hp, string goal, int? pinContextSize, List<ExecutionPlanDecisionDetail> decisions, List<string> warnings)
    {
        if (pinContextSize.HasValue && pinContextSize.Value > 0)
        {
            decisions.Add(new("CONTEXT_SIZE", pinContextSize.Value.ToString(), "User explicitly pinned context length.", "cli_pin"));
            return pinContextSize.Value;
        }

        int targetCtx = goal switch
        {
            "long-context" => Math.Min(32768, hp.ContextLength),
            "low-memory" => Math.Min(4096, hp.ContextLength),
            _ => Math.Min(8192, hp.ContextLength)
        };

        decisions.Add(new("CONTEXT_SIZE", targetCtx.ToString(), $"Context length selected for goal '{goal}'.", "auto_planner"));
        return targetCtx;
    }

    private static DType ResolveKvDtype(string goal, string? pinKvDtype, List<ExecutionPlanDecisionDetail> decisions, List<string> warnings)
    {
        if (!string.IsNullOrEmpty(pinKvDtype))
        {
            DType pinned = pinKvDtype.ToLowerInvariant() switch
            {
                "q8_0" => DType.Q8_0,
                _ => DType.Float32
            };
            decisions.Add(new("KV_DTYPE", pinned.ToString().ToLowerInvariant(), "User explicitly pinned KV cache dtype.", "cli_pin"));
            return pinned;
        }

        DType selected = goal switch
        {
            "throughput" or "long-context" => DType.Q8_0,
            "quality" => DType.Float32,
            _ => DType.Float32
        };

        if (selected == DType.Q8_0)
        {
            warnings.Add("Q8_0 KV cache is lossy and not bit-identical to FP32.");
        }

        decisions.Add(new("KV_DTYPE", selected.ToString().ToLowerInvariant(), $"KV dtype selected for goal '{goal}'.", "auto_planner"));
        return selected;
    }

    private static VulkanBackend? TryCreateVulkan()
    {
        try { return new VulkanBackend(); }
        catch { return null; }
    }
}
