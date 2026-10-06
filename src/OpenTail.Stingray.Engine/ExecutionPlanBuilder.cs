#nullable enable

using OpenTail.Stingray.Engine.Packaging;
using OpenTail.Stingray.Engine.Planning;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Backward-compatible facade delegating execution planning to <see cref="ExecutionPlanner"/>.
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
        IModelPackage package;
        if (Directory.Exists(modelPath) || modelPath.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            package = SafeTensorsModelPackage.Open(modelPath);
        }
        else
        {
            package = LooseGgufModelPackage.Open(modelPath);
        }

        var modelDesc = ModelDescription.FromPackage(package);
        var capabilities = BackendCapabilities.Detect(noGpuProbe);
        var request = new ExecutionRequest
        {
            Goal = goal,
            PinnedBackend = pinBackend,
            PinnedGpuLayers = pinGpuLayers,
            PinnedContextSize = pinContextSize,
            PinnedKvDtype = pinKvDtype,
            NoGpuProbe = noGpuProbe
        };

        return ExecutionPlanner.Plan(modelDesc, request, capabilities);
    }
}
