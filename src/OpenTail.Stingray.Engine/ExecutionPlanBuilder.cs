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
        => Build(new ExecutionRequest
        {
            ModelPath = modelPath,
            Goal = goal,
            PinnedBackend = pinBackend,
            PinnedGpuLayers = pinGpuLayers,
            PinnedContextSize = pinContextSize,
            PinnedKvDtype = pinKvDtype,
            NoGpuProbe = noGpuProbe
        });

    /// <summary>
    /// Plans an arbitrary <see cref="ExecutionRequest"/> (every execution-affecting input the planner knows). Like the CLI,
    /// server and ModelContext it bridges inherited <c>STINGRAY_*</c> variables into the request first, so equivalent
    /// requests yield equivalent plans from every frontend.
    /// </summary>
    public static ExecutionPlan Build(ExecutionRequest request)
    {
        string modelPath = request.ModelPath ?? throw new ArgumentException("ExecutionRequest.ModelPath is required.", nameof(request));
        bool noGpuProbe = request.NoGpuProbe;
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
        var capabilities = BackendCapabilities.Detect(noGpuProbe, request.DeviceIndex);
        return ExecutionPlanner.Plan(modelDesc, ExecutionRequestEnvironment.ApplyTo(request), capabilities);
    }
}
