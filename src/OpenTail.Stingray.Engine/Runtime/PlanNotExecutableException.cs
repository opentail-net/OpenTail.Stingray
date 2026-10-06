#nullable enable

using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Engine.Runtime;

/// <summary>
/// Thrown when an ExecutionPlan cannot be executed by RuntimeInstance due to invariant
/// violations, missing hardware capabilities, or incompatible model files.
/// Under plan-driven architecture, silent fallbacks are strictly prohibited (§5.1 of plan).
/// </summary>
public sealed class PlanNotExecutableException : InvalidOperationException
{
    public ExecutionPlan? Plan { get; }

    public PlanNotExecutableException(string message, ExecutionPlan? plan = null)
        : base(message)
    {
        Plan = plan;
    }

    public PlanNotExecutableException(string message, Exception innerException, ExecutionPlan? plan = null)
        : base(message, innerException)
    {
        Plan = plan;
    }
}
