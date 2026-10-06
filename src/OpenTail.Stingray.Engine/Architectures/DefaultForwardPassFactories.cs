namespace OpenTail.Stingray.Engine;

using OpenTail.Stingray.Core;

/// <summary>
/// Transitional placeholder and fallback forward-pass factory delegates.
/// </summary>
internal static class DefaultForwardPassFactories
{
    public static IForwardPass CreatePlaceholder(ArchitectureLoadContext ctx) =>
        throw new NotImplementedException($"Architecture '{ctx.Probe.Architecture ?? ctx.Decision.Kind?.ToString()}' has not yet migrated to an architecture-owned factory.");
}
