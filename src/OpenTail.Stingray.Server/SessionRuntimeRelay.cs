using OpenTail.Stingray.Sessions;

namespace OpenTail.Stingray.Server;

/// <summary>
/// Access to the server-owned hot-session runtime. A host can inject this service and expose its
/// own authenticated routing policy without having to depend on a loader implementation detail.
/// </summary>
public interface IServerSessionRuntime
{
    /// <summary>The active runtime, or <c>null</c> when the selected load lane cannot support sessions.</summary>
    HotSessionRuntime? Runtime { get; }

    /// <summary>Explains why <see cref="Runtime"/> is unavailable.</summary>
    string? UnavailabilityReason { get; }
}

internal sealed class SessionRuntimeRelay : IServerSessionRuntime
{
    private HotSessionRuntime? _runtime;
    private string? _unavailabilityReason;

    public SessionRuntimeRelay(string? unavailabilityReason) => _unavailabilityReason = unavailabilityReason;

    public HotSessionRuntime? Runtime => Volatile.Read(ref _runtime);
    public string? UnavailabilityReason => Volatile.Read(ref _unavailabilityReason);

    public void Set(HotSessionRuntime? runtime, string? unavailabilityReason)
    {
        Volatile.Write(ref _runtime, runtime);
        Volatile.Write(ref _unavailabilityReason, runtime is null ? unavailabilityReason : null);
    }
}
