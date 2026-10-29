namespace OpenTail.Stingray.Server;

/// <summary>Non-sensitive receipt of the concrete loader route selected for a built-in model.</summary>
public sealed record ServerRuntimeResolution(
    string Backend,
    string ForwardPass,
    string ModelFormat,
    int ContextSize);

/// <summary>Bridges lazy built-in model loading to diagnostics without entering inference loops.</summary>
public sealed class ServerRuntimeResolutionRelay
{
    private ServerRuntimeResolution? _resolution;
    public ServerRuntimeResolution? Resolution => Volatile.Read(ref _resolution);
    internal void Set(ServerRuntimeResolution? resolution) =>
        Interlocked.CompareExchange(ref _resolution, resolution, null);
}

/// <summary>
/// Architecture the loaded model resolved to. Null until a model has loaded, so diagnostics can tell
/// a resolved architecture from the <see cref="OpenTailStingrayServerOptions.Architecture"/> hint.
/// </summary>
public sealed class ServerLoadedArchitectureRelay
{
    private string? _architecture;
    public string? Architecture => Volatile.Read(ref _architecture);
    internal void Set(string architecture) =>
        Interlocked.CompareExchange(ref _architecture, architecture, null);
}
