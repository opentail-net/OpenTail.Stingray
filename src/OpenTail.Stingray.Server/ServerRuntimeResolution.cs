namespace OpenTail.Stingray.Server;

/// <summary>Non-sensitive receipt of the concrete loader route selected for a built-in model.</summary>
/// <param name="GpuLayers">Layers the execution plan placed on the GPU (0 for CPU). Null when the
/// loader did not record it.</param>
public sealed record ServerRuntimeResolution(
    string Backend,
    string ForwardPass,
    string ModelFormat,
    int ContextSize,
    int? GpuLayers = null);

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
    internal void Set(string? architecture)
    {
        // A caller-supplied EngineFactory may report no architecture; keep the hint fallback then.
        if (!string.IsNullOrWhiteSpace(architecture))
            Interlocked.CompareExchange(ref _architecture, architecture, null);
    }
}
