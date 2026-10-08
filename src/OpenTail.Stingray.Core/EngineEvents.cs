using System.Globalization;
using System.Text;

namespace OpenTail.Stingray.Core;

/// <summary>What happened. The generic payload slots of <see cref="EngineEvent"/> mean different things per kind.</summary>
public enum EngineEventKind
{
    /// <summary>A runtime instance was built. <c>Name</c> = architecture, <c>Ms</c> = build time, <c>Detail</c> = forward-pass kind.</summary>
    ModelLoaded,
    /// <summary>Prompt processed. <c>A</c> = prompt tokens, <c>Ms</c> = wall time.</summary>
    PrefillCompleted,
    /// <summary>Generation finished. <c>A</c> = tokens generated (incl. hidden thinking), <c>B</c> = visible tokens, <c>Ms</c> = wall time.</summary>
    DecodeCompleted,
    /// <summary>One MTP verify step. <c>A</c> = position, <c>B</c> = drafts proposed, <c>C</c> = drafts accepted.</summary>
    MtpStep,
    /// <summary>Process exit GC summary. <c>A</c>/<c>B</c>/<c>C</c> = gen0/gen1/gen2 collections, <c>Ms</c> = total pause, <c>Detail</c> = sizes.</summary>
    GcStats,
}

/// <summary>One telemetry event. A plain struct so emitting allocates nothing unless the caller builds <c>Detail</c>.</summary>
public readonly record struct EngineEvent(
    EngineEventKind Kind, string? Name = null, long A = 0, long B = 0, long C = 0, double Ms = 0, string? Detail = null);

/// <summary>Receives events. Implementations must be quick and thread-safe; exceptions are swallowed by the bus.</summary>
public interface IEngineEventSink
{
    void OnEvent(in EngineEvent e);
}

/// <summary>
/// A process-wide, in-process event bus. The engine reports what it did here and never decides who listens: the CLI, a server, a
/// benchmark or a profiler subscribe independently, and no logging code lives in the hot paths. With no subscriber
/// <see cref="Emit"/> is one volatile read. Telemetry must never affect inference, so a throwing sink is isolated.
/// Static and reflection-free (NativeAOT-safe).
/// </summary>
public static class EngineEvents
{
    private static readonly object s_gate = new();
    private static IEngineEventSink[] s_sinks = [];

    /// <summary>True when at least one sink is subscribed; guard any work done only to build an event.</summary>
    public static bool Enabled => Volatile.Read(ref s_sinks).Length != 0;

    /// <summary>Subscribes a sink; dispose the result to unsubscribe.</summary>
    public static IDisposable Subscribe(IEngineEventSink sink)
    {
        ArgumentNullException.ThrowIfNull(sink);
        lock (s_gate) s_sinks = [.. s_sinks, sink];
        return new Subscription(sink);
    }

    public static void Emit(in EngineEvent e)
    {
        var sinks = Volatile.Read(ref s_sinks);
        for (int i = 0; i < sinks.Length; i++)
        {
            try { sinks[i].OnEvent(e); }
            catch { /* a broken sink must not break inference */ }
        }
    }

    private sealed class Subscription(IEngineEventSink sink) : IDisposable
    {
        private int _disposed;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            lock (s_gate) s_sinks = s_sinks.Where(s => !ReferenceEquals(s, sink)).ToArray();
        }
    }

    /// <summary>One JSON object per event, invariant culture, hand-rolled so it needs no serializer or reflection.</summary>
    public static string ToJsonLine(in EngineEvent e)
    {
        var sb = new StringBuilder(128);
        sb.Append("{\"event\":\"").Append(e.Kind).Append('"');
        if (e.Name is not null) sb.Append(",\"name\":").Append(Quote(e.Name));
        if (e.A != 0) sb.Append(",\"a\":").Append(e.A.ToString(CultureInfo.InvariantCulture));
        if (e.B != 0) sb.Append(",\"b\":").Append(e.B.ToString(CultureInfo.InvariantCulture));
        if (e.C != 0) sb.Append(",\"c\":").Append(e.C.ToString(CultureInfo.InvariantCulture));
        if (e.Ms != 0) sb.Append(",\"ms\":").Append(e.Ms.ToString("0.###", CultureInfo.InvariantCulture));
        if (e.Detail is not null) sb.Append(",\"detail\":").Append(Quote(e.Detail));
        return sb.Append('}').ToString();
    }

    private static string Quote(string s)
    {
        var sb = new StringBuilder(s.Length + 2).Append('"');
        foreach (char c in s)
        {
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                default:
                    if (c < 0x20) sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                    else sb.Append(c);
                    break;
            }
        }
        return sb.Append('"').ToString();
    }
}
