using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Reports the counters a forward pass keeps for telemetry. Only the qwen4exp pass has counters so far. The values are
/// cumulative since the pass was loaded; the event is a snapshot, not a delta.
/// </summary>
public static class ForwardCounterEvents
{
    /// <summary>Emits <see cref="EngineEventKind.ForwardCounters"/> for a pass that keeps counters. Does nothing otherwise.</summary>
    public static void Emit(IForwardPass pass)
    {
        if (!EngineEvents.Enabled) return;
        if (pass is Qwen4ExpForwardPass q)
            EngineEvents.Emit(new(EngineEventKind.ForwardCounters, Name: Qwen4ExpArchitecture.Descriptor.Id,
                A: q.PleGatherRows, B: q.PleGatherBytes, C: q.PleTableBytes));
    }
}
