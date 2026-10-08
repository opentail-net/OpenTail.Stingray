using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Cli;

/// <summary>Writes each engine event to stderr as a JSON line (enabled by <c>STINGRAY_EVENTS=1</c>).</summary>
internal sealed class StderrJsonEventSink : IEngineEventSink
{
    public void OnEvent(in EngineEvent e) => Console.Error.WriteLine(EngineEvents.ToJsonLine(e));
}
