using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Feeds the <c>opentail.engine.*</c> instruments from <see cref="EngineEvents"/>. The CLI's run command has its own
/// decode loop (it does not go through <see cref="InferenceEngine"/>), but it already reports prefill and decode on the
/// event bus, so subscribing this sink gives CLI runs the same prefill/decode measurements as the server path.
/// Events carry no request identity, so one run is one request here. Counts and durations only.
/// </summary>
public sealed class EngineEventsTelemetryBridge : IEngineEventSink
{
    private readonly string _forwardPass;

    /// <param name="forwardPass">Closed label for the <c>opentail.forward_pass</c> attribute (for example <c>cli</c>).</param>
    public EngineEventsTelemetryBridge(string forwardPass = "cli") => _forwardPass = forwardPass;

    public void OnEvent(in EngineEvent e)
    {
        switch (e.Kind)
        {
            case EngineEventKind.PrefillCompleted:
                EngineTelemetry.RecordPrefillAndDecode(_forwardPass, newPromptTokens: (int)e.A, prefillSeconds: e.Ms / 1000.0, decodeTokens: -1, decodeSeconds: 0);
                break;
            case EngineEventKind.MtpStep:
                EngineTelemetry.RecordSpeculativeStep("mtp", proposed: e.B, accepted: e.C);
                break;
            case EngineEventKind.DecodeCompleted:
                EngineTelemetry.RecordPrefillAndDecode(_forwardPass, newPromptTokens: -1, prefillSeconds: 0, decodeTokens: (int)e.A, decodeSeconds: e.Ms / 1000.0);
                break;
        }
    }
}
