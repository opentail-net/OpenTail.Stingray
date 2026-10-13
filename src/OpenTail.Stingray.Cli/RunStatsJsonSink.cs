using System.Globalization;
using System.Text;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// STINGRAY_STATS_JSON=&lt;path&gt;: write one machine-readable result file at process exit, so benchmark scripts and the
/// Performance League read structured numbers instead of regex-matching printed lines. A deliberate, versioned contract
/// (schema_version), not a side effect of telemetry export. Built from the same <see cref="EngineEvents"/> the engine
/// already emits; hand-rolled JSON (NativeAOT-safe, no serializer). Counts and durations only, never content or paths.
///
/// Fields (all numbers are invariant-culture; durations in milliseconds):
///   schema_version, architecture, forward_pass, load_ms, prompt_tokens, prefill_ms, prefill_tokens_per_second,
///   decode_tokens (incl. hidden reasoning tokens), visible_tokens, decode_ms, decode_tokens_per_second,
///   speculative_steps, speculative_drafts_proposed, speculative_drafts_accepted,
///   perturbed (true when a diagnostic profile switch was on: never compare such a run).
/// A field is omitted when its event never happened. When a run has several prefill/decode phases the last one wins.
/// </summary>
internal sealed class RunStatsJsonSink : IEngineEventSink
{
    private readonly string _path;
    private readonly object _gate = new();
    private string? _architecture, _forwardPass;
    private double _loadMs = -1, _prefillMs = -1, _decodeMs = -1;
    private long _promptTokens = -1, _decodeTokens = -1, _visibleTokens = -1;
    private long _specSteps, _specProposed, _specAccepted;

    public RunStatsJsonSink(string path) => _path = path;

    public void OnEvent(in EngineEvent e)
    {
        lock (_gate)
        {
            switch (e.Kind)
            {
                case EngineEventKind.ModelLoaded:
                    _architecture = e.Name;
                    _loadMs = e.Ms;
                    const string prefix = "forwardPass=";
                    if (e.Detail is { } d && d.StartsWith(prefix, StringComparison.Ordinal)) _forwardPass = d[prefix.Length..];
                    break;
                case EngineEventKind.PrefillCompleted:
                    _promptTokens = e.A;
                    _prefillMs = e.Ms;
                    break;
                case EngineEventKind.DecodeCompleted:
                    _decodeTokens = e.A;
                    _visibleTokens = e.B;
                    _decodeMs = e.Ms;
                    break;
                case EngineEventKind.MtpStep:
                    _specSteps++;
                    _specProposed += e.B;
                    _specAccepted += e.C;
                    break;
            }
        }
    }

    public void WriteFile()
    {
        try
        {
            string json;
            lock (_gate) json = Render();
            File.WriteAllText(_path, json, new UTF8Encoding(false));
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"warning: STINGRAY_STATS_JSON could not be written: {ex.Message}");
        }
    }

    internal string Render()
    {
        var sb = new StringBuilder("{\n  \"schema_version\": 1");
        void Str(string name, string? v) { if (v is not null) sb.Append(",\n  \"").Append(name).Append("\": \"").Append(Escape(v)).Append('"'); }
        void Num(string name, double v, bool present) { if (present) sb.Append(",\n  \"").Append(name).Append("\": ").Append(v.ToString("0.###", CultureInfo.InvariantCulture)); }
        double Rate(long tokens, double ms) => ms > 0 ? tokens * 1000.0 / ms : 0;

        Str("architecture", _architecture);
        Str("forward_pass", _forwardPass);
        Num("load_ms", _loadMs, _loadMs >= 0);
        Num("prompt_tokens", _promptTokens, _promptTokens >= 0);
        Num("prefill_ms", _prefillMs, _prefillMs >= 0);
        Num("prefill_tokens_per_second", Rate(_promptTokens, _prefillMs), _prefillMs > 0 && _promptTokens >= 0);
        Num("decode_tokens", _decodeTokens, _decodeTokens >= 0);
        Num("visible_tokens", _visibleTokens, _visibleTokens >= 0);
        Num("decode_ms", _decodeMs, _decodeMs >= 0);
        Num("decode_tokens_per_second", Rate(_decodeTokens, _decodeMs), _decodeMs > 0 && _decodeTokens >= 0);
        Num("speculative_steps", _specSteps, _specSteps > 0);
        Num("speculative_drafts_proposed", _specProposed, _specSteps > 0);
        Num("speculative_drafts_accepted", _specAccepted, _specSteps > 0);
        bool perturbed = Environment.GetEnvironmentVariable("STINGRAY_MOE_PHASE_TIMING") == "1"
                         || Environment.GetEnvironmentVariable("STINGRAY_PROFILE_GPU_SPLIT") == "1";
        sb.Append(",\n  \"perturbed\": ").Append(perturbed ? "true" : "false").Append("\n}\n");
        return sb.ToString();
    }

    private static string Escape(string s) => s.Replace("\\", "\\\\", StringComparison.Ordinal).Replace("\"", "\\\"", StringComparison.Ordinal);
}
