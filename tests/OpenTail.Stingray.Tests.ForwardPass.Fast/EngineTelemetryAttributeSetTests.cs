using System.Diagnostics.Metrics;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Plan 5 C2, OpenTelemetry plan section 12: the attribute set of each instrument matches the checked-in table
/// (docs/reference/telemetry-instruments.md, column "Attributes"). The section 7 allow-list test checks every key against one global
/// list; this one checks each instrument against its own row, so a key that is allowed somewhere but undocumented for this instrument fails.
/// The direction tested is "emitted is a subset of documented": a documented key may be conditional (for example only on failure).
/// </summary>
[Collection("EngineTelemetry")]
public sealed class EngineTelemetryAttributeSetTests
{
    // The table abbreviates attribute names; these are the full tag keys (TelemetryAttributePolicy.AllowedAttributeKeys).
    private static readonly Dictionary<string, string> s_keyOf = new(StringComparer.Ordinal)
    {
        ["operation"] = "gen_ai.operation.name",
        ["model"] = "gen_ai.request.model",
        ["endpoint"] = "opentail.endpoint",
        ["outcome"] = "outcome",
        ["forward_pass"] = "opentail.forward_pass",
        ["architecture"] = "opentail.architecture",
        ["backend"] = "opentail.backend",
        ["backend.requested"] = "opentail.backend.requested",
        ["fallback.reason"] = "opentail.fallback.reason",
        ["reason"] = "opentail.reason",
        ["speculative.mode"] = "opentail.speculative.mode",
        ["sampling.path"] = "opentail.sampling.path",
        ["profile.phase"] = "opentail.profile.phase",
        ["scheduler"] = "opentail.diffusion.scheduler",
    };

    private static Dictionary<string, HashSet<string>> DocumentedAttributes(string table)
    {
        var result = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        foreach (string line in File.ReadAllLines(table))
        {
            if (!line.StartsWith("| `OpenTail.Stingray.", StringComparison.Ordinal)) continue;
            var cells = line.Trim().Trim('|').Split('|').Select(c => c.Trim().Trim('`')).ToArray();
            if (cells.Length < 8 || cells[0] != EngineTelemetry.MeterName) continue;
            var keys = new HashSet<string>(StringComparer.Ordinal);
            foreach (string abbreviation in cells[6].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                if (abbreviation == "none") continue;
                Assert.True(s_keyOf.TryGetValue(abbreviation, out string? key), $"row '{cells[1]}' lists the attribute '{abbreviation}', which this test cannot map to a tag key; add it to s_keyOf");
                keys.Add(key!);
            }
            result[cells[1]] = keys;
        }
        return result;
    }

    [Fact]
    public async Task EveryEmittedTag_IsDocumentedForThatInstrument()
    {
        string? table = EngineTelemetryContractTests.FindTable();
        Assert.SkipWhen(table is null, "repo docs not found from the test output folder");
        var documented = DocumentedAttributes(table!);

        var emitted = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        using var listener = new MeterListener();
        listener.InstrumentPublished = (i, l) => { if (i.Meter.Name == EngineTelemetry.MeterName) l.EnableMeasurementEvents(i); };
        void Record(Instrument i, ReadOnlySpan<KeyValuePair<string, object?>> tags)
        {
            lock (emitted)
            {
                if (!emitted.TryGetValue(i.Name, out var set)) emitted[i.Name] = set = new HashSet<string>(StringComparer.Ordinal);
                foreach (var kv in tags) set.Add(kv.Key);
            }
        }
        listener.SetMeasurementEventCallback<long>((i, v, t, s) => Record(i, t));
        listener.SetMeasurementEventCallback<double>((i, v, t, s) => Record(i, t));
        listener.SetMeasurementEventCallback<int>((i, v, t, s) => Record(i, t));
        listener.Start();

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 6 };
        var tokenizer = new InferenceEngineChunkTests.ScriptedTokenizer();
        var fwd = new InferenceEngineChunkTests.ScriptedForwardPass([0, 1, 6], tokenizer.VocabSize);
        using (var engine = new InferenceEngine(fwd, tokenizer, "mock", thinkTokenId: -1, endThinkTokenId: -1))
            await foreach (var _ in engine.GenerateChunksAsync("hello", sp)) { }
        using (var batching = new ContinuousBatchingEngine(
                   new ContinuousBatchingConstraintTests.FakeBatchedForwardPass(),
                   new ContinuousBatchingConstraintTests.CharTokenizer(), "test", maxBatchSize: 1))
            await foreach (var _ in batching.GenerateChunksAsync("hello", sp)) { }
        using (var load = EngineTelemetry.StartModelLoad())
            EngineTelemetry.RecordModelLoad(load.Activity, "llama", "ForwardPass", EngineTelemetry.OutcomeSuccess, 0.5);

        string[] problems;
        lock (emitted)
        {
            // Not vacuous: instruments with attributes really emitted them.
            Assert.True(emitted.Count(kv => kv.Value.Count > 0) >= 4, $"only {emitted.Count(kv => kv.Value.Count > 0)} instruments emitted attributes");
            problems = emitted
                .Select(kv => (Name: kv.Key, Extra: documented.TryGetValue(kv.Key, out var doc) ? kv.Value.Except(doc).Order(StringComparer.Ordinal).ToArray() : ["<instrument has no row in the table>"]))
                .Where(x => x.Extra.Length > 0)
                .Select(x => $"{x.Name}: {string.Join(", ", x.Extra)}")
                .ToArray();
        }
        Assert.True(problems.Length == 0, "tags emitted but not documented for the instrument: " + string.Join("; ", problems));
    }
}
