using System.Collections.Concurrent;
using System.Diagnostics.Metrics;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using OpenTail.Stingray.Audio.Whisper;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Server;
using Xunit;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Plan 5 C2, OpenTelemetry plan section 12: each server instrument emits only the attributes its row of the checked-in table
/// (docs/reference/telemetry-instruments.md, column "Attributes") documents. The engine meter has the same test
/// (<c>EngineTelemetryAttributeSetTests</c>). The direction is "emitted is a subset of documented": a documented key may be conditional.
/// Runs in the non-parallel telemetry collection: the meter listener is process-wide, so tags from tests running at the same time would mix in.
/// </summary>
[Collection(ModelLoadWaitTelemetryCollection.Name)]
public sealed class ServerTelemetryAttributeSetTests
{
    private const string MeterName = "OpenTail.Stingray.Server";

    // The table abbreviates attribute names; these are the full tag keys (TelemetryAttributePolicy.AllowedAttributeKeys).
    private static readonly Dictionary<string, string> s_keyOf = new(StringComparer.Ordinal)
    {
        ["operation"] = "gen_ai.operation.name",
        ["model"] = "gen_ai.request.model",
        ["endpoint"] = "opentail.endpoint",
        ["outcome"] = "outcome",
        ["error.type"] = "error.type",
        ["reason"] = "opentail.reason",
    };

    private static string? FindTable()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "docs", "reference", "telemetry-instruments.md");
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }

    private static Dictionary<string, HashSet<string>> DocumentedAttributes(string table)
    {
        var result = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        foreach (string line in File.ReadAllLines(table))
        {
            if (!line.StartsWith("| `OpenTail.Stingray.", StringComparison.Ordinal)) continue;
            var cells = line.Trim().Trim('|').Split('|').Select(c => c.Trim().Trim('`')).ToArray();
            if (cells.Length < 8 || cells[0] != MeterName) continue;
            var keys = new HashSet<string>(StringComparer.Ordinal);
            foreach (string abbreviation in cells[6].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                if (abbreviation == "none") continue;
                Assert.True(s_keyOf.TryGetValue(abbreviation, out string? key),
                    $"row '{cells[1]}' lists the attribute '{abbreviation}', which this test cannot map to a tag key; add it to s_keyOf");
                keys.Add(key!);
            }
            result[cells[1]] = keys;
        }
        return result;
    }

    private sealed class ThrowingProvider : IWhisperPipelineProvider
    {
        public WhisperPipeline Create(string? model) => throw new InvalidOperationException("secret provider failure");
    }

    private static byte[] SilentWav()
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        const int samples = 1600;
        w.Write("RIFF"u8); w.Write(36 + samples * 2); w.Write("WAVE"u8);
        w.Write("fmt "u8); w.Write(16); w.Write((short)1); w.Write((short)1); w.Write(16000); w.Write(32000); w.Write((short)2); w.Write((short)16);
        w.Write("data"u8); w.Write(samples * 2); w.Write(new byte[samples * 2]);
        w.Flush();
        return ms.ToArray();
    }

    [Fact]
    public async Task EveryEmittedServerTag_IsDocumentedForThatInstrument()
    {
        string? table = FindTable();
        Assert.SkipWhen(table is null, "repo docs not found from the test output folder");
        var documented = DocumentedAttributes(table!);

        var emitted = new ConcurrentDictionary<string, ConcurrentDictionary<string, byte>>(StringComparer.Ordinal);
        using var listener = new MeterListener();
        listener.InstrumentPublished = (i, l) => { if (i.Meter.Name == MeterName) l.EnableMeasurementEvents(i); };
        void Record(Instrument i, ReadOnlySpan<KeyValuePair<string, object?>> tags)
        {
            var set = emitted.GetOrAdd(i.Name, _ => new ConcurrentDictionary<string, byte>(StringComparer.Ordinal));
            foreach (var kv in tags) set[kv.Key] = 0;
        }
        listener.SetMeasurementEventCallback<long>((i, v, t, s) => Record(i, t));
        listener.SetMeasurementEventCallback<double>((i, v, t, s) => Record(i, t));
        listener.SetMeasurementEventCallback<int>((i, v, t, s) => Record(i, t));
        listener.Start();

        using var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s =>
            {
                s.AddSingleton<IInferenceEngine>(new FakeInferenceEngine("test-model"));
                s.AddSingleton<IWhisperPipelineProvider>(new ThrowingProvider());   // a non-chat operation that fails: error.type
            }));
        using var client = factory.CreateClient();
        foreach (bool stream in new[] { false, true })
        {
            using var ok = await client.PostAsJsonAsync("/v1/chat/completions", new
            {
                model = "test-model", stream, max_tokens = 4, messages = new[] { new { role = "user", content = "hi" } },
            });
            Assert.True(ok.IsSuccessStatusCode);
        }
        using var refused = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "test-model", max_tokens = 0, messages = new[] { new { role = "user", content = "hi" } },
        });
        Assert.False(refused.IsSuccessStatusCode);

        // A non-chat operation whose handler throws: the operation wrapper tags the duration with the exception type name (and rethrows).
        using (var form = new MultipartFormDataContent { { new ByteArrayContent(SilentWav()), "file", "a.wav" } })
        {
            try { using var failed = await client.PostAsync("/v1/audio/transcriptions", form); }
            catch (InvalidOperationException) { /* the exception reaches the test server client; the metric is what matters */ }
        }

        // Not vacuous: the request instruments really emitted attributes, and the failure path emitted error.type.
        Assert.True(emitted["gen_ai.server.request.duration"].ContainsKey("error.type"), "the failing non-chat operation did not emit error.type");
        Assert.Contains(emitted, kv => kv.Key == "gen_ai.server.request.duration" && kv.Value.Count > 0);
        string[] problems = emitted
            .Select(kv => (Name: kv.Key, Extra: documented.TryGetValue(kv.Key, out var doc)
                ? kv.Value.Keys.Except(doc).Order(StringComparer.Ordinal).ToArray()
                : ["<instrument has no row in the table>"]))
            .Where(x => x.Extra.Length > 0)
            .Select(x => $"{x.Name}: {string.Join(", ", x.Extra)}")
            .ToArray();
        Assert.True(problems.Length == 0, "tags emitted but not documented for the instrument: " + string.Join("; ", problems));
    }
}
