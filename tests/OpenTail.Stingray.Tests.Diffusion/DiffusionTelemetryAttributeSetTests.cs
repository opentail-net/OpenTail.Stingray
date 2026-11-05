using System.Diagnostics.Metrics;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Plan 5 C2, OpenTelemetry plan section 12: each diffusion instrument emits only the attributes its row of the checked-in table
/// (docs/reference/telemetry-instruments.md, column "Attributes") documents; the engine and server meters have the same test.
/// A successful loop and a failing one are both driven, since the outcome tag differs. Emitted is a subset of documented.
/// </summary>
[Collection("DiffusionTelemetry")]
public sealed class DiffusionTelemetryAttributeSetTests
{
    private static readonly Dictionary<string, string> s_keyOf = new(StringComparer.Ordinal)
    {
        ["scheduler"] = "opentail.diffusion.scheduler",
        ["outcome"] = "outcome",
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

    [Fact]
    public void EveryEmittedDiffusionTag_IsDocumentedForThatInstrument()
    {
        string? table = FindTable();
        Assert.SkipWhen(table is null, "repo docs not found from the test output folder");

        var documented = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        foreach (string line in File.ReadAllLines(table!))
        {
            if (!line.StartsWith("| `OpenTail.Stingray.", StringComparison.Ordinal)) continue;
            var cells = line.Trim().Trim('|').Split('|').Select(c => c.Trim().Trim('`')).ToArray();
            if (cells.Length < 8 || cells[0] != DiffusionTelemetry.MeterName) continue;
            var keys = new HashSet<string>(StringComparer.Ordinal);
            foreach (string abbreviation in cells[6].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                if (abbreviation == "none") continue;
                Assert.True(s_keyOf.TryGetValue(abbreviation, out string? key),
                    $"row '{cells[1]}' lists the attribute '{abbreviation}', which this test cannot map to a tag key; add it to s_keyOf");
                keys.Add(key!);
            }
            documented[cells[1]] = keys;
        }

        var emitted = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        using var listener = new MeterListener();
        listener.InstrumentPublished = (i, l) => { if (i.Meter.Name == DiffusionTelemetry.MeterName) l.EnableMeasurementEvents(i); };
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
        listener.Start();

        var scheduler = EulerFlowScheduler.Linear(4);
        scheduler.Denoise(new float[] { 1f, 2f, 3f, 4f }, (x, t) => x.Select(v => v * 0.5f + t).ToArray());
        Assert.ThrowsAny<Exception>(() => scheduler.Denoise(new float[] { 1f, 2f, 3f, 4f },
            (x, t) => throw new InvalidOperationException("injected")));

        string[] problems;
        lock (emitted)
        {
            // Not vacuous: all three instruments really emitted, and the loop instruments carried attributes.
            Assert.Equal(3, emitted.Count);
            Assert.All(emitted, kv => Assert.NotEmpty(kv.Value));
            problems = emitted
                .Select(kv => (Name: kv.Key, Extra: documented.TryGetValue(kv.Key, out var doc)
                    ? kv.Value.Except(doc).Order(StringComparer.Ordinal).ToArray()
                    : ["<instrument has no row in the table>"]))
                .Where(x => x.Extra.Length > 0)
                .Select(x => $"{x.Name}: {string.Join(", ", x.Extra)}")
                .ToArray();
        }
        Assert.True(problems.Length == 0, "tags emitted but not documented for the instrument: " + string.Join("; ", problems));
    }
}
