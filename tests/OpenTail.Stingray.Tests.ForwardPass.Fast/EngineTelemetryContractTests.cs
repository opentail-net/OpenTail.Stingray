using System.Diagnostics.Metrics;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// The engine meter's instruments must match the checked-in table (docs/reference/telemetry-instruments.md):
/// same names, types, units; every tier-1/2 row exists; nothing unlisted appears. Tier-3 rows are conditional.
/// </summary>
[Collection("EngineTelemetry")]
public sealed class EngineTelemetryContractTests
{
    internal sealed record Row(string Meter, string Name, string Type, string Unit, int Tier);

    internal static string? FindTable()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "docs", "reference", "telemetry-instruments.md");
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }

    internal static List<Row> ReadRows(string path)
    {
        var rows = new List<Row>();
        foreach (string line in File.ReadAllLines(path))
        {
            if (!line.StartsWith("| `OpenTail.Stingray.", StringComparison.Ordinal)) continue;
            var cells = line.Trim().Trim('|').Split('|').Select(c => c.Trim().Trim('`')).ToArray();
            if (cells.Length < 8 || !int.TryParse(cells[4], System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out int tier)) continue; // the spans table has different columns
            rows.Add(new Row(cells[0], cells[1], cells[2], cells[3], tier));
        }
        return rows;
    }

    internal static string TypeLabel(Instrument i) => i.GetType().Name switch
    {
        "Counter`1" => "counter",
        "Histogram`1" => "histogram",
        "UpDownCounter`1" => "up-down counter",
        "ObservableGauge`1" => "observable gauge",
        "ObservableCounter`1" => "observable counter",
        "ObservableUpDownCounter`1" => "observable up-down counter",
        var other => other,
    };

    [Fact]
    public void EngineMeterInstruments_MatchTheCheckedInTable()
    {
        string? table = FindTable();
        Assert.SkipWhen(table is null, "repo docs not found from the test output folder");
        var rows = ReadRows(table!).Where(r => r.Meter == EngineTelemetry.MeterName).ToDictionary(r => r.Name);

        _ = EngineTelemetry.Enabled; // runs the static initialiser, which creates the instruments
        var published = new Dictionary<string, Instrument>();
        using var listener = new MeterListener();
        listener.InstrumentPublished = (i, l) =>
        {
            if (i.Meter.Name == EngineTelemetry.MeterName) lock (published) published[i.Name] = i;
        };
        listener.Start();

        lock (published)
        {
            foreach (var (name, instrument) in published)
            {
                Assert.True(rows.TryGetValue(name, out var row), $"instrument '{name}' is published but missing from the table");
                Assert.Equal(row!.Type, TypeLabel(instrument));
                Assert.Equal(row.Unit, instrument.Unit);
            }
            foreach (var row in rows.Values.Where(r => r.Tier < 3))
                Assert.True(published.ContainsKey(row.Name), $"table row '{row.Name}' (tier {row.Tier}) is not published by the code");
        }
    }

    [Fact]
    public void TheTable_CoversEveryTier_AndEveryRowHasAKnownTier()
    {
        string? table = FindTable();
        Assert.SkipWhen(table is null, "repo docs not found from the test output folder");
        var rows = ReadRows(table!);
        Assert.NotEmpty(rows);
        Assert.All(rows, r => Assert.Contains(r.Tier, new[] { 1, 2, 3 }));
        Assert.Contains(rows, r => r.Tier == 1);
        Assert.Contains(rows, r => r.Tier == 2);
        Assert.Contains(rows, r => r.Tier == 3);
        Assert.Equal(rows.Count, rows.Select(r => r.Meter + "|" + r.Name).Distinct().Count());
    }
}
