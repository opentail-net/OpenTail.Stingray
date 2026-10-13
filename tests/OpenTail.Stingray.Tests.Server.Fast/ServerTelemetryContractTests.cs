using System.Diagnostics.Metrics;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// The server meter's instruments must match the checked-in table (docs/reference/telemetry-instruments.md):
/// same names, types and units, every row published, nothing unlisted.
/// </summary>
public sealed class ServerTelemetryContractTests
{
    private sealed record Row(string Meter, string Name, string Type, string Unit, int Tier);

    private static string? FindTable()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "docs", "reference", "telemetry-instruments.md");
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }

    private static List<Row> ReadRows(string path)
    {
        var rows = new List<Row>();
        foreach (string line in File.ReadAllLines(path))
        {
            if (!line.StartsWith("| `OpenTail.Stingray.", StringComparison.Ordinal)) continue;
            var cells = line.Trim().Trim('|').Split('|').Select(c => c.Trim().Trim('`')).ToArray();
            if (cells.Length < 8 || !int.TryParse(cells[4], System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out int tier)) continue;
            rows.Add(new Row(cells[0], cells[1], cells[2], cells[3], tier));
        }
        return rows;
    }

    private static string TypeLabel(Instrument i) => i.GetType().Name switch
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
    public void ServerMeterInstruments_MatchTheCheckedInTable()
    {
        string? table = FindTable();
        Assert.SkipWhen(table is null, "repo docs not found from the test output folder");
        var rows = ReadRows(table!).Where(r => r.Meter == "OpenTail.Stingray.Server").ToDictionary(r => r.Name);

        using var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureServices(s => s.AddSingleton<IInferenceEngine>(new FakeInferenceEngine("m"))));
        using var client = factory.CreateClient();   // starts the host: creates the observable instruments
        new ServerMetrics().RecordRequest();          // touches a static field, which runs the static initialiser that creates the request instruments

        var published = new Dictionary<string, Instrument>();
        using var listener = new MeterListener();
        listener.InstrumentPublished = (i, l) =>
        {
            if (i.Meter.Name == "OpenTail.Stingray.Server") lock (published) published[i.Name] = i;
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
                Assert.True(published.ContainsKey(row.Name), $"table row '{row.Name}' is not published by the code");
        }
    }
}
