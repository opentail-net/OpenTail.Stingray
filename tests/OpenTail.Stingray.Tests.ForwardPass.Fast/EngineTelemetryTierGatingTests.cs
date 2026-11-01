using System.Diagnostics.Metrics;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// Plan 5 C2, OpenTelemetry plan section 12: only tier 1 does work with no listener attached. The table's own definition
/// (docs/reference/telemetry-instruments.md): tier 2 "costs something only when a listener is attached", tier 3 is "explicit diagnostic mode,
/// off by default". Three properties are pinned for the engine meter:
/// <list type="number">
/// <item>With no listener, the tier-2 gate is closed (<c>EngineTelemetry.IterationEnabled</c> is what <c>ContinuousBatchingEngine</c> checks before it
/// takes the iteration timestamp).</item>
/// <item>Listening to tier-1 instruments only does not open it.</item>
/// <item>Listening to the tier-2 instruments opens it, and it closes again when that listener is gone.</item>
/// </list>
/// And tier-3 instruments are not published at all unless their diagnostic switch is set.
/// </summary>
[Collection("EngineTelemetry")]
public sealed class EngineTelemetryTierGatingTests
{
    [Fact]
    public void Tier2Work_IsGatedOnAListenerOfTierTwoInstruments_AndTier3InstrumentsExistOnlyInDiagnosticMode()
    {
        string? table = EngineTelemetryContractTests.FindTable();
        Assert.SkipWhen(table is null, "repo docs not found from the test output folder");
        var tierOf = EngineTelemetryContractTests.ReadRows(table!)
            .Where(r => r.Meter == EngineTelemetry.MeterName).ToDictionary(r => r.Name, r => r.Tier);
        Assert.Contains(tierOf, kv => kv.Value == 1);
        Assert.Contains(tierOf, kv => kv.Value == 2);
        Assert.Contains(tierOf, kv => kv.Value == 3);

        // Tier-3 output is labelled as perturbed in its own definition, so nobody mistakes it for a benchmark figure.
        foreach (string line in File.ReadAllLines(table!))
        {
            if (!line.StartsWith("| `" + EngineTelemetry.MeterName + "`", StringComparison.Ordinal)) continue;
            var cells = line.Trim().Trim('|').Split('|').Select(c => c.Trim().Trim('`')).ToArray();
            if (cells.Length >= 8 && tierOf.TryGetValue(cells[1], out int rowTier) && rowTier == 3)
                Assert.StartsWith("DIAGNOSTIC", cells[7], StringComparison.Ordinal);
        }

        _ = EngineTelemetry.Enabled;   // runs the static initialiser, which creates the instruments

        // 1. No listener: the tier-2 gate is closed.
        Assert.False(EngineTelemetry.IterationEnabled, "tier-2 work is enabled with no listener attached");

        MeterListener Listen(Func<int, bool> tierWanted, List<string>? published = null)
        {
            var l = new MeterListener();
            l.InstrumentPublished = (i, ml) =>
            {
                if (i.Meter.Name != EngineTelemetry.MeterName) return;
                published?.Add(i.Name);
                if (tierOf.TryGetValue(i.Name, out int tier) && tierWanted(tier)) ml.EnableMeasurementEvents(i);
            };
            l.Start();
            return l;
        }

        // 2. A listener on tier-1 instruments alone leaves it closed.
        var seen = new List<string>();
        using (var tierOneOnly = Listen(t => t == 1, seen))
            Assert.False(EngineTelemetry.IterationEnabled, "listening to tier-1 instruments enabled tier-2 work");

        // 3. A listener on the tier-2 instruments opens it, and it closes with the listener.
        using (var tierTwo = Listen(t => t == 2))
            Assert.True(EngineTelemetry.IterationEnabled, "listening to the tier-2 instruments did not enable them");
        Assert.False(EngineTelemetry.IterationEnabled, "tier-2 work stayed enabled after its listener went away");

        // Tier 3 exists only in diagnostic mode.
        Assert.SkipWhen(Environment.GetEnvironmentVariable("STINGRAY_PROFILE_GPU_SPLIT") == "1",
            "a tier-3 diagnostic switch is set in this environment");
        var tierThree = tierOf.Where(kv => kv.Value == 3).Select(kv => kv.Key).ToHashSet();
        Assert.DoesNotContain(seen, tierThree.Contains);
        Assert.True(seen.Count >= 10, $"only {seen.Count} engine instruments were published");   // not vacuous
    }
}
