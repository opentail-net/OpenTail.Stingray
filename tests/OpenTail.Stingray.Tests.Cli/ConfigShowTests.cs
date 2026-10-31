using Xunit;

namespace OpenTail.Stingray.Tests.Cli;

/// <summary>
/// <c>config --show</c> prints setting values, so it carries the same redaction obligation as
/// <c>list-env</c>. These tests drive <see cref="ConfigCommand.Render"/> with a fixed dictionary, so they
/// never depend on what the test machine has in its environment.
/// </summary>
public sealed class ConfigShowTests
{
    /// <summary>One sensitive segment per credential-shaped word the filter knows.</summary>
    public static TheoryData<string> SensitiveSegments => new()
    {
        "TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL", "APIKEY",
    };

    [Theory]
    [MemberData(nameof(SensitiveSegments))]
    public void EverySensitiveClassIsRedactedInTheReport(string segment)
    {
        string name = $"STINGRAY_TEST_{segment}";
        string secret = $"VALUE-MUST-NOT-APPEAR-{segment}";

        string report = string.Join("\n", ConfigCommand.Render(new Dictionary<string, string> { [name] = secret }));

        Assert.DoesNotContain(secret, report);
        Assert.Contains("<redacted>", report);
    }

    [Fact]
    public void AnUnknownSensitiveSettingIsRedactedToo()
    {
        // Unknown names are listed as "not read by this build". Their values are still printed by
        // list-env, so they must be masked here as well.
        var set = new Dictionary<string, string> { ["STINGRAY_NOT_A_REAL_SETTING_TOKEN"] = "SECRET-IN-UNKNOWN" };

        string report = string.Join("\n", ConfigCommand.Render(set));

        Assert.DoesNotContain("SECRET-IN-UNKNOWN", report);
        Assert.Contains("Set but not read by this build", report);
    }

    [Fact]
    public void AKnownOrdinarySettingShowsItsValueAndItsClass()
    {
        // Pick a registered name deterministically, so the test follows the registry instead of a hard-coded name.
        string name = KnownEnvironmentVariables.All.Where(n => !ListEnvCommand.IsSensitive(n)).OrderBy(n => n, StringComparer.Ordinal).First();
        string cls = EnvironmentVariableClasses.ClassOf(name) ?? throw new InvalidOperationException($"{name} has no class");

        string report = string.Join("\n", ConfigCommand.Render(new Dictionary<string, string> { [name] = "42" }));

        Assert.Contains($"{name}  42  [{cls}]  source: environment", report);
    }

    [Fact]
    public void EveryRegisteredNameHasAClass()
    {
        var missing = KnownEnvironmentVariables.All
            .Where(n => EnvironmentVariableClasses.ClassOf(n) is null)
            .OrderBy(n => n, StringComparer.Ordinal)
            .ToList();

        Assert.True(missing.Count == 0, "registered but unclassified: " + string.Join(", ", missing));
    }

    [Fact]
    public void ReportStatesThatOnlyTheEnvironmentIsResolved()
    {
        string report = string.Join("\n", ConfigCommand.Render(new Dictionary<string, string>()));

        Assert.Contains("only 'environment' is resolved here", report);
        Assert.Contains("CLI flags and built-in defaults are not read", report);
    }
}
