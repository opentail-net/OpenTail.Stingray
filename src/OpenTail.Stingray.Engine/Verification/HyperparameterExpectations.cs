using System.Globalization;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Engine.Verification;

/// <summary>Checks a golden's <see cref="GoldenFile.ExpectedHyperparameters"/> against what the descriptor resolved for the file.</summary>
public static class HyperparameterExpectations
{
    /// <summary>Failure messages; empty means every expectation holds. An unknown property name is itself a failure (a typo must not pass silently).</summary>
    public static IReadOnlyList<string> Check(ModelHyperparams hp, IReadOnlyDictionary<string, string>? expected)
    {
        var failures = new List<string>();
        if (expected is null) return failures;
        foreach (var (name, want) in expected)
        {
            var prop = typeof(ModelHyperparams).GetProperty(name,
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.IgnoreCase);
            if (prop is null) { failures.Add($"unknown ModelHyperparams property '{name}'"); continue; }
            string got = Format(prop.GetValue(hp));
            if (!string.Equals(got, want, StringComparison.OrdinalIgnoreCase))
                failures.Add($"{prop.Name}: expected {want}, resolved {got}");
        }
        return failures;
    }

    private static string Format(object? v) => v switch
    {
        null => "null",
        bool b => b ? "true" : "false",
        IFormattable f => f.ToString(null, CultureInfo.InvariantCulture),
        _ => v.ToString() ?? "null",
    };
}
