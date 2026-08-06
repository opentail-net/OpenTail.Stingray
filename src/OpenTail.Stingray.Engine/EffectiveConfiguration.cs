using System.Text.Json;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Resolves a small, ordered set of configuration candidates while retaining both the
/// winning source and overridden values. Frontends use this for diagnostic snapshots;
/// it does not mutate process environment or alter their runtime binding behavior.
/// </summary>
public static class EffectiveConfigurationResolver
{
    /// <summary>Precedence is the candidate order: first non-null value wins.</summary>
    public static EffectiveConfigurationSnapshot Resolve(IEnumerable<EffectiveConfigurationSetting> settings)
    {
        var values = new Dictionary<string, EffectiveConfigurationValue>(StringComparer.Ordinal);
        var diagnostics = new List<EffectiveConfigurationDiagnostic>();
        foreach (var setting in settings)
        {
            var candidates = setting.Candidates.ToArray();
            var chosen = candidates.FirstOrDefault(x => x.Value is not null)
                ?? new EffectiveConfigurationCandidate("default", setting.DefaultValue);
            string source = chosen.Source;
            object normalized = chosen.Value ?? setting.DefaultValue;
            if (chosen.Source == "environment" && chosen.Value is string raw)
            {
                if (TryParseEnvironmentValue(raw, setting.DefaultValue, out object? parsed))
                {
                    normalized = parsed!;
                }
                else
                {
                    normalized = setting.DefaultValue;
                    source = "default_after_invalid_environment";
                    diagnostics.Add(new("invalid", setting.Name,
                        $"environment value '{raw}' is invalid; using default '{setting.DefaultValue}'."));
                }
            }
            values[setting.Name] = new(ToJson(normalized), source);

            bool afterChosen = false;
            foreach (var candidate in candidates)
            {
                if (!afterChosen)
                {
                    afterChosen = candidate.Source == chosen.Source;
                    continue;
                }
                if (candidate.Value is not null && !Equals(Convert.ToString(candidate.Value), Convert.ToString(chosen.Value)))
                {
                    diagnostics.Add(new("conflict", setting.Name,
                        $"{candidate.Source} value '{candidate.Value}' is overridden by {chosen.Source} value '{chosen.Value}'."));
                }
            }
        }
        return new(values, diagnostics);
    }

    private static bool TryParseEnvironmentValue(string raw, object defaultValue, out object? value)
    {
        if (defaultValue is int)
        {
            bool parsed = int.TryParse(raw, System.Globalization.NumberStyles.Integer,
                System.Globalization.CultureInfo.InvariantCulture, out int integer);
            value = parsed ? integer : null;
            return parsed;
        }
        if (defaultValue is bool)
        {
            if (raw is "1" or "true" or "TRUE") { value = true; return true; }
            if (raw is "0" or "false" or "FALSE") { value = false; return true; }
            value = null;
            return false;
        }
        value = raw;
        return true;
    }

    private static JsonElement ToJson(object value)
    {
        string json = value switch
        {
            bool b => b ? "true" : "false",
            int i => i.ToString(System.Globalization.CultureInfo.InvariantCulture),
            long l => l.ToString(System.Globalization.CultureInfo.InvariantCulture),
            _ => "\"" + Convert.ToString(value, System.Globalization.CultureInfo.InvariantCulture)!
                .Replace("\\", "\\\\", StringComparison.Ordinal)
                .Replace("\"", "\\\"", StringComparison.Ordinal)
                .Replace("\r", "\\r", StringComparison.Ordinal)
                .Replace("\n", "\\n", StringComparison.Ordinal)
                .Replace("\t", "\\t", StringComparison.Ordinal) + "\"",
        };
        using var document = JsonDocument.Parse(json);
        return document.RootElement.Clone();
    }
}

public sealed record EffectiveConfigurationSetting(
    string Name, object DefaultValue, IReadOnlyList<EffectiveConfigurationCandidate> Candidates);

public sealed record EffectiveConfigurationCandidate(string Source, object? Value);
public sealed record EffectiveConfigurationValue(JsonElement Value, string Source);
public sealed record EffectiveConfigurationDiagnostic(string Kind, string Field, string Message);

public sealed record EffectiveConfigurationSnapshot(
    IReadOnlyDictionary<string, EffectiveConfigurationValue> Values,
    IReadOnlyList<EffectiveConfigurationDiagnostic> Diagnostics)
{
    public T Get<T>(string name)
    {
        JsonElement value = Values[name].Value;
        object result = typeof(T) == typeof(string) ? value.GetString()! :
            typeof(T) == typeof(int) ? value.ValueKind == JsonValueKind.Number ? value.GetInt32() : int.Parse(value.GetString()!, System.Globalization.CultureInfo.InvariantCulture) :
            typeof(T) == typeof(bool) ? value.ValueKind is JsonValueKind.True or JsonValueKind.False ? value.GetBoolean() : ParseBoolean(value.GetString()!) :
            throw new NotSupportedException($"Unsupported configuration value type '{typeof(T).Name}'.");
        return (T)result;

        static bool ParseBoolean(string raw) => raw switch { "1" => true, "0" => false, _ => bool.Parse(raw) };
    }
}
