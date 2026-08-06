using System.ComponentModel;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Reflection;

namespace OpenTail.Stingray.Cli.CommandLine;

/// <summary>One bindable command-line option, derived from a settings property.</summary>
internal sealed class OptionModel
{
    /// <summary>The settings property this option writes to.</summary>
    internal required PropertyInfo Property { get; init; }

    /// <summary>Accepted aliases, in declaration order (e.g. <c>-m</c>, <c>--model</c>).</summary>
    internal required string[] Aliases { get; init; }

    /// <summary>Value placeholder shown in help (<c>PATH</c>), or null.</summary>
    internal required string? Placeholder { get; init; }

    /// <summary>Help text.</summary>
    internal required string Description { get; init; }

    /// <summary>Declared default, or null.</summary>
    internal required object? DefaultValue { get; init; }

    /// <summary>True when presence alone sets the value (a boolean switch).</summary>
    internal required bool IsFlag { get; init; }

    /// <summary>True when the option may be repeated, accumulating into an array.</summary>
    internal required bool IsRepeatable { get; init; }

    /// <summary>Longest alias, used as the display name in help.</summary>
    internal string DisplayName => Aliases.OrderByDescending(a => a.Length).First();

    /// <summary>
    /// Reflect <typeparamref name="T"/>'s properties into option models. Results are cached
    /// per settings type.
    /// </summary>
    internal static IReadOnlyList<OptionModel> Describe<
        [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicProperties)] T>()
        where T : CommandSettings => OptionCache<T>.Options;

    private static class OptionCache<
        [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicProperties)] T>
        where T : CommandSettings
    {
        internal static readonly IReadOnlyList<OptionModel> Options = Build();

        private static List<OptionModel> Build()
        {
            var models = new List<OptionModel>();

            foreach (var prop in typeof(T).GetProperties(BindingFlags.Public | BindingFlags.Instance))
            {
                var attr = prop.GetCustomAttribute<CommandOptionAttribute>();
                if (attr is null) continue;

                // Template: "alias|alias|... [<PLACEHOLDER>]"
                string template = attr.Template.Trim();
                string? placeholder = null;

                int lt = template.IndexOf('<', StringComparison.Ordinal);
                if (lt >= 0)
                {
                    int gt = template.IndexOf('>', lt);
                    if (gt > lt) placeholder = template[(lt + 1)..gt];
                    template = template[..lt].Trim();
                }

                var aliases = template
                    .Split('|', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                    .ToArray();
                if (aliases.Length == 0) continue;

                Type target = Nullable.GetUnderlyingType(prop.PropertyType) ?? prop.PropertyType;
                bool repeatable = target.IsArray;

                models.Add(new OptionModel
                {
                    Property     = prop,
                    Aliases      = aliases,
                    Placeholder  = placeholder,
                    Description  = prop.GetCustomAttribute<DescriptionAttribute>()?.Description ?? "",
                    DefaultValue = prop.GetCustomAttribute<DefaultValueAttribute>()?.Value,
                    // A bool with no placeholder is a switch; "--flag <BOOL>" wants an explicit value.
                    IsFlag       = target == typeof(bool) && placeholder is null,
                    IsRepeatable = repeatable,
                });
            }

            return models;
        }
    }

    /// <summary>
    /// Convert <paramref name="raw"/> to this option's property type.
    /// Parsing is invariant-culture: CLI input must not vary with the machine's locale.
    /// </summary>
    internal bool TryConvert(string raw, out object? value, [NotNullWhen(false)] out string? error)
    {
        Type target = Nullable.GetUnderlyingType(Property.PropertyType) ?? Property.PropertyType;
        if (target.IsArray) target = target.GetElementType()!;

        error = null;

        if (target == typeof(string)) { value = raw; return true; }

        if (target == typeof(int))
        {
            if (int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out int i))
            { value = i; return true; }
            error = $"option '{DisplayName}' expects an integer, got '{raw}'.";
        }
        else if (target == typeof(long))
        {
            if (long.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out long l))
            { value = l; return true; }
            error = $"option '{DisplayName}' expects an integer, got '{raw}'.";
        }
        else if (target == typeof(float))
        {
            if (float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out float f))
            { value = f; return true; }
            error = $"option '{DisplayName}' expects a number, got '{raw}'.";
        }
        else if (target == typeof(bool))
        {
            if (TryParseBool(raw, out bool b)) { value = b; return true; }
            error = $"option '{DisplayName}' expects true/false, got '{raw}'.";
        }
        else
        {
            error = $"option '{DisplayName}' has unsupported type {target.Name}.";
        }

        value = null;
        return false;
    }

    /// <summary>Accept the spellings a CLI user actually types for a boolean.</summary>
    internal static bool TryParseBool(string raw, out bool value)
    {
        switch (raw.ToLowerInvariant())
        {
            case "true" or "1" or "yes" or "on":  value = true;  return true;
            case "false" or "0" or "no" or "off": value = false; return true;
            default:                              value = false; return false;
        }
    }
}
