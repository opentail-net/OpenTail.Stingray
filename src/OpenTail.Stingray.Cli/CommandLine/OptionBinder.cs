
namespace OpenTail.Stingray.Cli.CommandLine;

/// <summary>Binds an argument vector onto a settings instance.</summary>
internal static class OptionBinder
{
    /// <summary>
    /// Apply declared defaults, then parse <paramref name="args"/> over them.
    /// Returns false with a human-readable <paramref name="error"/> on bad input.
    /// </summary>
    internal static bool TryBind(
        CommandSettings settings,
        IReadOnlyList<OptionModel> options,
        string[] args,
        [NotNullWhen(false)] out string? error)
    {
        var byAlias = new Dictionary<string, OptionModel>(StringComparer.Ordinal);
        foreach (var opt in options)
            foreach (var alias in opt.Aliases)
                byAlias[alias] = opt;

        ApplyDefaults(settings, options);

        // Repeatable options accumulate across occurrences before being written once.
        var accumulated = new Dictionary<OptionModel, List<object?>>();

        for (int i = 0; i < args.Length; i++)
        {
            string arg = args[i];

            if (arg.Length == 0) continue;

            if (arg == "--")
            {
                error = "positional arguments are not supported.";
                return false;
            }

            if (arg[0] != '-')
            {
                error = $"unexpected argument '{arg}'.";
                return false;
            }

            // Split "--name=value" into name and inline value.
            string name = arg;
            string? inlineValue = null;
            int eq = arg.IndexOf('=', StringComparison.Ordinal);
            if (eq > 0)
            {
                name = arg[..eq];
                inlineValue = arg[(eq + 1)..];
            }

            if (!byAlias.TryGetValue(name, out var opt))
            {
                error = $"unknown option '{name}'.";
                return false;
            }

            string raw;
            if (opt.IsFlag)
            {
                // A switch is true by presence; "--flag=false" can still turn it off.
                raw = inlineValue ?? "true";
            }
            else if (inlineValue is not null)
            {
                raw = inlineValue;
            }
            else
            {
                // Consume the next token unconditionally — an option's value may itself look
                // like an option (e.g. "-g -1", "--temp -0.5").
                if (i + 1 >= args.Length)
                {
                    error = $"option '{name}' requires a value.";
                    return false;
                }
                raw = args[++i];
            }

            if (!opt.TryConvert(raw, out object? value, out string? convertError))
            {
                error = convertError;
                return false;
            }

            if (opt.IsRepeatable)
            {
                if (!accumulated.TryGetValue(opt, out var list))
                    accumulated[opt] = list = [];
                list.Add(value);
            }
            else
            {
                opt.Property.SetValue(settings, value);
            }
        }

        foreach (var (opt, values) in accumulated)
        {
            Type element = (Nullable.GetUnderlyingType(opt.Property.PropertyType)
                            ?? opt.Property.PropertyType).GetElementType()!;
            var array = Array.CreateInstance(element, values.Count);
            for (int i = 0; i < values.Count; i++) array.SetValue(values[i], i);
            opt.Property.SetValue(settings, array);
        }

        error = null;
        return true;
    }

    /// <summary>Seed properties carrying a <c>[DefaultValue]</c> before parsing.</summary>
    private static void ApplyDefaults(CommandSettings settings, IReadOnlyList<OptionModel> options)
    {
        foreach (var opt in options)
        {
            if (opt.DefaultValue is null) continue;

            // Init-only properties are settable via reflection: `init` is enforced by the
            // compiler, not the runtime.
            opt.Property.SetValue(settings, opt.DefaultValue);
        }
    }
}
