namespace OpenTail.Stingray.Cli;

/// <summary>
/// <c>stingray config --show</c>: the recognised <c>STINGRAY_*</c> settings that are set in this
/// process's environment, each with its value (redacted where sensitive), its source, and its class
/// from the inventory (<c>docs/reference/env-var-inventory.md</c>).
///
/// <para>Deliberately narrow. It reads the environment only: it does not see CLI flags, profile
/// files, or built-in defaults, so the only source it can resolve is <c>environment</c>. The output
/// says so, rather than implying a full effective-configuration report. <c>list-env</c> is the
/// broader listing with <c>--all</c> and <c>--json</c>; this command adds the class and source.</para>
/// </summary>
public sealed class ConfigCommand : Command<ConfigCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("--show")]
        [Description("Print the recognised settings that are set, with value (redacted where sensitive), source and class")]
        public bool Show { get; init; }
    }

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        if (!settings.Show)
        {
            Console.Error.WriteLine("Use: stingray config --show");
            return 2;
        }

        foreach (string line in Render(ReadEnvironment()))
            Console.WriteLine(line);
        return 0;
    }

    /// <summary>Every <c>STINGRAY_*</c> variable that is set, by name, in ordinal order.</summary>
    internal static SortedDictionary<string, string> ReadEnvironment()
    {
        var set = new SortedDictionary<string, string>(StringComparer.Ordinal);
        foreach (System.Collections.DictionaryEntry entry in Environment.GetEnvironmentVariables())
        {
            if (entry.Key is not string key) continue;
            if (!key.StartsWith(KnownEnvironmentVariables.Prefix, StringComparison.Ordinal)) continue;
            set[key] = entry.Value as string ?? "";
        }
        return set;
    }

    /// <summary>
    /// The printed report, as lines. Pure over its input so the redaction and class output can be
    /// tested without touching the real environment.
    /// </summary>
    internal static IReadOnlyList<string> Render(IReadOnlyDictionary<string, string> set)
    {
        var unknown = new HashSet<string>(KnownEnvironmentVariables.FindUnknown(set.Keys), StringComparer.Ordinal);
        var recognised = set.Keys.Where(k => !unknown.Contains(k)).OrderBy(k => k, StringComparer.Ordinal).ToList();
        var notRead = unknown.OrderBy(k => k, StringComparer.Ordinal).ToList();
        int width = set.Keys.Select(k => k.Length).DefaultIfEmpty(0).Max();

        var lines = new List<string>
        {
            $"Recognised STINGRAY_* settings that are set ({recognised.Count}):",
        };
        foreach (string name in recognised)
        {
            string cls = EnvironmentVariableClasses.ClassOf(name) ?? "unclassified";
            lines.Add($"  {name.PadRight(width)}  {Shown(name, set[name])}  [{cls}]  source: environment");
        }

        if (notRead.Count > 0)
        {
            lines.Add("");
            lines.Add($"Set but not read by this build ({notRead.Count}); no effect:");
            foreach (string name in notRead)
            {
                string? suggestion = KnownEnvironmentVariables.SuggestClosest(name);
                string hint = suggestion is null ? "" : $"  did you mean {suggestion}?";
                lines.Add($"  {name.PadRight(width)}  {Shown(name, set[name])}{hint}");
            }
        }

        lines.Add("");
        lines.Add("Sources: only 'environment' is resolved here. CLI flags and built-in defaults are not read by this");
        lines.Add("command, so a value shown may be overridden by a flag or ignored for the selected backend or model.");
        return lines;
    }

    /// <summary>Redaction is by name, the same rule <c>list-env</c> applies (see <c>ListEnvCommand.IsSensitive</c>).</summary>
    internal static string Shown(string name, string value) =>
        ListEnvCommand.IsSensitive(name) ? "<redacted>" : value;
}
