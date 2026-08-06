using System.ComponentModel;
using System.Text.Json;
using OpenTail.Stingray.Cli.CommandLine;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// Prints the <c>STINGRAY_*</c> settings currently active in this process's environment.
/// Usage: <c>opentail-llm-cli list-env</c>
///
/// <para>Answers "what configuration is this run actually using?", which today requires knowing
/// which of ~141 variables exist and checking each by hand. This is the environment slice of the
/// effective-configuration work in <c>docs/quality-of-life-improvements-plan.md</c> §7.3 — it does
/// NOT yet show CLI pins, profile files, or host configuration, and it reports what is SET rather
/// than what the engine resolved. Both are stated in the output so it cannot be mistaken for a
/// full effective-config report.</para>
/// </summary>
public sealed class ListEnvCommand : Command<ListEnvCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("--all")]
        [Description("Also list known settings that are NOT set (the full surface)")]
        public bool All { get; init; }

        [CommandOption("--json")]
        [Description("Emit machine-readable JSON instead of text")]
        public bool Json { get; init; }
    }

    /// <summary>
    /// Values are shown, so anything that could carry a credential is masked. Matching is on the
    /// NAME because the value's shape is not reliable — an access token looks like any other
    /// opaque string. Substrings rather than exact names, so a future credential setting is
    /// covered without anyone remembering to update this list.
    ///
    /// <para>Matching is on whole underscore-delimited SEGMENTS, not raw substrings. A substring
    /// test over-redacts: two real settings end in <c>_TOKENS</c> — token counts, not
    /// credentials — and masking their values would hide ordinary tuning information behind a
    /// security measure that protects nothing. The plural is a different word from the
    /// singular.</para>
    ///
    /// <para>Do not write example variable names in comments anywhere under <c>src/</c>: the
    /// inventory drift test scans source text for the literal prefix and would count them as real
    /// usages.</para>
    /// </summary>
    private static readonly string[] SensitiveNameSegments = ["TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL", "APIKEY"];

    internal static bool IsSensitive(string name)
    {
        foreach (Range segment in name.AsSpan().Split('_'))
        {
            ReadOnlySpan<char> part = name.AsSpan()[segment];
            foreach (string sensitive in SensitiveNameSegments)
                if (part.Equals(sensitive, StringComparison.Ordinal)) return true;
        }
        return false;
    }

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        var set = new SortedDictionary<string, string>(StringComparer.Ordinal);
        foreach (System.Collections.DictionaryEntry entry in Environment.GetEnvironmentVariables())
        {
            if (entry.Key is not string key) continue;
            if (!key.StartsWith(KnownEnvironmentVariables.Prefix, StringComparison.Ordinal)) continue;
            set[key] = entry.Value as string ?? "";
        }

        var unknown = new HashSet<string>(KnownEnvironmentVariables.FindUnknown(set.Keys), StringComparer.Ordinal);

        if (settings.Json)
        {
            WriteJson(set, unknown, settings.All);
            return 0;
        }

        if (set.Count == 0)
        {
            Console.WriteLine("No STINGRAY_* environment variables are set.");
        }
        else
        {
            Console.WriteLine($"Active STINGRAY_* environment settings ({set.Count}):");
            Console.WriteLine();
            int width = set.Keys.Max(k => k.Length);
            foreach ((string name, string value) in set)
            {
                string shown = IsSensitive(name) ? "<redacted>" : value;
                string mark = unknown.Contains(name) ? "  [UNKNOWN — not read by this build]" : "";
                Console.WriteLine($"  {name.PadRight(width)}  {shown}{mark}");
            }
        }

        if (unknown.Count > 0)
        {
            Console.WriteLine();
            foreach (string name in unknown.OrderBy(x => x, StringComparer.Ordinal))
            {
                string? suggestion = KnownEnvironmentVariables.SuggestClosest(name);
                Console.WriteLine(suggestion is null
                    ? $"  {name} is not read by this build and will have no effect."
                    : $"  {name} is not read by this build — did you mean {suggestion}?");
            }
        }

        if (settings.All)
        {
            var unset = KnownEnvironmentVariables.All
                .Where(v => !set.ContainsKey(v))
                .OrderBy(v => v, StringComparer.Ordinal)
                .ToList();
            Console.WriteLine();
            Console.WriteLine($"Known but not set ({unset.Count}):");
            Console.WriteLine();
            foreach (string name in unset) Console.WriteLine($"  {name}");
        }

        Console.WriteLine();
        Console.WriteLine("Note: this lists what is SET in the environment, not what the engine resolved —");
        Console.WriteLine("a setting may still be overridden by a CLI flag, or ignored as inapplicable to");
        Console.WriteLine("the selected backend or model. Membership in the known list means the engine");
        Console.WriteLine("reads the name, not that it is supported configuration; see docs/env-var-inventory.md.");
        return 0;
    }

    /// <summary>
    /// Hand-rolled with <see cref="Utf8JsonWriter"/> rather than a serializer: the project enables
    /// the AOT and trim analysers with warnings-as-errors, so reflection-based serialization would
    /// need a source-generated context. For a shape this small, writing it directly is less
    /// machinery than registering one.
    /// </summary>
    private static void WriteJson(SortedDictionary<string, string> set, HashSet<string> unknown, bool all)
    {
        using var stream = Console.OpenStandardOutput();
        using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true });
        writer.WriteStartObject();

        // Versioned so consumers can detect shape changes rather than guess (plan §7.1/§7.4).
        writer.WriteNumber("schemaVersion", 1);
        writer.WriteBoolean("valuesReflectEnvironmentOnly", true);

        writer.WriteStartArray("settings");
        foreach ((string name, string value) in set)
        {
            writer.WriteStartObject();
            writer.WriteString("name", name);
            bool sensitive = IsSensitive(name);
            writer.WriteBoolean("redacted", sensitive);
            writer.WriteString("value", sensitive ? null : value);
            writer.WriteBoolean("known", !unknown.Contains(name));
            if (unknown.Contains(name))
            {
                string? suggestion = KnownEnvironmentVariables.SuggestClosest(name);
                if (suggestion is not null) writer.WriteString("suggestion", suggestion);
            }
            writer.WriteEndObject();
        }
        writer.WriteEndArray();

        if (all)
        {
            writer.WriteStartArray("knownButNotSet");
            foreach (string name in KnownEnvironmentVariables.All
                         .Where(v => !set.ContainsKey(v))
                         .OrderBy(v => v, StringComparer.Ordinal))
                writer.WriteStringValue(name);
            writer.WriteEndArray();
        }

        writer.WriteEndObject();
        writer.Flush();
        stream.WriteByte((byte)'\n');
    }
}
