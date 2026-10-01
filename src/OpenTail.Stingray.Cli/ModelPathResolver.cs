namespace OpenTail.Stingray.Cli;

/// <summary>Where to look for a default checkpoint, and how to describe it when none is found.</summary>
/// <param name="Label">Model name used in the error ("Kokoro", "Fish-Speech (S2-Pro)").</param>
/// <param name="Kind">What the user should pass ("a Kokoro .gguf checkpoint").</param>
/// <param name="Example">An example path for the message.</param>
/// <param name="Candidates">Default locations, in preference order.</param>
/// <param name="DirMarker">When set, a candidate is a directory that must contain this file (e.g. "model.safetensors").</param>
/// <param name="Extra">Optional trailing sentence for the message.</param>
internal sealed record ModelSearch(string Label, string Kind, string Example, string[] Candidates, string? DirMarker = null, string Extra = "");

/// <summary>
/// One place for "find the default model file or tell the user how to supply one". Replaces the per-engine
/// copies of the same loop and message that had accumulated in <c>TtsCommand</c>.
/// </summary>
internal static class ModelPathResolver
{
    /// <summary>The first candidate that exists (a file, or a directory containing <paramref name="dirMarker"/>), else null.</summary>
    public static string? FindFirst(IEnumerable<string> candidates, string? dirMarker = null)
    {
        foreach (var c in candidates)
        {
            if (dirMarker is null ? File.Exists(c) : Directory.Exists(c) && File.Exists(Path.Combine(c, dirMarker)))
                return c;
        }
        return null;
    }

    /// <summary>Returns the first existing default, or throws <see cref="ArgumentException"/> with an actionable message.</summary>
    public static string Resolve(ModelSearch search) =>
        FindFirst(search.Candidates, search.DirMarker) ?? throw new ArgumentException(NotFoundMessage(search));

    /// <summary>A path the user passed explicitly must exist; the message names the option and the path.</summary>
    public static string RequireExisting(string label, string given) =>
        File.Exists(given) ? given : throw new ArgumentException($"{label} model file not found: '{given}'.");

    internal static string NotFoundMessage(ModelSearch s) =>
        $"No {s.Label} model found. Pass --model (-m) with a path to {s.Kind} (e.g. {s.Example}), or place one at that default path." +
        (s.Extra.Length == 0 ? "" : " " + s.Extra);
}
