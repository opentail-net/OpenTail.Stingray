namespace OpenTail.Stingray.Cli.Terminal;

/// <summary>
/// Console markup: <c>[style]text[/]</c>, with <c>[[</c> / <c>]]</c> as literal brackets.
///
/// <para>
/// Deliberately more forgiving than Spectre.Console: an unrecognised tag is emitted as
/// literal text rather than throwing. Spectre treats every <c>[</c> as markup and raises
/// "Could not find color or style" on anything it cannot parse, which is a live hazard for
/// help text and any string carrying JSON, file paths, or model output. Rendering is
/// best-effort and must never take down the process for a cosmetic concern.
/// </para>
/// </summary>
public static class Markup
{
    /// <summary>Escape a string so brackets survive markup rendering verbatim.</summary>
    public static string Escape(string text) =>
        text.Replace("[", "[[", StringComparison.Ordinal)
            .Replace("]", "]]", StringComparison.Ordinal);
}

/// <summary>String helpers mirroring Spectre.Console's markup extensions.</summary>
public static class MarkupExtensions
{
    /// <summary>Escape brackets so the string renders verbatim through markup.</summary>
    public static string EscapeMarkup(this string text) => Markup.Escape(text);
}

/// <summary>Style tokens understood by <see cref="MarkupRenderer"/>.</summary>
internal static class AnsiCodes
{
    internal static readonly string Esc = ((char)27).ToString();
    internal static readonly string Reset = Esc + "[0m";

    /// <summary>Map a single style token to its SGR parameter, or null if unrecognised.</summary>
    internal static string? Sgr(string token) => token switch
    {
        "bold" or "b"        => "1",
        "dim"                => "2",
        "italic" or "i"      => "3",
        "underline" or "u"   => "4",
        "black"              => "30",
        "red"                => "31",
        "green"              => "32",
        "yellow"             => "33",
        "blue"               => "34",
        "magenta"            => "35",
        "cyan"               => "36",
        "white"              => "37",
        "grey" or "gray"     => "90",
        _                    => null,
    };
}

/// <summary>
/// Translates markup into ANSI escape sequences, or into plain text when the destination
/// is not an interactive terminal (redirected output must stay pipe-friendly).
/// </summary>
internal static class MarkupRenderer
{
    /// <summary>
    /// Render <paramref name="markup"/>, emitting colour only when <paramref name="ansi"/>.
    /// Unrecognised tags pass through as literal text.
    /// </summary>
    internal static string Render(string markup, bool ansi)
    {
        // Fast path: nothing bracket-like to interpret.
        if (markup.IndexOf('[') < 0 && markup.IndexOf(']') < 0) return markup;

        var sb = new System.Text.StringBuilder(markup.Length);
        var stack = new List<string>();

        for (int i = 0; i < markup.Length; i++)
        {
            char c = markup[i];

            if (c == '[')
            {
                // "[[" is a literal '['.
                if (i + 1 < markup.Length && markup[i + 1] == '[') { sb.Append('['); i++; continue; }

                int close = markup.IndexOf(']', i + 1);
                if (close < 0) { sb.Append(c); continue; }   // unterminated: literal

                string tag = markup[(i + 1)..close];

                if (tag == "/")
                {
                    if (stack.Count > 0) stack.RemoveAt(stack.Count - 1);
                    if (ansi)
                    {
                        // No per-attribute "off" code is reliable across terminals, so reset
                        // and replay whatever styles are still open.
                        sb.Append(AnsiCodes.Reset);
                        foreach (var s in stack) sb.Append(s);
                    }
                    i = close;
                    continue;
                }

                if (TryBuildSgr(tag, out string sgr))
                {
                    stack.Add(sgr);
                    if (ansi) sb.Append(sgr);
                    i = close;
                    continue;
                }

                // Not a style we know — treat the '[' as ordinary text.
                sb.Append(c);
                continue;
            }

            if (c == ']')
            {
                // "]]" is a literal ']'.
                if (i + 1 < markup.Length && markup[i + 1] == ']') { sb.Append(']'); i++; continue; }
                sb.Append(c);
                continue;
            }

            sb.Append(c);
        }

        if (ansi && stack.Count > 0) sb.Append(AnsiCodes.Reset);
        return sb.ToString();
    }

    /// <summary>
    /// Build the SGR sequence for a whitespace-separated style tag ("bold red").
    /// Returns false if any token is unrecognised, so the caller can fall back to literal text.
    /// </summary>
    private static bool TryBuildSgr(string tag, out string sgr)
    {
        sgr = "";
        if (tag.Length == 0) return false;

        var parts = tag.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length == 0) return false;

        var codes = new List<string>(parts.Length);
        foreach (var part in parts)
        {
            // "on <colour>" (background) is accepted but rendered as foreground-only; the
            // alternative is dropping the whole tag to literal text, which reads worse.
            if (part.Equals("on", StringComparison.OrdinalIgnoreCase)) continue;

            string? code = AnsiCodes.Sgr(part.ToLowerInvariant());
            if (code is null) return false;
            codes.Add(code);
        }

        if (codes.Count == 0) return false;
        sgr = AnsiCodes.Esc + "[" + string.Join(';', codes) + "m";
        return true;
    }
}
