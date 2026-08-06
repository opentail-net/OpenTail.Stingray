namespace OpenTail.Stingray.Cli.Terminal;

/// <summary>
/// Minimal console renderer covering the surface this CLI actually uses, replacing
/// Spectre.Console. Colour is emitted only for an interactive stdout, so redirected or
/// piped output stays clean (<c>opentail-llm-cli -p "..." &gt; out.txt</c>).
/// </summary>
public static class AnsiConsole
{
    /// <summary>
    /// Whether to emit ANSI styling. False when stdout is redirected, when NO_COLOR is set
    /// (https://no-color.org/), or under a dumb terminal.
    /// </summary>
    internal static bool ColorEnabled { get; } = DetectColor();

    private static bool DetectColor()
    {
        if (Console.IsOutputRedirected) return false;
        if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("NO_COLOR"))) return false;
        if (string.Equals(Environment.GetEnvironmentVariable("TERM"), "dumb", StringComparison.OrdinalIgnoreCase))
            return false;
        return true;
    }

    /// <summary>Render markup and append a newline.</summary>
    public static void MarkupLine(string markup) =>
        Console.Out.WriteLine(MarkupRenderer.Render(markup, ColorEnabled));

    /// <summary>Render markup without a trailing newline.</summary>
    public static void Markup(string markup) =>
        Console.Out.Write(MarkupRenderer.Render(markup, ColorEnabled));

    /// <summary>Write a blank line.</summary>
    public static void WriteLine() => Console.Out.WriteLine();

    /// <summary>Write plain text (no markup interpretation) and append a newline.</summary>
    public static void WriteLine(string text) => Console.Out.WriteLine(text);

    /// <summary>Render a table to stdout.</summary>
    public static void Write(Table table) => table.Render(Console.Out, ColorEnabled);

    /// <summary>Begin a status/spinner scope.</summary>
    public static StatusBuilder Status() => new();
}

// ── Style / Spinner shims ────────────────────────────────────────────────────
// Present so the existing fluent call sites keep compiling unchanged. The spinner is
// cosmetic, so these carry no behaviour beyond naming.

/// <summary>A parsed style. Retained for call-site compatibility; not used for rendering.</summary>
public sealed class Style
{
    private Style() { }

    /// <summary>Parse a style expression. Unknown styles are accepted and ignored.</summary>
    public static Style Parse(string _) => new();
}

/// <summary>A spinner animation.</summary>
public sealed class Spinner
{
    internal string[] Frames { get; }

    private Spinner(string[] frames) => Frames = frames;

    /// <summary>Built-in spinners.</summary>
    public static class Known
    {
        /// <summary>Braille dot spinner.</summary>
        public static Spinner Dots { get; } =
            new(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]);
    }
}
