using OpenTail.Stingray.Cli.Terminal;

namespace OpenTail.Stingray.Cli.CommandLine;

/// <summary>
/// Renders <c>--help</c>. Descriptions pass through the markup renderer, so a stray bracket
/// in help text degrades to literal output instead of aborting the process.
/// </summary>
internal static class HelpRenderer
{
    private const int IndentWidth = 4;
    private const int GapWidth = 2;
    private const int MinDescriptionWidth = 24;
    private const int FallbackConsoleWidth = 100;

    internal static void WriteRootHelp(Configurator config, ICommand defaultCommand)
    {
        AnsiConsole.MarkupLine("[bold]USAGE:[/]");
        AnsiConsole.MarkupLine($"    {Markup.Escape(config.ApplicationName)} [dim]<OPTIONS> [COMMAND][/]");
        AnsiConsole.WriteLine();

        if (config.Commands.Count > 0)
        {
            AnsiConsole.MarkupLine("[bold]COMMANDS:[/]");
            var names = config.Commands.Select(c => c.Name).ToArray();
            int width = names.Max(n => n.Length);

            foreach (var cmd in config.Commands)
                WriteEntry(cmd.Name, width, cmd.Description);

            AnsiConsole.WriteLine();
        }

        WriteOptions(defaultCommand.Options);
    }

    internal static void WriteCommandHelp(Configurator config, CommandEntry command)
    {
        AnsiConsole.MarkupLine("[bold]USAGE:[/]");
        AnsiConsole.MarkupLine(
            $"    {Markup.Escape(config.ApplicationName)} {Markup.Escape(command.Name)} [dim]<OPTIONS>[/]");
        AnsiConsole.WriteLine();

        if (!string.IsNullOrWhiteSpace(command.Description))
        {
            foreach (string line in Wrap(command.Description, ConsoleWidth() - IndentWidth))
                AnsiConsole.MarkupLine($"    {Markup.Escape(line)}");
            AnsiConsole.WriteLine();
        }

        WriteOptions(command.Instance.Options);
    }

    private static void WriteOptions(IReadOnlyList<OptionModel> options)
    {
        AnsiConsole.MarkupLine("[bold]OPTIONS:[/]");

        var labels = new string[options.Count];
        for (int i = 0; i < options.Count; i++)
        {
            var o = options[i];
            string aliases = string.Join(", ", o.Aliases);
            labels[i] = o.Placeholder is null ? aliases : $"{aliases} <{o.Placeholder}>";
        }

        const string HelpLabel = "-h, --help";
        int width = labels.Length > 0
            ? Math.Max(labels.Max(l => l.Length), HelpLabel.Length)
            : HelpLabel.Length;

        WriteEntry(HelpLabel, width, "Prints help information");

        for (int i = 0; i < options.Count; i++)
        {
            var o = options[i];
            string description = o.Description;

            // Many descriptions already spell out their default inline ("... (default: 512)").
            // Appending another one from [DefaultValue] would render it twice, so only add the
            // suffix when the author hasn't already said it.
            if (o.DefaultValue is not null &&
                !description.Contains("default", StringComparison.OrdinalIgnoreCase))
            {
                string shown = o.DefaultValue as string ?? Convert.ToString(
                    o.DefaultValue, System.Globalization.CultureInfo.InvariantCulture) ?? "";
                if (shown.Length > 0)
                    description = description.Length > 0
                        ? $"{description} (default: {shown})"
                        : $"(default: {shown})";
            }

            WriteEntry(labels[i], width, description);
        }
    }

    /// <summary>Write one "label    description" row, wrapping the description column.</summary>
    private static void WriteEntry(string label, int labelWidth, string description)
    {
        int available = ConsoleWidth() - IndentWidth - labelWidth - GapWidth;
        if (available < MinDescriptionWidth) available = MinDescriptionWidth;

        var lines = Wrap(description, available);
        string indent = new(' ', IndentWidth);

        if (lines.Count == 0)
        {
            AnsiConsole.MarkupLine($"{indent}{Markup.Escape(label)}");
            return;
        }

        // Description text is author-controlled markup; render it rather than escaping so
        // existing [[ ]] escapes in option help resolve to literal brackets.
        AnsiConsole.MarkupLine(
            $"{indent}[green]{Markup.Escape(label.PadRight(labelWidth))}[/]{new string(' ', GapWidth)}{lines[0]}");

        string hanging = new(' ', IndentWidth + labelWidth + GapWidth);
        for (int i = 1; i < lines.Count; i++)
            AnsiConsole.MarkupLine($"{hanging}{lines[i]}");
    }

    /// <summary>Greedy word wrap. Operates on rendered width, ignoring markup tags.</summary>
    private static List<string> Wrap(string text, int width)
    {
        if (string.IsNullOrWhiteSpace(text)) return [];
        if (width < 1) width = MinDescriptionWidth;

        var lines = new List<string>();
        var current = new System.Text.StringBuilder();
        int visible = 0;

        foreach (string word in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            int wordWidth = MarkupRenderer.Render(word, ansi: false).Length;

            if (visible > 0 && visible + 1 + wordWidth > width)
            {
                lines.Add(current.ToString());
                current.Clear();
                visible = 0;
            }

            if (visible > 0) { current.Append(' '); visible++; }
            current.Append(word);
            visible += wordWidth;
        }

        if (current.Length > 0) lines.Add(current.ToString());
        return lines;
    }

    private static int ConsoleWidth()
    {
        if (Console.IsOutputRedirected) return FallbackConsoleWidth;
        try
        {
            int w = Console.WindowWidth;
            return w > 20 ? w : FallbackConsoleWidth;
        }
        catch (IOException)
        {
            return FallbackConsoleWidth;
        }
    }
}
