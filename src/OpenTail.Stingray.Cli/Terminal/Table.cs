namespace OpenTail.Stingray.Cli.Terminal;

/// <summary>Border presets for <see cref="Table"/>.</summary>
public enum TableBorder
{
    /// <summary>No border characters; columns separated by whitespace.</summary>
    None,

    /// <summary>Header underlined with a rule, no outer frame.</summary>
    Simple,
}

/// <summary>A single table column.</summary>
public sealed class TableColumn
{
    internal string Header { get; }
    internal bool RightAlign { get; private set; }
    internal bool NoWrapping { get; private set; }

    /// <summary>Create a column with the given (markup-capable) header.</summary>
    public TableColumn(string header) => Header = header;

    /// <summary>Never shrink or wrap this column, even under terminal-width pressure.</summary>
    public TableColumn NoWrap() { NoWrapping = true; return this; }

    /// <summary>Right-align this column's cells.</summary>
    public TableColumn RightAligned() { RightAlign = true; return this; }
}

/// <summary>
/// A plain text table sized to its content. Cells may contain markup; column widths are
/// measured on the <em>rendered</em> text so ANSI escapes never distort alignment.
///
/// <para>
/// When stdout is an interactive terminal, wrappable columns shrink and wrap to fit the
/// window. When redirected, no wrapping is applied — piped output stays one row per line so
/// it remains greppable.
/// </para>
/// </summary>
public sealed class Table
{
    private const int MinWrapWidth = 12;
    private const int ColumnGap = 2;

    private readonly List<TableColumn> _columns = [];
    private readonly List<string[]> _rows = [];
    private TableBorder _border = TableBorder.Simple;

    /// <summary>Set the border preset.</summary>
    public Table Border(TableBorder border) { _border = border; return this; }

    /// <summary>Append a column.</summary>
    public Table AddColumn(TableColumn column) { _columns.Add(column); return this; }

    /// <summary>Append a column with the given header.</summary>
    public Table AddColumn(string header) { _columns.Add(new TableColumn(header)); return this; }

    /// <summary>Append a row. Missing trailing cells are treated as empty.</summary>
    public Table AddRow(params string[] cells) { _rows.Add(cells); return this; }

    /// <summary>Visible text of a markup string, i.e. what width calculations must use.</summary>
    private static string Plain(string markup) => MarkupRenderer.Render(markup, ansi: false);

    internal void Render(TextWriter output, bool color)
    {
        if (_columns.Count == 0) return;

        var widths = NaturalWidths();
        ShrinkToTerminal(widths);

        WriteRow(output, color, [.. _columns.Select(c => c.Header)], widths);

        if (_border == TableBorder.Simple)
        {
            output.WriteLine(string.Join(
                new string(' ', ColumnGap), widths.Select(w => new string('-', w))));
        }

        foreach (var row in _rows)
            WriteRow(output, color, row, widths);
    }

    private int[] NaturalWidths()
    {
        var widths = new int[_columns.Count];
        for (int c = 0; c < _columns.Count; c++)
            widths[c] = Plain(_columns[c].Header).Length;

        foreach (var row in _rows)
            for (int c = 0; c < _columns.Count && c < row.Length; c++)
                widths[c] = Math.Max(widths[c], Plain(row[c] ?? "").Length);

        return widths;
    }

    /// <summary>
    /// Shrink wrappable columns so the table fits the terminal. No-op when output is
    /// redirected (unbounded width keeps piped rows on one line).
    /// </summary>
    private void ShrinkToTerminal(int[] widths)
    {
        if (Console.IsOutputRedirected) return;

        int available;
        try
        {
            available = Console.WindowWidth;
        }
        catch (IOException)
        {
            return;     // No console attached; leave widths alone.
        }
        if (available <= 0) return;

        int gaps = ColumnGap * (_columns.Count - 1);
        int total = widths.Sum() + gaps;
        int overflow = total - available;
        if (overflow <= 0) return;

        // Only wrappable columns give up space, widest first, down to MinWrapWidth.
        var shrinkable = Enumerable.Range(0, _columns.Count)
            .Where(c => !_columns[c].NoWrapping && widths[c] > MinWrapWidth)
            .OrderByDescending(c => widths[c])
            .ToList();

        foreach (int c in shrinkable)
        {
            if (overflow <= 0) break;
            int give = Math.Min(overflow, widths[c] - MinWrapWidth);
            widths[c] -= give;
            overflow -= give;
        }
    }

    private void WriteRow(TextWriter output, bool color, string[] cells, int[] widths)
    {
        // Wrap each cell to its column width, then emit the row as N physical lines.
        var wrapped = new List<string>[_columns.Count];
        int height = 1;

        for (int c = 0; c < _columns.Count; c++)
        {
            string cell = c < cells.Length ? cells[c] ?? "" : "";
            wrapped[c] = WrapCell(cell, widths[c]);
            height = Math.Max(height, wrapped[c].Count);
        }

        for (int line = 0; line < height; line++)
        {
            var sb = new System.Text.StringBuilder();

            for (int c = 0; c < _columns.Count; c++)
            {
                string piece = line < wrapped[c].Count ? wrapped[c][line] : "";

                // Pad against visible length; the rendered form may carry zero-width ANSI codes.
                string rendered = MarkupRenderer.Render(piece, color);
                int pad = Math.Max(0, widths[c] - Plain(piece).Length);

                if (c > 0) sb.Append(' ', ColumnGap);
                if (_columns[c].RightAlign) sb.Append(' ', pad).Append(rendered);
                else                        sb.Append(rendered).Append(' ', pad);
            }

            // Trailing padding is invisible but pollutes diffs and piped output.
            output.WriteLine(sb.ToString().TrimEnd());
        }
    }

    /// <summary>
    /// Split a cell into lines no wider than <paramref name="width"/>, preferring word
    /// boundaries. Wrapping operates on plain text: the cells that actually wrap are
    /// escaped data values, so there is no style to preserve across the break.
    /// </summary>
    private static List<string> WrapCell(string cell, int width)
    {
        string plain = Plain(cell);
        if (plain.Length <= width) return [cell];

        var lines = new List<string>();
        int pos = 0;

        while (pos < plain.Length)
        {
            int take = Math.Min(width, plain.Length - pos);

            // Prefer breaking at the last space inside the window.
            if (pos + take < plain.Length)
            {
                int space = plain.LastIndexOf(' ', pos + take - 1, take);
                if (space > pos) take = space - pos;
            }

            lines.Add(Markup.Escape(plain.Substring(pos, take).TrimEnd()));
            pos += take;

            while (pos < plain.Length && plain[pos] == ' ') pos++;
        }

        return lines;
    }
}
