using System.ComponentModel;
using OpenTail.Stingray.Cli.Terminal;
using OpenTail.Stingray.Cli.CommandLine;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// Dumps the tensor index of a GGUF model file (name, dtype, shape, byte size).
/// Usage:
///   opentail-llm-cli list-tensors -m model.gguf                       (all tensors)
///   opentail-llm-cli list-tensors -m model.gguf --layer 0             (only blk.0.*)
///   opentail-llm-cli list-tensors -m model.gguf --filter ssm          (substring match)
///   opentail-llm-cli list-tensors -m model.gguf --summary              (group by suffix, count + total size)
/// </summary>
public sealed class ListTensorsCommand : Command<ListTensorsCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-m|--model")]
        [Description("Path to GGUF model file")]
        public string? ModelPath { get; init; }

        [CommandOption("--layer")]
        [Description("Show only tensors for this layer index (matches blk.<N>.*)")]
        public int? Layer { get; init; }

        [CommandOption("--filter")]
        [Description("Case-insensitive substring filter on tensor name")]
        public string? Filter { get; init; }

        [CommandOption("--summary")]
        [Description("Group tensors by name suffix; show count and total bytes per group")]
        public bool Summary { get; init; }
    }

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        var modelPath = settings.ModelPath;
        if (modelPath is null || !File.Exists(modelPath))
        {
            AnsiConsole.MarkupLine("[red]Error:[/] No model file found. Use [yellow]-m <path>[/]");
            return 1;
        }

        if (modelPath.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
        {
            using var onnx = OnnxModelSession.TryLoad(modelPath);
            if (onnx == null)
            {
                AnsiConsole.MarkupLine("[red]Error:[/] Could not load ONNX model file. Ensure onnxruntime.dll is available.");
                return 1;
            }

            AnsiConsole.MarkupLine($"[bold]{Markup.Escape(Path.GetFileName(modelPath))}[/]  " +
                $"ONNX Graph  |  " +
                $"[cyan]{onnx.InputNames.Count}[/] inputs  |  " +
                $"[cyan]{onnx.OutputNames.Count}[/] outputs");
            AnsiConsole.WriteLine();

            var onnxTable = new Table()
                .Border(TableBorder.Simple)
                .AddColumn(new TableColumn("[bold]Type[/]").NoWrap())
                .AddColumn(new TableColumn("[bold]Tensor Name[/]"));

            foreach (var inName in onnx.InputNames) onnxTable.AddRow("Graph Input", Markup.Escape(inName));
            foreach (var outName in onnx.OutputNames) onnxTable.AddRow("Graph Output", Markup.Escape(outName));

            AnsiConsole.Write(onnxTable);
            return 0;
        }

        using var model = GgufModel.Open(modelPath);

        AnsiConsole.MarkupLine($"[bold]{Markup.Escape(Path.GetFileName(modelPath))}[/]  " +
            $"GGUF v{model.Header.Version}  |  " +
            $"[cyan]{model.Header.TensorCount}[/] tensors  |  " +
            $"[cyan]{model.Metadata.Count}[/] metadata keys");
        AnsiConsole.WriteLine();

        var tensors = (IEnumerable<GgufTensorInfo>)model.Tensors;

        if (settings.Layer is int layer)
        {
            var prefix = $"blk.{layer}.";
            tensors = tensors.Where(t => t.Name.StartsWith(prefix, StringComparison.Ordinal));
        }
        if (!string.IsNullOrEmpty(settings.Filter))
        {
            var f = settings.Filter;
            tensors = tensors.Where(t => t.Name.Contains(f, StringComparison.OrdinalIgnoreCase));
        }

        if (settings.Summary)
        {
            PrintSummary(tensors);
            return 0;
        }

        var table = new Table()
            .Border(TableBorder.Simple)
            .AddColumn(new TableColumn("[bold]Name[/]").NoWrap())
            .AddColumn(new TableColumn("[bold]DType[/]"))
            .AddColumn(new TableColumn("[bold]Shape[/]"))
            .AddColumn(new TableColumn("[bold]Bytes[/]").RightAligned());

        long totalBytes = 0;
        int rows = 0;
        foreach (var t in tensors.OrderBy(x => x.Name, StringComparer.Ordinal))
        {
            table.AddRow(
                Markup.Escape(t.Name),
                Markup.Escape(t.DType.ToString()),
                Markup.Escape("[" + string.Join(", ", t.Dimensions.Take(t.NDimensions)) + "]"),
                FormatBytes(t.ByteSize));
            totalBytes += t.ByteSize;
            rows++;
        }

        AnsiConsole.Write(table);
        AnsiConsole.MarkupLine($"[dim]{rows} tensors, total {FormatBytes(totalBytes)}[/]");
        return 0;
    }

    private static void PrintSummary(IEnumerable<GgufTensorInfo> tensors)
    {
        // Strip "blk.<N>." prefix so per-layer tensors collapse into one group.
        var groups = tensors
            .GroupBy(t => StripBlockPrefix(t.Name))
            .Select(g => new
            {
                Suffix     = g.Key,
                Count      = g.Count(),
                TotalBytes = g.Sum(t => t.ByteSize),
                Sample     = g.First(),
            })
            .OrderByDescending(g => g.TotalBytes);

        var table = new Table()
            .Border(TableBorder.Simple)
            .AddColumn(new TableColumn("[bold]Suffix[/]").NoWrap())
            .AddColumn(new TableColumn("[bold]Count[/]").RightAligned())
            .AddColumn(new TableColumn("[bold]DType[/]"))
            .AddColumn(new TableColumn("[bold]Shape (sample)[/]"))
            .AddColumn(new TableColumn("[bold]Total bytes[/]").RightAligned());

        foreach (var g in groups)
        {
            table.AddRow(
                Markup.Escape(g.Suffix),
                g.Count.ToString(),
                Markup.Escape(g.Sample.DType.ToString()),
                Markup.Escape("[" + string.Join(", ", g.Sample.Dimensions.Take(g.Sample.NDimensions)) + "]"),
                FormatBytes(g.TotalBytes));
        }
        AnsiConsole.Write(table);
    }

    private static string StripBlockPrefix(string name)
    {
        if (!name.StartsWith("blk.", StringComparison.Ordinal)) return name;
        int dot = name.IndexOf('.', 4);
        return dot < 0 ? name : "blk.*" + name[dot..];
    }

    private static string FormatBytes(long bytes)
    {
        if (bytes < 1024)             return $"{bytes} B";
        if (bytes < 1024L * 1024)     return $"{bytes / 1024.0:F1} KiB";
        if (bytes < 1024L * 1024 * 1024) return $"{bytes / (1024.0 * 1024):F1} MiB";
        return $"{bytes / (1024.0 * 1024 * 1024):F2} GiB";
    }
}
