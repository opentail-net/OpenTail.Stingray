
namespace OpenTail.Stingray.Cli;

/// <summary>
/// Lists GGUF models found on disk: <c>opentail-llm-cli list-models</c>
///
/// <para>The read-only slice of §6 Phase 3's model lifecycle. It deliberately does NOT implement a
/// model store, aliases, downloads, or verification — those need a manifest and network handling.
/// This answers only "what do I have, and does it parse?", which is the question that actually
/// comes up before every run and currently requires a file browser.</para>
///
/// <para>Reads the GGUF index only (memory mapped); no weights are loaded, so listing a directory
/// of large models stays fast.</para>
/// </summary>
public sealed class ListModelsCommand : Command<ListModelsCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-d|--dir <PATH>")]
        [Description("Directory to scan (default: ./models, then the current directory)")]
        public string? Directory { get; init; }

        [CommandOption("--deep")]
        [Description("Open each GGUF index to report architecture and tensor count (slower)")]
        public bool Deep { get; init; }
    }

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        string dir = settings.Directory
            ?? (System.IO.Directory.Exists("models") ? "models" : System.IO.Directory.GetCurrentDirectory());

        if (!System.IO.Directory.Exists(dir))
        {
            Console.Error.WriteLine($"list-models: directory not found: {dir}");
            return 1;
        }

        var files = System.IO.Directory
            .EnumerateFiles(dir, "*.gguf", SearchOption.TopDirectoryOnly)
            .OrderBy(f => Path.GetFileName(f), StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (files.Count == 0)
        {
            Console.WriteLine($"No .gguf files in {Path.GetFullPath(dir)}");
            return 0;
        }

        Console.WriteLine($"{files.Count} model(s) in {Path.GetFullPath(dir)}:");
        Console.WriteLine();

        foreach (string file in files)
        {
            cancellation.ThrowIfCancellationRequested();
            string name = Path.GetFileName(file);
            double gib = new FileInfo(file).Length / (1024.0 * 1024 * 1024);

            if (!settings.Deep)
            {
                Console.WriteLine($"  {name}  ({gib:F2} GiB)");
                continue;
            }

            // --deep opens the index. A model that fails here is exactly the case worth surfacing:
            // present on disk, plausible filename, and unusable.
            try
            {
                using var model = GgufModel.Open(file);
                string arch = model.Metadata.TryGetValue("general.architecture", out object? a)
                    ? Convert.ToString(a) ?? "unknown" : "unknown";
                Console.WriteLine($"  {name}  ({gib:F2} GiB)  arch={arch}  v{model.Header.Version}  {model.Tensors.Count} tensors");
            }
            // Catch broadly on purpose. A corrupt GGUF can fail in many ways — truncated header,
            // bad magic, absurd tensor counts driving overflow or out-of-range reads — and this is
            // an enumeration whose whole value is not stopping at the first bad file. A narrow
            // filter here was a real bug: garbage input escaped it and took down the listing.
            // OperationCanceledException is excluded so Ctrl-C still works.
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                Console.WriteLine($"  {name}  ({gib:F2} GiB)  UNREADABLE: {ex.GetType().Name}");
            }
        }

        Console.WriteLine();
        Console.WriteLine("This lists files on disk; it is not a managed model store. Use `plan -m <path>`");
        Console.WriteLine("to see whether a specific model is supported and how it would be placed.");
        return 0;
    }
}
