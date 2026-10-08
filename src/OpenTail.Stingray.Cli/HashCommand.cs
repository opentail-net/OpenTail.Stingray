using System.ComponentModel;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Cli;

/// <summary>
/// Prints a model file's SHA-256 and size, the identity a golden pins. The hash is cached beside the model (<c>&lt;file&gt;.sha256</c>, keyed by size
/// and modification time) so a large checkpoint is only read once.
/// </summary>
public sealed class HashCommand : Command<HashCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("-m|--model <PATH>")]
        [Description("Model file to fingerprint")]
        public string ModelPath { get; init; } = "";
    }

    protected override int Execute(Settings settings, CancellationToken cancellation)
    {
        if (!ModelPathResolver.TryRequireModelFile(settings.ModelPath, out int modelFileExit)) return modelFileExit;
        string last = "";
        var fp = ModelFingerprinter.Compute(settings.ModelPath, p =>
        {
            string pct = $"{p * 100:F0}%";
            if (pct != last) { last = pct; Console.Error.Write($"\rhashing {pct}  "); }
        });
        if (last.Length > 0) Console.Error.WriteLine();
        Console.WriteLine($"{fp.Sha256}  {fp.SizeBytes}  {Path.GetFileName(settings.ModelPath)}{(fp.FromCache ? "  (cached)" : "")}");
        return 0;
    }
}
