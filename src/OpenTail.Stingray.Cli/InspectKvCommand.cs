
namespace OpenTail.Stingray.Cli;

/// <summary>
/// Diagnostic command inspecting KV cache capacity, page distribution, forking and CoW statistics (§36 of Paged KV Cache Plan).
/// Usage: <c>stingray inspect-kv [--json]</c>
/// </summary>
public sealed class InspectKvCommand : Command<InspectKvCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandOption("--pages <PAGES>")]
        [Description("Simulate total page capacity (default: 65536)")]
        public int TotalPages { get; init; } = 65536;

        [CommandOption("--page-size <TOKENS>")]
        [Description("Tokens per page (default: 32)")]
        public int PageSizeTokens { get; init; } = 32;

        [CommandOption("--json")]
        [Description("Write machine-readable JSON snapshot to stdout")]
        public bool Json { get; init; }
    }

    protected override int Execute(Settings settings, CancellationToken ct = default)
    {
        using var cache = new CpuKvCache(settings.TotalPages, settings.PageSizeTokens);
        var stats = cache.GetStatistics();

        if (settings.Json)
        {
            var json = JsonSerializer.Serialize(stats, InspectKvJsonContext.Default.KvCacheStatistics);
            Console.WriteLine(json);
            return 0;
        }

        Console.WriteLine("KV CACHE DIAGNOSTICS");
        Console.WriteLine("──────────────────────────────────────────────────");
        Console.WriteLine($"Backend           : CPU");
        Console.WriteLine($"Page Size         : {cache.PageSizeTokens} tokens/page");
        Console.WriteLine($"Total Capacity    : {FormatBytes(stats.CapacityBytes)}");
        Console.WriteLine($"Used Memory       : {FormatBytes(stats.UsedBytes)}");
        Console.WriteLine($"Free Memory       : {FormatBytes(stats.FreeBytes)}");
        Console.WriteLine($"Reserved Memory   : {FormatBytes(stats.ReservedBytes)}");
        Console.WriteLine();
        Console.WriteLine("Pages");
        Console.WriteLine($"  Total Pages     : {stats.TotalPages}");
        Console.WriteLine($"  Used Pages      : {stats.UsedPages}");
        Console.WriteLine($"  Free Pages      : {stats.FreePages}");
        Console.WriteLine($"  Shared Pages    : {stats.SharedPages}");
        Console.WriteLine();
        Console.WriteLine("Activity Metrics");
        Console.WriteLine($"  Allocations     : {stats.Allocations}");
        Console.WriteLine($"  Releases        : {stats.Releases}");
        Console.WriteLine($"  Forks           : {stats.Forks}");
        Console.WriteLine($"  CoW Copies      : {stats.CopyOnWriteCopies}");
        Console.WriteLine($"  Evictions       : {stats.Evictions}");
        Console.WriteLine("──────────────────────────────────────────────────");

        return 0;
    }

    private static string FormatBytes(long bytes)
    {
        string[] units = { "B", "KiB", "MiB", "GiB", "TiB" };
        double val = bytes;
        int i = 0;
        while (val >= 1024 && i < units.Length - 1)
        {
            val /= 1024;
            i++;
        }
        return $"{val:F2} {units[i]}";
    }
}

[JsonSourceGenerationOptions(WriteIndented = true)]
[JsonSerializable(typeof(KvCacheStatistics))]
internal partial class InspectKvJsonContext : JsonSerializerContext
{
}
