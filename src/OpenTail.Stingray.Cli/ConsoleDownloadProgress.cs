namespace OpenTail.Stingray.Cli;

/// <summary>Prints a one-line, in-place download percentage (every 5%) for <c>pull</c> and <c>setup</c>.</summary>
internal sealed class ConsoleDownloadProgress
{
    private int _lastPercent = -1;
    private bool _printed;

    public void Report(long bytes, long? total)
    {
        if (total is not { } t || t <= 0) return;
        int percent = (int)(bytes * 100 / t);
        if (percent == _lastPercent || percent % 5 != 0) return;
        Console.Write($"\r  {percent,3}%  {FormatBytes(bytes)} / {FormatBytes(t)}   ");
        _lastPercent = percent;
        _printed = true;
    }

    /// <summary>Ends the progress line, if one was started.</summary>
    public void Finish()
    {
        if (_printed) Console.WriteLine();
    }

    public static string FormatBytes(long bytes)
    {
        if (bytes < 1024) return $"{bytes} B";
        if (bytes < 1024L * 1024) return $"{bytes / 1024.0:F1} KiB";
        if (bytes < 1024L * 1024 * 1024) return $"{bytes / (1024.0 * 1024):F1} MiB";
        return $"{bytes / (1024.0 * 1024 * 1024):F2} GiB";
    }
}
