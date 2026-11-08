using System.Diagnostics;
using System.Globalization;
using System.Text.RegularExpressions;

namespace OpenTail.Stingray.Vision;

/// <summary>What a clip reports before any frame is read.</summary>
public sealed record VideoClipInfo(int Width, int Height, double Fps);

/// <summary>
/// Supplies decoded frames. The managed core never decodes video itself; a host plugs in a source, as TensorSharp does.
/// </summary>
public interface IVideoFrameSource
{
    VideoClipInfo Probe(string path);

    /// <summary>Frames as tightly packed RGB24 (width * height * 3 bytes), sampled at <paramref name="fps"/> and scaled to the size.</summary>
    IEnumerable<byte[]> ReadFrames(string path, double fps, int width, int height);
}

/// <summary>
/// Decodes with an operator-installed ffmpeg executable. The executable is found through <c>STINGRAY_FFMPEG</c>, then PATH.
/// It is never bundled or downloaded.
/// </summary>
public sealed partial class FfmpegVideoFrameSource : IVideoFrameSource
{
    private readonly string _ffmpeg;

    public FfmpegVideoFrameSource(string? ffmpegPath = null)
    {
        _ffmpeg = ffmpegPath ?? Locate() ?? throw new FileNotFoundException(
            "ffmpeg was not found. Set STINGRAY_FFMPEG to its path, or put ffmpeg on PATH.");
    }

    /// <summary>The ffmpeg path this source will run, or null when none is configured or found.</summary>
    public static string? Locate()
    {
        string? configured = Environment.GetEnvironmentVariable("STINGRAY_FFMPEG");
        if (!string.IsNullOrWhiteSpace(configured)) return File.Exists(configured) ? configured : null;
        string name = OperatingSystem.IsWindows() ? "ffmpeg.exe" : "ffmpeg";
        foreach (string dir in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator))
        {
            try
            {
                if (!string.IsNullOrWhiteSpace(dir) && File.Exists(Path.Combine(dir, name))) return Path.Combine(dir, name);
            }
            catch (ArgumentException) { }
        }
        return null;
    }

    public VideoClipInfo Probe(string path)
    {
        if (!File.Exists(path)) throw new FileNotFoundException("video not found", path);
        // `ffmpeg -i <file>` with no output exits non-zero by design; the stream description is on stderr.
        return ParseProbeOutput(RunCapture(["-hide_banner", "-i", path]), Path.GetFileName(path));
    }

    /// <summary>Reads the video stream's size and rate from the stderr of <c>ffmpeg -i</c>.</summary>
    public static VideoClipInfo ParseProbeOutput(string ffmpegStderr, string name = "clip")
    {
        Match size = VideoStreamSize().Match(ffmpegStderr);
        if (!size.Success) throw new InvalidDataException($"no video stream in '{name}'.");
        Match fps = VideoRate().Match(ffmpegStderr);
        if (!fps.Success) throw new InvalidDataException($"no video frame rate in '{name}'.");
        double rate = double.Parse(fps.Groups[1].Value, CultureInfo.InvariantCulture);
        if (!double.IsFinite(rate) || rate <= 0) throw new InvalidDataException($"invalid video frame rate in '{name}'.");
        return new VideoClipInfo(int.Parse(size.Groups[1].Value, CultureInfo.InvariantCulture),
            int.Parse(size.Groups[2].Value, CultureInfo.InvariantCulture), rate);
    }

    public IEnumerable<byte[]> ReadFrames(string path, double fps, int width, int height)
    {
        if (!double.IsFinite(fps) || fps <= 0) throw new ArgumentOutOfRangeException(nameof(fps), "the frame rate must be a positive number.");
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width), "the frame width must be positive.");
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height), "the frame height must be positive.");
        if (!File.Exists(path)) throw new FileNotFoundException("video not found", path);
        string filter = FormattableString.Invariant($"fps={fps},scale={width}:{height}");
        var psi = Start(["-hide_banner", "-loglevel", "error", "-i", path, "-vf", filter,
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]);
        using var proc = Process.Start(psi) ?? throw new InvalidOperationException("ffmpeg did not start.");
        Task<string> err = proc.StandardError.ReadToEndAsync();
        int frameBytes = width * height * 3;
        var buffer = new byte[frameBytes];
        var stream = proc.StandardOutput.BaseStream;
        while (true)
        {
            int got = 0;
            while (got < frameBytes)
            {
                int n = stream.Read(buffer, got, frameBytes - got);
                if (n == 0) break;
                got += n;
            }
            if (got < frameBytes) break; // end of clip; a partial trailing frame is dropped
            yield return (byte[])buffer.Clone();
        }
        proc.WaitForExit();
        if (proc.ExitCode != 0)
            throw new InvalidOperationException($"ffmpeg failed ({proc.ExitCode}): {err.Result.Trim()}");
    }

    private string RunCapture(string[] args)
    {
        using var proc = Process.Start(Start(args)) ?? throw new InvalidOperationException("ffmpeg did not start.");
        Task<string> err = proc.StandardError.ReadToEndAsync();
        proc.StandardOutput.ReadToEnd();
        proc.WaitForExit();
        return err.Result;
    }

    private ProcessStartInfo Start(string[] args)
    {
        var psi = new ProcessStartInfo(_ffmpeg)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        foreach (string a in args) psi.ArgumentList.Add(a);
        return psi;
    }

    [GeneratedRegex(@"Video:.*?,\s*(\d{2,5})x(\d{2,5})")]
    private static partial Regex VideoStreamSize();

    [GeneratedRegex(@"(\d+(?:\.\d+)?)\s*fps")]
    private static partial Regex VideoRate();
}

/// <summary>Turns RGB24 frames into the CHW float layout the vision encoder takes, and pairs them for the temporal kernels.</summary>
public static class VideoTemporalPairs
{
    /// <summary>RGB24 to CHW in [-1, 1], the same normalisation as the still-image path.</summary>
    public static float[] RgbToChw(byte[] rgb, int width, int height)
    {
        int plane = width * height;
        var chw = new float[3 * plane];
        for (int p = 0; p < plane; p++)
            for (int c = 0; c < 3; c++)
                chw[c * plane + p] = 2f * rgb[p * 3 + c] / 255f - 1f;
        return chw;
    }

    /// <summary>Consecutive frame pairs (frame 0 and frame 1 of each temporal patch). An odd trailing frame is repeated.</summary>
    public static IEnumerable<(float[] Frame0, float[] Frame1)> Pair(IEnumerable<byte[]> frames, int width, int height)
    {
        float[]? pending = null;
        foreach (byte[] rgb in frames)
        {
            float[] chw = RgbToChw(rgb, width, height);
            if (pending is null) { pending = chw; continue; }
            yield return (pending, chw);
            pending = null;
        }
        if (pending is not null) yield return (pending, pending);
    }
}
