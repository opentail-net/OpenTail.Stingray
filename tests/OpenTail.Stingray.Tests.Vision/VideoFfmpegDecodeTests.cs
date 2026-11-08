using System.Diagnostics;
using OpenTail.Stingray.Engine.Verification;
using OpenTail.Stingray.Vision;

namespace OpenTail.Stingray.Tests.Vision;

/// <summary>
/// Phase 5.3 (Qwen 3.8 plan): the ffmpeg frame source decodes a real clip, and its decoded frame pairs feed the two-frame
/// video path of the projector. The clip is generated here with ffmpeg from synthetic frames (test only). Skipped when
/// ffmpeg or the mmproj is absent.
/// </summary>
public sealed class VideoFfmpegDecodeTests
{
    private const int Side = 224;
    private const int Frames = 8;   // 2 s at 4 fps
    private const double Fps = 4.0;

    private static string? FindMmproj() => ModelLocator.FindDeep("mmproj-Qwen3.8-Flash-Next-F16.gguf");

    /// <summary>A frame whose content moves from one frame to the next, so consecutive frames differ.</summary>
    private static byte[] SyntheticFrame(int k)
    {
        var rgb = new byte[Side * Side * 3];
        int bar = 16 + k * 12;
        for (int y = 0; y < Side; y++)
            for (int x = 0; x < Side; x++)
            {
                int i = (y * Side + x) * 3;
                bool inBar = x >= bar && x < bar + 48;
                rgb[i] = (byte)(inBar ? 230 : 30);
                rgb[i + 1] = (byte)(y < Side / 2 ? 90 : 160);
                rgb[i + 2] = (byte)(x < Side / 2 ? 60 : 200);
            }
        return rgb;
    }

    /// <summary>Writes an MP4 clip from the synthetic frames by piping raw RGB into ffmpeg.</summary>
    private static string MakeClip(string ffmpeg)
    {
        string dir = Path.Combine(Path.GetTempPath(), "stingray-video-test");
        Directory.CreateDirectory(dir);
        string path = Path.Combine(dir, $"synthetic-{Guid.NewGuid():N}.mp4");
        var psi = new ProcessStartInfo(ffmpeg)
        {
            RedirectStandardInput = true,
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        foreach (string a in new[] { "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                     "-s", $"{Side}x{Side}", "-r", "4", "-i", "-", "-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p", path })
            psi.ArgumentList.Add(a);
        using var proc = Process.Start(psi)!;
        Task<string> err = proc.StandardError.ReadToEndAsync();
        Task drain = proc.StandardOutput.BaseStream.CopyToAsync(Stream.Null);
        for (int k = 0; k < Frames; k++)
        {
            byte[] f = SyntheticFrame(k);
            proc.StandardInput.BaseStream.Write(f, 0, f.Length);
        }
        proc.StandardInput.Close();
        proc.WaitForExit();
        drain.Wait();
        Assert.True(proc.ExitCode == 0, $"ffmpeg could not write the test clip: {err.Result}");
        return path;
    }

    /// <summary>The ffmpeg path only if it actually runs. A PATH entry can exist without a working target (for example a scoop shim).</summary>
    private static string? FfmpegOrSkip()
    {
        string? path = FfmpegVideoFrameSource.Locate();
        if (path is null) return null;
        try
        {
            using var proc = Process.Start(new ProcessStartInfo(path, "-version")
            {
                RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true,
            });
            proc!.StandardOutput.ReadToEnd();
            proc.StandardError.ReadToEnd();
            proc.WaitForExit();
            return proc.ExitCode == 0 ? path : null;
        }
        catch (System.ComponentModel.Win32Exception) { return null; }
    }

    [Fact]
    public void ProbeParser_ReadsSizeAndRateFromFfmpegStderr()
    {
        // The text ffmpeg prints for "ffmpeg -i clip.mp4"; no ffmpeg needed for this check.
        const string stderr = """
Input #0, mov,mp4, from 'clip.mp4':
  Duration: 00:00:02.00, bitrate: 102 kb/s
  Stream #0:0[0x1](und): Video: mpeg4 (Simple Profile), yuv420p, 224x224 [SAR 1:1 DAR 1:1], 102 kb/s, 4 fps, 4 tbr, 12288 tbn (default)
""";
        var info = OpenTail.Stingray.Vision.FfmpegVideoFrameSource.ParseProbeOutput(stderr, "clip.mp4");
        Assert.Equal(224, info.Width);
        Assert.Equal(224, info.Height);
        Assert.Equal(4.0, info.Fps);
    }

    [Fact]
    public void Probe_ReportsTheClipSizeAndRate()
    {
        string? ffmpeg = FfmpegOrSkip();
        Assert.SkipUnless(ffmpeg is not null, "ffmpeg is required (STINGRAY_FFMPEG or PATH).");
        string clip = MakeClip(ffmpeg!);
        try
        {
            var info = new FfmpegVideoFrameSource(ffmpeg).Probe(clip);
            Assert.Equal(Side, info.Width);
            Assert.Equal(Side, info.Height);
            Assert.InRange(info.Fps, 3.9, 4.1);
        }
        finally { File.Delete(clip); }
    }

    [Fact]
    public void ReadFrames_ReturnsTheClipFramesWithinCodecLoss()
    {
        string? ffmpeg = FfmpegOrSkip();
        Assert.SkipUnless(ffmpeg is not null, "ffmpeg is required (STINGRAY_FFMPEG or PATH).");
        string clip = MakeClip(ffmpeg!);
        try
        {
            var source = new FfmpegVideoFrameSource(ffmpeg);
            var frames = source.ReadFrames(clip, Fps, Side, Side).ToList();
            Assert.Equal(Frames, frames.Count);

            byte[] original = SyntheticFrame(0);
            double meanAbs = 0;
            for (int i = 0; i < original.Length; i++) meanAbs += Math.Abs(original[i] - frames[0][i]);
            meanAbs /= original.Length;
            Assert.True(meanAbs < 25, $"decoded frame 0 differs too much from the source (mean |diff| {meanAbs:F1})");
        }
        finally { File.Delete(clip); }
    }

    [Fact]
    public void DecodedPairs_RunThroughTheProjectorTwoFramePath()
    {
        string? ffmpeg = FfmpegOrSkip();
        Assert.SkipUnless(ffmpeg is not null, "ffmpeg is required (STINGRAY_FFMPEG or PATH).");
        string? mmproj = FindMmproj();
        Assert.SkipUnless(mmproj is not null, "mmproj-Qwen3.8-Flash-Next-F16.gguf is required (904 MB).");

        string clip = MakeClip(ffmpeg!);
        try
        {
            var source = new FfmpegVideoFrameSource(ffmpeg);
            var frames = source.ReadFrames(clip, Fps, Side, Side);
            using var vision = QwenVlVisionModel.Open(mmproj!);
            var encoder = new QwenVlVisionEncoder(vision);

            int pairs = 0;
            int firstTokens = -1;
            foreach (var (f0, f1) in VideoTemporalPairs.Pair(frames, Side, Side))
            {
                float[] tokens = encoder.ForwardVideo(f0, f1, Side, Side, out int n);
                if (firstTokens < 0) firstTokens = n;
                Assert.Equal(firstTokens, n);
                Assert.True(tokens.Length > 0);
                Assert.All(tokens.Take(256), v => Assert.True(float.IsFinite(v)));
                pairs++;
            }

            Assert.Equal(Frames / 2, pairs);
            Assert.Equal((Side / 28) * (Side / 28), firstTokens); // 16x16 patches merged 2x2
        }
        finally { File.Delete(clip); }
    }
}
