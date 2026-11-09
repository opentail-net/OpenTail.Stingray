using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Vision;

namespace OpenTail.Stingray.Server.Endpoints;

/// <summary>
/// Video input for the qwen4exp pass (Phase 5.3, server path): a base64 MP4 data URL, decoded by the operator's ffmpeg
/// (STINGRAY_FFMPEG, then PATH; never bundled or downloaded). Frames are sampled at <see cref="Fps"/>, resized to
/// <see cref="Side"/> pixels, and capped at <see cref="MaxFrames"/> (8 seconds at 4 fps). The engine does the splicing.
/// </summary>
internal static class VideoContent
{
    /// <summary>The placeholder the engine expands into the video's layout (<c>InferenceEngine.SpliceQwenVideos</c>).</summary>
    public const string Placeholder = "<|video_pad|>";

    public const int Side = 224;
    public const float Fps = 4.0f;
    public const int MaxFrames = 32;
    public const int MaxVideos = 1;

    public static void CheckCap(int count)
    {
        if (count >= MaxVideos)
            throw new ImageContentException($"at most {MaxVideos} video is supported per request.");
    }

    /// <summary>Decodes a base64 MP4 data URL (<c>data:video/mp4;base64,...</c>) into the projector's input.</summary>
    public static QwenVideoInput FromDataUrl(string url)
    {
        const string prefix = "data:";
        if (!url.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            throw new ImageContentException(
                "video_url must be a base64 data URL (data:video/mp4;base64,...); remote URL fetching is not supported.");
        int comma = url.IndexOf(',');
        if (comma < 0)
            throw new ImageContentException("malformed data URL (missing comma).");
        string meta = url[prefix.Length..comma];
        if (!meta.StartsWith("video/mp4", StringComparison.OrdinalIgnoreCase) || !meta.Contains("base64", StringComparison.OrdinalIgnoreCase))
            throw new ImageContentException("only base64-encoded MP4 video is supported.");

        byte[] bytes;
        try { bytes = Convert.FromBase64String(url[(comma + 1)..]); }
        catch (FormatException) { throw new ImageContentException("video data is not valid base64."); }

        string? ffmpeg = FfmpegVideoFrameSource.Locate();
        if (ffmpeg is null)
            throw new ImageContentException("video input needs ffmpeg on the server (set STINGRAY_FFMPEG or put ffmpeg on PATH).");

        string dir = Path.Combine(Path.GetTempPath(), "stingray-video-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            string path = Path.Combine(dir, "clip.mp4");
            File.WriteAllBytes(path, bytes);
            var source = new FfmpegVideoFrameSource(ffmpeg);
            var frames = source.ReadFrames(path, Fps, Side, Side).Take(MaxFrames)
                .Select(f => VideoTemporalPairs.RgbToChw(f, Side, Side)).ToList();
            if (frames.Count == 0)
                throw new ImageContentException("the video has no decodable frames.");
            return new QwenVideoInput(frames, Side, Fps);
        }
        catch (ImageContentException) { throw; }
        catch (Exception ex) { throw new ImageContentException($"video decode failed: {ex.Message}"); }
        finally
        {
            try { Directory.Delete(dir, recursive: true); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
    }
}
