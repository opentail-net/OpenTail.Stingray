using OpenTail.Stingray.Audio;

namespace OpenTail.Stingray.Diffusion.MiniMaxH3;

/// <summary>
/// Host output exporter for MiniMax-H3.
/// Writes decoded video frames (as PNG sequence or animated GIF) and decoded audio (as 32 kHz stereo sidecar WAV).
/// </summary>
public static class MiniMaxH3OutputExporter
{
    /// <summary>
    /// Export decoded RGB video frames to disk (.gif or image sequence directory).
    /// </summary>
    public static void ExportVideo(
        string outputPath,
        IReadOnlyList<float[]> framesRgb,
        int width,
        int height,
        int fps = 24)
    {
        VideoFrameExporter.Export(outputPath, framesRgb, width, height, fps);
    }

    /// <summary>
    /// Export decoded 32 kHz stereo PCM audio to a .wav sidecar file.
    /// </summary>
    public static void ExportAudioWav(
        string outputPath,
        float[] pcmStereo,
        int sampleRate = MiniMaxH3AudioVaeDecoder.SampleRate,
        int channels = MiniMaxH3AudioVaeDecoder.AudioChannels)
    {
        ArgumentNullException.ThrowIfNull(outputPath);
        ArgumentNullException.ThrowIfNull(pcmStereo);

        var dir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
        {
            Directory.CreateDirectory(dir);
        }

        WavWriter.WriteWav(outputPath, pcmStereo, sampleRate, channels);
    }
}
