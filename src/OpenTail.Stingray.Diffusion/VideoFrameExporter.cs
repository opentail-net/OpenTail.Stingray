namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// High-level video frame export utility supporting animated GIFs and numbered PNG image sequences.
/// </summary>
public static class VideoFrameExporter
{
    /// <summary>
    /// Exports a multi-frame video sequence to the specified output path or directory.
    /// Auto-detects .gif extensions or writes a numbered PNG image sequence.
    /// </summary>
    public static void Export(
        string outputPath,
        IReadOnlyList<float[]> framesRgb,
        int width,
        int height,
        int fps = 24)
    {
        ArgumentNullException.ThrowIfNull(outputPath);
        if (framesRgb == null || framesRgb.Count == 0)
            throw new ArgumentException("Frames collection cannot be null or empty.", nameof(framesRgb));

        if (outputPath.EndsWith(".gif", StringComparison.OrdinalIgnoreCase))
        {
            var dir = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }

            GifWriter.SaveGif(outputPath, framesRgb, width, height, fps);
        }
        else
        {
            // Directory or base path for PNG sequence
            string targetDir = Path.HasExtension(outputPath)
                ? (Path.GetDirectoryName(outputPath) ?? ".")
                : outputPath;

            string baseName = Path.HasExtension(outputPath)
                ? Path.GetFileNameWithoutExtension(outputPath)
                : "frame";

            if (!Directory.Exists(targetDir))
            {
                Directory.CreateDirectory(targetDir);
            }

            for (int i = 0; i < framesRgb.Count; i++)
            {
                string framePath = Path.Combine(targetDir, $"{baseName}_{i + 1:D4}.png");
                PngWriter.Write(framePath, framesRgb[i], width, height);
            }
        }
    }
}
