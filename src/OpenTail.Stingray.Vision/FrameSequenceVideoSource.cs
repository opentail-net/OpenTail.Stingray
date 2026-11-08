namespace OpenTail.Stingray.Vision;

/// <summary>
/// Frames already on disk: a directory of PNG files, one per frame, read in file-name order. No decoder is involved. A
/// directory has no frame rate, so the caller states the rate the frames were sampled at. Frames are not resampled in time
/// (a different <c>fps</c> is refused). Frames of another size are resized with the bicubic resampler the still-image path uses.
/// </summary>
public sealed class FrameSequenceVideoSource : IVideoFrameSource
{
    private readonly double _sourceFps;

    public FrameSequenceVideoSource(double sourceFps)
    {
        if (!double.IsFinite(sourceFps) || sourceFps <= 0)
            throw new ArgumentOutOfRangeException(nameof(sourceFps), "the frame rate must be a positive number.");
        _sourceFps = sourceFps;
    }

    /// <summary>The size of the first frame and the stated rate. <paramref name="path"/> is the frame directory.</summary>
    public VideoClipInfo Probe(string path)
    {
        string first = FrameFiles(path).FirstOrDefault() ?? throw new InvalidDataException($"no PNG frames in '{path}'.");
        ImageIO.LoadRgb(first, out int width, out int height);
        return new VideoClipInfo(width, height, _sourceFps);
    }

    public IEnumerable<byte[]> ReadFrames(string path, double fps, int width, int height)
    {
        if (fps != _sourceFps)
            throw new NotSupportedException($"the frames are sampled at {_sourceFps} fps; resampling to {fps} fps is not supported.");
        foreach (string file in FrameFiles(path))
        {
            byte[] rgb = ImageIO.LoadRgb(file, out int w, out int h);
            yield return w == width && h == height ? rgb : LlamaImgTool.ResizeBicubic(rgb, w, h, width, height);
        }
    }

    private static IEnumerable<string> FrameFiles(string directory)
    {
        if (!Directory.Exists(directory)) throw new DirectoryNotFoundException($"frame directory not found: {directory}");
        return Directory.EnumerateFiles(directory, "*.png").OrderBy(f => f, Comparer<string>.Create(NaturalCompare));
    }

    /// <summary>Orders "frame-2.png" before "frame-10.png": runs of digits compare by value, other text by character.</summary>
    private static int NaturalCompare(string a, string b)
    {
        int i = 0, j = 0;
        while (i < a.Length && j < b.Length)
        {
            if (char.IsDigit(a[i]) && char.IsDigit(b[j]))
            {
                int si = i, sj = j;
                while (i < a.Length && char.IsDigit(a[i])) i++;
                while (j < b.Length && char.IsDigit(b[j])) j++;
                string na = a[si..i].TrimStart('0'), nb = b[sj..j].TrimStart('0');
                int c = na.Length.CompareTo(nb.Length);
                if (c == 0) c = string.CompareOrdinal(na, nb);
                if (c != 0) return c;
            }
            else
            {
                int c = a[i].CompareTo(b[j]);
                if (c != 0) return c;
                i++; j++;
            }
        }
        return (a.Length - i).CompareTo(b.Length - j);
    }
}
