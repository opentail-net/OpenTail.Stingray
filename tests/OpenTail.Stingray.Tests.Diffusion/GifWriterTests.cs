using OpenTail.Stingray.Diffusion;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

public sealed class GifWriterTests
{
    [Fact]
    public void WriteGif_ProducesValidGif89aHeaderAndStructure()
    {
        int width = 32;
        int height = 32;
        int numFrames = 3;

        var frames = new List<float[]>();
        for (int f = 0; f < numFrames; f++)
        {
            float[] frame = new float[width * height * 3];
            float shade = (f + 1) / (float)numFrames;
            Array.Fill(frame, shade);
            frames.Add(frame);
        }

        using var ms = new MemoryStream();
        GifWriter.WriteGif(ms, frames, width, height, fps: 24);

        byte[] gifBytes = ms.ToArray();
        Assert.NotEmpty(gifBytes);

        // Header: "GIF89a"
        string header = System.Text.Encoding.ASCII.GetString(gifBytes.AsSpan(0, 6));
        Assert.Equal("GIF89a", header);

        // Dimensions: uint16 LE
        ushort readWidth = BitConverter.ToUInt16(gifBytes, 6);
        ushort readHeight = BitConverter.ToUInt16(gifBytes, 8);
        Assert.Equal(width, readWidth);
        Assert.Equal(height, readHeight);

        // Netscape loop block present
        string netscapeStr = System.Text.Encoding.ASCII.GetString(gifBytes);
        Assert.Contains("NETSCAPE2.0", netscapeStr);

        // Ends with GIF trailer ';' (0x3B)
        Assert.Equal((byte)0x3B, gifBytes[^1]);
    }

    [Fact]
    public void VideoFrameExporter_SaveAsGif_CreatesFileOnDisk()
    {
        string tempPath = Path.Combine(Path.GetTempPath(), $"stingray_test_{Guid.NewGuid():N}.gif");
        try
        {
            int width = 16;
            int height = 16;
            var frames = new List<float[]>
            {
                new float[16 * 16 * 3],
                new float[16 * 16 * 3]
            };

            VideoFrameExporter.Export(tempPath, frames, width, height, fps: 12);

            Assert.True(File.Exists(tempPath));
            Assert.True(new FileInfo(tempPath).Length > 100);
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
    }
}
