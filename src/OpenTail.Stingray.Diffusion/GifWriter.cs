using System.Buffers.Binary;

namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// 100% native managed, zero-dependency GIF89a animated image writer.
/// Encodes multi-frame RGB video/image sequences into standard looping .gif files.
/// </summary>
public static class GifWriter
{
    /// <summary>
    /// Encodes a sequence of float RGB frames ([0.0, 1.0] in [3, height, width] or [height * width * 3]) to a GIF file.
    /// </summary>
    public static void SaveGif(
        string path,
        IReadOnlyList<float[]> framesRgb,
        int width,
        int height,
        int fps = 24)
    {
        using var stream = File.Create(path);
        WriteGif(stream, framesRgb, width, height, fps);
    }

    /// <summary>
    /// Writes a sequence of float RGB frames to an output stream as an animated GIF89a.
    /// </summary>
    public static void WriteGif(
        Stream stream,
        IReadOnlyList<float[]> framesRgb,
        int width,
        int height,
        int fps = 24)
    {
        if (framesRgb == null || framesRgb.Count == 0)
            throw new ArgumentException("Frames collection cannot be null or empty.", nameof(framesRgb));
        if (width <= 0 || height <= 0)
            throw new ArgumentException("Width and height must be positive.");

        int delayCentiseconds = Math.Max(1, (int)Math.Round(100.0 / Math.Max(1, fps)));

        // 1. Header: "GIF89a"
        stream.Write([(byte)'G', (byte)'I', (byte)'F', (byte)'8', (byte)'9', (byte)'a']);

        // 2. Logical Screen Descriptor (7 bytes)
        Span<byte> lsd = stackalloc byte[7];
        BinaryPrimitives.WriteUInt16LittleEndian(lsd[0..2], (ushort)width);
        BinaryPrimitives.WriteUInt16LittleEndian(lsd[2..4], (ushort)height);
        lsd[4] = 0xF7; // Global Color Table Present (1), 8 bits color res (111), Not sorted (0), 256 colors (111)
        lsd[5] = 0x00; // Background color index
        lsd[6] = 0x00; // Pixel aspect ratio
        stream.Write(lsd);

        // 3. Global Color Table (256 entries * 3 bytes RGB = 768 bytes, 3-3-2 uniform color palette)
        byte[] palette = GenerateUniformPalette();
        stream.Write(palette);

        // 4. Netscape 2.0 Loop Extension (for infinite looping)
        stream.Write([
            0x21, 0xFF, 0x0B, // Extension, Application Label, 11 bytes
            (byte)'N', (byte)'E', (byte)'T', (byte)'S', (byte)'C', (byte)'A', (byte)'P', (byte)'E', (byte)'2', (byte)'.', (byte)'0',
            0x03, 0x01, 0x00, 0x00, // Sub-block length 3, Loop sub-block, Loop count = 0 (infinite)
            0x00 // Sub-block terminator
        ]);

        // 5. Frame Loop
        byte[] indexedPixels = new byte[width * height];
        Span<byte> gce = stackalloc byte[8];
        Span<byte> id = stackalloc byte[10];

        for (int f = 0; f < framesRgb.Count; f++)
        {
            var frame = framesRgb[f];
            QuantizeRgbToPalette(frame, indexedPixels, width, height);

            // Graphic Control Extension (8 bytes)
            gce[0] = 0x21; // Extension
            gce[1] = 0xF9; // Graphic Control Label
            gce[2] = 0x04; // Byte count
            gce[3] = 0x04; // Packed: Disposal Method 1 (do not dispose), No transparency
            BinaryPrimitives.WriteUInt16LittleEndian(gce[4..6], (ushort)delayCentiseconds);
            gce[6] = 0x00; // Transparent color index
            gce[7] = 0x00; // Block terminator
            stream.Write(gce);

            // Image Descriptor (10 bytes)
            id[0] = 0x2C; // Image Separator
            BinaryPrimitives.WriteUInt16LittleEndian(id[1..3], 0); // Left
            BinaryPrimitives.WriteUInt16LittleEndian(id[3..5], 0); // Top
            BinaryPrimitives.WriteUInt16LittleEndian(id[5..7], (ushort)width);
            BinaryPrimitives.WriteUInt16LittleEndian(id[7..9], (ushort)height);
            id[9] = 0x00; // Packed: No local color table, Not interlaced
            stream.Write(id);

            // LZW Compressed Image Data
            CompressLzw(stream, indexedPixels, colorDepth: 8);
        }

        // 6. Trailer: 0x3B (';')
        stream.WriteByte(0x3B);
    }

    private static byte[] GenerateUniformPalette()
    {
        // 3-3-2 RGB uniform color palette (8 red, 8 green, 4 blue = 256 colors)
        byte[] pal = new byte[256 * 3];
        for (int i = 0; i < 256; i++)
        {
            int r = (i >> 5) & 0x07;
            int g = (i >> 2) & 0x07;
            int b = i & 0x03;

            pal[i * 3 + 0] = (byte)((r * 255) / 7);
            pal[i * 3 + 1] = (byte)((g * 255) / 7);
            pal[i * 3 + 2] = (byte)((b * 255) / 3);
        }
        return pal;
    }

    private static void QuantizeRgbToPalette(float[] rgb, byte[] output, int width, int height)
    {
        int pixelCount = width * height;
        bool isPlanar = rgb.Length >= pixelCount * 3 && (rgb.Length == pixelCount * 3);

        if (isPlanar)
        {
            // Planar format: [R: 0..N-1, G: N..2N-1, B: 2N..3N-1]
            int planeOffsetG = pixelCount;
            int planeOffsetB = pixelCount * 2;

            for (int i = 0; i < pixelCount; i++)
            {
                int r = (int)(Math.Clamp(rgb[i], 0f, 1f) * 7.0f + 0.5f);
                int g = (int)(Math.Clamp(rgb[planeOffsetG + i], 0f, 1f) * 7.0f + 0.5f);
                int b = (int)(Math.Clamp(rgb[planeOffsetB + i], 0f, 1f) * 3.0f + 0.5f);

                output[i] = (byte)((r << 5) | (g << 2) | b);
            }
        }
        else
        {
            // Interleaved format: [R, G, B, R, G, B, ...]
            for (int i = 0; i < pixelCount; i++)
            {
                int off = i * 3;
                int r = (int)(Math.Clamp(rgb[off], 0f, 1f) * 7.0f + 0.5f);
                int g = (int)(Math.Clamp(rgb[off + 1], 0f, 1f) * 7.0f + 0.5f);
                int b = (int)(Math.Clamp(rgb[off + 2], 0f, 1f) * 3.0f + 0.5f);

                output[i] = (byte)((r << 5) | (g << 2) | b);
            }
        }
    }

    private static void CompressLzw(Stream stream, ReadOnlySpan<byte> pixels, int colorDepth)
    {
        int initCodeSize = Math.Max(2, colorDepth);
        stream.WriteByte((byte)initCodeSize);

        int clearCode = 1 << initCodeSize;
        int endOfInfoCode = clearCode + 1;
        int nextCode = clearCode + 2;
        int codeSize = initCodeSize + 1;
        int maxCode = 1 << codeSize;

        const int TableSize = 5003;
        int[] hKeys = new int[TableSize];
        int[] hValues = new int[TableSize];
        Array.Fill(hKeys, -1);

        var writer = new BitStreamWriter(stream);
        writer.WriteBits(clearCode, codeSize);

        if (pixels.Length == 0)
        {
            writer.WriteBits(endOfInfoCode, codeSize);
            writer.Flush();
            stream.WriteByte(0x00);
            return;
        }

        int prefix = pixels[0];

        for (int i = 1; i < pixels.Length; i++)
        {
            int c = pixels[i];
            int key = (prefix << 8) | c;
            int hash = (key * 0x45d9f3b) % TableSize;
            if (hash < 0) hash += TableSize;

            int foundCode = -1;
            while (hKeys[hash] != -1)
            {
                if (hKeys[hash] == key)
                {
                    foundCode = hValues[hash];
                    break;
                }
                hash = (hash + 1) % TableSize;
            }

            if (foundCode != -1)
            {
                prefix = foundCode;
            }
            else
            {
                writer.WriteBits(prefix, codeSize);

                if (nextCode < 4096)
                {
                    hKeys[hash] = key;
                    hValues[hash] = nextCode++;

                    if (nextCode > maxCode && codeSize < 12)
                    {
                        codeSize++;
                        maxCode = 1 << codeSize;
                    }
                }
                else
                {
                    // Table full: issue clear code and reset
                    writer.WriteBits(clearCode, codeSize);
                    Array.Fill(hKeys, -1);
                    codeSize = initCodeSize + 1;
                    maxCode = 1 << codeSize;
                    nextCode = clearCode + 2;
                }

                prefix = c;
            }
        }

        writer.WriteBits(prefix, codeSize);
        writer.WriteBits(endOfInfoCode, codeSize);
        writer.Flush();
        stream.WriteByte(0x00); // Block terminator
    }

    private sealed class BitStreamWriter
    {
        private readonly Stream _stream;
        private readonly byte[] _buffer = new byte[255];
        private int _bufIndex = 0;
        private int _accumulator = 0;
        private int _bitCount = 0;

        public BitStreamWriter(Stream stream)
        {
            _stream = stream;
        }

        public void WriteBits(int value, int count)
        {
            _accumulator |= (value << _bitCount);
            _bitCount += count;

            while (_bitCount >= 8)
            {
                _buffer[_bufIndex++] = (byte)(_accumulator & 0xFF);
                _accumulator >>= 8;
                _bitCount -= 8;

                if (_bufIndex == 255)
                {
                    _stream.WriteByte(255);
                    _stream.Write(_buffer, 0, 255);
                    _bufIndex = 0;
                }
            }
        }

        public void Flush()
        {
            if (_bitCount > 0)
            {
                _buffer[_bufIndex++] = (byte)(_accumulator & 0xFF);
                _accumulator = 0;
                _bitCount = 0;
            }

            if (_bufIndex > 0)
            {
                _stream.WriteByte((byte)_bufIndex);
                _stream.Write(_buffer, 0, _bufIndex);
                _bufIndex = 0;
            }
        }
    }
}
