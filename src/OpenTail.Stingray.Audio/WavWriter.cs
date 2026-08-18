using System.Buffers.Binary;

namespace OpenTail.Stingray.Audio;

/// <summary>
/// Dithering mode for float-to-16-bit PCM quantization.
/// </summary>
public enum DitherMode
{
    /// <summary>No dithering applied (direct truncation/rounding).</summary>
    None = 0,

    /// <summary>Triangular Probability Density Function (TPDF) 2 LSB peak-to-peak dither with silence thresholding.</summary>
    Tpdf = 1
}

/// <summary>
/// High-performance 16-bit PCM RIFF WAVE audio file writer with optional TPDF dithering.
/// </summary>
public static class WavWriter
{
    private const double SilenceThreshold = 1e-5; // ~ -100 dBFS

    /// <summary>
    /// Writes float audio samples in [-1.0, 1.0] to a standard 16-bit PCM WAV file.
    /// </summary>
    public static void WriteWav(
        string path,
        ReadOnlySpan<float> samples,
        int sampleRate = 24000,
        int channels = 1,
        DitherMode dither = DitherMode.Tpdf)
    {
        using var stream = File.Create(path);
        WriteWav(stream, samples, sampleRate, channels, dither);
    }

    /// <summary>
    /// Writes float audio samples in [-1.0, 1.0] to a stream in WAV format.
    /// </summary>
    public static void WriteWav(
        Stream stream,
        ReadOnlySpan<float> samples,
        int sampleRate = 24000,
        int channels = 1,
        DitherMode dither = DitherMode.Tpdf)
    {
        int bytesPerSample = 2; // 16-bit
        int dataChunkSize = samples.Length * bytesPerSample;
        int riffChunkSize = 36 + dataChunkSize;

        Span<byte> header = stackalloc byte[44];

        // "RIFF"
        header[0] = (byte)'R'; header[1] = (byte)'I'; header[2] = (byte)'F'; header[3] = (byte)'F';
        BinaryPrimitives.WriteInt32LittleEndian(header[4..8], riffChunkSize);

        // "WAVE"
        header[8] = (byte)'W'; header[9] = (byte)'A'; header[10] = (byte)'V'; header[11] = (byte)'E';

        // "fmt " chunk
        header[12] = (byte)'f'; header[13] = (byte)'m'; header[14] = (byte)'t'; header[15] = (byte)' ';
        BinaryPrimitives.WriteInt32LittleEndian(header[16..20], 16); // Subchunk1Size for PCM
        BinaryPrimitives.WriteInt16LittleEndian(header[20..22], 1);  // AudioFormat 1 = PCM
        BinaryPrimitives.WriteInt16LittleEndian(header[22..24], (short)channels);
        BinaryPrimitives.WriteInt32LittleEndian(header[24..28], sampleRate);
        BinaryPrimitives.WriteInt32LittleEndian(header[28..32], sampleRate * channels * bytesPerSample); // ByteRate
        BinaryPrimitives.WriteInt16LittleEndian(header[32..34], (short)(channels * bytesPerSample));     // BlockAlign
        BinaryPrimitives.WriteInt16LittleEndian(header[34..36], 16); // BitsPerSample

        // "data" chunk
        header[36] = (byte)'d'; header[37] = (byte)'a'; header[38] = (byte)'t'; header[39] = (byte)'a';
        BinaryPrimitives.WriteInt32LittleEndian(header[40..44], dataChunkSize);

        stream.Write(header);

        // Convert float samples to 16-bit signed PCM with optional TPDF dither
        byte[] pcmBuffer = new byte[Math.Min(4096, samples.Length * 2)];
        int offset = 0;
        uint rngState = 0x853c49e6; // Fast deterministic XorShift32 seed

        while (offset < samples.Length)
        {
            int count = Math.Min(pcmBuffer.Length / 2, samples.Length - offset);
            for (int i = 0; i < count; i++)
            {
                short sample16 = QuantizeSample(samples[offset + i], dither, ref rngState);
                BinaryPrimitives.WriteInt16LittleEndian(pcmBuffer.AsSpan(i * 2, 2), sample16);
            }
            stream.Write(pcmBuffer, 0, count * 2);
            offset += count;
        }
    }

    /// <summary>
    /// Converts float audio samples in [-1.0, 1.0] to a standalone WAV byte array.
    /// </summary>
    public static byte[] ToWavBytes(
        ReadOnlySpan<float> samples,
        int sampleRate = 24000,
        int channels = 1,
        DitherMode dither = DitherMode.Tpdf)
    {
        using var ms = new MemoryStream();
        WriteWav(ms, samples, sampleRate, channels, dither);
        return ms.ToArray();
    }

    private static short QuantizeSample(float sample, DitherMode dither, ref uint rngState)
    {
        double clamped = Math.Clamp((double)sample, -1.0, 1.0);
        double scaled = clamped < 0 ? clamped * 32768.0 : clamped * 32767.0;

        if (dither == DitherMode.Tpdf && Math.Abs(clamped) > SilenceThreshold)
        {
            // 2-point XorShift32 uniform PRNG -> Triangular PDF
            rngState ^= rngState << 13;
            rngState ^= rngState >> 17;
            rngState ^= rngState << 5;
            double r1 = (rngState & 0x00FFFFFF) / 16777216.0;

            rngState ^= rngState << 13;
            rngState ^= rngState >> 17;
            rngState ^= rngState << 5;
            double r2 = (rngState & 0x00FFFFFF) / 16777216.0;

            scaled += (r1 + r2 - 1.0);
        }

        double quantized = Math.Round(scaled, MidpointRounding.AwayFromZero);
        return (short)Math.Clamp(quantized, short.MinValue, short.MaxValue);
    }
}
