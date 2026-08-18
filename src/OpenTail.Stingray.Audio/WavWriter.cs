using System.Buffers.Binary;

namespace OpenTail.Stingray.Audio;

/// <summary>
/// High-performance 16-bit PCM RIFF WAVE audio file writer.
/// </summary>
public static class WavWriter
{
    /// <summary>
    /// Writes float audio samples in [-1.0, 1.0] to a standard 16-bit PCM WAV file.
    /// </summary>
    public static void WriteWav(string path, ReadOnlySpan<float> samples, int sampleRate = 24000, int channels = 1)
    {
        using var stream = File.Create(path);
        WriteWav(stream, samples, sampleRate, channels);
    }

    /// <summary>
    /// Writes float audio samples in [-1.0, 1.0] to a stream in WAV format.
    /// </summary>
    public static void WriteWav(Stream stream, ReadOnlySpan<float> samples, int sampleRate = 24000, int channels = 1)
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

        // Convert float samples to 16-bit signed PCM
        byte[] pcmBuffer = new byte[Math.Min(4096, samples.Length * 2)];
        int offset = 0;

        while (offset < samples.Length)
        {
            int count = Math.Min(pcmBuffer.Length / 2, samples.Length - offset);
            for (int i = 0; i < count; i++)
            {
                float s = Math.Clamp(samples[offset + i], -1.0f, 1.0f);
                short sample16 = (short)(s * 32767.0f);
                BinaryPrimitives.WriteInt16LittleEndian(pcmBuffer.AsSpan(i * 2, 2), sample16);
            }
            stream.Write(pcmBuffer, 0, count * 2);
            offset += count;
        }
    }

    /// <summary>
    /// Converts float audio samples in [-1.0, 1.0] to a standalone WAV byte array.
    /// </summary>
    public static byte[] ToWavBytes(ReadOnlySpan<float> samples, int sampleRate = 24000, int channels = 1)
    {
        using var ms = new MemoryStream();
        WriteWav(ms, samples, sampleRate, channels);
        return ms.ToArray();
    }
}
