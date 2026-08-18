using System.Buffers.Binary;

namespace OpenTail.Stingray.Audio;

/// <summary>
/// Lightweight 16-bit / 32-bit PCM RIFF WAVE audio file reader.
/// </summary>
public static class WavReader
{
    /// <summary>
    /// Reads audio samples from a WAV file as normalized [-1.0, 1.0] float samples.
    /// Returns samples array and detected sample rate.
    /// </summary>
    public static (float[] Samples, int SampleRate, int Channels) ReadWav(string path)
    {
        using var stream = File.OpenRead(path);
        return ReadWav(stream);
    }

    /// <summary>
    /// Reads audio samples from a stream in RIFF WAV format.
    /// </summary>
    public static (float[] Samples, int SampleRate, int Channels) ReadWav(Stream stream)
    {
        using var reader = new BinaryReader(stream);

        // RIFF header
        byte[] riff = reader.ReadBytes(4);
        if (riff.Length < 4 || riff[0] != 'R' || riff[1] != 'I' || riff[2] != 'F' || riff[3] != 'F')
        {
            throw new InvalidDataException("Not a valid RIFF WAV file.");
        }

        reader.ReadInt32(); // riff chunk size

        byte[] wave = reader.ReadBytes(4);
        if (wave.Length < 4 || wave[0] != 'W' || wave[1] != 'A' || wave[2] != 'V' || wave[3] != 'E')
        {
            throw new InvalidDataException("Not a valid WAVE stream.");
        }

        int sampleRate = 16000;
        int channels = 1;
        int bitsPerSample = 16;
        float[]? samples = null;

        while (stream.Position < stream.Length)
        {
            byte[] chunkIdBytes = reader.ReadBytes(4);
            if (chunkIdBytes.Length < 4) break;

            string chunkId = System.Text.Encoding.ASCII.GetString(chunkIdBytes);
            int chunkSize = reader.ReadInt32();

            if (chunkId == "fmt ")
            {
                short audioFormat = reader.ReadInt16();
                channels = reader.ReadInt16();
                sampleRate = reader.ReadInt32();
                reader.ReadInt32(); // byteRate
                reader.ReadInt16(); // blockAlign
                bitsPerSample = reader.ReadInt16();

                // Skip any extra format bytes
                if (chunkSize > 16)
                {
                    reader.ReadBytes(chunkSize - 16);
                }
            }
            else if (chunkId == "data")
            {
                int bytesPerSample = bitsPerSample / 8;
                int totalSamples = chunkSize / bytesPerSample;
                int frameCount = totalSamples / channels;

                samples = new float[frameCount];

                if (bitsPerSample == 16)
                {
                    for (int f = 0; f < frameCount; f++)
                    {
                        float sum = 0f;
                        for (int ch = 0; ch < channels; ch++)
                        {
                            short s = reader.ReadInt16();
                            sum += s / 32768.0f;
                        }
                        samples[f] = sum / channels;
                    }
                }
                else if (bitsPerSample == 32)
                {
                    for (int f = 0; f < frameCount; f++)
                    {
                        float sum = 0f;
                        for (int ch = 0; ch < channels; ch++)
                        {
                            float s = reader.ReadSingle();
                            sum += s;
                        }
                        samples[f] = sum / channels;
                    }
                }
                else if (bitsPerSample == 8)
                {
                    for (int f = 0; f < frameCount; f++)
                    {
                        float sum = 0f;
                        for (int ch = 0; ch < channels; ch++)
                        {
                            byte b = reader.ReadByte();
                            sum += (b - 128) / 128.0f;
                        }
                        samples[f] = sum / channels;
                    }
                }
                else
                {
                    throw new NotSupportedException($"Unsupported bits per sample: {bitsPerSample}");
                }
                break;
            }
            else
            {
                // Skip unknown chunk
                reader.ReadBytes(chunkSize);
            }
        }

        return (samples ?? [], sampleRate, channels);
    }
}
