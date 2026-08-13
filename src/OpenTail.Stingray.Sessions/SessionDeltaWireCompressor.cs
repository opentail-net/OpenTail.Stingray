using System;
using System.IO;
using System.IO.Compression;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Supported compression algorithms for transport-neutral wire delta compression.
/// </summary>
public enum CompressionAlgorithm
{
    None,
    Brotli,
    GZip
}

[JsonSourceGenerationOptions(WriteIndented = false)]
[JsonSerializable(typeof(SessionDelta))]
internal partial class SessionDeltaJsonContext : JsonSerializerContext
{
}

/// <summary>
/// Transport-neutral serialization and wire compression utilities for <see cref="SessionDelta"/>.
/// </summary>
public static class SessionDeltaWireCompressor
{
    /// <summary>
    /// Serializes a <see cref="SessionDelta"/> instance to a JSON string.
    /// </summary>
    public static string SerializeJson(SessionDelta delta)
    {
        ArgumentNullException.ThrowIfNull(delta);
        return JsonSerializer.Serialize(delta, SessionDeltaJsonContext.Default.SessionDelta);
    }

    /// <summary>
    /// Deserializes a JSON string back into a <see cref="SessionDelta"/> instance.
    /// </summary>
    public static SessionDelta DeserializeJson(string json)
    {
        ArgumentException.ThrowIfNullOrEmpty(json);
        var delta = JsonSerializer.Deserialize(json, SessionDeltaJsonContext.Default.SessionDelta);
        return delta ?? throw new FormatException("Failed to deserialize SessionDelta from JSON.");
    }

    /// <summary>
    /// Serializes and compresses a <see cref="SessionDelta"/> into a binary payload using the specified algorithm.
    /// </summary>
    public static byte[] Compress(SessionDelta delta, CompressionAlgorithm algorithm = CompressionAlgorithm.Brotli)
    {
        ArgumentNullException.ThrowIfNull(delta);
        string json = SerializeJson(delta);
        byte[] uncompressedBytes = System.Text.Encoding.UTF8.GetBytes(json);

        if (algorithm == CompressionAlgorithm.None)
        {
            return uncompressedBytes;
        }

        using var memoryStream = new MemoryStream();
        if (algorithm == CompressionAlgorithm.Brotli)
        {
            using (var brotliStream = new BrotliStream(memoryStream, CompressionLevel.Optimal, leaveOpen: true))
            {
                brotliStream.Write(uncompressedBytes, 0, uncompressedBytes.Length);
                brotliStream.Flush();
            }
        }
        else if (algorithm == CompressionAlgorithm.GZip)
        {
            using (var gzipStream = new GZipStream(memoryStream, CompressionLevel.Optimal, leaveOpen: true))
            {
                gzipStream.Write(uncompressedBytes, 0, uncompressedBytes.Length);
                gzipStream.Flush();
            }
        }
        else
        {
            throw new NotSupportedException($"Unsupported compression algorithm '{algorithm}'.");
        }

        return memoryStream.ToArray();
    }

    /// <summary>
    /// Decompresses and deserializes a binary payload back into a <see cref="SessionDelta"/> instance using the specified algorithm.
    /// </summary>
    public static SessionDelta Decompress(byte[] compressedBytes, CompressionAlgorithm algorithm = CompressionAlgorithm.Brotli)
    {
        ArgumentNullException.ThrowIfNull(compressedBytes);
        if (compressedBytes.Length == 0)
        {
            throw new ArgumentException("Compressed payload cannot be empty.", nameof(compressedBytes));
        }

        if (algorithm == CompressionAlgorithm.None)
        {
            string uncompressedJson = System.Text.Encoding.UTF8.GetString(compressedBytes);
            return DeserializeJson(uncompressedJson);
        }

        using var inputStream = new MemoryStream(compressedBytes);
        using var outputStream = new MemoryStream();

        if (algorithm == CompressionAlgorithm.Brotli)
        {
            using var brotliStream = new BrotliStream(inputStream, CompressionMode.Decompress);
            brotliStream.CopyTo(outputStream);
        }
        else if (algorithm == CompressionAlgorithm.GZip)
        {
            using var gzipStream = new GZipStream(inputStream, CompressionMode.Decompress);
            gzipStream.CopyTo(outputStream);
        }
        else
        {
            throw new NotSupportedException($"Unsupported compression algorithm '{algorithm}'.");
        }

        string json = System.Text.Encoding.UTF8.GetString(outputStream.ToArray());
        return DeserializeJson(json);
    }
}
