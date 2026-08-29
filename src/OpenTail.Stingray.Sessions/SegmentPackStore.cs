namespace OpenTail.Stingray.Sessions;


/// <summary>
/// Persists sealed immutable KV segment blocks to .pack files on disk for cold storage eviction (Milestone 3 Phase 2).
/// Magic tag: 0x4F545350 (OTSP), version 1.
///
/// <para><b>Whole-Pack Integrity Check (#6):</b> Computes and validates a trailing SHA-256 checksum over both
/// pack header fields and payload bytes to detect header bit flips or storage corruptions.</para>
/// </summary>
public static class SegmentPackStore
{
    private const uint Magic = 0x4F545350; // OTSP
    private const ushort Version = 1;
    private static readonly Encoding Utf8Strict = new UTF8Encoding(false, true);

    public static string GetPackPath(string storageDirectory, string blockId)
    {
        return Path.Combine(storageDirectory, $"{blockId}.pack");
    }

    /// <summary>
    /// Enumeration pattern matching every pack whose block id starts with <paramref name="blockIdPrefix"/>.
    /// Lives here so the <c>.pack</c> extension stays defined next to <see cref="GetPackPath"/>
    /// rather than being re-spelled by each caller that needs to sweep.
    /// </summary>
    public static string GetPackSearchPattern(string blockIdPrefix) => $"{blockIdPrefix}*.pack";

    public static SegmentBlockRef SaveBlock(string storageDirectory, string blockId, int startTokenPos, int tokenCount, ReadOnlySpan<byte> payload)
    {
        ArgumentException.ThrowIfNullOrEmpty(storageDirectory);
        ArgumentException.ThrowIfNullOrEmpty(blockId);

        if (!Directory.Exists(storageDirectory))
        {
            Directory.CreateDirectory(storageDirectory);
        }

        byte[] payloadHash = SHA256.HashData(payload);
        string payloadHashHex = Convert.ToHexStringLower(payloadHash);
        var checksum = new ContentDigest(payloadHashHex);

        using var ms = new MemoryStream();
        using (var writer = new BinaryWriter(ms, Utf8Strict, leaveOpen: true))
        {
            writer.Write(Magic);
            writer.Write(Version);
            writer.Write(blockId);
            writer.Write(startTokenPos);
            writer.Write(tokenCount);
            writer.Write((long)payload.Length); // Uncompressed
            writer.Write((long)payload.Length); // Compressed (1:1)
            writer.Write(payloadHashHex);
            writer.Write(payload);
            writer.Flush();
        }

        byte[] packBytes = ms.ToArray();
        byte[] wholePackHash = SHA256.HashData(packBytes);

        string packPath = GetPackPath(storageDirectory, blockId);
        string tempPath = packPath + $".tmp_{Guid.NewGuid():N}";

        using (var fs = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None, 4096, FileOptions.None))
        {
            fs.Write(packBytes, 0, packBytes.Length);
            fs.Write(wholePackHash, 0, 32);
            fs.Flush(flushToDisk: true);
        }

        File.Move(tempPath, packPath, overwrite: true);

        return new SegmentBlockRef(
            blockId,
            startTokenPos,
            tokenCount,
            payload.Length,
            payload.Length,
            checksum);
    }

    public static byte[] LoadBlock(string packFilePath)
    {
        ArgumentException.ThrowIfNullOrEmpty(packFilePath);
        if (!File.Exists(packFilePath))
            throw new FileNotFoundException($"Segment pack file '{packFilePath}' not found.", packFilePath);

        byte[] allBytes = File.ReadAllBytes(packFilePath);
        if (allBytes.Length < 32 + 32)
            throw new SessionJournalFormatException("Segment pack file is too short.");

        int packLen = allBytes.Length - 32;
        ReadOnlySpan<byte> packBytes = allBytes.AsSpan(0, packLen);
        ReadOnlySpan<byte> expectedWholeHash = allBytes.AsSpan(packLen, 32);

        // Finding #6: Whole-pack SHA-256 checksum check (header + payload)
        byte[] actualWholeHash = SHA256.HashData(packBytes);
        if (!CryptographicOperations.FixedTimeEquals(expectedWholeHash, actualWholeHash))
            throw new SessionJournalFormatException($"Whole-pack SHA-256 checksum validation failed for '{packFilePath}'.");

        using var ms = new MemoryStream(packBytes.ToArray(), writable: false);
        using var reader = new BinaryReader(ms, Utf8Strict, leaveOpen: true);

        uint magic = reader.ReadUInt32();
        if (magic != Magic)
            throw new SessionJournalFormatException($"Invalid segment pack magic 0x{magic:X8}, expected 0x{Magic:X8}.");

        ushort version = reader.ReadUInt16();
        if (version != Version)
            throw new SessionJournalFormatException($"Unsupported segment pack version {version}.");

        string blockId = reader.ReadString();
        int startTokenPos = reader.ReadInt32();
        int tokenCount = reader.ReadInt32();
        long uncompressedBytes = reader.ReadInt64();
        long compressedBytes = reader.ReadInt64();
        string expectedPayloadHashHex = reader.ReadString();

        if (uncompressedBytes < 0 || uncompressedBytes > 100 * 1024 * 1024)
            throw new SessionJournalFormatException($"Invalid segment block length {uncompressedBytes}.");

        byte[] payload = reader.ReadBytes((int)uncompressedBytes);
        byte[] actualPayloadHash = SHA256.HashData(payload);
        string actualPayloadHashHex = Convert.ToHexStringLower(actualPayloadHash);

        if (!string.Equals(expectedPayloadHashHex, actualPayloadHashHex, StringComparison.OrdinalIgnoreCase))
            throw new SessionJournalFormatException($"Payload checksum mismatch for block '{blockId}'. Expected {expectedPayloadHashHex}, got {actualPayloadHashHex}.");

        return payload;
    }
}
