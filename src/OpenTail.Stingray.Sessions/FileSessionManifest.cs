namespace OpenTail.Stingray.Sessions;

using System;
using System.Collections.Immutable;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using OpenTail.Stingray.Core;

public sealed record SegmentBlockRef(
    string BlockId,
    int StartTokenPosition,
    int TokenCount,
    long UncompressedBytes,
    long CompressedBytes,
    ContentDigest BlockChecksum);

public sealed record SessionManifestEnvelope(
    SessionId SessionId,
    SessionRevision Revision,
    SessionStateABI Abi,
    string CompatibilityKey,
    StatePayloadHash PayloadHash,
    ImmutableArray<SegmentBlockRef> Blocks);

/// <summary>
/// Versioned binary manifest file parser and encoder for durable session state on disk (Milestone 3 Phase 2).
/// Magic tag: 0x4F54534D (OTSM), version 1.
///
/// <para><b>Whole-Manifest Integrity Check (#6):</b> Computes and validates a trailing SHA-256 checksum over all manifest header,
/// ABI, compatibility key, and block list payload bytes to detect bit flips or filesystem corruption.</para>
///
/// <para><b>Durability Note (#9):</b> Emits an explicit <c>Flush(flushToDisk: true)</c> on the temporary file before performing an
/// atomic <c>File.Move</c> (replace) operation.</para>
/// </summary>
public static class FileSessionManifest
{
    private const uint Magic = 0x4F54534D; // OTSM

    /// <summary>
    /// Current manifest version.
    ///
    /// <para><b>v3 changed the MEANING of the revision field</b> without changing its layout: it now
    /// carries the session store's turn counter — the value <c>RunTurnAsync</c> compares
    /// <c>expected_revision</c> against — where v1/v2 wrote the cursor's accepted-position count.
    /// The two coincide only while every turn accepts exactly one position, which is why a restored
    /// session and a live one used to disagree about what "revision" meant.</para>
    ///
    /// <para>v1/v2 files stay readable and are NOT migrated. The optimistic-concurrency contract
    /// needs the three sources (store, manifest, published token) to AGREE, not to hold a
    /// particular number: a legacy position count is simply a larger opaque seed for the same
    /// monotonic counter, and every value the client subsequently reads and echoes comes from the
    /// store. Rejecting those files would strand working sessions to correct a number nobody can
    /// observe.</para>
    /// </summary>
    private const ushort Version = 3;
    private static readonly Encoding Utf8Strict = new UTF8Encoding(false, true);

    public static byte[] Encode(SessionManifestEnvelope manifest)
    {
        ArgumentNullException.ThrowIfNull(manifest);

        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms, Utf8Strict, leaveOpen: true);

        writer.Write(Magic);
        writer.Write(Version);
        writer.Write(manifest.SessionId.Value.ToByteArray());
        writer.Write(manifest.Revision.Value);

        writer.Write(manifest.Abi.ModelFingerprint);
        writer.Write(manifest.Abi.KvBytesPerToken);
        writer.Write(manifest.Abi.MaxSequenceLength);
        writer.Write((byte)manifest.Abi.ModelFormat);

        writer.Write(manifest.CompatibilityKey);
        writer.Write(manifest.PayloadHash.Value.Value);

        var blocks = manifest.Blocks.IsDefault ? ImmutableArray<SegmentBlockRef>.Empty : manifest.Blocks;
        writer.Write(blocks.Length);

        foreach (var block in blocks)
        {
            writer.Write(block.BlockId);
            writer.Write(block.StartTokenPosition);
            writer.Write(block.TokenCount);
            writer.Write(block.UncompressedBytes);
            writer.Write(block.CompressedBytes);
            writer.Write(block.BlockChecksum.Value);
        }

        writer.Flush();
        byte[] payloadBytes = ms.ToArray();

        // Finding #6: Compute whole-manifest SHA-256 checksum
        byte[] hash = SHA256.HashData(payloadBytes);

        byte[] result = new byte[payloadBytes.Length + 32];
        Buffer.BlockCopy(payloadBytes, 0, result, 0, payloadBytes.Length);
        Buffer.BlockCopy(hash, 0, result, payloadBytes.Length, 32);

        return result;
    }

    public static SessionManifestEnvelope Decode(ReadOnlySpan<byte> data)
    {
        if (data.Length < 32 + 32) // Min header + trailing 32-byte hash
            throw new SessionJournalFormatException("Manifest data payload is too short.");

        int payloadLen = data.Length - 32;
        ReadOnlySpan<byte> payloadBytes = data[..payloadLen];
        ReadOnlySpan<byte> expectedHash = data[payloadLen..];

        // Finding #6: Verify whole-manifest SHA-256 checksum
        byte[] actualHash = SHA256.HashData(payloadBytes);
        if (!CryptographicOperations.FixedTimeEquals(expectedHash, actualHash))
            throw new SessionJournalFormatException("Whole-manifest SHA-256 checksum validation failed.");

        using var ms = new MemoryStream(payloadBytes.ToArray(), writable: false);
        using var reader = new BinaryReader(ms, Utf8Strict, leaveOpen: true);

        uint magic = reader.ReadUInt32();
        if (magic != Magic)
            throw new SessionJournalFormatException($"Invalid manifest magic 0x{magic:X8}, expected 0x{Magic:X8}.");

        ushort version = reader.ReadUInt16();
        if (version is not (1 or 2 or Version))
            throw new SessionJournalFormatException($"Unsupported manifest version {version}.");

        Guid sessionGuid = new(reader.ReadBytes(16));
        long revisionVal = reader.ReadInt64();

        string modelFp = reader.ReadString();
        long kvBytes = reader.ReadInt64();
        int maxSeqLen = reader.ReadInt32();
        ModelFormat modelFormat = ModelFormat.Gguf;
        if (version >= 2)
        {
            byte formatByte = reader.ReadByte();
            if (!Enum.IsDefined(typeof(ModelFormat), (ModelFormat)formatByte))
                throw new SessionJournalFormatException("Manifest contains an invalid model format.");
            modelFormat = (ModelFormat)formatByte;
        }
        var abi = new SessionStateABI(modelFp, kvBytes, maxSeqLen, modelFormat);

        string compatKey = reader.ReadString();
        string hashHex = reader.ReadString();
        var payloadHash = new StatePayloadHash(new ContentDigest(hashHex));

        int blockCount = reader.ReadInt32();
        if (blockCount < 0 || blockCount > 100_000)
            throw new SessionJournalFormatException($"Invalid manifest block count {blockCount}.");

        var blocksBuilder = ImmutableArray.CreateBuilder<SegmentBlockRef>(blockCount);
        for (int i = 0; i < blockCount; i++)
        {
            string blockId = reader.ReadString();
            int startPos = reader.ReadInt32();
            int tokenCount = reader.ReadInt32();
            long uncompressedBytes = reader.ReadInt64();
            long compressedBytes = reader.ReadInt64();
            string checksumHex = reader.ReadString();

            blocksBuilder.Add(new SegmentBlockRef(
                blockId,
                startPos,
                tokenCount,
                uncompressedBytes,
                compressedBytes,
                new ContentDigest(checksumHex)));
        }

        // Finding #12: Reject trailing unread bytes within payload stream
        if (ms.Position != ms.Length)
            throw new SessionJournalFormatException($"Manifest payload contains {ms.Length - ms.Position} unread trailing bytes.");

        return new SessionManifestEnvelope(
            new SessionId(sessionGuid),
            new SessionRevision(revisionVal),
            abi,
            compatKey,
            payloadHash,
            blocksBuilder.ToImmutable());
    }

    public static void SaveAtomic(string manifestPath, SessionManifestEnvelope manifest)
    {
        ArgumentException.ThrowIfNullOrEmpty(manifestPath);

        byte[] encoded = Encode(manifest);
        string tempPath = manifestPath + $".tmp_{Guid.NewGuid():N}";

        var dir = Path.GetDirectoryName(manifestPath);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
        {
            Directory.CreateDirectory(dir);
        }

        using (var fs = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None, 4096, FileOptions.None))
        {
            fs.Write(encoded, 0, encoded.Length);
            fs.Flush(flushToDisk: true);
        }

        File.Move(tempPath, manifestPath, overwrite: true);
    }

    public static SessionManifestEnvelope Load(string manifestPath)
    {
        ArgumentException.ThrowIfNullOrEmpty(manifestPath);
        if (!File.Exists(manifestPath))
            throw new FileNotFoundException($"Session manifest file '{manifestPath}' not found.", manifestPath);

        byte[] bytes = File.ReadAllBytes(manifestPath);
        return Decode(bytes);
    }
}
