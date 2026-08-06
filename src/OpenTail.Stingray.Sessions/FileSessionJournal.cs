namespace OpenTail.Stingray.Sessions;

using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;

public enum SessionJournalRecordKind : ushort
{
    OperationCommitted = 1,
    RevisionCheckpoint = 2,
    CursorState = 3
}

public sealed record SessionJournalRecord(
    SessionJournalRecordKind Kind,
    SessionId SessionId,
    SessionRevision Revision,
    SessionOperationId OperationId,
    byte[] Payload) : IEquatable<SessionJournalRecord>
{
    public bool Equals(SessionJournalRecord? other)
    {
        if (other is null) return false;
        if (ReferenceEquals(this, other)) return true;
        return Kind == other.Kind
            && SessionId.Equals(other.SessionId)
            && Revision.Equals(other.Revision)
            && OperationId.Equals(other.OperationId)
            && (Payload == other.Payload || (Payload is not null && other.Payload is not null && Payload.AsSpan().SequenceEqual(other.Payload)));
    }

    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(Kind);
        hash.Add(SessionId);
        hash.Add(Revision);
        hash.Add(OperationId);
        if (Payload is not null) hash.AddBytes(Payload);
        return hash.ToHashCode();
    }
}

public sealed class SessionJournalFormatException(string message, Exception? innerException = null)
    : IOException(message, innerException);

/// <summary>
/// Crash-consistent, append-only file journal for session revisions, operations, and cursors (Milestone 3).
///
/// <para><b>Magic Tag (Finding #9):</b> Uses <c>0x4F54534A</c> (<c>OTSJ</c> - OpenTail Session Journal),
/// written in standard little-endian order as <c>0x4A 0x53 0x54 0x4F</c> ('J', 'S', 'T', 'O').</para>
///
/// <para><b>Durability (Finding #12):</b> Emits an explicit <c>Flush(flushToDisk: true)</c> (fsync) after
/// every record write to guarantee durability across unexpected power or process crashes.</para>
/// </summary>
public sealed class FileSessionJournal : IDisposable
{
    private const uint Magic = 0x4F54534A; // OTSJ
    private const ushort Version = 1;
    private const int MaxAllowedPayloadBytes = 100 * 1024 * 1024; // 100MB bound

    /// <summary>Fixed on-disk record header: kind(2) + sessionId(16) + revision(8) + operationId(16)
    /// + payloadLength(4). Derived in one place so the writer and the recovery reader cannot drift.</summary>
    private const int HeaderBytes = sizeof(ushort) + 16 + sizeof(long) + 16 + sizeof(int);

    private const int HashBytes = 32;
    private static readonly Encoding Utf8Strict = new UTF8Encoding(false, true);

    private readonly FileStream _stream;
    private readonly object _gate = new();
    private bool _disposed;

    public string FilePath { get; }

    /// <summary>
    /// Bytes discarded by the most recent <see cref="RecoverAndTruncateCorruptTail"/> call.
    /// </summary>
    /// <remarks>
    /// Recovery cannot distinguish "the process died mid-write" from "these bytes are damaged", and
    /// in an append-only journal both are confined to the tail, so both are handled by truncating.
    /// That is the correct behaviour but it is also silent data loss, so the amount is surfaced here
    /// rather than left for a caller to infer. A non-zero value after startup recovery is worth
    /// logging: zero or a partial trailing record is routine after a crash, anything larger is not.
    /// </remarks>
    public long LastRecoveryTruncatedBytes { get; private set; }

    public FileSessionJournal(string filePath)
    {
        ArgumentException.ThrowIfNullOrEmpty(filePath);
        FilePath = Path.GetFullPath(filePath);

        var directory = Path.GetDirectoryName(FilePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        bool isNewFile = !File.Exists(FilePath) || new FileInfo(FilePath).Length == 0;

        _stream = new FileStream(
            FilePath,
            FileMode.OpenOrCreate,
            FileAccess.ReadWrite,
            FileShare.Read,
            bufferSize: 4096,
            FileOptions.None);

        if (isNewFile)
        {
            WriteHeader();
        }
        else
        {
            ValidateHeader();
        }
    }

    private void WriteHeader()
    {
        using var writer = new BinaryWriter(_stream, Utf8Strict, leaveOpen: true);
        writer.Write(Magic);
        writer.Write(Version);
        _stream.Flush(flushToDisk: true);
    }

    private void ValidateHeader()
    {
        _stream.Position = 0;
        using var reader = new BinaryReader(_stream, Utf8Strict, leaveOpen: true);
        uint magic = reader.ReadUInt32();
        if (magic != Magic)
            throw new SessionJournalFormatException($"Invalid journal magic header 0x{magic:X8}, expected 0x{Magic:X8}.");
        ushort version = reader.ReadUInt16();
        if (version != Version)
            throw new SessionJournalFormatException($"Unsupported journal version {version}, expected {Version}.");
    }

    private static byte[] BuildHeaderBuffer(SessionJournalRecord record)
    {
        using var ms = new MemoryStream(HeaderBytes);
        using var writer = new BinaryWriter(ms, Utf8Strict);
        writer.Write((ushort)record.Kind);
        writer.Write(record.SessionId.Value.ToByteArray());
        writer.Write(record.Revision.Value);
        writer.Write(record.OperationId.Value.ToByteArray());
        writer.Write(record.Payload?.Length ?? 0);
        writer.Flush();
        return ms.ToArray();
    }

    private static byte[] ComputeWholeRecordHash(byte[] headerBuffer, byte[] payload)
    {
        using var sha = SHA256.Create();
        sha.TransformBlock(headerBuffer, 0, headerBuffer.Length, null, 0);
        sha.TransformFinalBlock(payload, 0, payload.Length);
        return sha.Hash!;
    }

    public void Append(SessionJournalRecord record)
    {
        ArgumentNullException.ThrowIfNull(record);
        ThrowIfDisposed();

        byte[] payload = record.Payload ?? [];
        byte[] headerBuffer = BuildHeaderBuffer(record);
        byte[] wholeRecordHash = ComputeWholeRecordHash(headerBuffer, payload);

        lock (_gate)
        {
            _stream.Seek(0, SeekOrigin.End);
            _stream.Write(headerBuffer, 0, headerBuffer.Length);
            _stream.Write(payload, 0, payload.Length);
            _stream.Write(wholeRecordHash, 0, wholeRecordHash.Length);

            // Finding #12: explicit fsync per commit guarantees durability
            _stream.Flush(flushToDisk: true);
        }
    }

    public IReadOnlyList<SessionJournalRecord> RecoverAndTruncateCorruptTail()
    {
        ThrowIfDisposed();
        var records = new List<SessionJournalRecord>();

        lock (_gate)
        {
            ValidateHeader();

            long validPosition = sizeof(uint) + sizeof(ushort);

            while (_stream.Position < _stream.Length)
            {
                if (_stream.Length - _stream.Position < HeaderBytes)
                {
                    // Incomplete trailing record header -> corrupt tail
                    break;
                }

                long entryStart = _stream.Position;
                byte[] headerBuffer = new byte[HeaderBytes];
                // ReadExactly, not Read: a single Read may legally return fewer bytes than asked for
                // (payloads here can exceed the FileStream buffer), and treating a short read as a
                // corrupt tail would truncate perfectly valid committed records. The length check
                // above guarantees the bytes are present, so a short read is a stream-level fault,
                // not end-of-data.
                _stream.ReadExactly(headerBuffer, 0, HeaderBytes);

                using var headerReader = new BinaryReader(new MemoryStream(headerBuffer), Utf8Strict);
                ushort kindVal = headerReader.ReadUInt16();
                Guid sessionGuid = new(headerReader.ReadBytes(16));
                long revisionVal = headerReader.ReadInt64();
                Guid opGuid = new(headerReader.ReadBytes(16));
                int payloadLen = headerReader.ReadInt32();

                // An implausible length is what a torn write looks like: the process died partway
                // through this record and the length field holds whatever was already on disk. That
                // is the ordinary crash case this method exists to survive, so it truncates like any
                // other damaged tail. Throwing here would make a single interrupted write render the
                // whole journal unopenable and lose every VALID record before it — strictly worse
                // than the data loss it was trying to report. The loss is surfaced instead through
                // LastRecoveryTruncatedBytes.
                if (payloadLen < 0 || payloadLen > MaxAllowedPayloadBytes)
                {
                    _stream.Position = entryStart;
                    break;
                }

                if (_stream.Length - _stream.Position < (long)payloadLen + HashBytes)
                {
                    // Incomplete trailing payload or checksum -> corrupt tail
                    _stream.Position = entryStart;
                    break;
                }

                byte[] payload = new byte[payloadLen];
                _stream.ReadExactly(payload, 0, payloadLen);
                byte[] expectedHash = new byte[HashBytes];
                _stream.ReadExactly(expectedHash, 0, HashBytes);

                // Whole-record SHA-256 spanning header + payload, so a bit flip in the revision,
                // session id or length is caught rather than silently accepted as a valid record.
                byte[] actualHash = ComputeWholeRecordHash(headerBuffer, payload);
                if (!CryptographicOperations.FixedTimeEquals(expectedHash, actualHash))
                {
                    // Whole-record checksum mismatch -> corrupt tail
                    _stream.Position = entryStart;
                    break;
                }

                records.Add(new SessionJournalRecord(
                    (SessionJournalRecordKind)kindVal,
                    new SessionId(sessionGuid),
                    new SessionRevision(revisionVal),
                    new SessionOperationId(opGuid),
                    payload));

                validPosition = _stream.Position;
            }

            // Drop everything after the last record that verified, and report how much was lost.
            LastRecoveryTruncatedBytes = Math.Max(0, _stream.Length - validPosition);
            if (_stream.Length > validPosition)
            {
                _stream.SetLength(validPosition);
                _stream.Flush(flushToDisk: true);
            }

            return records;
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(FileSessionJournal));
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true;
            _stream.Dispose();
        }
    }
}
