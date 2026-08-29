namespace OpenTail.Stingray.Sessions;


/// <summary>
/// Bounded binary representation of terminal completed operations retained across a cold-session
/// restore. This deliberately stores only successful results: an interrupted in-flight turn has
/// no safe replay result, whereas a committed turn can be returned idempotently after restart.
/// </summary>
internal static class SessionOperationLedgerCodec
{
    private const uint Magic = 0x4F544F50; // OTOP
    private const ushort Version = 1;
    internal const int MaxPayloadBytes = 1 * 1024 * 1024;
    private const int MaxRecords = 256;
    private const int MaxChunksPerRecord = 4096;
    private static readonly UTF8Encoding Utf8Strict = new(false, true);

    /// <summary>
    /// Encodes as many newest completed results as fit. The hot store already has its own bounded
    /// retention window; this second byte ceiling keeps disk-backed reconnect receipts bounded
    /// even when a model emits unusually large textual chunks.
    /// </summary>
    public static byte[] Encode(IReadOnlyCollection<SessionOperationSnapshot> operations)
    {
        ArgumentNullException.ThrowIfNull(operations);
        var frames = new List<byte[]>();
        int used = sizeof(uint) + sizeof(ushort) + sizeof(int);

        foreach (var operation in operations
                     .Where(x => x.State == SessionOperationState.Completed)
                     .OrderByDescending(x => x.CompletedAt)
                     .Take(MaxRecords))
        {
            byte[] frame = EncodeOperation(operation);
            if (frame.Length > MaxPayloadBytes - used)
                continue;
            frames.Add(frame);
            used += frame.Length;
        }

        frames.Reverse(); // oldest-first is stable and makes inspection/recovery deterministic.
        using var stream = new MemoryStream(used);
        using var writer = new BinaryWriter(stream, Utf8Strict, leaveOpen: true);
        writer.Write(Magic);
        writer.Write(Version);
        writer.Write(frames.Count);
        foreach (byte[] frame in frames)
            writer.Write(frame);
        writer.Flush();
        return stream.ToArray();
    }

    public static ImmutableArray<SessionOperationSnapshot> Decode(SessionId expectedSessionId, ReadOnlySpan<byte> payload)
    {
        if (payload.Length > MaxPayloadBytes)
            throw new SessionJournalFormatException("Persisted operation ledger exceeds its bounded payload size.");

        using var stream = new MemoryStream(payload.ToArray(), writable: false);
        using var reader = new BinaryReader(stream, Utf8Strict, leaveOpen: true);
        if (reader.ReadUInt32() != Magic)
            throw new SessionJournalFormatException("Invalid persisted operation ledger magic.");
        if (reader.ReadUInt16() != Version)
            throw new SessionJournalFormatException("Unsupported persisted operation ledger version.");
        int count = reader.ReadInt32();
        if (count < 0 || count > MaxRecords)
            throw new SessionJournalFormatException("Persisted operation ledger has an invalid record count.");

        var result = ImmutableArray.CreateBuilder<SessionOperationSnapshot>(count);
        for (int i = 0; i < count; i++)
        {
            SessionOperationSnapshot operation = DecodeOperation(reader);
            if (operation.SessionId != expectedSessionId)
                throw new SessionJournalFormatException("Persisted operation ledger belongs to a different session.");
            result.Add(operation);
        }
        if (stream.Position != stream.Length)
            throw new SessionJournalFormatException("Persisted operation ledger has unread trailing bytes.");
        return result.ToImmutable();
    }

    private static byte[] EncodeOperation(SessionOperationSnapshot operation)
    {
        if (operation.State != SessionOperationState.Completed || operation.CommittedRevision is null
            || operation.CompletedAt is null)
            throw new ArgumentException("Only completed operations with a committed revision can be persisted.", nameof(operation));

        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Utf8Strict, leaveOpen: true);
        writer.Write(operation.SessionId.Value.ToByteArray());
        writer.Write(operation.OperationId.Value.ToByteArray());
        WriteString(writer, operation.RequestDigest.Value);
        writer.Write(operation.CommittedRevision.Value.Value);
        writer.Write(operation.CreatedAt.UtcDateTime.Ticks);
        writer.Write(operation.CompletedAt.Value.UtcDateTime.Ticks);
        var chunks = operation.ResultChunks ?? [];
        if (chunks.Count > MaxChunksPerRecord)
            throw new SessionJournalFormatException("Completed operation has too many result chunks to persist.");
        writer.Write(chunks.Count);
        foreach (var chunk in chunks)
        {
            writer.Write((byte)chunk.Kind);
            writer.Write(chunk.PromptTokens);
            writer.Write(chunk.TruncatedByMaxTokens);
            writer.Write(chunk.TruncatedByResourceBudget);
            WriteString(writer, chunk.Text);
        }
        writer.Flush();
        return stream.ToArray();
    }

    private static SessionOperationSnapshot DecodeOperation(BinaryReader reader)
    {
        var sessionId = new SessionId(new Guid(ReadExactly(reader, 16)));
        var operationId = new SessionOperationId(new Guid(ReadExactly(reader, 16)));
        var digest = SessionRequestDigest.FromCanonicalValue(ReadString(reader));
        var revision = new SessionRevision(reader.ReadInt64());
        var createdAt = new DateTimeOffset(new DateTime(reader.ReadInt64(), DateTimeKind.Utc));
        var completedAt = new DateTimeOffset(new DateTime(reader.ReadInt64(), DateTimeKind.Utc));
        int chunkCount = reader.ReadInt32();
        if (chunkCount < 0 || chunkCount > MaxChunksPerRecord)
            throw new SessionJournalFormatException("Persisted operation has an invalid result chunk count.");
        var chunks = new GenerateChunk[chunkCount];
        for (int i = 0; i < chunks.Length; i++)
        {
            byte kind = reader.ReadByte();
            if (!Enum.IsDefined(typeof(GenerateChunkKind), (GenerateChunkKind)kind))
                throw new SessionJournalFormatException("Persisted operation contains an invalid chunk kind.");
            int promptTokens = reader.ReadInt32();
            bool truncatedByMaxTokens = reader.ReadBoolean();
            bool truncatedByResourceBudget = reader.ReadBoolean();
            chunks[i] = new GenerateChunk((GenerateChunkKind)kind, ReadString(reader), promptTokens,
                truncatedByMaxTokens, truncatedByResourceBudget);
        }
        return new SessionOperationSnapshot(sessionId, operationId, digest, SessionOperationState.Completed,
            FencingEpoch: 0, revision, createdAt, completedAt, FailureReason: null, ResultChunks: chunks);
    }

    private static void WriteString(BinaryWriter writer, string value)
    {
        ArgumentNullException.ThrowIfNull(value);
        byte[] bytes = Utf8Strict.GetBytes(value);
        writer.Write(bytes.Length);
        writer.Write(bytes);
    }

    private static string ReadString(BinaryReader reader)
    {
        int length = reader.ReadInt32();
        if (length < 0 || length > MaxPayloadBytes)
            throw new SessionJournalFormatException("Persisted operation contains an invalid string length.");
        return Utf8Strict.GetString(ReadExactly(reader, length));
    }

    private static byte[] ReadExactly(BinaryReader reader, int length)
    {
        byte[] bytes = reader.ReadBytes(length);
        if (bytes.Length != length)
            throw new SessionJournalFormatException("Persisted operation ledger ended unexpectedly.");
        return bytes;
    }
}
