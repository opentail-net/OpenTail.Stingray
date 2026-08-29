
namespace OpenTail.Stingray.Sessions;

/// <summary>Limits enforced before a cursor decoder allocates attacker-controlled collections.</summary>
public sealed record SessionCursorCodecLimits(
    int MaxPayloadBytes = 4 * 1024 * 1024,
    int MaxSections = 64,
    int MaxSegments = 16_384,
    int MaxTokensPerSegment = 1_048_576,
    int MaxStringBytes = 16 * 1024)
{
    public void Validate()
    {
        if (MaxPayloadBytes <= 0 || MaxSections <= 0 || MaxSegments < 0 || MaxTokensPerSegment <= 0 || MaxStringBytes <= 0)
            throw new ArgumentOutOfRangeException(nameof(SessionCursorCodecLimits));
    }
}

/// <summary>Raised when a cursor envelope is malformed, incompatible or exceeds declared limits.</summary>
public sealed class SessionCursorFormatException(string message) : IOException(message);

/// <summary>An unrecognised optional envelope section preserved byte-for-byte across rewrites.</summary>
public sealed record SessionCursorOptionalSection(ushort Id, ImmutableArray<byte> Payload);

/// <summary>The known cursor plus opaque optional sections retained for forward-compatible rewrites.</summary>
public sealed record SessionCursorEnvelope(
    SessionCursor Cursor,
    ImmutableArray<SessionCursorOptionalSection> OptionalSections);

/// <summary>
/// Versioned, bounded binary envelope for the authoritative cursor and execution log. This is not
/// a KV-cache codec: active-state ABI, compatibility and payload sections belong to Milestone 2.
/// </summary>
public static class SessionCursorCodec
{
    private const uint Magic = 0x4F545343; // OTSC
    private const ushort LegacyVersion = 1;
    private const ushort Version = 2;
    private const ushort CursorSectionId = 1;
    private const byte RequiredSection = 1;
    private const int EnvelopeHeaderBytes = sizeof(uint) + sizeof(ushort) + sizeof(ushort);
    private const int DirectoryEntryBytes = sizeof(ushort) + sizeof(byte) + sizeof(int) + sizeof(int);
    private const byte TokenSegmentKind = 1;
    private const byte AtomicSegmentKind = 2;
    private static readonly UTF8Encoding StrictUtf8 = new(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

    public static byte[] Encode(SessionCursor cursor, SessionCursorCodecLimits? limits = null)
        => Encode(new SessionCursorEnvelope(cursor, []), limits);

    /// <summary>Encodes a cursor while preserving opaque optional sections from a prior decode.</summary>
    public static byte[] Encode(SessionCursorEnvelope envelope, SessionCursorCodecLimits? limits = null)
    {
        ArgumentNullException.ThrowIfNull(envelope);
        ArgumentNullException.ThrowIfNull(envelope.Cursor);
        var effectiveLimits = limits ?? new SessionCursorCodecLimits();
        effectiveLimits.Validate();
        var optionalSections = envelope.OptionalSections.IsDefault ? [] : envelope.OptionalSections;
        if (optionalSections.Length + 1 > effectiveLimits.MaxSections)
            throw new SessionCursorFormatException("Cursor section count exceeds its configured limit.");
        if (optionalSections.Any(section => section.Id == CursorSectionId)
            || optionalSections.GroupBy(section => section.Id).Any(group => group.Count() != 1))
            throw new SessionCursorFormatException("Optional sections must have unique non-cursor identifiers.");
        int cursorBytes = GetCursorSectionLength(envelope.Cursor, effectiveLimits);
        int payloadBytes = checked(EnvelopeHeaderBytes + checked((optionalSections.Length + 1) * DirectoryEntryBytes) + cursorBytes);
        foreach (var section in optionalSections)
        {
            if (section.Payload.IsDefault)
                throw new SessionCursorFormatException("Optional section payload must not be default.");
            payloadBytes = checked(payloadBytes + section.Payload.Length);
        }
        if (payloadBytes > effectiveLimits.MaxPayloadBytes)
            throw new SessionCursorFormatException("Cursor payload exceeds the configured byte limit.");
        using var stream = new MemoryStream(payloadBytes);
        using var writer = new BinaryWriter(stream, StrictUtf8, leaveOpen: true);
        writer.Write(Magic);
        writer.Write(Version);
        writer.Write((ushort)(optionalSections.Length + 1));
        writer.Write(CursorSectionId);
        writer.Write(RequiredSection);
        int offset = EnvelopeHeaderBytes + (optionalSections.Length + 1) * DirectoryEntryBytes;
        writer.Write(offset);
        writer.Write(cursorBytes);
        offset += cursorBytes;
        foreach (var section in optionalSections)
        {
            writer.Write(section.Id);
            writer.Write((byte)0);
            writer.Write(offset);
            writer.Write(section.Payload.Length);
            offset += section.Payload.Length;
        }
        WriteCursorSection(writer, envelope.Cursor);
        foreach (var section in optionalSections) writer.Write(section.Payload.AsSpan());
        writer.Flush();
        return stream.ToArray();
    }

    private static void WriteCursorSection(BinaryWriter writer, SessionCursor cursor)
    {
        writer.Write(cursor.AcceptedPositionCount);
        writer.Write(cursor.MaterializedPositionCount);
        writer.Write(cursor.NextLogicalPosition);
        writer.Write(cursor.PhysicalSlotCount);
        writer.Write((byte)cursor.Coverage);
        writer.Write(cursor.ExecutionLog.Length);
        foreach (var segment in cursor.ExecutionLog)
        {
            switch (segment)
            {
                case TokenSegment tokens:
                    writer.Write(TokenSegmentKind);
                    writer.Write(tokens.TokenIds.Length);
                    foreach (int token in tokens.TokenIds) writer.Write(token);
                    break;
                case AtomicExecutionSegment atomic:
                    writer.Write(AtomicSegmentKind);
                    WriteUtf8(writer, atomic.Kind);
                    writer.Write(atomic.Positions);
                    WriteUtf8(writer, atomic.CanonicalInputDigest.Value);
                    break;
                default:
                    throw new NotSupportedException($"Execution segment '{segment.GetType().Name}' has no cursor codec representation.");
            }
        }
    }

    public static SessionCursor Decode(ReadOnlySpan<byte> payload, SessionCursorCodecLimits? limits = null)
        => DecodeEnvelope(payload, limits).Cursor;

    /// <summary>Decodes a cursor and retains opaque optional sections for a later compatible rewrite.</summary>
    public static SessionCursorEnvelope DecodeEnvelope(ReadOnlySpan<byte> payload, SessionCursorCodecLimits? limits = null)
    {
        var effectiveLimits = limits ?? new SessionCursorCodecLimits();
        effectiveLimits.Validate();
        if (payload.Length > effectiveLimits.MaxPayloadBytes)
            throw new SessionCursorFormatException("Cursor payload exceeds the configured byte limit.");

        using var stream = new MemoryStream(payload.ToArray(), writable: false);
        using var reader = new BinaryReader(stream, StrictUtf8, leaveOpen: true);
        try
        {
            if (reader.ReadUInt32() != Magic) throw new SessionCursorFormatException("Cursor payload has an invalid magic value.");
            ushort version = reader.ReadUInt16();
            if (version == LegacyVersion)
                return new SessionCursorEnvelope(DecodeCursorSection(payload.Slice(sizeof(uint) + sizeof(ushort)), effectiveLimits), []);
            if (version != Version) throw new SessionCursorFormatException("Cursor payload uses an unsupported format version.");
            int sectionCount = reader.ReadUInt16();
            if (sectionCount > effectiveLimits.MaxSections)
                throw new SessionCursorFormatException("Cursor section count exceeds its configured limit.");
            int directoryEnd = checked(EnvelopeHeaderBytes + checked(sectionCount * DirectoryEntryBytes));
            if (directoryEnd > payload.Length) throw new SessionCursorFormatException("Cursor section directory is truncated.");
            var sections = new List<Section>(sectionCount);
            for (int index = 0; index < sectionCount; index++)
            {
                ushort id = reader.ReadUInt16();
                byte flags = reader.ReadByte();
                int offset = reader.ReadInt32();
                int length = reader.ReadInt32();
                if ((flags & ~RequiredSection) != 0 || offset < directoryEnd || length < 0
                    || offset > payload.Length - length)
                    throw new SessionCursorFormatException("Cursor section directory contains an invalid entry.");
                if (sections.Any(section => section.Id == id))
                    throw new SessionCursorFormatException("Cursor section directory contains a duplicate section identifier.");
                sections.Add(new Section(id, flags, offset, length));
            }
            int expectedOffset = directoryEnd;
            foreach (var section in sections.OrderBy(section => section.Offset))
            {
                if (section.Offset != expectedOffset)
                    throw new SessionCursorFormatException("Cursor sections must cover the payload contiguously without gaps or overlap.");
                expectedOffset = checked(expectedOffset + section.Length);
            }
            if (expectedOffset != payload.Length)
                throw new SessionCursorFormatException("Cursor section directory does not cover the complete payload.");
            foreach (var section in sections.Where(section => section.Id != CursorSectionId && (section.Flags & RequiredSection) != 0))
                throw new SessionCursorFormatException($"Cursor payload contains unknown required section {section.Id}.");
            var cursorSection = sections.SingleOrDefault(section => section.Id == CursorSectionId);
            if (cursorSection == default || (cursorSection.Flags & RequiredSection) == 0)
                throw new SessionCursorFormatException("Cursor payload is missing its required cursor section.");

            var optionalSections = ImmutableArray.CreateBuilder<SessionCursorOptionalSection>();
            foreach (var section in sections)
            {
                if (section.Id != CursorSectionId)
                    optionalSections.Add(new SessionCursorOptionalSection(section.Id,
                        ImmutableArray.CreateRange(payload.Slice(section.Offset, section.Length).ToArray())));
            }
            return new SessionCursorEnvelope(
                DecodeCursorSection(payload.Slice(cursorSection.Offset, cursorSection.Length), effectiveLimits),
                optionalSections.ToImmutable());
        }
        catch (EndOfStreamException ex)
        {
            throw new SessionCursorFormatException($"Cursor payload is truncated: {ex.Message}");
        }
        catch (DecoderFallbackException ex)
        {
            throw new SessionCursorFormatException($"Cursor payload contains invalid UTF-8: {ex.Message}");
        }
        catch (ArgumentException ex)
        {
            throw new SessionCursorFormatException($"Cursor payload violates an invariant: {ex.Message}");
        }
    }

    private static int GetCursorSectionLength(SessionCursor cursor, SessionCursorCodecLimits limits)
    {
        if (cursor.ExecutionLog.Length > limits.MaxSegments)
            throw new SessionCursorFormatException("Cursor segment count exceeds its configured limit.");
        int length = 21; // four counts, coverage and segment count
        foreach (var segment in cursor.ExecutionLog)
        {
            length = checked(length + 1);
            switch (segment)
            {
                case TokenSegment tokens:
                    if (tokens.TokenIds.Length > limits.MaxTokensPerSegment)
                        throw new SessionCursorFormatException("Cursor token count exceeds its configured limit.");
                    length = checked(length + sizeof(int) + checked(tokens.TokenIds.Length * sizeof(int)));
                    break;
                case AtomicExecutionSegment atomic:
                    length = checked(length + GetUtf8FieldLength(atomic.Kind, limits)
                        + sizeof(int) + GetUtf8FieldLength(atomic.CanonicalInputDigest.Value, limits));
                    break;
                default:
                    throw new NotSupportedException($"Execution segment '{segment.GetType().Name}' has no cursor codec representation.");
            }
        }
        return length;
    }

    private static SessionCursor DecodeCursorSection(ReadOnlySpan<byte> payload, SessionCursorCodecLimits limits)
    {
        using var stream = new MemoryStream(payload.ToArray(), writable: false);
        using var reader = new BinaryReader(stream, StrictUtf8, leaveOpen: true);
        int accepted = reader.ReadInt32();
        int materialized = reader.ReadInt32();
        int next = reader.ReadInt32();
        int physicalSlots = reader.ReadInt32();
        var coverage = (StateCoverage)reader.ReadByte();
        if (!Enum.IsDefined(coverage)) throw new SessionCursorFormatException("Cursor payload has an unknown state coverage value.");
        int segmentCount = ReadBoundedCount(reader, limits.MaxSegments, "segment count");
        var segments = ImmutableArray.CreateBuilder<ExecutionSegment>(segmentCount);
        for (int index = 0; index < segmentCount; index++)
        {
            switch (reader.ReadByte())
            {
                case TokenSegmentKind:
                {
                    int tokenCount = ReadBoundedCount(reader, limits.MaxTokensPerSegment, "token count");
                    var tokens = new int[tokenCount];
                    for (int token = 0; token < tokenCount; token++) tokens[token] = reader.ReadInt32();
                    segments.Add(new TokenSegment(tokens));
                    break;
                }
                case AtomicSegmentKind:
                {
                    string kind = ReadUtf8(reader, limits);
                    int positions = reader.ReadInt32();
                    string digest = ReadUtf8(reader, limits);
                    segments.Add(new AtomicExecutionSegment(kind, positions, new ContentDigest(digest)));
                    break;
                }
                default:
                    throw new SessionCursorFormatException("Cursor payload contains an unknown mandatory segment kind.");
            }
        }
        if (stream.Position != stream.Length) throw new SessionCursorFormatException("Cursor section contains trailing bytes.");
        return new SessionCursor(segments.MoveToImmutable(), accepted, materialized, next, physicalSlots, coverage);
    }

    private static int GetUtf8FieldLength(string value, SessionCursorCodecLimits limits)
    {
        int byteCount = StrictUtf8.GetByteCount(value);
        if (byteCount > limits.MaxStringBytes)
            throw new SessionCursorFormatException("Cursor string byte count exceeds its configured limit.");
        return checked(sizeof(int) + byteCount);
    }

    private static int ReadBoundedCount(BinaryReader reader, int max, string name)
    {
        int value = reader.ReadInt32();
        if (value < 0 || value > max) throw new SessionCursorFormatException($"Cursor {name} exceeds its configured limit.");
        return value;
    }

    private static void WriteUtf8(BinaryWriter writer, string value)
    {
        byte[] bytes = StrictUtf8.GetBytes(value);
        writer.Write(bytes.Length);
        writer.Write(bytes);
    }

    private static string ReadUtf8(BinaryReader reader, SessionCursorCodecLimits limits)
    {
        int byteCount = ReadBoundedCount(reader, limits.MaxStringBytes, "string byte count");
        byte[] bytes = reader.ReadBytes(byteCount);
        if (bytes.Length != byteCount) throw new EndOfStreamException();
        return StrictUtf8.GetString(bytes);
    }

    private readonly record struct Section(ushort Id, byte Flags, int Offset, int Length);
}
