using System.Collections.Immutable;
using System.Security.Cryptography;
using System.Text;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Sessions;

/// <summary>ABI contract parameters defining active session KV cache compatibility.</summary>
public sealed record SessionStateABI(
    string ModelFingerprint,
    long KvBytesPerToken,
    int MaxSequenceLength,
    ModelFormat ModelFormat = ModelFormat.Gguf);

/// <summary>Canonical in-memory active state envelope exported for lossless state serialization.</summary>
public sealed record SessionStateEnvelope(
    SessionId SessionId,
    SessionCursorEnvelope CursorEnvelope,
    SessionStateABI Abi,
    string CompatibilityKey,
    StatePayloadHash PayloadHash,
    ImmutableArray<SessionCursorOptionalSection> OptionalSections);

/// <summary>
/// Versioned, bounded binary codec for exporting and importing canonical active session state (Milestone 2).
/// Enforces section directory isolation, ABI compatibility validation, payload integrity hashing,
/// and hostile length field protection.
/// </summary>
public static class SessionStateCodec
{
    private const uint Magic = 0x4F545353; // OTSS (OpenTail Session State)
    private const ushort Version = 2;

    public const ushort CursorSectionId = 1;
    public const ushort AbiSectionId = 2;
    public const ushort CompatibilitySectionId = 3;
    public const ushort PayloadHashSectionId = 4;
    public const ushort SessionIdSectionId = 5;
    /// <summary>Required model-container discriminator, introduced in state codec version 2.</summary>
    public const ushort ModelFormatSectionId = 7;
    /// <summary>
    /// RETIRED. KV pages briefly rode in an optional section with this id; they cannot, because
    /// <see cref="SessionCursorCodecLimits.MaxPayloadBytes"/> caps an envelope at 4 MB while a single
    /// cache block costs ~6 MB. KV now travels out of band in segment packs
    /// (<c>HotSession.ExportKvBytes</c> + <c>SegmentPackStore</c>). The id stays reserved so a future
    /// section does not silently reuse it and collide with any envelope written in that window.
    /// </summary>
    public const ushort KvCacheDataSectionId = 6;

    private const byte RequiredSection = 1;
    private const int EnvelopeHeaderBytes = sizeof(uint) + sizeof(ushort) + sizeof(ushort);
    private const int DirectoryEntryBytes = sizeof(ushort) + sizeof(byte) + sizeof(int) + sizeof(int);

    private static readonly UTF8Encoding StrictUtf8 = new(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

    /// <summary>Computes a deterministic SHA-256 compatibility key for a model ABI specification.</summary>
    /// <remarks>
    /// <para><b>GGUF hashes exactly as it did before the format discriminator existed.</b> The format is
    /// appended to the hashed string only for non-GGUF containers, so a session written by an older
    /// build still computes the same key today. Hashing <c>ModelFormat.Gguf</c> unconditionally would
    /// change every historical key — which would silently defeat the version-1 read support in
    /// <see cref="Decode"/>, since a v1 envelope's stored key could then never match a recomputed one.</para>
    ///
    /// <para>The key still fences the formats apart: a SafeTensors session and a GGUF session over the
    /// same weights produce different keys, which is the property that matters. Do not "simplify" this
    /// into a single interpolation — the asymmetry is the point, and no test can notice the difference
    /// until a real pre-existing session fails to restore.</para>
    /// </remarks>
    public static string ComputeCompatibilityKey(SessionStateABI abi)
    {
        ArgumentNullException.ThrowIfNull(abi);
        using var sha256 = SHA256.Create();
        var raw = abi.ModelFormat == ModelFormat.Gguf
            ? $"{abi.ModelFingerprint}:{abi.KvBytesPerToken}:{abi.MaxSequenceLength}"
            : $"{abi.ModelFingerprint}:{abi.KvBytesPerToken}:{abi.MaxSequenceLength}:{abi.ModelFormat}";
        var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(raw));
        return Convert.ToHexStringLower(hashBytes);
    }

    public static byte[] Encode(SessionStateEnvelope envelope, SessionCursorCodecLimits? limits = null)
    {
        ArgumentNullException.ThrowIfNull(envelope);
        ArgumentNullException.ThrowIfNull(envelope.CursorEnvelope);
        ArgumentNullException.ThrowIfNull(envelope.Abi);
        var effectiveLimits = limits ?? new SessionCursorCodecLimits();
        effectiveLimits.Validate();

        var optionalSections = envelope.OptionalSections.IsDefault ? [] : envelope.OptionalSections;
        int totalSections = 6 + optionalSections.Length;
        if (totalSections > effectiveLimits.MaxSections)
            throw new SessionCursorFormatException("Session state section count exceeds configured limit.");

        var cursorBytes = SessionCursorCodec.Encode(envelope.CursorEnvelope, effectiveLimits);
        var abiBytes = EncodeAbi(envelope.Abi);
        var compatBytes = StrictUtf8.GetBytes(envelope.CompatibilityKey);
        var hashBytes = StrictUtf8.GetBytes(envelope.PayloadHash.Value.Value);
        var sessionIdBytes = StrictUtf8.GetBytes(envelope.SessionId.Value.ToString("N"));
        byte[] modelFormatBytes = [(byte)envelope.Abi.ModelFormat];

        int payloadBytes = checked(EnvelopeHeaderBytes + checked(totalSections * DirectoryEntryBytes)
            + cursorBytes.Length + abiBytes.Length + compatBytes.Length + hashBytes.Length
            + sessionIdBytes.Length + modelFormatBytes.Length);

        foreach (var sec in optionalSections)
        {
            if (sec.Payload.IsDefault)
                throw new SessionCursorFormatException("Optional section payload must not be default.");
            payloadBytes = checked(payloadBytes + sec.Payload.Length);
        }

        if (payloadBytes > effectiveLimits.MaxPayloadBytes)
            throw new SessionCursorFormatException("Session state payload exceeds configured byte limit.");

        using var stream = new MemoryStream(payloadBytes);
        using var writer = new BinaryWriter(stream, StrictUtf8, leaveOpen: true);

        writer.Write(Magic);
        writer.Write(Version);
        writer.Write((ushort)totalSections);

        int offset = EnvelopeHeaderBytes + (totalSections * DirectoryEntryBytes);

        // Section 1: Cursor
        WriteDirectoryEntry(writer, CursorSectionId, RequiredSection, ref offset, cursorBytes.Length);
        // Section 2: ABI
        WriteDirectoryEntry(writer, AbiSectionId, RequiredSection, ref offset, abiBytes.Length);
        // Section 3: CompatibilityKey
        WriteDirectoryEntry(writer, CompatibilitySectionId, RequiredSection, ref offset, compatBytes.Length);
        // Section 4: PayloadHash
        WriteDirectoryEntry(writer, PayloadHashSectionId, RequiredSection, ref offset, hashBytes.Length);
        // Section 5: SessionId
        WriteDirectoryEntry(writer, SessionIdSectionId, RequiredSection, ref offset, sessionIdBytes.Length);
        // Section 6: ModelFormat
        WriteDirectoryEntry(writer, ModelFormatSectionId, RequiredSection, ref offset, modelFormatBytes.Length);

        // Optional sections
        foreach (var sec in optionalSections)
        {
            WriteDirectoryEntry(writer, sec.Id, 0, ref offset, sec.Payload.Length);
        }

        // Section payloads
        writer.Write(cursorBytes);
        writer.Write(abiBytes);
        writer.Write(compatBytes);
        writer.Write(hashBytes);
        writer.Write(sessionIdBytes);
        writer.Write(modelFormatBytes);
        foreach (var sec in optionalSections) writer.Write(sec.Payload.AsSpan());

        writer.Flush();
        return stream.ToArray();
    }

    public static SessionStateEnvelope Decode(ReadOnlySpan<byte> data, SessionCursorCodecLimits? limits = null)
    {
        var effectiveLimits = limits ?? new SessionCursorCodecLimits();
        effectiveLimits.Validate();

        if (data.Length < EnvelopeHeaderBytes || data.Length > effectiveLimits.MaxPayloadBytes)
            throw new SessionCursorFormatException("Invalid session state payload length.");

        using var stream = new MemoryStream(data.ToArray(), writable: false);
        using var reader = new BinaryReader(stream, StrictUtf8, leaveOpen: true);

        uint magic = reader.ReadUInt32();
        if (magic != Magic)
            throw new SessionCursorFormatException($"Invalid session state magic 0x{magic:X8}, expected 0x{Magic:X8}.");

        ushort version = reader.ReadUInt16();
        if (version is not 1 and not Version)
            throw new SessionCursorFormatException($"Unsupported session state version {version}.");

        ushort sectionCount = reader.ReadUInt16();
        int minSectionCount = version == 1 ? 5 : 6;
        if (sectionCount < minSectionCount || sectionCount > effectiveLimits.MaxSections)
            throw new SessionCursorFormatException($"Invalid section count {sectionCount}.");

        var entries = new (ushort id, byte flags, int offset, int length)[sectionCount];
        for (int i = 0; i < sectionCount; i++)
        {
            ushort id = reader.ReadUInt16();
            byte flags = reader.ReadByte();
            int offset = reader.ReadInt32();
            int length = reader.ReadInt32();

            if (offset < EnvelopeHeaderBytes || length < 0 || checked(offset + length) > data.Length)
                throw new SessionCursorFormatException($"Invalid directory entry bounds for section {id}.");

            entries[i] = (id, flags, offset, length);
        }

        SessionId? sessionId = null;
        SessionCursorEnvelope? cursorEnv = null;
        SessionStateABI? abi = null;
        string? compatKey = null;
        StatePayloadHash? payloadHash = null;
        ModelFormat modelFormat = ModelFormat.Gguf;
        bool hasModelFormat = version == 1;
        var optionalSections = ImmutableArray.CreateBuilder<SessionCursorOptionalSection>();

        foreach (var (id, flags, offset, length) in entries)
        {
            var sectionData = data.Slice(offset, length);
            switch (id)
            {
                case CursorSectionId:
                    cursorEnv = SessionCursorCodec.DecodeEnvelope(sectionData, effectiveLimits);
                    break;
                case AbiSectionId:
                    abi = DecodeAbi(sectionData, effectiveLimits);
                    break;
                case CompatibilitySectionId:
                    compatKey = StrictUtf8.GetString(sectionData);
                    break;
                case PayloadHashSectionId:
                    var hashStr = StrictUtf8.GetString(sectionData);
                    payloadHash = new StatePayloadHash(new ContentDigest(hashStr));
                    break;
                case SessionIdSectionId:
                    var idStr = StrictUtf8.GetString(sectionData);
                    sessionId = new SessionId(Guid.Parse(idStr));
                    break;
                case ModelFormatSectionId:
                    if (length != 1 || !Enum.IsDefined(typeof(ModelFormat), (ModelFormat)sectionData[0]))
                        throw new SessionCursorFormatException("Invalid model format section.");
                    modelFormat = (ModelFormat)sectionData[0];
                    hasModelFormat = true;
                    break;
                default:
                    if ((flags & RequiredSection) != 0)
                        throw new SessionCursorFormatException($"Unsupported required section {id}.");
                    optionalSections.Add(new SessionCursorOptionalSection(id, sectionData.ToArray().ToImmutableArray()));
                    break;
            }
        }

        if (sessionId is null || cursorEnv is null || abi is null || compatKey is null || payloadHash is null || !hasModelFormat)
            throw new SessionCursorFormatException("Missing required section in session state payload.");

        return new SessionStateEnvelope(sessionId.Value, cursorEnv,
            abi with { ModelFormat = modelFormat }, compatKey, payloadHash.Value, optionalSections.ToImmutable());
    }

    private static void WriteDirectoryEntry(BinaryWriter writer, ushort id, byte flags, ref int offset, int length)
    {
        writer.Write(id);
        writer.Write(flags);
        writer.Write(offset);
        writer.Write(length);
        offset = checked(offset + length);
    }

    private static byte[] EncodeAbi(SessionStateABI abi)
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, StrictUtf8);
        writer.Write(abi.ModelFingerprint);
        writer.Write(abi.KvBytesPerToken);
        writer.Write(abi.MaxSequenceLength);
        writer.Flush();
        return stream.ToArray();
    }

    private static SessionStateABI DecodeAbi(ReadOnlySpan<byte> data, SessionCursorCodecLimits limits)
    {
        using var stream = new MemoryStream(data.ToArray(), writable: false);
        using var reader = new BinaryReader(stream, StrictUtf8);
        string modelFp = reader.ReadString();
        if (StrictUtf8.GetByteCount(modelFp) > limits.MaxStringBytes)
            throw new SessionCursorFormatException("ABI model fingerprint exceeds string byte limit.");
        long kvBytes = reader.ReadInt64();
        int maxSeqLen = reader.ReadInt32();
        return new SessionStateABI(modelFp, kvBytes, maxSeqLen);
    }
}
