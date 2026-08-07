namespace OpenTail.Stingray.Sessions;

using System;
using System.Collections.Immutable;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

/// <summary>
/// Session runtime wrapper extending <see cref="HotSessionRuntime"/> with cold eviction to disk and
/// restoration from a manifest plus segment pack (Milestone 3 Phase 2 / R2, partial).
/// </summary>
/// <remarks>
/// <para><b>What is durable: cursor metadata plus the cache bytes for opt-in cache types.</b>
/// <see cref="EvictToDisk"/> keeps the cursor envelope and physical KV pages separate: the bounded
/// envelope remains a cursor/ABI codec, while the potentially large page stream is split into
/// <see cref="SegmentPackStore"/> blocks. The current CPU <see cref="PagedKvCache"/> opts in through
/// <see cref="IPersistableSequenceKvCache"/>.</para>
///
/// <para><b>What is not proven yet: restart-safe product continuation.</b> A real GGUF must still
/// be exercised through process exit, a new runtime, restore, and greedy continuation compared
/// token-by-token with fresh full replay. Backends with windowed, recurrent, compressed, or
/// device-resident caches are not covered merely because the cursor can be restored.</para>
///
/// <para>What IS sound here: atomic manifest writes with a whole-manifest SHA-256, per-block
/// SHA-256 verification on load, identity-preserving restore, real committed revisions, ABI
/// compatibility checks on import, and a startup sweep of orphaned temp files.</para>
/// </remarks>
public sealed class ColdSessionRuntime
{
    private readonly HotSessionRuntime _hotRuntime;
    private readonly ContinuousBatchingEngine _engine;
    private readonly string _storageDirectory;
    private readonly ModelFormat _modelFormat;

    public ColdSessionRuntime(HotSessionRuntime hotRuntime, ContinuousBatchingEngine engine, string storageDirectory,
        ModelFormat modelFormat = ModelFormat.Gguf)
    {
        ArgumentNullException.ThrowIfNull(hotRuntime);
        ArgumentNullException.ThrowIfNull(engine);
        ArgumentException.ThrowIfNullOrEmpty(storageDirectory);

        _hotRuntime = hotRuntime;
        _engine = engine;
        _storageDirectory = Path.GetFullPath(storageDirectory);
        _modelFormat = modelFormat;

        if (!Directory.Exists(_storageDirectory))
        {
            Directory.CreateDirectory(_storageDirectory);
        }

        // Cleanup stale orphaned temp files from prior crashed runs (#8)
        SweepStaleTempFiles(_storageDirectory);
    }

    public HotSessionRuntime HotRuntime => _hotRuntime;
    public string StorageDirectory => _storageDirectory;

    /// <summary>Creates a new session with a caller-selected opaque identifier.</summary>
    public HotSession Create(SessionId? sessionId = null, string? modelKey = null) =>
        _hotRuntime.Create(sessionId, modelKey);

    public HotSession Create(SessionAddress address) => _hotRuntime.Create(address);

    /// <summary>Opens an active session or restores its verified cold representation.</summary>
    public HotSession Open(SessionId sessionId)
    {
        try
        {
            return _hotRuntime.Open(sessionId);
        }
        catch (SessionNotFoundException)
        {
            string manifestPath = Path.Combine(_storageDirectory, $"{sessionId.Value:N}.manifest");
            if (!File.Exists(manifestPath)) throw;
            return RestoreColdSession(FileSessionManifest.Load(manifestPath));
        }
    }

    public HotSession OpenOrCreate(SessionAddress address)
    {
        var sessionId = address.ToSessionId();

        try
        {
            // If active in RAM, return hot session immediately
            return _hotRuntime.Open(sessionId);
        }
        catch (SessionNotFoundException)
        {
            // Search for cold manifest on disk
            string manifestPath = Path.Combine(_storageDirectory, $"{sessionId.Value:N}.manifest");
            if (File.Exists(manifestPath))
            {
                var manifest = FileSessionManifest.Load(manifestPath);
                return RestoreColdSession(manifest);
            }

            // Otherwise create new active hot session
            return _hotRuntime.Create(address);
        }
    }

    /// <summary>Deletes active and cold state for a session. Missing sessions return <c>false</c>.</summary>
    public bool Delete(SessionId sessionId)
    {
        bool removed = _hotRuntime.Delete(sessionId);
        string manifestPath = Path.Combine(_storageDirectory, $"{sessionId.Value:N}.manifest");
        if (!File.Exists(manifestPath)) return removed;

        var manifest = FileSessionManifest.Load(manifestPath);
        foreach (var block in manifest.Blocks)
        {
            string packPath = SegmentPackStore.GetPackPath(_storageDirectory, block.BlockId);
            if (File.Exists(packPath)) File.Delete(packPath);
        }
        File.Delete(manifestPath);
        return true;
    }

    /// <summary>
    /// Evicts an active hot session from RAM to disk manifests and segment pack files (#1, #5, #11).
    /// </summary>
    /// <summary>Bytes of KV per segment pack.</summary>
    /// <remarks>
    /// Kept well under <c>SegmentPackStore</c>'s 100 MB per-block ceiling. A whole session's KV blows
    /// straight past that — a 24-layer 2048-wide cache costs 6 MB per 16 tokens, so ~256 tokens would
    /// exceed a single pack. Splitting across packs is what the manifest's ordered block list is for.
    /// </remarks>
    private const int KvChunkBytes = 32 * 1024 * 1024;

    private const string CursorBlockPrefix = "cur_";
    private const string KvBlockPrefix = "kv_";

    /// <summary>
    /// Evicts an active hot session to disk: the cursor envelope in one pack, the physical KV pages
    /// across as many packs as they need, and a manifest listing all of them in order.
    /// </summary>
    /// <exception cref="InvalidOperationException">A turn is still running. Export reads the cursor,
    /// block table and pages without synchronisation, so a live turn would produce a stream whose
    /// header disagrees with its body.</exception>
    public SessionManifestEnvelope EvictToDisk(HotSession session, string modelFingerprint)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentException.ThrowIfNullOrEmpty(modelFingerprint);

        if (session.IsInUse)
            throw new InvalidOperationException(
                $"Session '{session.SessionId}' has a turn in flight and cannot be evicted. " +
                "KV export is not synchronised against a running turn.");

        byte[] exportedState = session.ExportState(modelFingerprint, modelFormat: _modelFormat);
        var envelope = SessionStateCodec.Decode(exportedState);
        string sid = session.SessionId.Value.ToString("N");

        var blocks = ImmutableArray.CreateBuilder<SegmentBlockRef>();

        // Block 0 is always the cursor envelope; RestoreColdSession relies on that ordering.
        blocks.Add(SegmentPackStore.SaveBlock(
            _storageDirectory, CursorBlockPrefix + sid,
            startTokenPos: 0,
            tokenCount: session.Cursor.AcceptedPositionCount,
            payload: exportedState));

        // Blocks 1..N are the KV byte stream in order. Token positions do not describe a byte chunk,
        // so they are recorded as zero rather than given a misleading value.
        byte[]? kvBytes = session.ExportKvBytes();
        if (kvBytes is { Length: > 0 })
        {
            int chunks = (kvBytes.Length + KvChunkBytes - 1) / KvChunkBytes;
            for (int i = 0; i < chunks; i++)
            {
                int offset = i * KvChunkBytes;
                int len = Math.Min(KvChunkBytes, kvBytes.Length - offset);
                blocks.Add(SegmentPackStore.SaveBlock(
                    _storageDirectory, $"{KvBlockPrefix}{sid}_{i:D5}",
                    startTokenPos: 0, tokenCount: 0,
                    payload: kvBytes.AsSpan(offset, len)));
            }
        }

        var manifest = new SessionManifestEnvelope(
            session.SessionId,
            session.CommittedRevision,
            envelope.Abi,
            envelope.CompatibilityKey,
            envelope.PayloadHash,
            blocks.ToImmutable());

        string manifestPath = Path.Combine(_storageDirectory, $"{sid}.manifest");
        FileSessionManifest.SaveAtomic(manifestPath, manifest);

        // Reclaim packs this session wrote earlier that the new manifest no longer lists. Block ids
        // are deterministic (kv_{sid}_{index}), so re-eviction overwrites in place — but an eviction
        // that produces FEWER blocks than its predecessor leaves the surplus high-index packs on
        // disk, referenced by nothing. Restore is unaffected because it only loads ids the manifest
        // names, so this is a disk leak rather than a correctness bug; it is swept here, AFTER the
        // manifest commits, so a crash mid-sweep can only ever leave extra bytes, never remove a
        // pack the live manifest still points at.
        PruneUnreferencedPacks(sid, manifest);

        _hotRuntime.Delete(session.SessionId);
        return manifest;
    }

    /// <summary>Deletes this session's packs that <paramref name="manifest"/> does not reference.</summary>
    private void PruneUnreferencedPacks(string sid, SessionManifestEnvelope manifest)
    {
        var referenced = new HashSet<string>(StringComparer.Ordinal);
        foreach (var block in manifest.Blocks)
            referenced.Add(SegmentPackStore.GetPackPath(_storageDirectory, block.BlockId));

        try
        {
            // Scoped to this session's own prefixes: another session's packs are never candidates.
            // Materialise before deleting — EnumerateFiles is lazy, so removing entries mid-walk can
            // skip siblings or throw, and the catch below would hide it as a silent no-op.
            foreach (string prefix in new[] { CursorBlockPrefix + sid, KvBlockPrefix + sid })
            {
                string[] candidates = Directory.GetFiles(
                    _storageDirectory, SegmentPackStore.GetPackSearchPattern(prefix));
                foreach (string path in candidates)
                    if (!referenced.Contains(path))
                        File.Delete(path);
            }
        }
        catch (IOException)
        {
            // Best effort: a stale pack costs disk, never correctness. Never fail an evicted turn
            // because housekeeping lost a race with another reader.
        }
        catch (UnauthorizedAccessException)
        {
        }
    }

    private HotSession RestoreColdSession(SessionManifestEnvelope manifest)
    {
        if (manifest.Blocks.Length == 0)
            throw new SessionJournalFormatException($"Manifest for session '{manifest.SessionId}' contains no state blocks.");

        var cursorBlock = manifest.Blocks[0];
        if (!cursorBlock.BlockId.StartsWith(CursorBlockPrefix, StringComparison.Ordinal))
            throw new SessionJournalFormatException(
                $"Manifest for session '{manifest.SessionId}' does not begin with a cursor block (found '{cursorBlock.BlockId}').");

        byte[] exportedState = SegmentPackStore.LoadBlock(
            SegmentPackStore.GetPackPath(_storageDirectory, cursorBlock.BlockId));

        // Reassemble the KV stream from the remaining blocks, in manifest order. Each pack verifies
        // its own SHA-256 on load, so a damaged chunk fails here rather than being imported.
        byte[] kvBytes = [];
        if (manifest.Blocks.Length > 1)
        {
            long total = 0;
            for (int i = 1; i < manifest.Blocks.Length; i++) total += manifest.Blocks[i].UncompressedBytes;
            kvBytes = new byte[total];
            int written = 0;
            for (int i = 1; i < manifest.Blocks.Length; i++)
            {
                byte[] chunk = SegmentPackStore.LoadBlock(
                    SegmentPackStore.GetPackPath(_storageDirectory, manifest.Blocks[i].BlockId));
                Buffer.BlockCopy(chunk, 0, kvBytes, written, chunk.Length);
                written += chunk.Length;
            }
            if (written != kvBytes.Length)
                throw new SessionJournalFormatException(
                    $"Session '{manifest.SessionId}' KV stream is {written} bytes but the manifest declares {kvBytes.Length}.");
        }

        return _hotRuntime.ImportState(exportedState, kvBytes, manifest.Abi.ModelFingerprint,
            expectedModelFormat: _modelFormat);
    }

    private static void SweepStaleTempFiles(string directory)
    {
        try
        {
            foreach (var tempFile in Directory.EnumerateFiles(directory, "*.tmp_*"))
            {
                try
                {
                    File.Delete(tempFile);
                }
                catch
                {
                    // Best effort cleanup
                }
            }
        }
        catch
        {
            // Ignore directory enumeration failures during startup sweep
        }
    }
}
