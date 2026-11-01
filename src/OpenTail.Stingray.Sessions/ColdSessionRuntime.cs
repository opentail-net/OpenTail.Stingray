namespace OpenTail.Stingray.Sessions;


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
/// <para><b>What is proven, and only for one lane.</b> The bar this doc used to set — a real GGUF
/// carried through process exit, a new runtime, restore, and greedy continuation compared
/// token-by-token against fresh full replay — is met by
/// <c>HotSessionGreedyReplayTests.ColdSession_RealModel_CrossProcessRestore_MatchesFullGreedyReplay</c>,
/// which spawns an actual child process. It is <c>Assert.SkipUnless</c>-gated on the model
/// fixture, so it reports as SKIPPED rather than passing where <c>models/</c> is absent — read a
/// green run accordingly.</para>
///
/// <para><b>What that does NOT extend to:</b> backends with windowed, recurrent, compressed, or
/// device-resident caches. Being able to restore a cursor is not evidence for those; the proof
/// covers the CPU-dense GGUF lane it was written against and nothing wider.</para>
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
        SweepOrphanPacks(_storageDirectory);
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
    private const string OperationBlockPrefix = "ops_";
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

        // An exclusive lease on the turn gate, held through export, manifest publication and deletion: IsInUse alone clears while a turn is still
        // committing, which let a manifest pair a new cursor with the previous revision and operation ledger.
        using var exclusive = session.TryAcquireExclusive()
            ?? throw new InvalidOperationException(
                $"Session '{session.SessionId}' has a turn in flight or committing and cannot be evicted. " +
                "KV export is not synchronised against a running turn.");

        byte[] exportedState = session.ExportState(modelFingerprint, modelFormat: _modelFormat);
        var envelope = SessionStateCodec.Decode(exportedState);
        string sid = session.SessionId.Value.ToString("N");
        // Never overwrite packs named by the currently committed manifest. A crash after writing
        // a replacement pack but before atomically replacing that manifest used to leave the old
        // manifest pointing at a partially replaced cache. Generation-scoped ids make publication
        // one-way: until the new manifest is durable, the old manifest and every pack it names are
        // untouched; after publication, stale generations are only best-effort housekeeping.
        string generation = Guid.NewGuid().ToString("N");

        var blocks = ImmutableArray.CreateBuilder<SegmentBlockRef>();

        // Block 0 is always the cursor envelope; RestoreColdSession relies on that ordering.
        blocks.Add(SegmentPackStore.SaveBlock(
            _storageDirectory, $"{CursorBlockPrefix}{sid}_{generation}",
            startTokenPos: 0,
            tokenCount: session.Cursor.AcceptedPositionCount,
            payload: exportedState));

        // A completed response is part of the idempotency contract, not merely diagnostic text.
        // Keep its bounded ledger in an independent pack so cursor/KV restoration remains valid
        // for old manifests and large output cannot bloat the cursor envelope.
        var operations = _hotRuntime.GetSessionSnapshot(session.SessionId).Operations;
        if (operations.Any(operation => operation.State == SessionOperationState.Completed))
        {
            byte[] operationLedger = SessionOperationLedgerCodec.Encode(operations);
            blocks.Add(SegmentPackStore.SaveBlock(
                _storageDirectory, $"{OperationBlockPrefix}{sid}_{generation}",
                startTokenPos: 0, tokenCount: 0, payload: operationLedger));
        }

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
                    _storageDirectory, $"{KvBlockPrefix}{sid}_{generation}_{i:D5}",
                    startTokenPos: 0, tokenCount: 0,
                    payload: kvBytes.AsSpan(offset, len)));
            }
        }

        // Persist the STORE's revision — the turn counter RunTurnAsync validates expected_revision
        // against — not the cursor's accepted-position count. Writing the position count is what
        // made the durable and live paths disagree: a restored session came back seeded with a
        // position count while a live one counted turns, so the two lanes could not both be right.
        // Manifest v3 records this meaning; see FileSessionManifest.
        var manifest = new SessionManifestEnvelope(
            session.SessionId,
            _hotRuntime.GetSessionSnapshot(session.SessionId).CommittedRevision,
            envelope.Abi,
            envelope.CompatibilityKey,
            envelope.PayloadHash,
            blocks.ToImmutable());

        string manifestPath = Path.Combine(_storageDirectory, $"{sid}.manifest");
        FileSessionManifest.SaveAtomic(manifestPath, manifest);

        // Reclaim packs from earlier generations only AFTER publishing the new manifest. A crash
        // in this best-effort sweep can leave extra bytes, but can never remove a pack the live
        // manifest points at.
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
            foreach (string prefix in new[] { CursorBlockPrefix + sid, OperationBlockPrefix + sid, KvBlockPrefix + sid })
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

        SegmentBlockRef? operationBlock = null;
        var kvBlocks = new List<SegmentBlockRef>();
        foreach (var block in manifest.Blocks.Skip(1))
        {
            if (block.BlockId.StartsWith(OperationBlockPrefix, StringComparison.Ordinal))
            {
                if (operationBlock is not null)
                    throw new SessionJournalFormatException($"Manifest for session '{manifest.SessionId}' contains multiple operation ledgers.");
                operationBlock = block;
            }
            else if (block.BlockId.StartsWith(KvBlockPrefix, StringComparison.Ordinal))
            {
                kvBlocks.Add(block);
            }
            else
            {
                throw new SessionJournalFormatException($"Manifest for session '{manifest.SessionId}' contains an unknown block '{block.BlockId}'.");
            }
        }

        var restoredOperations = operationBlock is null
            ? ImmutableArray<SessionOperationSnapshot>.Empty
            : SessionOperationLedgerCodec.Decode(manifest.SessionId, SegmentPackStore.LoadBlock(
                SegmentPackStore.GetPackPath(_storageDirectory, operationBlock.BlockId)));

        // Reassemble the KV stream in manifest order. Each pack verifies its own SHA-256 on load,
        // so a damaged chunk fails here rather than being imported.
        byte[] kvBytes = [];
        if (kvBlocks.Count > 0)
        {
            long total = 0;
            foreach (var block in kvBlocks) total += block.UncompressedBytes;
            kvBytes = new byte[total];
            int written = 0;
            foreach (var block in kvBlocks)
            {
                byte[] chunk = SegmentPackStore.LoadBlock(
                    SegmentPackStore.GetPackPath(_storageDirectory, block.BlockId));
                Buffer.BlockCopy(chunk, 0, kvBytes, written, chunk.Length);
                written += chunk.Length;
            }
            if (written != kvBytes.Length)
                throw new SessionJournalFormatException(
                    $"Session '{manifest.SessionId}' KV stream is {written} bytes but the manifest declares {kvBytes.Length}.");
        }

        // Seed the store with the PERSISTED revision, so a restored session's concurrency token
        // continues the same counter a live one uses instead of restarting from a position count.
        var session = _hotRuntime.ImportState(exportedState, kvBytes, manifest.Abi.ModelFingerprint,
            expectedModelFormat: _modelFormat, committedRevision: manifest.Revision);
        session.RestoreCompletedOperations(restoredOperations);
        return session;
    }

    /// <summary>
    /// Deletes leftover <c>.tmp_{guid}</c> files from an interrupted <see cref="SegmentPackStore.SaveBlock"/>
    /// write. The glob is deliberately narrow: <c>SegmentPackStore</c> is the only writer of this
    /// pattern (<c>packPath + ".tmp_" + Guid</c>), so it cannot collide with any other store's temp
    /// files sharing this directory — notably <c>FileSessionStore</c>, whose atomic-write temp files
    /// are named <c>finalPath + ".tmp"</c> with no trailing GUID and would otherwise also match a
    /// broader <c>*.tmp*</c> glob. The age guard is defense in depth against sweeping a temp file
    /// still being written by a concurrent operation in this same store.
    /// </summary>
    private static void SweepStaleTempFiles(string directory)
    {
        var staleThreshold = DateTime.UtcNow - TimeSpan.FromMinutes(5);
        try
        {
            foreach (var tempFile in Directory.EnumerateFiles(directory, "*.tmp_*"))
            {
                try
                {
                    if (File.GetLastWriteTimeUtc(tempFile) <= staleThreshold)
                    {
                        File.Delete(tempFile);
                    }
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

    /// <summary>
    /// Deletes pack files that no manifest in this directory references, once they are older than the grace period.
    /// </summary>
    /// <remarks>
    /// Covers two crash windows that <see cref="PruneUnreferencedPacks"/> cannot reach: packs written by an eviction that
    /// never published its manifest, and the previous generation left behind when a crash interrupted pruning. A pack is
    /// a candidate only if its name is one of this runtime's block prefixes, then a session id, then a generation (both
    /// 32 lowercase hex digits), with an optional five-digit KV index. A session whose manifest exists but cannot be loaded
    /// keeps all of its packs. A missing manifest means nothing is referenced, so only the grace period protects its packs.
    /// Best effort: a failure on one file, or on the directory, never makes construction throw.
    /// </remarks>
    private static void SweepOrphanPacks(string directory)
    {
        // Same window as SweepStaleTempFiles. An EvictToDisk in another thread or runtime may have written packs
        // whose manifest is not yet published; those are younger than this window and are never candidates.
        var cutoff = DateTime.UtcNow - TimeSpan.FromMinutes(5);
        var referencedBySession = new Dictionary<string, HashSet<string>?>(StringComparer.Ordinal);
        try
        {
            foreach (string prefix in new[] { CursorBlockPrefix, OperationBlockPrefix, KvBlockPrefix })
            {
                string[] candidates = Directory.GetFiles(directory, SegmentPackStore.GetPackSearchPattern(prefix));
                foreach (string path in candidates)
                {
                    try
                    {
                        string blockId = Path.GetFileNameWithoutExtension(path);
                        if (!TryReadSessionId(blockId, prefix, out string sid)) continue;

                        if (!referencedBySession.TryGetValue(sid, out var referenced))
                        {
                            referenced = LoadReferencedPackNames(directory, sid);
                            referencedBySession[sid] = referenced;
                        }
                        if (referenced is null) continue; // manifest unreadable: this session's packs are unprovable, keep them
                        if (referenced.Contains(Path.GetFileName(path))) continue;
                        if (File.GetLastWriteTimeUtc(path) > cutoff) continue;

                        File.Delete(path); // orphan sweep
                    }
                    catch (IOException)
                    {
                        // Best effort: a stale pack costs disk, never correctness.
                    }
                    catch (UnauthorizedAccessException)
                    {
                    }
                }
            }
        }
        catch (IOException)
        {
            // Best effort: the directory listing failed; nothing is deleted.
        }
        catch (UnauthorizedAccessException)
        {
        }
    }

    /// <summary>True when <paramref name="blockId"/> is <paramref name="prefix"/> + 32 lowercase hex (session) + "_" + 32 lowercase hex (generation) + optional "_" + 5 digits.</summary>
    private static bool TryReadSessionId(string blockId, string prefix, out string sid)
    {
        const int Hex = 32;
        sid = string.Empty;
        if (!blockId.StartsWith(prefix, StringComparison.Ordinal)) return false;

        string rest = blockId[prefix.Length..];
        if (rest.Length < Hex + 1 + Hex || rest[Hex] != '_') return false;
        if (!IsLowerHex(rest, 0, Hex) || !IsLowerHex(rest, Hex + 1, Hex)) return false;

        string tail = rest[(Hex + 1 + Hex)..];
        if (tail.Length != 0 && !(tail.Length == 6 && tail[0] == '_' && IsDigits(tail, 1, 5))) return false;

        sid = rest[..Hex];
        return true;
    }

    private static bool IsLowerHex(string s, int start, int count)
    {
        for (int i = start; i < start + count; i++)
        {
            char c = s[i];
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
        }
        return true;
    }

    private static bool IsDigits(string s, int start, int count)
    {
        for (int i = start; i < start + count; i++)
            if (s[i] < '0' || s[i] > '9') return false;
        return true;
    }

    /// <returns>The pack file names this session's manifest references. An empty set when there is no manifest;
    /// <c>null</c> when a manifest exists but cannot be read, which means nothing about its packs can be proven.</returns>
    private static HashSet<string>? LoadReferencedPackNames(string directory, string sid)
    {
        string manifestPath = Path.Combine(directory, $"{sid}.manifest");
        if (!File.Exists(manifestPath)) return new HashSet<string>(StringComparer.Ordinal);

        try
        {
            var manifest = FileSessionManifest.Load(manifestPath);
            var names = new HashSet<string>(StringComparer.Ordinal);
            foreach (var block in manifest.Blocks)
                names.Add(Path.GetFileName(SegmentPackStore.GetPackPath(directory, block.BlockId)));
            return names;
        }
        catch (Exception)
        {
            // Corrupt, truncated, unreadable, or replaced while being read: prove nothing, delete nothing.
            return null;
        }
    }
}
