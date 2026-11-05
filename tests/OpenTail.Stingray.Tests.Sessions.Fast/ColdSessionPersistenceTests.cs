namespace OpenTail.Stingray.Tests.Sessions.Fast;


public sealed class ColdSessionPersistenceTests
{
    private const int NonStopToken = 7;
    private const int Eos = 31;

    private static SessionRequestDigest Digest(string val) => SessionRequestDigest.FromCanonicalValue(val);

    private sealed class Tokenizer : ITokenizer
    {
        public int VocabSize => 64;
        public int BosTokenId => 0;
        public int EosTokenId => Eos;
        public int UnknownTokenId => 0;
        public int PadTokenId => Eos;
        public bool AddBosToken => false;
        public IReadOnlyCollection<int> EogTokenIds => [Eos];
        public IReadOnlyList<int> Encode(string text) => [1, 2];
        public string Decode(IEnumerable<int> tokens) => "tok";
        public byte[] DecodeBytes(int token) => [(byte)('a' + (token % 26))];
    }

    private sealed class ProductionPagedKvForwardPass : IBatchedForwardPass
    {
        private static readonly float[] NonStopLogits = CreateLogits(NonStopToken);
        public bool SnapKvEnabled => false;
        public long KvBytesPerToken => 10;
        public int MaxSeqLen => 64;
        public bool PrefillDequantCacheActive => false;

        public ISequenceKvCache CreateCache() => new PagedKvCache(numLayers: 1, numKvHeads: 1, headDim: 4);

        public ReadOnlySpan<float> PrefillWithCache(IReadOnlyList<int> tokens, ISequenceKvCache cache, int startPos = 0)
        {
            var paged = Assert.IsType<PagedKvCache>(cache);
            float[] key = [1f, 2f, 3f, 4f];
            float[] val = [10f, 20f, 30f, 40f];
            for (int i = 0; i < tokens.Count; i++)
            {
                paged.Append(0, key, val);
                paged.IncrementPosition();
            }
            return NonStopLogits;
        }

        public float[]?[] PrefillPackedMulti(ReadOnlyMemory<int>[] chunks, int[] startPos, ISequenceKvCache[] caches, bool[] wantLogits) =>
            throw new NotSupportedException();

        public float[][] BatchForwardMulti(int[] tokens, int[] positions, ISequenceKvCache[] caches)
        {
            float[] key = [1f, 2f, 3f, 4f];
            float[] val = [10f, 20f, 30f, 40f];
            for (int i = 0; i < caches.Length; i++)
            {
                var paged = Assert.IsType<PagedKvCache>(caches[i]);
                paged.Append(0, key, val);
                paged.IncrementPosition();
            }
            return Enumerable.Repeat(NonStopLogits, tokens.Length).ToArray();
        }

        private static float[] CreateLogits(int winner)
        {
            var logits = new float[64];
            logits[winner] = 10f;
            return logits;
        }
    }

    [Fact]
    public unsafe void PagedKvCache_ExportAndImport_PreservesFloat32PagesAndDimensions()
    {
        using var cache1 = new PagedKvCache(numLayers: 2, numKvHeads: 2, headDim: 4, bf16Store: false);

        float[] key = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];
        float[] val = [10f, 20f, 30f, 40f, 50f, 60f, 70f, 80f];

        // Fill 20 positions to cross the PageSize=16 block boundary
        for (int p = 0; p < 20; p++)
        {
            cache1.Append(0, key, val);
            cache1.Append(1, key, val);
            cache1.IncrementPosition();
        }

        Assert.Equal(20, cache1.Length);
        Assert.Equal(20, cache1.LogicalLength);

        byte[] exportedBytes = cache1.ExportKvState();
        Assert.NotNull(exportedBytes);
        Assert.True(exportedBytes.Length > 36);

        using var cache2 = new PagedKvCache(numLayers: 2, numKvHeads: 2, headDim: 4, bf16Store: false);
        cache2.ImportKvState(exportedBytes);

        Assert.Equal(20, cache2.Length);
        Assert.Equal(20, cache2.LogicalLength);

        // Verify page float values match exactly across layers and positions
        for (int pos = 0; pos < 20; pos++)
        {
            for (int l = 0; l < 2; l++)
            {
                float* kPtr = cache2.KeyAt(l, pos);
                for (int d = 0; d < 8; d++)
                {
                    Assert.Equal(key[d], kPtr[d]);
                }
            }
        }
    }

    [Fact]
    public unsafe void PagedKvCache_ExportAndImport_PreservesBf16StorePages()
    {
        using var cache1 = new PagedKvCache(numLayers: 2, numKvHeads: 2, headDim: 4, bf16Store: true);

        float[] key = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];
        float[] val = [10f, 20f, 30f, 40f, 50f, 60f, 70f, 80f];

        for (int p = 0; p < 20; p++)
        {
            cache1.Append(0, key, val);
            cache1.Append(1, key, val);
            cache1.IncrementPosition();
        }

        byte[] exportedBytes = cache1.ExportKvState();
        Assert.NotNull(exportedBytes);

        using var cache2 = new PagedKvCache(numLayers: 2, numKvHeads: 2, headDim: 4, bf16Store: true);
        cache2.ImportKvState(exportedBytes);

        Assert.Equal(20, cache2.Length);
        Assert.Equal(20, cache2.LogicalLength);

        // Verify BF16 16-bit key pointer values match
        for (int pos = 0; pos < 20; pos++)
        {
            ushort* kPtr1 = cache1.Bf16KeyAt(0, pos);
            ushort* kPtr2 = cache2.Bf16KeyAt(0, pos);
            for (int d = 0; d < 8; d++)
            {
                Assert.Equal(kPtr1[d], kPtr2[d]);
            }
        }
    }

    [Fact]
    public void FileSessionManifest_SaveAndLoad_PreservesMetadata()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"opentail_manifest_test_{Guid.NewGuid():N}");
        string manifestPath = Path.Combine(tempDir, "test.manifest");
        try
        {
            var id = SessionId.New();
            var rev = new SessionRevision(5);
            var abi = new SessionStateABI("model-test", 10, 2048, ModelFormat.SafeTensors);
            var hash = StatePayloadHash.Compute(System.Text.Encoding.UTF8.GetBytes("payload"));
            var blockChecksum = ContentDigest.FromCanonicalBytes(System.Text.Encoding.UTF8.GetBytes("block"));
            var blockRef = new SegmentBlockRef("blk_1", 0, 10, 100, 100, blockChecksum);

            var manifest = new SessionManifestEnvelope(id, rev, abi, "compat-key", hash, ImmutableArray.Create(blockRef));

            FileSessionManifest.SaveAtomic(manifestPath, manifest);
            var loaded = FileSessionManifest.Load(manifestPath);

            Assert.Equal(manifest.SessionId, loaded.SessionId);
            Assert.Equal(manifest.Revision, loaded.Revision);
            Assert.Equal(manifest.Abi.ModelFingerprint, loaded.Abi.ModelFingerprint);
            Assert.Equal(ModelFormat.SafeTensors, loaded.Abi.ModelFormat);
            Assert.Equal(manifest.CompatibilityKey, loaded.CompatibilityKey);
            Assert.Single(loaded.Blocks);
            Assert.Equal("blk_1", loaded.Blocks[0].BlockId);
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void SegmentPackStore_SaveAndLoadBlock_VerifiesChecksum()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"opentail_pack_test_{Guid.NewGuid():N}");
        try
        {
            byte[] payload = System.Text.Encoding.UTF8.GetBytes("hello-segment-pack-data");
            var blockRef = SegmentPackStore.SaveBlock(tempDir, "blk_alpha", 0, 5, payload);

            Assert.Equal("blk_alpha", blockRef.BlockId);
            Assert.Equal(5, blockRef.TokenCount);

            string packPath = SegmentPackStore.GetPackPath(tempDir, "blk_alpha");
            byte[] loadedPayload = SegmentPackStore.LoadBlock(packPath);

            Assert.Equal(payload, loadedPayload);
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void CowSessionSnapshot_BranchAndMerge_PreservesParentBlocks()
    {
        var parentId = SessionId.New();
        var childId = SessionId.New();
        var rev = new SessionRevision(1);

        var block1 = new SegmentBlockRef("parent_blk_1", 0, 10, 100, 100, ContentDigest.FromCanonicalBytes([1]));
        var parentBlocks = ImmutableArray.Create(block1);

        var branchInfo = CowSessionSnapshot.CreateBranch(parentId, childId, rev, parentBlocks);
        Assert.Equal(parentId, branchInfo.ParentSessionId);
        Assert.Equal(childId, branchInfo.ChildSessionId);

        var childBlock = new SegmentBlockRef("child_blk_2", 10, 5, 50, 50, ContentDigest.FromCanonicalBytes([2]));
        var merged = CowSessionSnapshot.MergeBranch(branchInfo, ImmutableArray.Create(childBlock));

        Assert.Equal(2, merged.Length);
        Assert.Equal("parent_blk_1", merged[0].BlockId);
        Assert.Equal("child_blk_2", merged[1].BlockId);
    }

    [Fact]
    public async Task ColdSession_WithPagedKvCache_EvictsToDiskAndRestoresExactKv()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"opentail_cold_test_{Guid.NewGuid():N}");
        try
        {
            var fwd = new ProductionPagedKvForwardPass();
            using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
            var hotRuntime = new HotSessionRuntime(engine, new Tokenizer());
            var coldRuntime = new ColdSessionRuntime(hotRuntime, engine, tempDir, ModelFormat.SafeTensors);

            var address = new SessionAddress("tenant1", "coder", "thread1", "test");

            int expectedAcceptedPositions;
            SessionRevision expectedRevision;
            byte[]? expectedKvBytes;
            SessionManifestEnvelope? manifest;

            using (var session = coldRuntime.Create(address))
            {
                var sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 2 };
                await session.RunTurnAsync("hello", sampling, SessionRevision.Initial, SessionOperationId.New(), Digest("hello"));

                expectedAcceptedPositions = session.Cursor.AcceptedPositionCount;
                expectedRevision = hotRuntime.GetSessionSnapshot(session.SessionId).CommittedRevision;
                expectedKvBytes = session.ExportKvBytes();
                Assert.NotNull(expectedKvBytes);
                Assert.True(expectedKvBytes!.Length > 0, "The production cache must export real pages.");

                // Evict session to disk
                manifest = coldRuntime.EvictToDisk(session, "test");
                Assert.Equal(ModelFormat.SafeTensors, manifest.Abi.ModelFormat);
            }

            // Verify hot session is no longer active in RAM
            Assert.Throws<SessionNotFoundException>(() => hotRuntime.Open(address.ToSessionId()));

            // KV travels OUT OF BAND, not inside the cursor envelope: the envelope cannot carry it
            // (SessionCursorCodecLimits.MaxPayloadBytes caps it at 4 MB, one cache block is ~6 MB).
            // Block 0 is the cursor envelope. A bounded completed-operation ledger may follow;
            // KV blocks retain their own prefix and are reassembled in manifest order.
            string sid = address.ToSessionId().Value.ToString("N");
            Assert.True(manifest!.Blocks.Length >= 2, "Expected a cursor block plus at least one KV block.");
            Assert.StartsWith("cur_", manifest.Blocks[0].BlockId);
            var kvBlocks = manifest.Blocks.Where(block => block.BlockId.StartsWith("kv_", StringComparison.Ordinal)).ToArray();
            Assert.NotEmpty(kvBlocks);

            byte[] cursorPack = SegmentPackStore.LoadBlock(
                SegmentPackStore.GetPackPath(tempDir, manifest.Blocks[0].BlockId));
            var envelope = SessionStateCodec.Decode(cursorPack);
            Assert.Empty(envelope.OptionalSections);

            // The first KV pack begins the PKVC stream.
            byte[] firstKvPack = SegmentPackStore.LoadBlock(
                SegmentPackStore.GetPackPath(tempDir, kvBlocks[0].BlockId));
            Assert.Equal(0x504B5643u, BitConverter.ToUInt32(firstKvPack.AsSpan(0, 4)));

            // Restore cold session from disk manifest & segment packs
            using var restoredSession = coldRuntime.Open(address.ToSessionId());
            Assert.NotNull(restoredSession);
            Assert.Equal(address.ToSessionId(), restoredSession.SessionId);

            Assert.Equal(expectedAcceptedPositions, restoredSession.Cursor.AcceptedPositionCount);
            Assert.Equal(expectedRevision, hotRuntime.GetSessionSnapshot(restoredSession.SessionId).CommittedRevision);

            // The whole point of the test: the KV stream survives chunking across packs and
            // reassembly byte for byte.
            Assert.Equal(expectedKvBytes, restoredSession.ExportKvBytes());

            // Execute Turn 2 on restored cold session against restored PagedKvCache
            var turn2Sampling = new SamplingParams { Temperature = 0f, MaxNewTokens = 2 };
            var turn2Result = await restoredSession.RunTurnAsync(
                "world",
                turn2Sampling,
                hotRuntime.GetSessionSnapshot(restoredSession.SessionId).CommittedRevision,
                SessionOperationId.New(),
                Digest("world"));

            Assert.NotNull(turn2Result);
            Assert.NotEmpty(turn2Result.Chunks);
            Assert.True(restoredSession.Cursor.AcceptedPositionCount > expectedAcceptedPositions);

            restoredSession.Dispose();
            Assert.True(coldRuntime.Delete(address.ToSessionId()));
            Assert.False(File.Exists(Path.Combine(tempDir, $"{sid}.manifest")));
            Assert.All(manifest.Blocks, block => Assert.False(File.Exists(
                SegmentPackStore.GetPackPath(tempDir, block.BlockId))));
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public async Task ColdSession_Open_RejectsACorruptPersistedKvPack()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"opentail_cold_corrupt_{Guid.NewGuid():N}");
        try
        {
            var fwd = new ProductionPagedKvForwardPass();
            using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
            var hot = new HotSessionRuntime(engine, new Tokenizer());
            var cold = new ColdSessionRuntime(hot, engine, tempDir, ModelFormat.SafeTensors);
            var address = new SessionAddress("tenant", "role", "thread", "test");

            using (var session = cold.Create(address))
            {
                await session.RunTurnAsync("hello", new SamplingParams { Temperature = 0f, MaxNewTokens = 2 },
                    SessionRevision.Initial, SessionOperationId.New(), Digest("hello"));
                var manifest = cold.EvictToDisk(session, "test");
                var kvBlock = Assert.Single(manifest.Blocks.Where(block =>
                    block.BlockId.StartsWith("kv_", StringComparison.Ordinal)));
                string packPath = SegmentPackStore.GetPackPath(tempDir, kvBlock.BlockId);
                byte[] bytes = File.ReadAllBytes(packPath);
                bytes[^1] ^= 0x80; // damage the framed payload/checksum after a valid atomic write.
                File.WriteAllBytes(packPath, bytes);
            }

            Assert.Throws<SessionJournalFormatException>(() => cold.Open(address.ToSessionId()));
            Assert.Throws<SessionJournalFormatException>(() => cold.OpenOrCreate(address));
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }

    /// <summary>
    /// Completed operation output is part of the restart idempotency contract. The ledger is a
    /// separately checksummed segment pack, so corruption must fail closed rather than restoring
    /// the KV state and then silently forgetting a response that a caller may retry.
    /// </summary>
    [Fact]
    public async Task ColdSession_Open_RejectsACorruptPersistedOperationLedger()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"opentail_cold_ops_corrupt_{Guid.NewGuid():N}");
        try
        {
            var fwd = new ProductionPagedKvForwardPass();
            using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
            var hot = new HotSessionRuntime(engine, new Tokenizer());
            var cold = new ColdSessionRuntime(hot, engine, tempDir, ModelFormat.SafeTensors);
            var address = new SessionAddress("tenant", "role", "thread", "test");

            using (var session = cold.Create(address))
            {
                await session.RunTurnAsync("hello", new SamplingParams { Temperature = 0f, MaxNewTokens = 2 },
                    SessionRevision.Initial, SessionOperationId.New(), Digest("hello"));
                var manifest = cold.EvictToDisk(session, "test");
                var ledgerBlock = Assert.Single(manifest.Blocks.Where(block =>
                    block.BlockId.StartsWith("ops_", StringComparison.Ordinal)));
                string packPath = SegmentPackStore.GetPackPath(tempDir, ledgerBlock.BlockId);
                byte[] bytes = File.ReadAllBytes(packPath);
                bytes[^1] ^= 0x80;
                File.WriteAllBytes(packPath, bytes);
            }

            Assert.Throws<SessionJournalFormatException>(() => cold.Open(address.ToSessionId()));
            Assert.Throws<SessionJournalFormatException>(() => cold.OpenOrCreate(address));
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void FileSessionManifest_RejectsInvalidMagic()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"opentail_manifest_magic_{Guid.NewGuid():N}");
        string manifestPath = Path.Combine(tempDir, "bad.manifest");
        try
        {
            Directory.CreateDirectory(tempDir);
            byte[] corruptData = [0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
            File.WriteAllBytes(manifestPath, corruptData);

            Assert.Throws<SessionJournalFormatException>(() => FileSessionManifest.Load(manifestPath));
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }

    /// <summary>
    /// A cache exported as BF16 must import into an F32 cache and vice versa.
    /// </summary>
    /// <remarks>
    /// Under <c>STINGRAY_KV_STORE=auto</c> a cache narrows to BF16 once it passes the crossover,
    /// so every session long enough to be worth evicting exports <c>bf16=true</c> — while the cache
    /// restoring it starts F32 and has not yet crossed its own threshold. Refusing that mismatch
    /// failed on exactly the sessions cold storage exists for.
    /// </remarks>
    [Fact]
    public unsafe void PagedKvCache_ImportAcrossElementFormats_ConvertsInsteadOfThrowing()
    {
        using var bf16Cache = new PagedKvCache(numLayers: 1, numKvHeads: 2, headDim: 4, bf16Store: true);

        // Values exactly representable in BF16 (8 mantissa bits), so the round trip is lossless
        // and any difference is a conversion bug rather than expected precision loss.
        float[] key = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];
        float[] val = [10f, 20f, 30f, 40f, 50f, 60f, 70f, 80f];
        for (int p = 0; p < 20; p++)
        {
            bf16Cache.Append(0, key, val);
            bf16Cache.IncrementPosition();
        }

        byte[] bf16Stream = bf16Cache.ExportKvState();

        // BF16 stream -> F32 cache.
        using var f32Cache = new PagedKvCache(numLayers: 1, numKvHeads: 2, headDim: 4, bf16Store: false);
        f32Cache.ImportKvState(bf16Stream);
        Assert.Equal(20, f32Cache.Length);
        for (int pos = 0; pos < 20; pos++)
        {
            float* k = f32Cache.KeyAt(0, pos);
            for (int d = 0; d < 8; d++) Assert.Equal(key[d], k[d]);
        }

        // F32 stream -> BF16 cache, back to where it started.
        byte[] f32Stream = f32Cache.ExportKvState();
        using var roundTrip = new PagedKvCache(numLayers: 1, numKvHeads: 2, headDim: 4, bf16Store: true);
        roundTrip.ImportKvState(f32Stream);
        Assert.Equal(20, roundTrip.Length);
        Assert.Equal(bf16Stream, roundTrip.ExportKvState());
    }

    /// <summary>
    /// An empty cache exports exactly the 36-byte v2 PKVC header and must round-trip.
    /// </summary>
    /// <remarks>
    /// Historical, and stated in the numbers of its own era: when the header was 35 bytes the
    /// import guard required <c>&gt;= 36</c>, one byte too many, so an empty cache's export was
    /// rejected as "buffer too small" despite being complete and valid. PKVC v2 added the
    /// per-layer-geometry flag, making the current header 36 — but the guard is deliberately still
    /// 35, because that is the size of a v1 stream and those must keep importing. Assert the exact
    /// length here rather than a lower bound: an off-by-one in either direction has already
    /// shipped once, and only an equality check catches the next one.
    /// </remarks>
    [Fact]
    public void PagedKvCache_ExportAndImport_EmptyCacheRoundTrips()
    {
        using var empty = new PagedKvCache(numLayers: 2, numKvHeads: 2, headDim: 4);
        byte[] stream = empty.ExportKvState();
        Assert.Equal(36, stream.Length);

        using var target = new PagedKvCache(numLayers: 2, numKvHeads: 2, headDim: 4);
        target.ImportKvState(stream);
        Assert.Equal(0, target.Length);
    }

    /// <summary>
    /// A rewound cache must not export blocks past its length.
    /// </summary>
    /// <remarks><c>TruncateTo</c> is soft — it keeps pages allocated — so exporting
    /// <c>_allocatedBlocks</c> shipped dead pages. At 256 KB per layer per block that is the
    /// difference between fitting a storage bound and not.</remarks>
    [Fact]
    public void PagedKvCache_Export_ExcludesBlocksPastLength()
    {
        using var cache = new PagedKvCache(numLayers: 1, numKvHeads: 2, headDim: 4);
        float[] v = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f];
        for (int p = 0; p < 48; p++) { cache.Append(0, v, v); cache.IncrementPosition(); }
        int fullLength = cache.ExportKvState().Length;

        cache.TruncateTo(8);   // one live block; pages for the other two stay allocated
        int truncatedLength = cache.ExportKvState().Length;

        Assert.True(truncatedLength < fullLength,
            $"Rewound export should shed dead blocks, but was {truncatedLength} vs {fullLength}.");

        using var target = new PagedKvCache(numLayers: 1, numKvHeads: 2, headDim: 4);
        target.ImportKvState(cache.ExportKvState());
        Assert.Equal(8, target.Length);
    }

    /// <summary>
    /// KV larger than one segment pack must survive chunking and reassembly.
    /// </summary>
    /// <remarks>
    /// KV used to ride inside the cursor envelope, which <c>SessionCursorCodecLimits</c> caps at
    /// 4 MB — about 11 tokens of a production-sized cache. It now travels out of band, split across
    /// packs. This uses a cache wide enough that the export clears the old envelope ceiling.
    /// </remarks>
    [Fact]
    public void ColdStorage_KvLargerThanEnvelopeLimit_ChunksAndReassembles()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"opentail_kvchunk_{Guid.NewGuid():N}");
        try
        {
            // kvDim 512 -> 16 positions x 512 x 2 regions x 4 B = 64 KB per page per layer.
            // 192 positions = 12 blocks; 12 x 8 layers x 64 KB = 6 MB, past the 4 MB envelope cap.
            const int layers = 8;
            using var cache = new PagedKvCache(numLayers: layers, numKvHeads: 64, headDim: 8);
            float[] vec = new float[512];
            for (int i = 0; i < vec.Length; i++) vec[i] = i * 0.5f;
            for (int p = 0; p < 192; p++)
            {
                for (int l = 0; l < layers; l++) cache.Append(l, vec, vec);
                cache.IncrementPosition();
            }

            byte[] kv = cache.ExportKvState();
            Assert.True(kv.Length > 4 * 1024 * 1024,
                $"Test needs a payload past the 4 MB envelope cap; got {kv.Length} bytes.");

            // Store it the way ColdSessionRuntime does: split across packs, reassemble in order.
            const int chunkBytes = 1 * 1024 * 1024;
            int chunks = (kv.Length + chunkBytes - 1) / chunkBytes;
            var refs = new List<SegmentBlockRef>();
            for (int i = 0; i < chunks; i++)
            {
                int off = i * chunkBytes;
                int len = Math.Min(chunkBytes, kv.Length - off);
                refs.Add(SegmentPackStore.SaveBlock(tempDir, $"kv_{i:D5}", 0, 0, kv.AsSpan(off, len)));
            }
            Assert.True(chunks > 1, "Expected the payload to span multiple packs.");

            byte[] reassembled = new byte[kv.Length];
            int written = 0;
            foreach (var r in refs)
            {
                byte[] part = SegmentPackStore.LoadBlock(SegmentPackStore.GetPackPath(tempDir, r.BlockId));
                Buffer.BlockCopy(part, 0, reassembled, written, part.Length);
                written += part.Length;
            }
            Assert.Equal(kv.Length, written);
            Assert.Equal(kv, reassembled);

            using var restored = new PagedKvCache(numLayers: layers, numKvHeads: 64, headDim: 8);
            restored.ImportKvState(reassembled);
            Assert.Equal(192, restored.Length);
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }

    /// <summary>
    /// Each eviction writes a new pack generation before atomically publishing its manifest. Once
    /// published, earlier generations and unrelated stale packs must be reclaimed; otherwise
    /// durable sessions slowly leak disk space.
    /// </summary>
    [Fact]
    public void EvictToDisk_ReEviction_ReclaimsPacksTheNewManifestNoLongerReferences()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"opentail_prune_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        try
        {
            var fwd = new ProductionPagedKvForwardPass();
            using var engine = new ContinuousBatchingEngine(fwd, new Tokenizer(), "test", maxBatchSize: 1);
            var hotRuntime = new HotSessionRuntime(engine, new Tokenizer());
            var coldRuntime = new ColdSessionRuntime(hotRuntime, engine, tempDir, ModelFormat.SafeTensors);

            var session = coldRuntime.Create();
            string sid = session.SessionId.Value.ToString("N");
            var manifest = coldRuntime.EvictToDisk(session, "test");

            // Model a stale pack under this session's prefix that no manifest lists.
            string stale = SegmentPackStore.GetPackPath(tempDir, $"kv_{sid}_99999");
            File.WriteAllBytes(stale, [0xDE, 0xAD]);
            Assert.True(File.Exists(stale));

            var restored = coldRuntime.Open(session.SessionId);
            var nextManifest = coldRuntime.EvictToDisk(restored, "test");

            Assert.False(File.Exists(stale), "an unreferenced pack survived re-eviction.");
            Assert.NotEqual(manifest.Blocks[0].BlockId, nextManifest.Blocks[0].BlockId);
            Assert.All(manifest.Blocks, block => Assert.False(File.Exists(
                SegmentPackStore.GetPackPath(tempDir, block.BlockId))));

            // And the packs the manifest DOES name are untouched — pruning must not be overzealous.
            foreach (var block in nextManifest.Blocks)
                Assert.True(File.Exists(SegmentPackStore.GetPackPath(tempDir, block.BlockId)),
                    $"pruning removed referenced block '{block.BlockId}'.");
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }

    /// <summary>
    /// <see cref="HotSession.IsInUse"/> clears when the engine finishes, while the turn still publishes its cursor and commits its revision and
    /// operation record. Eviction used to pass its check there and export a cursor that did not match the revision and ledger it wrote beside it.
    /// The seam parks the turn after the cursor is published and before the store commit; eviction must be refused and publish nothing.
    /// </summary>
    [Fact]
    public async Task EvictToDisk_WhileATurnIsCommitting_IsRefused_AndPublishesNothing()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"opentail_cold_commit_race_{Guid.NewGuid():N}");
        try
        {
            using var engine = new ContinuousBatchingEngine(new ProductionPagedKvForwardPass(), new Tokenizer(), "test", maxBatchSize: 1);
            var hot = new HotSessionRuntime(engine, new Tokenizer());
            var cold = new ColdSessionRuntime(hot, engine, tempDir, ModelFormat.SafeTensors);
            using var session = cold.Create(new SessionAddress("tenant", "role", "thread", "test"));

            using var atCommit = new ManualResetEventSlim();
            using var release = new ManualResetEventSlim();
            session.FaultBeforeCommitForTests = () =>
            {
                atCommit.Set();
                Assert.True(release.Wait(TimeSpan.FromSeconds(10)));
            };
            var turn = session.RunTurnAsync("hello", new SamplingParams { Temperature = 0f, MaxNewTokens = 2 },
                SessionRevision.Initial, SessionOperationId.New(), Digest("hello"));
            Assert.True(atCommit.Wait(TimeSpan.FromSeconds(10)));

            Assert.False(session.IsInUse);   // the premise: the engine is done, yet the turn has not committed
            Assert.Throws<InvalidOperationException>(() => cold.EvictToDisk(session, "test"));
            Assert.Empty(Directory.GetFiles(tempDir, "*.manifest"));
            Assert.Empty(Directory.GetFiles(tempDir, "*.pack"));

            release.Set();
            var result = await turn;
            var manifest = cold.EvictToDisk(session, "test");   // once the turn has committed, eviction works
            Assert.Equal(result.Operation.CommittedRevision, manifest.Revision);
            Assert.Single(Directory.GetFiles(tempDir, "*.manifest"));
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }

    private static string NewOrphanTestDirectory(string name)
    {
        string dir = Path.Combine(Path.GetTempPath(), $"opentail_orphan_{name}_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void DeleteDirectory(string dir)
    {
        if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
    }

    /// <summary>Creates a session with real KV pages and a completed operation, evicts it, and returns its id and pack paths.</summary>
    private static async Task<(SessionId SessionId, List<string> PackPaths)> EvictSessionWithRealPacksAsync(string dir)
    {
        using var engine = new ContinuousBatchingEngine(new ProductionPagedKvForwardPass(), new Tokenizer(), "test", maxBatchSize: 1);
        var cold = new ColdSessionRuntime(new HotSessionRuntime(engine, new Tokenizer()), engine, dir, ModelFormat.SafeTensors);
        var session = cold.Create();
        await session.RunTurnAsync("hello", new SamplingParams { Temperature = 0f, MaxNewTokens = 2 },
            SessionRevision.Initial, SessionOperationId.New(), Digest("hello"));
        var manifest = cold.EvictToDisk(session, "test");
        var packs = manifest.Blocks.Select(b => SegmentPackStore.GetPackPath(dir, b.BlockId)).ToList();
        Assert.NotEmpty(packs);
        return (session.SessionId, packs);
    }

    private static void AgeFiles(IEnumerable<string> paths) =>
        paths.ToList().ForEach(p => File.SetLastWriteTimeUtc(p, DateTime.UtcNow.AddMinutes(-10)));

    private static ColdSessionRuntime ColdRuntimeOver(string dir, out ContinuousBatchingEngine engine)
    {
        engine = new ContinuousBatchingEngine(new ProductionPagedKvForwardPass(), new Tokenizer(), "test", maxBatchSize: 1);
        return new ColdSessionRuntime(new HotSessionRuntime(engine, new Tokenizer()), engine, dir, ModelFormat.SafeTensors);
    }

    [Fact]
    public void ColdOrphanSweep_OldUnreferencedPack_IsRemovedOnNextConstruction()
    {
        string dir = NewOrphanTestDirectory("old");
        try
        {
            string orphan = SegmentPackStore.GetPackPath(dir, $"kv_{Guid.NewGuid():N}_{Guid.NewGuid():N}_00000");
            File.WriteAllBytes(orphan, [1, 2, 3]);
            AgeFiles([orphan]);

            _ = ColdRuntimeOver(dir, out var engine);
            using (engine)
                Assert.False(File.Exists(orphan), "an old unreferenced pack survived the startup sweep.");
        }
        finally { DeleteDirectory(dir); }
    }

    [Fact]
    public void ColdOrphanSweep_RecentUnreferencedPack_IsKept()
    {
        string dir = NewOrphanTestDirectory("recent");
        try
        {
            // Stands in for an EvictToDisk that has written its packs but not yet published its manifest.
            string inFlight = SegmentPackStore.GetPackPath(dir, $"cur_{Guid.NewGuid():N}_{Guid.NewGuid():N}");
            File.WriteAllBytes(inFlight, [1, 2, 3]);

            _ = ColdRuntimeOver(dir, out var engine);
            using (engine)
                Assert.True(File.Exists(inFlight), "a pack inside the grace period was deleted.");
        }
        finally { DeleteDirectory(dir); }
    }

    [Fact]
    public async Task ColdOrphanSweep_PacksNamedByAValidManifest_SurviveEvenWhenOld_AndSessionRestores()
    {
        string dir = NewOrphanTestDirectory("referenced");
        try
        {
            var (sessionId, packs) = await EvictSessionWithRealPacksAsync(dir);
            AgeFiles(packs);

            var cold = ColdRuntimeOver(dir, out var engine);
            using (engine)
            {
                foreach (string pack in packs)
                    Assert.True(File.Exists(pack), $"a pack named by a valid manifest was swept: {Path.GetFileName(pack)}");

                using var restored = cold.Open(sessionId);
                Assert.NotNull(restored);
            }
        }
        finally { DeleteDirectory(dir); }
    }

    [Fact]
    public async Task ColdOrphanSweep_CorruptManifest_KeepsThatSessionsOldPacks()
    {
        string dir = NewOrphanTestDirectory("corrupt");
        try
        {
            var (sessionId, packs) = await EvictSessionWithRealPacksAsync(dir);
            AgeFiles(packs);
            File.WriteAllBytes(Path.Combine(dir, $"{sessionId.Value:N}.manifest"), [0xDE, 0xAD, 0xBE, 0xEF]);

            _ = ColdRuntimeOver(dir, out var engine);
            using (engine)
                foreach (string pack in packs)
                    Assert.True(File.Exists(pack), $"the sweep deleted a pack whose manifest is corrupt: {Path.GetFileName(pack)}");
        }
        finally { DeleteDirectory(dir); }
    }

    [Fact]
    public async Task ColdOrphanSweep_UnrelatedFileAndAnotherSessionsPack_AreKept()
    {
        string dir = NewOrphanTestDirectory("unrelated");
        try
        {
            string notes = Path.Combine(dir, "notes.txt");
            File.WriteAllText(notes, "not a pack");
            // Matches the pack search pattern but its middle is not a session id, so it is not this runtime's file.
            string odd = SegmentPackStore.GetPackPath(dir, "kv_not-a-session_00000");
            File.WriteAllBytes(odd, [9, 9]);

            var (_, otherPacks) = await EvictSessionWithRealPacksAsync(dir);
            AgeFiles(otherPacks);
            AgeFiles([notes, odd]);

            _ = ColdRuntimeOver(dir, out var engine);
            using (engine)
            {
                Assert.True(File.Exists(notes), "the sweep deleted an unrelated file.");
                Assert.True(File.Exists(odd), "the sweep deleted a file whose name is not a session pack.");
                foreach (string pack in otherPacks)
                    Assert.True(File.Exists(pack), $"the sweep deleted another session's referenced pack: {Path.GetFileName(pack)}");
            }
        }
        finally { DeleteDirectory(dir); }
    }

    /// <summary>
    /// Opening or creating a session is look up, then restore or create; the hot store refuses a second session with the same id. Callers
    /// arriving together (for example right after a restart) used to both miss and both try, and one failed with a duplicate. Both must now
    /// succeed and receive the same session. Many rounds of simultaneous callers, each on a fresh address, make a lost race near-certain.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ConcurrentOpenOrCreate_ForOneAddress_AllSucceed_AndShareOneSession(bool existingColdManifest)
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"opentail_cold_single_flight_{Guid.NewGuid():N}");
        try
        {
            using var engine = new ContinuousBatchingEngine(new ProductionPagedKvForwardPass(), new Tokenizer(), "test", maxBatchSize: 1);
            var hot = new HotSessionRuntime(engine, new Tokenizer());
            var cold = new ColdSessionRuntime(hot, engine, tempDir, ModelFormat.SafeTensors);

            for (int round = 0; round < 25; round++)
            {
                var address = new SessionAddress("tenant", "role", $"thread-{round}", "test");
                if (existingColdManifest)
                {
                    var seed = cold.OpenOrCreate(address);
                    await seed.RunTurnAsync("hello", new SamplingParams { Temperature = 0f, MaxNewTokens = 2 },
                        SessionRevision.Initial, SessionOperationId.New(), Digest("hello"));
                    cold.EvictToDisk(seed, "test");   // leaves a manifest and no hot session
                }

                const int callers = 16;
                using var start = new Barrier(callers);
                var opened = await Task.WhenAll(Enumerable.Range(0, callers).Select(_ => Task.Run(() =>
                {
                    start.SignalAndWait();
                    return cold.OpenOrCreate(address);
                })));

                Assert.All(opened, s => Assert.Same(opened[0], s));
            }
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, recursive: true);
        }
    }
}