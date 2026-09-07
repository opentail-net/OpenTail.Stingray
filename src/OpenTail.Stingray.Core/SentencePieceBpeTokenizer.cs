namespace OpenTail.Stingray.Core;

/// <summary>
/// Real SentencePiece BPE tokenizer -- inference-time segmentation only (no training). A
/// genuinely different algorithm from <see cref="UnigramTokenizer"/> (Viterbi lattice DP): BPE
/// instead greedily merges adjacent symbol pairs in score-priority order, exactly like classic
/// byte-pair-encoding but operating on SentencePiece's normalized/metaspace-escaped text.
///
/// <para><b>Real algorithm, transcribed directly from the vendored, authoritative source</b> --
/// `examples/audio.cpp/external/sentencepiece/src/bpe_model.cc`'s <c>Model::SampleEncode(text,
/// alpha=0.0)</c> (the exact function <c>Model::Encode</c> delegates to; alpha=0 disables the
/// unrelated BPE-dropout sampling feature) and `model_interface.cc`'s <c>InitializePieces</c>/
/// <c>PieceToId</c>/<c>ByteToPiece</c>: split the normalized input into one-Unicode-scalar initial
/// symbols (or a longer match when a USER_DEFINED special-token string occurs at that position --
/// such a symbol is frozen and never merged further); seed a max-priority queue with every
/// adjacent symbol pair whose concatenated text is itself a real vocabulary piece (NORMAL,
/// USER_DEFINED, or UNUSED type -- CONTROL/UNKNOWN/BYTE pieces are looked up by exact string match
/// only, never as merge targets), keyed by that piece's real stored `score` (ties broken by the
/// LEFT-most pair, matching the reference's `left &gt; h2-&gt;left` comparator exactly); repeatedly
/// pop the highest-priority still-valid pair, merge it (extending the left symbol, splicing out
/// the right one), and re-seed any newly-adjacent pairs -- until the queue is empty. Finally, any
/// surviving symbol whose exact text is not itself a real vocabulary piece resolves to the
/// tokenizer's `&lt;unk&gt;` id and -- since MOSS-TTS-Nano's real checkpoint has
/// `trainer_spec.byte_fallback=true` (confirmed via `MossTtsSpmProtoDumpDebugTest`, not guessed)
/// -- is decomposed into one real per-byte `&lt;0xXX&gt;` BYTE-type piece per UTF-8 byte instead of a
/// single opaque UNK token (SentencePiece's real `ByteToPiece`/`PopulateSentencePieceText` byte
/// fallback path).</para>
///
/// <para><b>Known gap, not worked around</b> (same discipline as <see cref="UnigramTokenizer"/>'s
/// documented gap): the real `precompiled_charsmap` (a compiled darts double-array trie mapping
/// arbitrary input substrings to Unicode-normalized replacements -- MOSS-TTS-Nano's real
/// checkpoint carries a real 237561-byte one, confirmed via `MossTtsSpmProtoDumpDebugTest`, name
/// `nmt_nfkc`) is NOT implemented; plain Unicode NFKC is applied as a stand-in, identical to the
/// real pipeline for plain-ASCII input and possibly divergent on exotic/non-ASCII input until this
/// gap is closed.</para>
/// </summary>
public sealed class SentencePieceBpeTokenizer
{
    private readonly Dictionary<string, (int Id, float Score)> _pieces = new(StringComparer.Ordinal); // NORMAL/USER_DEFINED/UNUSED
    private readonly Dictionary<string, int> _reserved = new(StringComparer.Ordinal); // CONTROL/UNKNOWN/BYTE, exact-match only
    private readonly List<string> _userDefinedSymbols = [];
    private readonly int _unkId;
    private readonly bool _byteFallback;

    private const char MetaspaceChar = '▁';

    private SentencePieceBpeTokenizer(
        Dictionary<string, (int, float)> pieces,
        Dictionary<string, int> reserved,
        List<string> userDefinedSymbols,
        int unkId,
        bool byteFallback)
    {
        _pieces = pieces;
        _reserved = reserved;
        _userDefinedSymbols = userDefinedSymbols;
        _unkId = unkId;
        _byteFallback = byteFallback;
    }

    /// <summary>
    /// Loads a real raw SentencePiece <c>ModelProto</c> (<c>tokenizer.model</c>) via a minimal
    /// hand-rolled protobuf wire-format reader (skips unknown fields; fields aren't guaranteed
    /// ordered) -- deliberately not a full protobuf runtime, matching this project's NativeAOT/
    /// trim-friendly, dependency-light conventions. Real field numbers per
    /// `sentencepiece_model.proto`'s <c>ModelProto</c>/<c>SentencePiece</c>/<c>TrainerSpec</c>
    /// messages (not guessed -- confirmed field-by-field against the vendored .proto).
    /// </summary>
    public static SentencePieceBpeTokenizer FromModelFile(string modelPath)
        => FromModelBytes(File.ReadAllBytes(modelPath));

    public static SentencePieceBpeTokenizer FromModelBytes(byte[] data)
    {
        var pieces = new Dictionary<string, (int, float)>(StringComparer.Ordinal);
        var reserved = new Dictionary<string, int>(StringComparer.Ordinal);
        var userDefined = new List<string>();
        int unkId = -1;
        bool byteFallback = false;

        int i = 0;
        int pieceIndex = 0;
        while (i < data.Length)
        {
            long tag = ReadVarint(data, ref i);
            int fieldNo = (int)(tag >> 3);
            int wire = (int)(tag & 7);

            if (wire == 0) { ReadVarint(data, ref i); }
            else if (wire == 5) { i += 4; }
            else if (wire == 1) { i += 8; }
            else if (wire == 2)
            {
                long len = ReadVarint(data, ref i);
                int start = i;
                i += (int)len;

                if (fieldNo == 1) // repeated SentencePiece pieces
                {
                    ParsePiece(data, start, (int)len, pieceIndex, pieces, reserved, userDefined, ref unkId);
                    pieceIndex++;
                }
                else if (fieldNo == 2) // TrainerSpec
                {
                    int j = start;
                    while (j < start + len)
                    {
                        long subTag = ReadVarint(data, ref j);
                        int subField = (int)(subTag >> 3);
                        int subWire = (int)(subTag & 7);
                        if (subWire == 0)
                        {
                            long v = ReadVarint(data, ref j);
                            if (subField == 35) byteFallback = v != 0;
                        }
                        else if (subWire == 5) j += 4;
                        else if (subWire == 1) j += 8;
                        else if (subWire == 2)
                        {
                            long subLen = ReadVarint(data, ref j);
                            j += (int)subLen;
                        }
                    }
                }
                // fieldNo == 3 (NormalizerSpec) and others: skip, not needed (see class doc's known gap).
            }
            else
            {
                throw new InvalidDataException($"Unexpected protobuf wire type {wire} at offset {i}.");
            }
        }

        if (unkId < 0) throw new InvalidDataException("SentencePiece model has no UNKNOWN piece.");
        return new SentencePieceBpeTokenizer(pieces, reserved, userDefined, unkId, byteFallback);
    }

    private static void ParsePiece(
        byte[] data, int start, int len, int pieceIndex,
        Dictionary<string, (int, float)> pieces,
        Dictionary<string, int> reserved,
        List<string> userDefined,
        ref int unkId)
    {
        string text = "";
        float score = 0f;
        int type = 1; // default NORMAL

        int j = start;
        while (j < start + len)
        {
            long subTag = ReadVarint(data, ref j);
            int subField = (int)(subTag >> 3);
            int subWire = (int)(subTag & 7);
            if (subWire == 5)
            {
                if (subField == 2) score = BitConverter.ToSingle(data, j);
                j += 4;
            }
            else if (subWire == 0)
            {
                long v = ReadVarint(data, ref j);
                if (subField == 3) type = (int)v;
            }
            else if (subWire == 2)
            {
                long subLen = ReadVarint(data, ref j);
                if (subField == 1) text = Encoding.UTF8.GetString(data, j, (int)subLen);
                j += (int)subLen;
            }
        }

        if (text.Length == 0) throw new InvalidDataException("SentencePiece piece text must not be empty.");

        // Real ModelInterface::InitializePieces type routing (NORMAL=1, UNKNOWN=2, CONTROL=3,
        // USER_DEFINED=4, UNUSED=5, BYTE=6): NORMAL/USER_DEFINED/UNUSED are merge-eligible "pieces_";
        // everything else (CONTROL/UNKNOWN/BYTE) is exact-match-only "reserved_id_map_".
        bool isNormalPiece = type is 1 or 4 or 5;
        if (isNormalPiece)
        {
            pieces[text] = (pieceIndex, score);
            if (type == 4) userDefined.Add(text);
        }
        else
        {
            reserved[text] = pieceIndex;
            if (type == 2)
            {
                if (unkId >= 0) throw new InvalidDataException("SentencePiece model defines <unk> more than once.");
                unkId = pieceIndex;
            }
        }
    }

    private static long ReadVarint(byte[] data, ref int i)
    {
        long result = 0; int shift = 0;
        while (true)
        {
            byte b = data[i++];
            result |= (long)(b & 0x7F) << shift;
            if ((b & 0x80) == 0) break;
            shift += 7;
        }
        return result;
    }

    /// <summary>
    /// Real preprocessing stand-in: NFKC (see class doc's "Known gap" for the real
    /// `precompiled_charsmap`) -&gt; Metaspace (collapse whitespace runs to a single `▁`, always
    /// prepend one at the start -- SentencePiece's real "dummy prefix" behavior, same as
    /// <see cref="UnigramTokenizer"/>'s identical stage since normalization is shared across model
    /// types).
    /// </summary>
    private static string Preprocess(string text)
    {
        string normalized = text.Normalize(NormalizationForm.FormKC);
        var sb = new StringBuilder(normalized.Length + 1);
        bool prevWasSpace = false;
        foreach (char c in normalized)
        {
            if (char.IsWhiteSpace(c))
            {
                if (!prevWasSpace) sb.Append(MetaspaceChar);
                prevWasSpace = true;
            }
            else
            {
                sb.Append(c);
                prevWasSpace = false;
            }
        }
        if (sb.Length == 0 || sb[0] != MetaspaceChar) sb.Insert(0, MetaspaceChar);
        return sb.ToString();
    }

    private sealed class Symbol
    {
        public int Start;
        public int Length; // 0 once merged away (spliced out)
        public int Prev = -1;
        public int Next = -1;
        public bool Freeze;
    }

    private readonly struct SymbolPair(int left, int right, float score, int size)
    {
        public readonly int Left = left;
        public readonly int Right = right;
        public readonly float Score = score;
        public readonly int Size = size; // combined byte length at the time this pair was queued
    }

    /// <summary>Max-priority by (Score desc, Left asc) -- matches the reference's
    /// `SymbolPairComparator` exactly (leftmost pair wins on an exact score tie).</summary>
    private sealed class SymbolPairPriority(float score, int left) : IComparable<SymbolPairPriority>
    {
        public readonly float Score = score;
        public readonly int Left = left;

        // .NET PriorityQueue dequeues the SMALLEST priority first, so invert the comparison to get
        // "highest score first, then lowest Left first" (both are "we want this popped sooner").
        public int CompareTo(SymbolPairPriority? other)
        {
            if (other is null) return -1;
            int byScore = other.Score.CompareTo(Score); // higher score => "smaller" => popped first
            if (byScore != 0) return byScore;
            return Left.CompareTo(other.Left); // lower left => "smaller" => popped first
        }
    }

    /// <summary>Real BPE segmentation (no special tokens added). Matches
    /// `bpe_model.cc`'s `Model::Encode` == `SampleEncode(text, alpha=0.0)`.</summary>
    public List<int> Encode(string text)
    {
        string prepped = Preprocess(text);
        var bytes = Encoding.UTF8.GetBytes(prepped);
        int n = bytes.Length;
        if (n == 0) return [];

        var symbols = new List<Symbol>(n);
        int pos = 0;
        while (pos < n)
        {
            int matchLen = MatchUserDefinedPrefix(bytes, pos, n);
            bool frozen = matchLen > 0;
            if (matchLen == 0) matchLen = Utf8ScalarByteLength(bytes, pos, n);

            symbols.Add(new Symbol
            {
                Start = pos,
                Length = matchLen,
                Prev = symbols.Count == 0 ? -1 : symbols.Count - 1,
                Freeze = frozen,
            });
            if (symbols.Count > 1) symbols[^2].Next = symbols.Count - 1;
            pos += matchLen;
        }

        var queue = new PriorityQueue<SymbolPair, SymbolPairPriority>();

        void MaybeAddPair(int left, int right)
        {
            if (left < 0 || right < 0) return;
            if (symbols[left].Freeze || symbols[right].Freeze) return;
            int combinedLen = symbols[left].Length + symbols[right].Length;
            string piece = Encoding.UTF8.GetString(bytes, symbols[left].Start, combinedLen);
            if (!_pieces.TryGetValue(piece, out var entry)) return;
            var pair = new SymbolPair(left, right, entry.Score, combinedLen);
            queue.Enqueue(pair, new SymbolPairPriority(entry.Score, left));
        }

        for (int k = 1; k < symbols.Count; k++) MaybeAddPair(k - 1, k);

        while (queue.Count > 0)
        {
            var top = queue.Dequeue();
            var left = symbols[top.Left];
            var right = symbols[top.Right];
            if (left.Length == 0 || right.Length == 0 || left.Length + right.Length != top.Size) continue;

            left.Length += right.Length;
            left.Next = right.Next;
            if (right.Next >= 0) symbols[right.Next].Prev = top.Left;
            right.Length = 0;

            MaybeAddPair(left.Prev, top.Left);
            MaybeAddPair(top.Left, left.Next);
        }

        var ids = new List<int>();
        for (int idx = 0; idx != -1; idx = symbols[idx].Next)
        {
            var sym = symbols[idx];
            string piece = Encoding.UTF8.GetString(bytes, sym.Start, sym.Length);
            int id = LookupId(piece);
            if (id == _unkId && _byteFallback)
            {
                for (int b = sym.Start; b < sym.Start + sym.Length; b++)
                {
                    string bytePiece = $"<0x{bytes[b]:X2}>";
                    ids.Add(_reserved.TryGetValue(bytePiece, out var byteId) ? byteId : _unkId);
                }
            }
            else
            {
                ids.Add(id);
            }
        }
        return ids;
    }

    private int LookupId(string piece)
    {
        if (_reserved.TryGetValue(piece, out var reservedId)) return reservedId;
        if (_pieces.TryGetValue(piece, out var entry)) return entry.Id;
        return _unkId;
    }

    private int MatchUserDefinedPrefix(byte[] bytes, int pos, int n)
    {
        int best = 0;
        foreach (var symbol in _userDefinedSymbols)
        {
            var symBytes = Encoding.UTF8.GetBytes(symbol);
            if (symBytes.Length <= best || pos + symBytes.Length > n) continue;
            bool match = true;
            for (int k = 0; k < symBytes.Length; k++)
            {
                if (bytes[pos + k] != symBytes[k]) { match = false; break; }
            }
            if (match) best = symBytes.Length;
        }
        return best;
    }

    private static int Utf8ScalarByteLength(byte[] bytes, int start, int n)
    {
        byte b = bytes[start];
        int len = b < 0x80 ? 1 : b >> 5 == 0b110 ? 2 : b >> 4 == 0b1110 ? 3 : b >> 3 == 0b11110 ? 4 : 1;
        return Math.Min(len, n - start);
    }
}
