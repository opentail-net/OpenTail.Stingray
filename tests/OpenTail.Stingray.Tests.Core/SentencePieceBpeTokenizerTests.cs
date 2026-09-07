namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Structural conformance tests for <see cref="SentencePieceBpeTokenizer"/> against small,
/// hand-built toy `ModelProto` byte streams (real protobuf wire format, hand-encoded) -- verifies
/// the BPE merge-priority/tie-break/byte-fallback algorithm without needing a real checkpoint. See
/// <c>MossTtsSentencePieceBpeTokenizerRealWeightsTests</c> (Audio test project) for the real
/// checkpoint smoke test.
/// </summary>
public sealed class SentencePieceBpeTokenizerTests
{
    private static void WriteVarint(MemoryStream ms, long value)
    {
        ulong v = (ulong)value;
        do
        {
            byte b = (byte)(v & 0x7F);
            v >>= 7;
            if (v != 0) b |= 0x80;
            ms.WriteByte(b);
        } while (v != 0);
    }

    private static void WriteTag(MemoryStream ms, int fieldNo, int wireType) => WriteVarint(ms, ((long)fieldNo << 3) | (uint)wireType);

    private static void WriteLengthDelimited(MemoryStream ms, int fieldNo, byte[] payload)
    {
        WriteTag(ms, fieldNo, 2);
        WriteVarint(ms, payload.Length);
        ms.Write(payload);
    }

    /// <summary>Builds one `SentencePiece { piece, score, type }` sub-message.</summary>
    private static byte[] BuildPiece(string text, float score, int type)
    {
        using var ms = new MemoryStream();
        WriteLengthDelimited(ms, 1, Encoding.UTF8.GetBytes(text));
        WriteTag(ms, 2, 5); // fixed32 score
        ms.Write(BitConverter.GetBytes(score));
        if (type != 1) // NORMAL is the default, real encoders may omit it but we write explicitly for clarity except default
        {
            WriteTag(ms, 3, 0);
            WriteVarint(ms, type);
        }
        return ms.ToArray();
    }

    private static byte[] BuildTrainerSpec(bool byteFallback)
    {
        using var ms = new MemoryStream();
        WriteTag(ms, 35, 0);
        WriteVarint(ms, byteFallback ? 1 : 0);
        return ms.ToArray();
    }

    /// <summary>
    /// Builds a real-wire-format toy `ModelProto`: pieces in the given order (index = vocab id,
    /// matching the real SentencePiece convention), plus a `byte_fallback` TrainerSpec.
    /// </summary>
    private static byte[] BuildModel((string text, float score, int type)[] pieces, bool byteFallback = true)
    {
        using var ms = new MemoryStream();
        foreach (var (text, score, type) in pieces)
            WriteLengthDelimited(ms, 1, BuildPiece(text, score, type));
        WriteLengthDelimited(ms, 2, BuildTrainerSpec(byteFallback));
        return ms.ToArray();
    }

    // Piece type constants matching sentencepiece_model.proto's SentencePiece.Type enum.
    private const int Normal = 1, Unknown = 2, Control = 3, UserDefined = 4, Unused = 5, Byte = 6;

    private static (string text, float score, int type)[] BaseByteFallbackPieces()
    {
        var pieces = new List<(string, float, int)>
        {
            ("<unk>", 0f, Unknown),
            ("<s>", 0f, Control),
            ("</s>", 0f, Control),
        };
        for (int b = 0; b < 256; b++) pieces.Add(($"<0x{b:X2}>", 0f, Byte));
        return pieces.ToArray();
    }

    [Fact]
    public void Encode_MergesHighestScoringAdjacentPairFirst()
    {
        // Toy vocab: base chars 'a','b','c' plus merges "ab" (score 1.0) and "bc" (score 2.0).
        // "abc" should merge "bc" first (higher score), leaving "a"+"bc" -- NOT "ab"+"c" -- since
        // "a"+"bc" is not itself a registered piece, final ids = [id('a'), id('bc')].
        var pieces = BaseByteFallbackPieces().Concat(
        [
            ("a", 0f, Normal), ("b", 0f, Normal), ("c", 0f, Normal),
            ("ab", 1.0f, Normal), ("bc", 2.0f, Normal),
        ]).ToArray();
        // Preprocess prepends ▁ before the first symbol; "abc" -> "▁abc" -> initial symbols ▁,a,b,c.
        // Register "▁a" explicitly so the leading ▁ merges predictably, keeping this test focused
        // on the a/b/c merge-priority order rather than the dummy-prefix handling.
        pieces = pieces.Append(("▁a", 0.5f, Normal)).ToArray();
        var model = BuildModel(pieces);
        var tok = SentencePieceBpeTokenizer.FromModelBytes(model);
        var idOfA = IndexOf(pieces, "▁a");
        var idOfBc = IndexOf(pieces, "bc");

        var ids = tok.Encode("abc");
        Assert.Equal([idOfA, idOfBc], ids);
    }

    [Fact]
    public void Encode_TiesBreakToTheLeftmostPair()
    {
        // "aaa" with a single merge rule "aa" (only one score value in vocab) -- the leftmost pair
        // must merge first: "aa"+"a" (not "a"+"aa"), matching the reference's left>right tie-break.
        var pieces = BaseByteFallbackPieces().Concat(
        [
            ("▁a", 0f, Normal), ("a", 0f, Normal),
            ("▁aa", 1f, Normal), ("aa", 1f, Normal),
        ]).ToArray();
        var model = BuildModel(pieces);
        var tok = SentencePieceBpeTokenizer.FromModelBytes(model);

        var ids = tok.Encode("aaa");
        // "▁aaa" -> symbols [▁, a, a, a]. Adjacent pairs: (▁,a)->"▁a" score0, (a,a)->"aa" score1 (x2, left=1 and left=2).
        // Highest score is "aa" -- tie between left=1 and left=2 -- leftmost (left=1) wins first:
        // merges (1,2) -> "aa" spanning original a[1..3). Then re-seed (0,1)="▁"+"aa"="▁aa" (in vocab,
        // score1) and (1,3)="aa"+"a"="aaa" (not in vocab). Next pop: "▁aa" (score1, left=0) beats
        // nothing else pending -> merges into "▁aaa" (not in vocab, so this merge is impossible --
        // MaybeAddPair only enqueues pairs whose concatenation IS a vocab piece, so "▁aa"+"a" is
        // never enqueued since "▁aaa" isn't a piece). Final symbols: ["▁aa", "a"].
        int idOfLeadAa = IndexOf(pieces, "▁aa");
        int idOfA = IndexOf(pieces, "a");
        Assert.Equal([idOfLeadAa, idOfA], ids);
    }

    [Fact]
    public void Encode_UnmatchedCharacter_FallsBackToUtf8Bytes()
    {
        var pieces = BaseByteFallbackPieces();
        var model = BuildModel(pieces);
        var tok = SentencePieceBpeTokenizer.FromModelBytes(model);

        // "€" (U+20AC) is not in the toy vocab at all -- with byte_fallback=true it must decompose
        // into its 3 real UTF-8 bytes (0xE2 0x82 0xAC), each looked up as its own <0xXX> piece.
        var ids = tok.Encode("€");
        int b1 = IndexOf(pieces, "<0xE2>");
        int b2 = IndexOf(pieces, "<0x82>");
        int b3 = IndexOf(pieces, "<0xAC>");

        // The leading dummy-prefix ▁ is also unmatched here (no ▁ piece registered), so it too
        // decomposes into its UTF-8 bytes (0xE2 0x96 0x81) before the € bytes.
        int sp1 = IndexOf(pieces, "<0xE2>");
        int sp2 = IndexOf(pieces, "<0x96>");
        int sp3 = IndexOf(pieces, "<0x81>");
        Assert.Equal([sp1, sp2, sp3, b1, b2, b3], ids);
    }

    [Fact]
    public void Encode_UserDefinedSymbol_IsFrozenAndNeverMerged()
    {
        var pieces = BaseByteFallbackPieces().Concat(
        [
            ("<|special|>", 0f, UserDefined),
            ("▁", 0f, Normal),
        ]).ToArray();
        var model = BuildModel(pieces);
        var tok = SentencePieceBpeTokenizer.FromModelBytes(model);

        var ids = tok.Encode("<|special|>");
        int specialId = IndexOf(pieces, "<|special|>");
        // Preprocess prepends a dummy ▁ before the whole input (SentencePiece's real behavior --
        // the special-token text itself contains no whitespace so it stays one run).
        int leadSpaceId = IndexOf(pieces, "▁");
        Assert.Equal([leadSpaceId, specialId], ids);
    }

    private static int IndexOf((string text, float score, int type)[] pieces, string text)
    {
        int idx = Array.FindIndex(pieces, p => p.text == text);
        Assert.True(idx >= 0, $"test vocab is missing expected piece '{text}'");
        return idx;
    }
}
