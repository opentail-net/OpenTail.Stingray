using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Tests.Core;

public sealed class BertGgufVocabTests
{
    // llama.cpp's converter stores BERT WordPiece as "phantom space" tokens: word-start "w" -> "▁w",
    // continuation "##x" -> "x", [SPECIALS] verbatim. FromGgufVocab must undo that or every word becomes [UNK].
    private static readonly string[] Vocab =
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "▁hello", "▁wor", "ld", "▁[", "▁]"];

    [Fact]
    public void FromGgufVocab_RestoresWordPieceContinuations()
    {
        var tok = BertWordPieceTokenizer.FromGgufVocab(Vocab);
        Assert.Equal([4, 5, 6], tok.EncodeIds("Hello world"));   // lower-cased; "wor" + "##ld"
        Assert.Equal(2, tok.ClsTokenId);
        Assert.Equal(3, tok.SepTokenId);
        Assert.Equal(1, tok.UnkTokenId);
    }

    [Fact]
    public void FromGgufVocab_KeepsBracketPunctuationDistinctFromSpecialTokens()
    {
        var tok = BertWordPieceTokenizer.FromGgufVocab(Vocab);
        Assert.Equal(7, tok.TokenToId("["));   // "▁[" is the real "[" punctuation token, not a special
        Assert.Equal(2, tok.TokenToId("[CLS]"));
    }

    [Fact]
    public void EncoderTokenizer_FromWordPiece_AddsClsAndSep()
    {
        var enc = EncoderTokenizer.FromWordPiece(BertWordPieceTokenizer.FromGgufVocab(Vocab));
        Assert.Equal([2, 4, 5, 6, 3], enc.Encode("hello world").Ids);
        Assert.Equal([2, 4, 3, 5, 6, 3], enc.EncodePair("hello", "world").Ids);
        Assert.Equal([0, 0, 0, 1, 1, 1], enc.EncodePair("hello", "world").TypeIds);
    }
}
