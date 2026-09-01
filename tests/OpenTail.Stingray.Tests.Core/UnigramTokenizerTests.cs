
namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Golden verification for <see cref="UnigramTokenizer"/> against the real Hugging Face
/// `tokenizers` Python package (`Tokenizer.from_file(...).encode(text, add_special_tokens=False)
/// .ids`), run on Parler-TTS's real `tokenizer.json`. See `UnigramTokenizer`'s class doc for the
/// real algorithm's source derivation and the documented `precompiled_charsmap` gap (this test
/// set is plain-ASCII by design, where that gap does not apply).
/// </summary>
public sealed class UnigramTokenizerTests
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Encode_RealParlerTokenizer_MatchesGoldenIds()
    {
        string? tokenizerPath = FindRepoFile("scratch-llamacpp-ref/parler-tokenizer/tokenizer.json");
        string? goldenPath = FindRepoFile("scratch-llamacpp-ref/parler-tokenizer/unigram_golden.json");
        Assert.SkipUnless(tokenizerPath != null && goldenPath != null,
            "Parler-TTS real tokenizer.json / golden fixture not found");

        var tok = UnigramTokenizer.FromTokenizerJson(tokenizerPath!);

        using var doc = JsonDocument.Parse(File.ReadAllBytes(goldenPath!));
        foreach (var entry in doc.RootElement.EnumerateArray())
        {
            string text = entry.GetProperty("text").GetString()!;
            var expected = new List<int>();
            foreach (var idEl in entry.GetProperty("ids").EnumerateArray()) expected.Add(idEl.GetInt32());
            // Golden was captured with add_special_tokens=True, which appends T5's real EOS (id 1);
            // UnigramTokenizer.Encode is segmentation-only, so drop the trailing EOS before comparing.
            expected.RemoveAt(expected.Count - 1);

            var actual = tok.Encode(text);
            Assert.Equal(expected, actual);
        }
    }

    [Fact]
    public void Encode_RealParlerTokenizer_MatchesGoldenIds_HarderCases()
    {
        string? tokenizerPath = FindRepoFile("scratch-llamacpp-ref/parler-tokenizer/tokenizer.json");
        string? goldenPath = FindRepoFile("scratch-llamacpp-ref/parler-tokenizer/unigram_golden2.json");
        Assert.SkipUnless(tokenizerPath != null && goldenPath != null,
            "Parler-TTS real tokenizer.json / harder golden fixture not found");

        var tok = UnigramTokenizer.FromTokenizerJson(tokenizerPath!);

        using var doc = JsonDocument.Parse(File.ReadAllBytes(goldenPath!));
        foreach (var entry in doc.RootElement.EnumerateArray())
        {
            string text = entry.GetProperty("text").GetString()!;
            var expected = new List<int>();
            foreach (var idEl in entry.GetProperty("ids").EnumerateArray()) expected.Add(idEl.GetInt32());

            var actual = tok.Encode(text);
            Assert.Equal(expected, actual);
        }
    }

    /// <summary>
    /// <see cref="UnigramTokenizer.FromGgufVocab"/> is the GGUF-path factory (used when
    /// `tokenizer.ggml.model=t5`, unblocking `minicpm`/`internlm2`/`ernie4_5`/`baichuan`/
    /// `orion`/`nanbeige`). With no token-type array it falls back to the same bracket heuristic
    /// as <see cref="UnigramTokenizer.FromTokenizerJson"/>, so re-loading Parler's real
    /// tokenizer.json's own vocab/scores through the GGUF-shaped factory must produce byte-for-byte
    /// identical segmentation to the tokenizer.json path on the same golden fixture.
    /// </summary>
    [Fact]
    public void FromGgufVocab_NoTokenTypes_MatchesFromTokenizerJson_OnRealVocab()
    {
        string? tokenizerPath = FindRepoFile("scratch-llamacpp-ref/parler-tokenizer/tokenizer.json");
        string? goldenPath = FindRepoFile("scratch-llamacpp-ref/parler-tokenizer/unigram_golden2.json");
        Assert.SkipUnless(tokenizerPath != null && goldenPath != null,
            "Parler-TTS real tokenizer.json / harder golden fixture not found");

        using var doc0 = JsonDocument.Parse(File.ReadAllBytes(tokenizerPath!));
        var vocab = doc0.RootElement.GetProperty("model").GetProperty("vocab");
        int n = vocab.GetArrayLength();
        var pieces = new string[n];
        var scores = new float[n];
        int i = 0;
        foreach (var entry in vocab.EnumerateArray())
        {
            pieces[i] = entry[0].GetString() ?? "";
            scores[i] = entry[1].GetSingle();
            i++;
        }
        int unkId = doc0.RootElement.GetProperty("model").TryGetProperty("unk_id", out var u) ? u.GetInt32() : 0;

        var jsonTok = UnigramTokenizer.FromTokenizerJson(tokenizerPath!);
        var ggufTok = UnigramTokenizer.FromGgufVocab(pieces, scores, unkId, tokenTypes: null);

        using var doc = JsonDocument.Parse(File.ReadAllBytes(goldenPath!));
        foreach (var entry in doc.RootElement.EnumerateArray())
        {
            string text = entry.GetProperty("text").GetString()!;
            Assert.Equal(jsonTok.Encode(text), ggufTok.Encode(text));
        }
    }

    /// <summary>
    /// When a real GGUF `tokenizer.ggml.token_type` array is supplied, it must be used directly
    /// (llama_token_type NORMAL=1) rather than the bracket heuristic -- a piece that looks like a
    /// bracketed control token (e.g. "&lt;x&gt;") but is typed NORMAL (type=1) must still count
    /// toward the UNK-fallback minimum score, and vice versa for an unbracketed piece typed
    /// CONTROL (type=3).
    /// </summary>
    [Fact]
    public void FromGgufVocab_WithTokenTypes_UsesRealTypeArray_NotBracketHeuristic()
    {
        // 0: "▁" NORMAL, 1: "a" NORMAL but low score, 2: "<x>" NORMAL (bracketed but real-typed
        // NORMAL) with a very low score that only affects the UNK floor if actually counted,
        // 3: "unused" CONTROL (unbracketed but real-typed CONTROL) which must be EXCLUDED from the
        // NORMAL-minimum even though the bracket heuristic would have included it.
        string[] pieces = ["▁", "a", "<x>", "unused"];
        float[] scores = [-1f, -2f, -100f, -1000f];
        int[] tokenTypes = [1, 1, 1, 3];

        var tok = UnigramTokenizer.FromGgufVocab(pieces, scores, unkId: 0, tokenTypes);

        // "b" has no vocab entry at all -> forces a UNK edge whose score is
        // (min NORMAL score) - 10. Real-typed NORMAL must include piece 2's -100 (bracket
        // heuristic would have excluded it), and must NOT include piece 3's -1000 (bracket
        // heuristic would have included it, being unbracketed).
        var ids = tok.Encode("▁b");
        // "▁" segments normally (piece 0); "b" has no match, hits the UNK path with id 0 (unkId).
        Assert.Equal([0, 0], ids);
    }
}
