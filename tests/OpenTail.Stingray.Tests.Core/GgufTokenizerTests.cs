
namespace OpenTail.Stingray.Tests.Core;

public sealed class GgufTokenizerTests
{
    private static string? FindModelPath()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, "models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf");
            if (File.Exists(candidate)) return candidate;
            var parent = Directory.GetParent(dir);
            if (parent == null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static GgufTokenizer? CreateTokenizer()
    {
        var path = FindModelPath();
        if (path is null) return null;
        using var model = GgufModel.Open(path);
        return GgufTokenizer.FromGgufModel(model);
    }

    [Fact]
    public void FromGgufModel_LoadsSuccessfully()
    {
        var tokenizer = CreateTokenizer();
        Assert.SkipUnless(tokenizer is not null, "model fixture not present in this environment");

        Assert.Equal(49152, tokenizer.VocabSize);
        Assert.Equal(1, tokenizer.BosTokenId);
        Assert.Equal(2, tokenizer.EosTokenId);
        Assert.Equal(0, tokenizer.UnknownTokenId);
        Assert.False(tokenizer.AddBosToken);
    }

    [Fact]
    public void Encode_SimpleText_ReturnsTokenIds()
    {
        var tokenizer = CreateTokenizer();
        Assert.SkipUnless(tokenizer is not null, "model fixture not present in this environment");

        var ids = tokenizer.Encode("Hello");
        Assert.NotEmpty(ids);
        Assert.True(ids.All(id => id >= 0 && id < tokenizer.VocabSize));
    }

    [Fact]
    public void Encode_IndentedCode_StaysWithinVocab()
    {
        // Issue #267: CodeGenTokenizer injects model-independent consecutive-whitespace tokens
        // at ids beyond this GGUF's 49152-row embedding (e.g. an 8-space run → id 50280). Feeding
        // one to the GPU embedding gather reads out of bounds and aborts the CUDA context (error
        // 700). The tokenizer must decompose such tokens into in-vocab byte tokens so every id is
        // addressable in the embedding table.
        var tokenizer = CreateTokenizer();
        Assert.SkipUnless(tokenizer is not null, "model fixture not present in this environment");

        foreach (var text in new[]
                 {
                     "        private const int PageSize = 16;", // the original repro (8-space indent)
                     "    if (x) {\n        return;\n    }",      // 4- and 8-space runs + newlines/tabs
                     new string(' ', 8) + "x",
                 })
        {
            var ids = tokenizer.Encode(text);
            Assert.NotEmpty(ids);
            Assert.All(ids, id => Assert.InRange(id, 0, tokenizer.VocabSize - 1));
        }
    }

    [Fact]
    public void Encode_IndentedCode_RoundTripsThroughDecode()
    {
        // Semantic guard for the #267 remap: the decomposed in-vocab tokens must still decode
        // back to the original indented text. A remap that produced wrong-but-in-vocab ids would
        // pass the in-range check but fail here.
        var tokenizer = CreateTokenizer();
        Assert.SkipUnless(tokenizer is not null, "model fixture not present in this environment");

        var text = "    if (x) {\n        return 0;\n    }";
        var ids = tokenizer.Encode(text);
        Assert.All(ids, id => Assert.InRange(id, 0, tokenizer.VocabSize - 1));
        Assert.Equal(text, tokenizer.Decode(ids));
    }

    [Fact]
    public void Encode_MultiSpaceRun_DecomposesToInVocabSpaceTokens()
    {
        // Issue #267: a run of spaces must survive encoding — not dropped, and not emitted as an
        // id outside the embedding table.
        //
        // What this test must NOT assert is that more spaces yield more tokens. It used to, and
        // that was wrong: it described the CodeGenTokenizer path (which decomposed the 2–8-space
        // tokens at ids 50280–50286 into repeated single-space tokens) rather than SmolLM2 itself.
        // With the model's declared `smollm` pre-tokenizer, llama.cpp encodes a whole run as one
        // token — 4 spaces + "X" is [333, 2273] and 8 spaces + "X" is [415, 2273], both length 2.
        // Exact-ID parity for those two cases lives in PreTokenizerParityTests, where the reference
        // values are recorded; this test keeps the in-range and preservation guarantees.
        var tokenizer = CreateTokenizer();
        Assert.SkipUnless(tokenizer is not null, "model fixture not present in this environment");

        foreach (int spaces in (int[])[2, 4, 8])
        {
            var text = new string(' ', spaces) + "X";
            var ids = tokenizer.Encode(text);

            Assert.NotEmpty(ids);
            Assert.All(ids, id => Assert.InRange(id, 0, tokenizer.VocabSize - 1));
            // The whitespace is preserved rather than silently collapsed or dropped.
            Assert.Equal(text, tokenizer.Decode(ids));
        }
    }

    [Fact]
    public void Decode_RoundTrips_SimpleText()
    {
        var tokenizer = CreateTokenizer();
        Assert.SkipUnless(tokenizer is not null, "model fixture not present in this environment");

        var text = "Hello, world!";
        var ids = tokenizer.Encode(text);
        var decoded = tokenizer.Decode(ids);

        Assert.Equal(text, decoded);
    }

    [Fact]
    public void Decode_RoundTrips_LongerText()
    {
        var tokenizer = CreateTokenizer();
        Assert.SkipUnless(tokenizer is not null, "model fixture not present in this environment");

        var text = "The quick brown fox jumps over the lazy dog.";
        var ids = tokenizer.Encode(text);
        var decoded = tokenizer.Decode(ids);

        Assert.Equal(text, decoded);
    }

    [Fact]
    public void Encode_EmptyString_ReturnsEmpty()
    {
        var tokenizer = CreateTokenizer();
        Assert.SkipUnless(tokenizer is not null, "model fixture not present in this environment");

        var ids = tokenizer.Encode("");
        Assert.Empty(ids);
    }

    [Fact]
    public void Encode_MultipleWords_ProducesMultipleTokens()
    {
        var tokenizer = CreateTokenizer();
        Assert.SkipUnless(tokenizer is not null, "model fixture not present in this environment");

        var ids = tokenizer.Encode("This is a test of the tokenizer");
        // A sentence with common words should produce several tokens
        Assert.True(ids.Count >= 3, $"Expected at least 3 tokens, got {ids.Count}");
    }

    [Fact]
    public void Decode_RoundTrips_SpecialCharacters()
    {
        var tokenizer = CreateTokenizer();
        Assert.SkipUnless(tokenizer is not null, "model fixture not present in this environment");

        var text = "x = 42; // comment\nprint(x)";
        var ids = tokenizer.Encode(text);
        var decoded = tokenizer.Decode(ids);

        Assert.Equal(text, decoded);
    }

    [Fact]
    public void Decode_RoundTrips_Unicode()
    {
        var tokenizer = CreateTokenizer();
        Assert.SkipUnless(tokenizer is not null, "model fixture not present in this environment");

        var text = "café résumé naïve";
        var ids = tokenizer.Encode(text);
        var decoded = tokenizer.Decode(ids);

        Assert.Equal(text, decoded);
    }

    [Fact]
    public void DecodeBytes_PerTokenStream_ReassemblesMultiByteUnicode()
    {
        var tokenizer = CreateTokenizer();
        Assert.SkipUnless(tokenizer is not null, "model fixture not present in this environment");

        // Multi-byte UTF-8 (3-byte CJK and curly quotes, em-dash) is the regression
        // case for issue #13: a single character is split across token boundaries.
        var text = "你好，世界 — “hello”";
        var ids = tokenizer.Encode(text);

        // Concat all per-token DecodeBytes output and verify it equals UTF-8 of original.
        var bytes = new System.Collections.Generic.List<byte>();
        foreach (var id in ids)
            bytes.AddRange(tokenizer.DecodeBytes(id));

        Assert.Equal(System.Text.Encoding.UTF8.GetBytes(text), bytes.ToArray());
    }

    [Fact]
    public void DecodeBytes_StreamedThroughUtf8Decoder_ProducesNoReplacementChars()
    {
        var tokenizer = CreateTokenizer();
        Assert.SkipUnless(tokenizer is not null, "model fixture not present in this environment");

        var text = "你好世界";
        var ids = tokenizer.Encode(text);

        var dec = new Utf8StreamDecoder();
        var sb = new System.Text.StringBuilder();
        foreach (var id in ids)
            sb.Append(dec.Append(tokenizer.DecodeBytes(id)));
        sb.Append(dec.Flush());

        var output = sb.ToString();
        Assert.Equal(text, output);
        Assert.DoesNotContain('�', output);
    }

    [Fact]
    public void DecodeBytes_AsciiToken_RoundTripsThroughUtf8()
    {
        var tokenizer = CreateTokenizer();
        Assert.SkipUnless(tokenizer is not null, "model fixture not present in this environment");

        var ids = tokenizer.Encode("Hello, world!");
        var bytes = new System.Collections.Generic.List<byte>();
        foreach (var id in ids)
            bytes.AddRange(tokenizer.DecodeBytes(id));

        Assert.Equal("Hello, world!", System.Text.Encoding.UTF8.GetString(bytes.ToArray()));
    }
}
