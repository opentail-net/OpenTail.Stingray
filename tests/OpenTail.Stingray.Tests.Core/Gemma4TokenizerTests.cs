using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Tests.Core;

public sealed class Gemma4TokenizerTests
{
    /// <summary>
    /// Resolves ANY local Gemma-4 E4B GGUF, by family rather than by checkpoint filename.
    ///
    /// <para>This previously named <c>E:\models\gemma-4-E4B-it-Q8_0.gguf</c> — one absolute path,
    /// one exact quantisation. Everything asserted here (vocab size, BOS/EOS/UNK ids, special
    /// tokens, encode/decode round-trip) is a property of the TOKENIZER, which is identical across
    /// quantisations of the same model. Pinning the quant meant these tests skipped on a machine
    /// that had a perfectly usable E4B GGUF sitting in the repo's own <c>models/</c> directory.</para>
    /// </summary>
    private static string? FindGemma4E4BModel()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var models = Path.Combine(dir, "models");
            if (Directory.Exists(models))
            {
                foreach (var candidate in Directory.EnumerateFiles(models, "*E4B*.gguf"))
                {
                    // mmproj files share the name prefix but carry a vision projector, not a text
                    // model with a tokenizer.
                    if (!Path.GetFileName(candidate).Contains("mmproj", StringComparison.OrdinalIgnoreCase))
                        return candidate;
                }
            }
            if (Directory.GetParent(dir) is not { } parent) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static GgufTokenizer? CreateTokenizer()
    {
        var path = FindGemma4E4BModel();
        if (path is null) return null;
        using var model = GgufModel.Open(path);
        return GgufTokenizer.FromGgufModel(model);
    }

    [Fact]
    public void Gemma4_Tokenizer_LoadsFromGguf()
    {
        var tokenizer = CreateTokenizer();
        Assert.SkipUnless(tokenizer is not null, "model fixture not present in this environment");

        Assert.Equal(262_144, tokenizer.VocabSize);
        Assert.Equal(2, tokenizer.BosTokenId);
        Assert.Equal(3, tokenizer.UnknownTokenId);

        // Gemma-4 exports DISAGREE about which token is the configured EOS: some declare
        // <turn|> (106), others the literal <eos> (1). Both are legitimate, so pinning one made
        // this an assertion about a particular checkpoint rather than about Gemma 4.
        int eos = tokenizer.SpecialTokens["<eos>"];
        int turn = tokenizer.SpecialTokens["<turn|>"];
        Assert.Contains(tokenizer.EosTokenId, new[] { eos, turn });

        // What generation actually depends on: BOTH end tokens are in the stop set, whichever one
        // the export happened to configure. Miss this and a model whose EOS is <eos> runs straight
        // through <turn|>, decoding the turn terminator as literal text.
        Assert.Contains(eos, tokenizer.EogTokenIds);
        Assert.Contains(turn, tokenizer.EogTokenIds);
    }

    [Fact]
    public void Gemma4_Tokenizer_RoundTripsHello()
    {
        var tokenizer = CreateTokenizer();
        Assert.SkipUnless(tokenizer is not null, "model fixture not present in this environment");

        const string text = "Hello world";
        var ids = tokenizer.Encode(text);
        Assert.NotEmpty(ids);

        var decoded = tokenizer.Decode(ids);
        Assert.Equal(text, decoded.TrimStart());
    }

    [Fact]
    public void Gemma4_Tokenizer_HandlesSpecialTokens()
    {
        var tokenizer = CreateTokenizer();
        Assert.SkipUnless(tokenizer is not null, "model fixture not present in this environment");

        // In Gemma 4, only token-type-3 control tokens are special; the metadata EOS
        // for this model is <turn|> at id 106 (literal <eos> at id 1 is a normal token).
        // Verify control tokens like <bos> and <turn|> survive as single IDs through encode.
        Assert.True(tokenizer.SpecialTokens.ContainsKey("<bos>"));
        Assert.True(tokenizer.SpecialTokens.ContainsKey("<turn|>"));

        var ids = tokenizer.Encode("<bos>Hello<turn|>");
        Assert.Contains(tokenizer.BosTokenId, ids);
        // Assert the id of the token actually written, not EosTokenId — those coincide only on
        // exports that configure <turn|> as EOS.
        Assert.Contains(tokenizer.SpecialTokens["<turn|>"], ids);
    }

    [Fact]
    public void Gemma4_DecodeBytes_RoundTripsAscii()
    {
        var tokenizer = CreateTokenizer();
        Assert.SkipUnless(tokenizer is not null, "model fixture not present in this environment");

        const string text = "Hello, world!";
        var ids = tokenizer.Encode(text);
        var bytes = new List<byte>();
        foreach (var id in ids)
            bytes.AddRange(tokenizer.DecodeBytes(id));

        var roundTripped = System.Text.Encoding.UTF8.GetString(bytes.ToArray());
        Assert.Equal(text, roundTripped.TrimStart());
    }
}
