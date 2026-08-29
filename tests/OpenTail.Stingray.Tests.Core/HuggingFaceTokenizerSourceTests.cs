
namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Phase 1 item 3: a Hugging Face package's tokenizer assets must normalise into the same
/// <see cref="TokenizerSource"/> the GGUF path produces, and refuse what it cannot honour.
/// </summary>
public sealed class HuggingFaceTokenizerSourceTests
{
    [Fact]
    public void Load_MinimalBpePackage_NormalisesVocabularyAndMerges()
    {
        using var dir = new TempDir();
        dir.WriteTokenizerJson();

        var result = HuggingFaceTokenizerSource.Load(dir.Path);

        Assert.True(result.IsUsable, string.Join("; ", result.Rejections));
        var source = result.Source!;
        Assert.Equal(["<unk>", "<s>", "</s>", "a", "b", "ab"], source.Tokens);
        Assert.Equal(["a b"], source.Merges);
        Assert.Equal("hf-bpe", source.ModelFamily);
    }

    /// <summary>
    /// A non-BPE model would encode without error through a BPE constructor and disagree with the
    /// model's training — silently wrong rather than failing.
    /// </summary>
    [Theory]
    [InlineData("Unigram")]
    [InlineData("WordPiece")]
    [InlineData("WordLevel")]
    public void Load_NonBpeTokenizerModel_IsRefused(string modelType)
    {
        using var dir = new TempDir();
        dir.WriteTokenizerJson(modelType: modelType);

        var result = HuggingFaceTokenizerSource.Load(dir.Path);

        Assert.False(result.IsUsable);
        var r = Assert.Single(result.Rejections);
        Assert.Equal(ModelPackageRejectionKind.MissingTokenizer, r.Kind);
        Assert.Equal(modelType, r.Subject);
    }

    /// <summary>An id no token claims would surface later as a decode fault; refuse it here.</summary>
    [Fact]
    public void Load_VocabularyWithAnUnassignedId_IsRefused()
    {
        using var dir = new TempDir();
        dir.WriteTokenizerJson(vocab: """{"<unk>":0,"<s>":1,"gap":4}""");

        var result = HuggingFaceTokenizerSource.Load(dir.Path);

        Assert.False(result.IsUsable);
        Assert.Contains(result.Rejections,
            x => x.Kind == ModelPackageRejectionKind.MalformedPackage && x.Detail.Contains("unassigned id"));
    }

    /// <summary>Merges are written either as "a b" or as ["a","b"] depending on tokenizers version.</summary>
    [Fact]
    public void Load_PairFormMerges_AreNormalisedToTheSpaceSeparatedForm()
    {
        using var dir = new TempDir();
        dir.WriteTokenizerJson(merges: """[["a","b"],["ab","c"]]""");

        var result = HuggingFaceTokenizerSource.Load(dir.Path);

        Assert.True(result.IsUsable, string.Join("; ", result.Rejections));
        Assert.Equal(["a b", "ab c"], result.Source!.Merges);
    }

    [Fact]
    public void Load_AddedSpecialTokens_BecomeSpecialTokensAndControlTypes()
    {
        using var dir = new TempDir();
        dir.WriteTokenizerJson(addedTokens:
            """[{"id":1,"content":"<s>","special":true},{"id":2,"content":"</s>","special":true}]""");

        var result = HuggingFaceTokenizerSource.Load(dir.Path);

        Assert.True(result.IsUsable, string.Join("; ", result.Rejections));
        var source = result.Source!;
        Assert.Equal(1, source.AdditionalSpecialTokens["<s>"]);
        Assert.Equal(2, source.AdditionalSpecialTokens["</s>"]);
        Assert.NotNull(source.TokenTypes);
        Assert.Equal(TokenizerSource.ControlTokenType, source.TokenTypes![1]);
    }

    [Fact]
    public void Load_TokenizerConfigSidecar_SuppliesSpecialIdsTemplateAndAddBos()
    {
        using var dir = new TempDir();
        dir.WriteTokenizerJson();
        File.WriteAllText(Path.Combine(dir.Path, "tokenizer_config.json"), """
            {
              "bos_token": "<s>",
              "eos_token": {"content": "</s>"},
              "unk_token": "<unk>",
              "add_bos_token": true,
              "chat_template": "{{ messages[0].content }}"
            }
            """);

        var result = HuggingFaceTokenizerSource.Load(dir.Path);

        Assert.True(result.IsUsable, string.Join("; ", result.Rejections));
        var source = result.Source!;
        Assert.Equal(1, source.BosTokenId);
        Assert.Equal(2, source.EosTokenId);
        Assert.Equal(0, source.UnknownTokenId);
        Assert.Equal(2, source.PadTokenId);          // falls back to EOS
        Assert.True(source.AddBosToken);
        Assert.Equal("{{ messages[0].content }}", source.ChatTemplate);
    }

    [Fact]
    public void Load_SpecialTokensMapSidecar_IsAlsoConsulted()
    {
        using var dir = new TempDir();
        dir.WriteTokenizerJson();
        File.WriteAllText(Path.Combine(dir.Path, "special_tokens_map.json"),
            """{"bos_token":"<s>","eos_token":"</s>"}""");

        var result = HuggingFaceTokenizerSource.Load(dir.Path);

        Assert.True(result.IsUsable, string.Join("; ", result.Rejections));
        Assert.Equal(1, result.Source!.BosTokenId);
        Assert.Equal(2, result.Source!.EosTokenId);
    }

    /// <summary>Sidecars carry names and defaults, not semantics — a broken one must not block loading.</summary>
    [Fact]
    public void Load_MalformedSidecar_FallsBackToDefaults()
    {
        using var dir = new TempDir();
        dir.WriteTokenizerJson();
        File.WriteAllText(Path.Combine(dir.Path, "tokenizer_config.json"), "{ broken ");

        var result = HuggingFaceTokenizerSource.Load(dir.Path);

        Assert.True(result.IsUsable, string.Join("; ", result.Rejections));
        Assert.False(result.Source!.AddBosToken);
    }

    [Fact]
    public void Load_MissingTokenizerJson_IsReportedNotThrown()
    {
        using var dir = new TempDir();

        var result = HuggingFaceTokenizerSource.Load(dir.Path);

        Assert.False(result.IsUsable);
        Assert.Contains(result.Rejections, x => x.Kind == ModelPackageRejectionKind.MissingTokenizer);
    }

    [Fact]
    public void Load_MalformedTokenizerJson_IsReportedNotThrown()
    {
        using var dir = new TempDir();
        File.WriteAllText(Path.Combine(dir.Path, "tokenizer.json"), "{ nope ");

        var result = HuggingFaceTokenizerSource.Load(dir.Path);

        Assert.False(result.IsUsable);
        Assert.Contains(result.Rejections, x => x.Kind == ModelPackageRejectionKind.MalformedPackage);
    }

    /// <summary>
    /// The point of the abstraction: a package vocabulary reaches the same construction path GGUF
    /// uses, so both formats produce the same tokenizer for the same vocabulary.
    /// </summary>
    [Fact]
    public void Load_ThenFromSource_ProducesAWorkingTokenizer()
    {
        using var dir = new TempDir();
        dir.WriteTokenizerJson(addedTokens:
            """[{"id":1,"content":"<s>","special":true},{"id":2,"content":"</s>","special":true}]""");
        File.WriteAllText(Path.Combine(dir.Path, "tokenizer_config.json"),
            """{"bos_token":"<s>","eos_token":"</s>","unk_token":"<unk>"}""");

        var result = HuggingFaceTokenizerSource.Load(dir.Path);
        Assert.True(result.IsUsable, string.Join("; ", result.Rejections));

        var tokenizer = GgufTokenizer.FromSource(result.Source!);

        Assert.Equal(6, tokenizer.VocabSize);
        Assert.Equal(1, tokenizer.BosTokenId);
        Assert.Equal(2, tokenizer.EosTokenId);
        Assert.Equal("ab", tokenizer.Decode([5]));
    }

    [Fact]
    public void FromSource_EmptyVocabulary_Throws()
    {
        var source = new TokenizerSource { Tokens = [] };

        Assert.Throws<InvalidDataException>(() => GgufTokenizer.FromSource(source));
    }

    private sealed class TempDir : IDisposable
    {
        public string Path { get; } = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(), "opentail-hftok-" + Guid.NewGuid().ToString("N"));

        public TempDir() => Directory.CreateDirectory(Path);

        public void WriteTokenizerJson(
            string modelType = "BPE",
            string vocab = """{"<unk>":0,"<s>":1,"</s>":2,"a":3,"b":4,"ab":5}""",
            string merges = """["a b"]""",
            string addedTokens = "[]")
        {
            File.WriteAllText(System.IO.Path.Combine(Path, "tokenizer.json"), $$"""
                {
                  "added_tokens": {{addedTokens}},
                  "model": {
                    "type": "{{modelType}}",
                    "vocab": {{vocab}},
                    "merges": {{merges}}
                  }
                }
                """);
        }

        public void Dispose()
        {
            try { Directory.Delete(Path, recursive: true); } catch { }
        }
    }
}
