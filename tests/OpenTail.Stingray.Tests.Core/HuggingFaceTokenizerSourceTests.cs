
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
    /// A real HF "fast tokenizer" BPE export of a SentencePiece-family vocabulary (Gemma/T5Gemma)
    /// declares a `normalizer` step replacing literal spaces with U+2581 ('▁') before merging --
    /// found 2026-09-02 porting Stable Audio 3's T5Gemma text conditioner, where this loader
    /// accepted the real tokenizer.json as plain BPE (`model.type: "BPE"`) and silently produced
    /// wrong token ids because that substitution never ran. Detecting it routes the source through
    /// `GgufTokenizer`'s existing Gemma/Llama SPM machinery (`ModelFamily="gemma"`) instead of the
    /// plain byte-level BPE path, AND marks the merges list as rank-priority (real BPE, not
    /// GGUF-Llama's score-based algorithm -- see <see cref="TokenizerSource.MergesAreRankPriority"/>).
    /// </summary>
    [Fact]
    public void Load_SentencePieceStyleNormalizer_RoutesToGemmaFamilyWithRankPriorityMerges()
    {
        using var dir = new TempDir();
        dir.WriteTokenizerJson(normalizer: """{"type":"Replace","pattern":{"String":" "},"content":"▁"}""");

        var result = HuggingFaceTokenizerSource.Load(dir.Path);

        Assert.True(result.IsUsable, string.Join("; ", result.Rejections));
        var source = result.Source!;
        Assert.Equal("gemma", source.ModelFamily);
        Assert.True(source.MergesAreRankPriority);
    }

    /// <summary>
    /// Real BPE merge order is priority-ranked (earliest-declared merge applies first wherever it
    /// is found in the sequence), NOT "leftmost mergeable pair wins on a score tie" (the real
    /// algorithm for genuine llama.cpp SPM, `tokenizer.ggml.model=llama` -- see
    /// <see cref="GgufTokenizer.SpmMergePiecesByScore"/>'s doc comment). A vocab where these two
    /// algorithms disagree: pieces `[a, b, c]`, merges `["b c", "a b"]` (rank 0 then rank 1). Real
    /// rank-priority BPE applies the LOWEST-rank mergeable pair wherever it occurs -- "b c" (rank 0)
    /// beats "a b" (rank 1) even though "a b" is leftmost -- giving `[a, bc]`. The leftmost-tie
    /// algorithm this loader silently used before this fix would instead merge "a b" first (the
    /// first mergeable pair scanning left-to-right, since both have score 0.0), giving `[ab, c]`.
    /// This is exactly the class of divergence that showed up on a real T5Gemma prompt containing
    /// "arpeggio" before this fix ("▁arp"+"egg"+"io" instead of the real "▁ar"+"pe"+"ggio").
    /// </summary>
    [Fact]
    public void Encode_SentencePieceStyleBpe_UsesRankPriorityNotLeftmostTie()
    {
        using var dir = new TempDir();
        dir.WriteTokenizerJson(
            vocab: """{"<unk>":0,"a":1,"b":2,"c":3,"ab":4,"bc":5,"abc":6}""",
            merges: """["b c", "a b"]""",
            normalizer: """{"type":"Replace","pattern":{"String":" "},"content":"▁"}""");

        var result = HuggingFaceTokenizerSource.Load(dir.Path);
        Assert.True(result.IsUsable, string.Join("; ", result.Rejections));

        var tok = GgufTokenizer.FromSource(result.Source!);
        var ids = tok.Encode("abc");

        Assert.Equal([1, 5], ids); // "a" (id 1) + "bc" (id 5) -- the real rank-priority result
    }

    /// <summary>
    /// Real-checkpoint regression test for the fix above: the real bundled T5Gemma tokenizer
    /// (Stable Audio 3's text conditioner, see docs/057-stable-audio-3-implementation-plan.md),
    /// encoding the exact prompt whose real ids were captured from HF `transformers` for
    /// <c>StableAudioT5GemmaEncoderGoldenParityTests</c>. Skips (does not fail) when the local
    /// checkpoint isn't present, same convention as this project's other real-weight fixtures.
    /// </summary>
    [Fact]
    public void Encode_RealT5GemmaTokenizer_MatchesRealTransformersIds()
    {
        string? dir = FindRepoDir("models/stable-audio-3-t5gemma");
        if (dir is null) return; // skip: needs the local T5Gemma checkpoint (tokenizer.json lives alongside the weights)

        var result = HuggingFaceTokenizerSource.Load(dir);
        Assert.True(result.IsUsable, string.Join("; ", result.Rejections));

        var tok = GgufTokenizer.FromSource(result.Source!);
        var ids = tok.Encode("A beautiful piano arpeggio grows into a grand cinematic climax");

        // Real ids captured via HF `transformers`' AutoTokenizer for this exact prompt -- see
        // tests/OpenTail.Stingray.Tests.Diffusion/TestData/StableAudioT5GemmaGolden/ids.bin
        // (the first 12 non-padding entries there).
        Assert.Equal([235280, 4964, 16748, 813, 554, 16194, 26075, 1280, 476, 4497, 106852, 82923], ids);
    }

    private static string? FindRepoDir(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath.Replace('/', Path.DirectorySeparatorChar));
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
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
            string addedTokens = "[]",
            string? normalizer = null)
        {
            string normalizerJson = normalizer is null ? "null" : normalizer;
            File.WriteAllText(System.IO.Path.Combine(Path, "tokenizer.json"), $$"""
                {
                  "added_tokens": {{addedTokens}},
                  "normalizer": {{normalizerJson}},
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
