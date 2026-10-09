namespace OpenTail.Stingray.Cli;

/// <summary>Tokenizer metadata shapes that have historically decided whether an "unknown" architecture runs at all.</summary>
public enum TokenizerShape
{
    /// <summary>No <c>tokenizer.ggml.model</c>: no tokenizer can be built from this file alone.</summary>
    NoModel,
    /// <summary>model=llama, scores but no merges: the minicpm/xverse/orion class, handled by <c>GgufTokenizer.SpmMergePiecesByScore</c>.</summary>
    SpmScoresOnly,
    /// <summary>model=llama with neither merges nor scores: likely tokenizes to near-character fragments.</summary>
    SpmNoMergesNoScores,
    /// <summary>model=t5: Unigram-LM, routed through <c>UnigramTokenizer.FromGgufVocab</c>.</summary>
    Unigram,
    /// <summary>model=gpt2 (BPE) without merges.</summary>
    BpeNoMerges,
    /// <summary>Nothing noteworthy.</summary>
    Ordinary,
}

public sealed record TokenizerTriage(TokenizerShape Shape, string Model, bool HasMerges, bool HasScores, string? Note);

public sealed record LayerTensor(string Name, string DType, string Shape);

/// <summary>
/// Triage rules shared by <c>admit-arch</c> (prints them) and <c>scout</c> (reports them as findings/blockers), so the two commands
/// cannot drift apart on what counts as a suspicious tokenizer or what the layer-0 inventory is. Pure functions over metadata/index.
/// </summary>
public static class ArchitectureTriage
{
    public static TokenizerTriage ClassifyTokenizer(IReadOnlyDictionary<string, object> metadata)
    {
        bool hasMerges = metadata.ContainsKey("tokenizer.ggml.merges");
        bool hasScores = metadata.ContainsKey("tokenizer.ggml.scores");
        if (!metadata.TryGetValue("tokenizer.ggml.model", out var raw))
            return new TokenizerTriage(TokenizerShape.NoModel, "(absent)", hasMerges, hasScores, null);

        string model = Convert.ToString(raw, System.Globalization.CultureInfo.InvariantCulture) ?? "";
        return (model, hasMerges, hasScores) switch
        {
            ("llama", false, true) => new(TokenizerShape.SpmScoresOnly, model, hasMerges, hasScores,
                "scores-only SPM shape (the minicpm/xverse/orion class). Already handled by GgufTokenizer.SpmMergePiecesByScore."),
            ("llama", false, false) => new(TokenizerShape.SpmNoMergesNoScores, model, hasMerges, hasScores,
                "neither merges nor scores present - likely fragments to near-character level; check tokenizer output carefully."),
            ("t5", _, _) => new(TokenizerShape.Unigram, model, hasMerges, hasScores,
                "Unigram-LM (real llama.cpp LLAMA_VOCAB_TYPE_UGM) - routed through UnigramTokenizer.FromGgufVocab."),
            ("gpt2", false, _) => new(TokenizerShape.BpeNoMerges, model, hasMerges, hasScores,
                "BPE (gpt2) without tokenizer.ggml.merges."),
            _ => new(TokenizerShape.Ordinary, model, hasMerges, hasScores, null),
        };
    }

    /// <summary>Tensors of one layer (<c>blk.&lt;layer&gt;.*</c>), ordinal-sorted, for comparison against a known-working architecture.</summary>
    public static IReadOnlyList<LayerTensor> LayerInventory(IEnumerable<GgufTensorInfo> tensors, int layer = 0)
    {
        string prefix = $"blk.{layer}.";
        return tensors.Where(t => t.Name.StartsWith(prefix, StringComparison.Ordinal))
            .OrderBy(t => t.Name, StringComparer.Ordinal)
            .Select(t => new LayerTensor(t.Name, t.DType.ToString(), "[" + string.Join(",", t.Dimensions.Take(t.NDimensions)) + "]"))
            .ToArray();
    }
}
