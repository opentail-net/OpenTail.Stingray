using OpenTail.Stingray.Engine.Scout;
namespace OpenTail.Stingray.Tests.Cli;

/// <summary>The triage rules shared by admit-arch and scout.</summary>
public sealed class ArchitectureTriageTests
{
    private static TokenizerTriage Classify(string? model, bool merges, bool scores)
    {
        var md = new Dictionary<string, object>();
        if (model is not null) md["tokenizer.ggml.model"] = model;
        if (merges) md["tokenizer.ggml.merges"] = new string[] { "a b" };
        if (scores) md["tokenizer.ggml.scores"] = new float[] { 1f };
        return ArchitectureTriage.ClassifyTokenizer(md);
    }

    [Theory]
    [InlineData(null, false, false, TokenizerShape.NoModel)]
    [InlineData("llama", false, true, TokenizerShape.SpmScoresOnly)]
    [InlineData("llama", false, false, TokenizerShape.SpmNoMergesNoScores)]
    [InlineData("llama", true, true, TokenizerShape.Ordinary)]
    [InlineData("t5", false, true, TokenizerShape.Unigram)]
    [InlineData("gpt2", false, false, TokenizerShape.BpeNoMerges)]
    [InlineData("gpt2", true, false, TokenizerShape.Ordinary)]
    public void Tokenizer_shapes_are_classified(string? model, bool merges, bool scores, TokenizerShape expected) =>
        Assert.Equal(expected, Classify(model, merges, scores).Shape);

    [Fact]
    public void Layer_inventory_returns_only_that_layer_sorted_with_shapes()
    {
        GgufTensorInfo T(string n, params long[] d) => new(n, d.Length, d, DType.Q4_K, 0);
        var inv = ArchitectureTriage.LayerInventory([T("blk.1.x", 1), T("blk.0.b", 2, 3), T("blk.0.a", 4), T("blk.10.a", 5), T("token_embd.weight", 6)]);
        Assert.Equal(["blk.0.a", "blk.0.b"], inv.Select(i => i.Name));
        Assert.Equal("[2,3]", inv[1].Shape);
    }
}
