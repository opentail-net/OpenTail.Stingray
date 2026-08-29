
namespace OpenTail.Stingray.Tests.Audio.Fast;

/// <summary>SCRATCH/throwaway: probes Fish Speech infra reuse. Delete after use.</summary>
public sealed class FishSpeechScratchTests : HeavyTestBase
{
    [Fact]
    public void ForwardPass_Constructs_And_ForwardEmbedding_Runs()
    {
        string? modelPath = FindModelPath("s2-pro-q4_k_m.gguf");
        Assert.SkipUnless(modelPath != null, "models/s2-pro-q4_k_m.gguf not found");

        using var model = GgufModel.Open(modelPath!);
        var source = new FishSpeechTensorSource(model, numLayers: 36);
        var hp = ModelHyperparams.FromGgufMetadata(source.Metadata, source);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(source, backend, hp, maxContextLength: 512);

        Assert.Equal(128, hp.HeadDim); // regression guard for the metadata-passing bug fixed this fire
        Assert.True(fwd.SupportsEmbeddingInput);
        var emb = new float[hp.EmbeddingDim];
        emb[0] = 0.01f;
        var logits = fwd.ForwardEmbedding(emb, 0);
        Assert.Equal(hp.VocabSize, logits.Length);
        Assert.False(float.IsNaN(logits[0]));
    }

    [Fact]
    public void Tokenizer_Loads_From_S2CppExamplesDir()
    {
        string? tokDir = FindRepoDir("examples/s2.cpp");
        Assert.SkipUnless(tokDir != null, "examples/s2.cpp not found");

        var result = HuggingFaceTokenizerSource.Load(tokDir!);
        Assert.True(result.IsUsable, string.Join("; ", result.Rejections.Select(r => r.Detail)));
        Assert.NotNull(result.Source);

        var tokenizer = GgufTokenizer.FromSource(result.Source!);
        var imEndIds = tokenizer.Encode("<|im_end|>");
        var voiceIds = tokenizer.Encode("<|voice|>");
        Assert.Single(imEndIds);
        Assert.Single(voiceIds);
    }

    [Fact]
    public void GenerateSemanticTokens_RealWeights_ProducesInRangeTokens()
    {
        string? modelPath = FindModelPath("s2-pro-q4_k_m.gguf");
        string? tokDir = FindRepoDir("examples/s2.cpp");
        Assert.SkipUnless(modelPath != null && tokDir != null, "S2 Pro GGUF or examples/s2.cpp not found");

        using var pipeline = new FishSpeechPipeline(modelPath!, tokDir!);
        var tokens = pipeline.GenerateSemanticTokens("Hello, this is a test.", maxTokens: 30);

        Assert.NotEmpty(tokens);
        foreach (var t in tokens)
            Assert.InRange(t, 0, 4095); // real codebook_size

        File.WriteAllText(Path.Combine(Path.GetTempPath(), "fishspeech_semantic_tokens.txt"),
            string.Join(",", tokens));
    }

    private static string? FindModelPath(string fileName)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static string? FindRepoDir(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }
}
