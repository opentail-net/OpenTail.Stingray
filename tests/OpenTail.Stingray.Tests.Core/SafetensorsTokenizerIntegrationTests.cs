using System.Text;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Core;

public sealed class SafetensorsTokenizerIntegrationTests : IDisposable
{
    private readonly string _tempDir;

    public SafetensorsTokenizerIntegrationTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "opentail_st_tok_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
        {
            try { Directory.Delete(_tempDir, recursive: true); } catch { }
        }
    }

    [Fact]
    public void HuggingFaceTokenizerSource_LoadsValidBpeTokenizerJson()
    {
        string packageDir = Path.Combine(_tempDir, "bpe_package");
        Directory.CreateDirectory(packageDir);

        var vocabSb = new StringBuilder();
        vocabSb.Append("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"<bos>\":0,\"<eos>\":1,\"h\":2,\"el\":3,\"hel\":4},\"merges\":[\"h el\"]}}");
        File.WriteAllText(Path.Combine(packageDir, "tokenizer.json"), vocabSb.ToString());

        string configJson = "{\"add_bos_token\":true,\"bos_token\":\"<bos>\",\"eos_token\":\"<eos>\"}";
        File.WriteAllText(Path.Combine(packageDir, "tokenizer_config.json"), configJson);

        var result = HuggingFaceTokenizerSource.Load(packageDir);

        Assert.True(result.IsUsable);
        Assert.NotNull(result.Source);
        Assert.Equal(5, result.Source.Tokens.Length);
        Assert.Single(result.Source.Merges);

        var ggufTok = GgufTokenizer.FromSource(result.Source);
        Assert.Equal(5, ggufTok.VocabSize);
    }

    [Fact]
    public void HuggingFaceTokenizerSource_RefusesNonBpeModel()
    {
        string packageDir = Path.Combine(_tempDir, "unigram_package");
        Directory.CreateDirectory(packageDir);

        string tokenizerJson = "{\"model\":{\"type\":\"Unigram\",\"vocab\":[]}}";
        File.WriteAllText(Path.Combine(packageDir, "tokenizer.json"), tokenizerJson);

        var result = HuggingFaceTokenizerSource.Load(packageDir);

        Assert.False(result.IsUsable);
        Assert.NotEmpty(result.Rejections);
        Assert.Contains(result.Rejections, r => r.Kind == ModelPackageRejectionKind.MissingTokenizer);
    }

    [Theory]
    [InlineData("{")]
    [InlineData("[]")]
    [InlineData("{\"model\":[]}")]
    [InlineData("{\"model\":{\"type\":\"BPE\",\"vocab\":[]}}")]
    public void HuggingFaceTokenizerSource_MalformedCorpus_IsRefused(string tokenizerJson)
    {
        string packageDir = Path.Combine(_tempDir, "mutated_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(packageDir);
        File.WriteAllText(Path.Combine(packageDir, "tokenizer.json"), tokenizerJson);

        var result = HuggingFaceTokenizerSource.Load(packageDir);

        Assert.False(result.IsUsable);
        Assert.NotEmpty(result.Rejections);
    }

    [Fact]
    public void ModelPackageInspector_DetectsSentencePieceAssets()
    {
        string packageDir = Path.Combine(_tempDir, "spm_package");
        Directory.CreateDirectory(packageDir);

        File.WriteAllBytes(Path.Combine(packageDir, "tokenizer.model"), new byte[] { 0x08, 0x01, 0x12, 0x05, 0x74, 0x65, 0x73, 0x74, 0x00 });

        string configJson = "{\"model_type\":\"llama\",\"hidden_size\":64,\"num_hidden_layers\":2,\"num_attention_heads\":4,\"intermediate_size\":128,\"vocab_size\":32000}";
        File.WriteAllText(Path.Combine(packageDir, "config.json"), configJson);

        File.WriteAllBytes(Path.Combine(packageDir, "model.safetensors"), new byte[64]);

        var report = ModelPackageInspector.Inspect(packageDir);
        Assert.Equal(ModelPackageTokenizerFamily.SentencePiece, report.TokenizerFamily);
    }
}
