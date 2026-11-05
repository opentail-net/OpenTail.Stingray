
namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class GgufAudioAndEmbeddingRealWeightsTests
{
    private static string? FindModelPath(string fileName) => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(fileName);

    [Fact]
    public void DumpKokoroAndParaformerTensors()
    {
        string? kokoroPath = FindModelPath("kokoro-82m-q8_0.gguf");
        if (kokoroPath != null)
        {
            using var model = GgufModel.Open(kokoroPath);
            var sb = new StringBuilder();
            sb.AppendLine($"Kokoro Tensors Count: {model.Tensors.Count}");
            foreach (var t in model.Tensors)
            {
                sb.AppendLine($"{t.Name} | {t.DType} | dims=[{string.Join(",", t.Dimensions)}]");
            }
            File.WriteAllText(@"C:\Git-Public\OpenTail.Stingray\kokoro_tensors.txt", sb.ToString());
        }

        string? bgePath = FindModelPath("bge-small-en-v1.5-q8_0.gguf");
        if (bgePath != null)
        {
            using var model = GgufModel.Open(bgePath);
            var sb = new StringBuilder();
            sb.AppendLine($"BGE Tensors Count: {model.Tensors.Count}");
            foreach (var t in model.Tensors)
            {
                sb.AppendLine($"{t.Name} | {t.DType} | dims=[{string.Join(",", t.Dimensions)}]");
            }
            File.WriteAllText(@"C:\Git-Public\OpenTail.Stingray\bge_tensors.txt", sb.ToString());
        }
    }

    [Fact]
    public void BgeSmall_GGUF_RealModelFile_LoadsAndInspectsMetadata()
    {
        string? modelPath = FindModelPath("bge-small-en-v1.5-q8_0.gguf");
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "BGE-Small GGUF must contain tensors");
        Assert.True(model.Metadata.Count > 0, "BGE-Small GGUF must contain metadata");
    }

    [Fact]
    public void AllMiniLM_GGUF_RealModelFile_LoadsAndInspectsMetadata()
    {
        string? modelPath = FindModelPath("all-MiniLM-L6-v2-Q8_0.gguf");
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "all-MiniLM-L6-v2 GGUF must contain tensors");
        Assert.True(model.Metadata.Count > 0, "all-MiniLM-L6-v2 GGUF must contain metadata");
    }

    [Fact]
    public void Kokoro82M_GGUF_RealModelFile_LoadsAndInspectsMetadata()
    {
        string? modelPath = FindModelPath("kokoro-82m-q8_0.gguf");
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "Kokoro-82M GGUF must contain tensors");
        Assert.True(model.Metadata.Count > 0, "Kokoro-82M GGUF must contain metadata");
    }

    [Fact]
    public void KokoroVoice_GGUF_RealModelFile_LoadsAndInspectsMetadata()
    {
        string? modelPath = FindModelPath("kokoro-voice-af_heart.gguf");
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "Kokoro Voice GGUF must contain tensors");
    }

    [Fact]
    public void Paraformer_GGUF_RealModelFile_LoadsAndInspectsMetadata()
    {
        string? modelPath = FindModelPath("_models/paraformer-q8.gguf");
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "Paraformer GGUF must contain tensors");
        Assert.True(model.Metadata.Count > 0, "Paraformer GGUF must contain metadata");
        Assert.Equal("paraformer", Convert.ToString(model.Metadata["general.architecture"]));
        Assert.True(model.Metadata.ContainsKey("pf.vocab"), "Paraformer GGUF must contain pf.vocab");
    }
}
