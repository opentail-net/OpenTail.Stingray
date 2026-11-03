
namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class SmolLm2RealWeightsTests
{
    private static string? FindModelPath(string fileName) => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(fileName);

    [Fact]
    public void SmolLM2_135M_RealModelFile_LoadsAndInspectsMetadata()
    {
        string? modelPath = FindModelPath("SmolLM2-135M-Instruct-Q4_K_M.gguf");
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "SmolLM2 135M GGUF must contain tensors");
        Assert.True(model.Metadata.Count > 0, "SmolLM2 135M GGUF must contain metadata");
    }

    [Fact]
    public async Task SmolLM2_135M_RealModel_ExecutesPrefillAndGreedyDecode()
    {
        string? modelPath = FindModelPath("SmolLM2-135M-Instruct-Q4_K_M.gguf");
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var model = GgufModel.Open(modelPath);
        var hp = OpenTail.Stingray.Engine.ArchitectureModelResolver.ResolveHyperparams(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        var fwd = new OpenTail.Stingray.Engine.ForwardPass(model, backend, hp, maxContextLength: 512);
        using var engine = new InferenceEngine(fwd, tokenizer, "smollm2-135m");

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 6 };
        var sb = new StringBuilder();
        await foreach (var tok in engine.GenerateAsync("The capital of France is", sp))
        {
            sb.Append(tok);
        }

        string generated = sb.ToString();
        Assert.NotNull(generated);
        Assert.NotEmpty(generated);
    }
}
