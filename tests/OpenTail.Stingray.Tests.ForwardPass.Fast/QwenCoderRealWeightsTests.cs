
namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class QwenCoderRealWeightsTests
{
    private static string? FindModelPath(string fileName) => OpenTail.Stingray.Engine.Verification.ModelLocator.FindDeep(fileName);

    [Fact]
    public void QwenCoder_05B_RealModelFile_LoadsAndInspectsMetadata()
    {
        string? modelPath = FindModelPath("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf");
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "Qwen2.5-Coder 0.5B GGUF must contain tensors");
        Assert.True(model.Metadata.Count > 0, "Qwen2.5-Coder 0.5B GGUF must contain metadata");
    }

    [Fact]
    public async Task QwenCoder_05B_RealModel_ExecutesPrefillAndGreedyDecode()
    {
        string? modelPath = FindModelPath("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf");
        if (modelPath is null) Assert.Skip("modelPath not found (checkpoint or fixture missing).");

        using var model = GgufModel.Open(modelPath);
        var hp = OpenTail.Stingray.Engine.ArchitectureModelResolver.ResolveHyperparams(model.Metadata);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new CpuBackend();
        var fwd = new OpenTail.Stingray.Engine.ForwardPass(model, backend, hp, maxContextLength: 512);
        using var engine = new InferenceEngine(fwd, tokenizer, "qwen2.5-coder-0.5b");

        var sp = new SamplingParams { Temperature = 0f, MaxNewTokens = 6 };
        var sb = new StringBuilder();
        await foreach (var tok in engine.GenerateAsync("def add(a, b):\n    return", sp))
        {
            sb.Append(tok);
        }

        string generated = sb.ToString();
        Assert.NotNull(generated);
        Assert.NotEmpty(generated);
    }
}
