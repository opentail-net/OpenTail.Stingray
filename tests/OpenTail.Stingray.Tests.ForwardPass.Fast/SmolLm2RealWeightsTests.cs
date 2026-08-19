using System;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class SmolLm2RealWeightsTests
{
    private static string? FindModelPath(string fileName)
    {
        string[] absoluteCandidates =
        {
            $@"C:\Git-Public\OpenTail.Stingray\models\{fileName}",
            $@"C:\p\opentail-llm\models\{fileName}",
            $@"E:\models\{fileName}",
        };
        foreach (var p in absoluteCandidates)
        {
            if (File.Exists(p)) return p;
        }

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

    [Fact]
    public void SmolLM2_135M_RealModelFile_LoadsAndInspectsMetadata()
    {
        string? modelPath = FindModelPath("SmolLM2-135M-Instruct-Q4_K_M.gguf");
        if (modelPath is null) return;

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "SmolLM2 135M GGUF must contain tensors");
        Assert.True(model.Metadata.Count > 0, "SmolLM2 135M GGUF must contain metadata");
    }

    [Fact]
    public async Task SmolLM2_135M_RealModel_ExecutesPrefillAndGreedyDecode()
    {
        string? modelPath = FindModelPath("SmolLM2-135M-Instruct-Q4_K_M.gguf");
        if (modelPath is null) return;

        using var model = GgufModel.Open(modelPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata);
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
