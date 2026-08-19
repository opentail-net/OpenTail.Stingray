using System;
using System.IO;
using System.Text;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class GgufAudioAndEmbeddingRealWeightsTests
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
        if (modelPath is null) return;

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "BGE-Small GGUF must contain tensors");
        Assert.True(model.Metadata.Count > 0, "BGE-Small GGUF must contain metadata");
    }

    [Fact]
    public void AllMiniLM_GGUF_RealModelFile_LoadsAndInspectsMetadata()
    {
        string? modelPath = FindModelPath("all-MiniLM-L6-v2-Q8_0.gguf");
        if (modelPath is null) return;

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "all-MiniLM-L6-v2 GGUF must contain tensors");
        Assert.True(model.Metadata.Count > 0, "all-MiniLM-L6-v2 GGUF must contain metadata");
    }

    [Fact]
    public void Kokoro82M_GGUF_RealModelFile_LoadsAndInspectsMetadata()
    {
        string? modelPath = FindModelPath("kokoro-82m-q8_0.gguf");
        if (modelPath is null) return;

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "Kokoro-82M GGUF must contain tensors");
        Assert.True(model.Metadata.Count > 0, "Kokoro-82M GGUF must contain metadata");
    }

    [Fact]
    public void KokoroVoice_GGUF_RealModelFile_LoadsAndInspectsMetadata()
    {
        string? modelPath = FindModelPath("kokoro-voice-af_heart.gguf");
        if (modelPath is null) return;

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "Kokoro Voice GGUF must contain tensors");
    }

    [Fact]
    public void Paraformer_GGUF_RealModelFile_LoadsAndInspectsMetadata()
    {
        string? modelPath = FindModelPath("paraformer-q8.gguf");
        if (modelPath is null) return;

        using var model = GgufModel.Open(modelPath);
        Assert.NotNull(model);
        Assert.True(model.Tensors.Count > 0, "Paraformer GGUF must contain tensors");
        Assert.True(model.Metadata.Count > 0, "Paraformer GGUF must contain metadata");
    }
}
