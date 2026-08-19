using System;
using System.IO;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

public sealed class DSparkRealWeightsTests
{
    private const string DSparkDirName = "dspark_qwen3_4b_block7";

    private static string? FindModelDir(string dirName)
    {
        string[] absoluteCandidates =
        {
            $@"C:\Git-Public\OpenTail.Stingray\models\{dirName}",
            $@"C:\p\opentail-llm\models\{dirName}",
            $@"E:\models\{dirName}",
        };
        foreach (var p in absoluteCandidates)
        {
            if (Directory.Exists(p)) return p;
        }

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", dirName);
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void DSparkDraftModel_RealWeights_LoadsAndInitializesSuccessfully()
    {
        string? modelDir = FindModelDir(DSparkDirName);
        if (modelDir is null) return;

        string configPath = Path.Combine(modelDir, "config.json");
        string weightsPath = Path.Combine(modelDir, "model.safetensors");

        if (!File.Exists(configPath) || !File.Exists(weightsPath)) return;

        var cfg = DSparkConfig.FromJsonFile(configPath);
        Assert.NotNull(cfg);
        Assert.True(cfg.HiddenSize > 0);
        Assert.True(cfg.VocabSize > 0);

        using var st = SafetensorsLoader.Open(weightsPath);
        Assert.NotNull(st);
        Assert.True(st.TensorCount > 0, "DSpark weights safetensors must contain tensors");

        using var draft = new DSparkDraftModel(cfg, st, maxContextLength: 512);
        Assert.NotNull(draft);
        Assert.Equal(cfg.VocabSize, draft.VocabSize);
    }
}
