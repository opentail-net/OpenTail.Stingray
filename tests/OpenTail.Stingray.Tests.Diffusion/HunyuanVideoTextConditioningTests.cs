using OpenTail.Stingray.Diffusion.HunyuanVideo;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Real-weight verification of <see cref="HunyuanVideoTextConditioning.Encode"/> against the real
/// llava-llama-3-8b-v1_1 checkpoint, independent of the VAE (not yet downloaded -- see docs/088's
/// HunyuanVideo item; this test only exercises the text-encoder half of the pipeline).
/// </summary>
public sealed class HunyuanVideoTextConditioningTests
{
    private const string TextEncoderFileName = "llava-llama-3-8b-v1_1-int4.gguf";

    private static string? FindModelPath(string fileName)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", "_models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void HunyuanVideoTextConditioning_RealWeights_ProducesFiniteCroppedEmbeddings()
    {
        string? modelPath = FindModelPath(TextEncoderFileName);
        if (modelPath is null) return;

        using var model = GgufModel.Open(modelPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new Cpu.CpuBackend();
        using var forward = new Engine.ForwardPass(model, backend, hp);

        var (embeds, nTokens) = HunyuanVideoTextConditioning.Encode(forward, tokenizer, "A red apple on a wooden table");

        Assert.True(nTokens > 0, "Encode must return at least one kept token after crop_start.");
        Assert.Equal(nTokens * HunyuanVideoModel.TextDim, embeds.Length);
        Assert.All(embeds, v => Assert.True(float.IsFinite(v), "Text-conditioning embedding must be finite"));

        double sumSq = 0;
        foreach (var v in embeds) sumSq += (double)v * v;
        double rms = Math.Sqrt(sumSq / embeds.Length);
        Assert.True(rms > 1e-4, $"Text-conditioning embedding RMS too small ({rms}), likely degenerate/all-zero");
    }

    [Fact]
    public void HunyuanVideoTextConditioning_RealWeights_DifferentPromptsProduceDifferentEmbeddings()
    {
        string? modelPath = FindModelPath(TextEncoderFileName);
        if (modelPath is null) return;

        using var model = GgufModel.Open(modelPath);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new Cpu.CpuBackend();
        using var forward = new Engine.ForwardPass(model, backend, hp);

        var (embedsA, nA) = HunyuanVideoTextConditioning.Encode(forward, tokenizer, "A red apple on a wooden table");
        var (embedsB, nB) = HunyuanVideoTextConditioning.Encode(forward, tokenizer, "A blue whale swimming in the ocean");

        int minTokens = Math.Min(nA, nB);
        Assert.True(minTokens > 0);

        double diffSq = 0;
        int compareLen = minTokens * HunyuanVideoModel.TextDim;
        for (int i = 0; i < compareLen; i++)
        {
            double d = embedsA[i] - embedsB[i];
            diffSq += d * d;
        }
        double diffRms = Math.Sqrt(diffSq / compareLen);
        Assert.True(diffRms > 1e-3, $"Two genuinely different prompts must not produce near-identical embeddings (diffRms={diffRms})");
    }
}
