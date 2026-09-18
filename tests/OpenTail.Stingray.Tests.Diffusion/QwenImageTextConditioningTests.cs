namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Real smoke test confirming `qwen2vl`'s newly-admitted text-only NEOX RoPE dispatch
/// (ModelGraph.cs, docs/089) runs correctly against the real Qwen2.5-VL-7B-Instruct checkpoint --
/// a real forward pass producing finite, non-degenerate hidden states via the same
/// `IForwardPass.ExtractHiddenStates` mechanism confirmed suitable for Qwen Image's text
/// conditioning (final layer only, unlike FLUX.2's multi-layer Mistral recipe).
/// </summary>
public sealed class QwenImageTextConditioningTests
{
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
    public void Qwen2Vl_RealWeights_TextOnlyForwardPassProducesFiniteHiddenStates()
    {
        string? modelPath = FindModelPath("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf");
        Assert.SkipUnless(modelPath != null, "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf not found");

        using var model = GgufModel.Open(modelPath!);
        Assert.Equal("qwen2vl", model.Metadata["general.architecture"]);

        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new OpenTail.Stingray.Cpu.CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp);

        // Real Qwen Image recipe (docs/089): ChatML template, real final-layer extraction.
        var tokens = tokenizer.Encode(
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\na red apple on a wooden table<|im_end|>\n<|im_start|>assistant\n"
        ).ToList();
        Assert.True(tokens.Count > 0);

        var hidden = new float[tokens.Count * hp.EmbeddingDim];
        fwd.ExtractHiddenStates(tokens, hidden);

        bool allZero = true;
        foreach (var v in hidden)
        {
            Assert.True(float.IsFinite(v), "Qwen2.5-VL hidden states contain NaN/Inf");
            if (v != 0f) allZero = false;
        }
        Assert.False(allZero, "Qwen2.5-VL hidden states are all-zero -- likely a wiring bug");
    }

    [Fact]
    public void QwenImageTextConditioning_Encode_RealWeights_ProducesFiniteConditioning()
    {
        string? modelPath = FindModelPath("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf");
        Assert.SkipUnless(modelPath != null, "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf not found");

        using var model = GgufModel.Open(modelPath!);
        var hp = ModelHyperparams.FromGgufMetadata(model.Metadata, model);
        var tokenizer = GgufTokenizer.FromGgufModel(model);
        using var backend = new OpenTail.Stingray.Cpu.CpuBackend();
        using var fwd = new Engine.ForwardPass(model, backend, hp);

        var (embeds, nTokens) = OpenTail.Stingray.Diffusion.QwenImage.QwenImageTextConditioning.Encode(
            fwd, tokenizer, "a red apple on a wooden table");

        Assert.True(nTokens > 0);
        Assert.Equal(nTokens * OpenTail.Stingray.Diffusion.QwenImage.QwenImageModel.ContextDim, embeds.Length);
        foreach (var v in embeds)
            Assert.True(float.IsFinite(v), "QwenImageTextConditioning.Encode output contains NaN/Inf");
    }
}
