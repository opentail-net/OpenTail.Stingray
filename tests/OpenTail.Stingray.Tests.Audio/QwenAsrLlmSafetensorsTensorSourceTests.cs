
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Verifies the real, canonical Hugging Face `Qwen/Qwen3-ASR-0.6B` Safetensors checkpoint
/// (`thinker.model.*`/`thinker.lm_head.*`, BF16, bias-free GQA decoder with Q/K RMSNorm) runs
/// through this engine's real, unmodified `ForwardPass` via
/// <see cref="QwenAsrLlmSafetensorsTensorSource"/> -- the Safetensors counterpart of
/// <see cref="QwenAsrLlmTensorSourceTests"/>'s GGUF-based adapter test, same architectural bet,
/// real weights confirmed to match exactly (28 layers, hidden=1024, 16/8 heads, head_dim=128,
/// ffn=3072, vocab=151936).
/// </summary>
public sealed class QwenAsrLlmSafetensorsTensorSourceTests : HeavyTestBase
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static QwenAsrLlmSafetensorsTensorSource Open(string path) => new(
        path,
        numLayers: 28, hiddenDim: 1024, numHeads: 16, numKvHeads: 8, headDim: 128,
        ffDim: 3072, vocabSize: 151936, ropeTheta: 1_000_000f, rmsNormEps: 1e-6f);

    [Fact]
    public void Adapter_MapsRealTensors_AllLayersAndHeadPresent()
    {
        string? path = FindRepoFile("models/qwen3-asr-0.6b-hf/model.safetensors");
        Assert.SkipUnless(path != null, "models/qwen3-asr-0.6b-hf/model.safetensors not found");

        using var source = Open(path!);

        Assert.Equal("qwen3", source.Metadata["general.architecture"]);
        Assert.NotNull(source.FindTensor("token_embd.weight"));
        Assert.NotNull(source.FindTensor("output.weight"));
        Assert.NotNull(source.FindTensor("output_norm.weight"));
        Assert.NotNull(source.FindTensor("blk.0.attn_q.weight"));
        Assert.NotNull(source.FindTensor("blk.0.attn_q_norm.weight"));
        Assert.NotNull(source.FindTensor("blk.0.attn_k_norm.weight"));
        Assert.NotNull(source.FindTensor("blk.27.ffn_down.weight"));
        Assert.Null(source.FindTensor("blk.28.attn_q.weight")); // only 28 layers (0..27)
    }

    [Fact]
    public void Adapter_RunsRealForwardPass_ProducesFiniteNonDegenerateLogits()
    {
        string? path = FindRepoFile("models/qwen3-asr-0.6b-hf/model.safetensors");
        Assert.SkipUnless(path != null, "models/qwen3-asr-0.6b-hf/model.safetensors not found");

        using var source = Open(path!);
        var hp = ModelHyperparams.FromGgufMetadata(source.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(source, backend, hp);

        var prompt = new int[] { 785, 3974, 13876, 25, 1863 };
        var logits = fwd.Prefill(prompt).ToArray();

        Assert.Equal(151936, logits.Length);
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        foreach (var v in logits)
        {
            Assert.False(float.IsNaN(v) || float.IsInfinity(v), "logit is NaN/Inf");
            if (v < min) min = v;
            if (v > max) max = v;
        }
        Assert.True(max - min > 1.0f, $"logits look degenerate: min={min}, max={max}");
    }
}
