
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Verifies CosyVoice2's LLM backbone (a plain Qwen2ForCausalLM, confirmed against its real
/// config -- see docs/audio-review-progress.md's CosyVoice section) runs through this engine's
/// real, unmodified `ForwardPass` via <see cref="CosyVoiceLlmTensorSource"/>, the same
/// architectural bet already validated for QwenASR's LLM half
/// (<see cref="QwenAsrLlmTensorSourceTests"/>).
/// </summary>
public sealed class CosyVoiceLlmTensorSourceTests : HeavyTestBase
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

    private static CosyVoiceLlmTensorSource Open(string path) => new(
        path,
        numLayers: 24, hiddenDim: 896, numHeads: 14, numKvHeads: 2, headDim: 64,
        ffDim: 4864, vocabSize: 151936, ropeTheta: 1_000_000f, rmsNormEps: 1e-6f);

    [Fact]
    public void Adapter_MapsRealTensors_AllLayersAndHeadPresent()
    {
        string? path = FindRepoFile("models/cosyvoice2_llm.safetensors");
        Assert.SkipUnless(path != null, "models/cosyvoice2_llm.safetensors not found");

        using var source = Open(path!);

        Assert.Equal("qwen2", source.Metadata["general.architecture"]);
        Assert.NotNull(source.FindTensor("token_embd.weight"));
        Assert.NotNull(source.FindTensor("output.weight"));
        Assert.NotNull(source.FindTensor("output_norm.weight"));
        Assert.NotNull(source.FindTensor("blk.0.attn_q.weight"));
        Assert.NotNull(source.FindTensor("blk.0.attn_q.bias"));
        Assert.NotNull(source.FindTensor("blk.23.ffn_down.weight"));
        Assert.Null(source.FindTensor("blk.24.attn_q.weight")); // only 24 layers (0..23)
    }

    [Fact]
    public void Adapter_LoadsRealSpeechTokenTensors_CorrectShapesAndFinite()
    {
        string? path = FindRepoFile("models/cosyvoice2_llm.safetensors");
        Assert.SkipUnless(path != null, "models/cosyvoice2_llm.safetensors not found");

        using var source = Open(path!);

        Assert.NotNull(source.SpeechEmbeddingWeight);
        Assert.Equal(6564 * 896, source.SpeechEmbeddingWeight!.Length);
        Assert.NotNull(source.LlmDecoderWeight);
        Assert.Equal(6564 * 896, source.LlmDecoderWeight!.Length);
        Assert.NotNull(source.LlmDecoderBias);
        Assert.Equal(6564, source.LlmDecoderBias!.Length);
        Assert.NotNull(source.LlmEmbeddingWeight);
        Assert.Equal(2 * 896, source.LlmEmbeddingWeight!.Length);

        foreach (var v in source.SpeechEmbeddingWeight) Assert.False(float.IsNaN(v) || float.IsInfinity(v));
        foreach (var v in source.LlmDecoderWeight) Assert.False(float.IsNaN(v) || float.IsInfinity(v));
    }

    /// <summary>
    /// The real fix for the "ForwardPass has no embedding-injection/output-head-override API"
    /// blocker flagged in the previous iteration: EnableSpeechGenerationMode swaps in a
    /// synthetic combined embedding table (real text rows + real speech rows) and the real
    /// speech-vocab head, entirely via the existing IModelTensorSource seam -- no Engine
    /// changes. This proves it end-to-end: prefill a token sequence that runs into the
    /// speech-vocab id range and confirm real, finite, non-degenerate logits over the real
    /// 6564-entry speech vocabulary come back.
    /// </summary>
    [Fact]
    public void SpeechGenerationMode_RunsRealForwardPass_ProducesFiniteSpeechVocabLogits()
    {
        string? path = FindRepoFile("models/cosyvoice2_llm.safetensors");
        Assert.SkipUnless(path != null, "models/cosyvoice2_llm.safetensors not found");

        using var source = Open(path!);
        source.EnableSpeechGenerationMode();

        Assert.NotNull(source.FindTensor("token_embd.weight"));
        // +2 for the real `llm_embedding.weight` sos/task_id rows, appended as synthetic vocab
        // entries this session (see CosyVoiceLlmTensorSource.SosTaskTokenIdBase).
        Assert.Equal(151936 + 6564 + 2, source.FindTensor("token_embd.weight")!.Value.Dimensions[1]);
        Assert.Equal(6564, source.FindTensor("output.weight")!.Value.Dimensions[1]);

        var hp = ModelHyperparams.FromGgufMetadata(source.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(source, backend, hp);

        // A few real text tokens followed by a real speech token (offset into the combined
        // vocab via SpeechTokenIdOffset) -- exercises both embedding halves in one sequence.
        var prompt = new int[] { 785, 3974, source.SpeechTokenIdOffset + 12 };
        var rawLogits = fwd.Prefill(prompt).ToArray();

        Assert.Equal(6564, rawLogits.Length);
        var logits = new float[rawLogits.Length];
        for (int i = 0; i < logits.Length; i++)
            logits[i] = rawLogits[i] + source.LlmDecoderBias![i]; // ForwardPass has no final-layer bias support

        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        foreach (var v in logits)
        {
            Assert.False(float.IsNaN(v) || float.IsInfinity(v), "logit is NaN/Inf");
            if (v < min) min = v;
            if (v > max) max = v;
        }
        Assert.True(max - min > 1.0f, $"logits look degenerate: min={min}, max={max}");
    }

    [Fact]
    public void Adapter_RunsRealForwardPass_ProducesFiniteNonDegenerateLogits()
    {
        string? path = FindRepoFile("models/cosyvoice2_llm.safetensors");
        Assert.SkipUnless(path != null, "models/cosyvoice2_llm.safetensors not found");

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
