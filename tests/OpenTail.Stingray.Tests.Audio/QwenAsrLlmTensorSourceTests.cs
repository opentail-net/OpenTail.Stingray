
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Verifies the qwen3asr -> qwen3 metadata-remapping adapter (see docs/audio-review-
/// progress.md's QwenASR section) actually produces something the real Engine accepts and can
/// find weights through -- not yet a full generation-loop test, just confirms the adapter's
/// core claim: `ModelCompatibility.ValidateForTextGeneration`-equivalent checks pass and
/// `blk.0.attn_q.weight`-style tensors resolve through the filtered/remapped view.
/// </summary>
public sealed class QwenAsrLlmTensorSourceTests : HeavyTestBase
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

    [Fact]
    public void Adapter_RemapsMetadata_ToStandardQwen3Keys()
    {
        string? path = FindRepoFile("models/qwen3-asr-0.6b-q4_k.gguf");
        Assert.SkipUnless(path != null, "models/qwen3-asr-0.6b-q4_k.gguf not found");

        using var inner = GgufModel.Open(path!);
        var adapter = new QwenAsrLlmTensorSource(inner);

        Assert.Equal("qwen3", adapter.Metadata["general.architecture"]);
        Assert.Equal(1024, System.Convert.ToInt32(adapter.Metadata["qwen3.embedding_length"]));
        Assert.Equal(16, System.Convert.ToInt32(adapter.Metadata["qwen3.attention.head_count"]));
        Assert.Equal(8, System.Convert.ToInt32(adapter.Metadata["qwen3.attention.head_count_kv"]));
        Assert.Equal(128, System.Convert.ToInt32(adapter.Metadata["qwen3.attention.key_length"]));
        Assert.Equal(28, System.Convert.ToInt32(adapter.Metadata["qwen3.block_count"]));
        Assert.Equal(3072, System.Convert.ToInt32(adapter.Metadata["qwen3.feed_forward_length"]));
        Assert.Equal(151936, System.Convert.ToInt32(adapter.Metadata["qwen3.vocab_size"]));
    }

    [Fact]
    public void Adapter_ExposesOnlyLlmTensors_NotAudioTensors()
    {
        string? path = FindRepoFile("models/qwen3-asr-0.6b-q4_k.gguf");
        Assert.SkipUnless(path != null, "models/qwen3-asr-0.6b-q4_k.gguf not found");

        using var inner = GgufModel.Open(path!);
        var adapter = new QwenAsrLlmTensorSource(inner);

        Assert.NotNull(adapter.FindTensor("blk.0.attn_q.weight"));
        Assert.NotNull(adapter.FindTensor("token_embd.weight"));
        Assert.NotNull(adapter.FindTensor("output.weight"));
        Assert.NotNull(adapter.FindTensor("output_norm.weight"));

        Assert.Null(adapter.FindTensor("audio.conv.1.weight"));
        Assert.Null(adapter.FindTensor("audio.blk.0.attn_q.weight"));

        foreach (var t in adapter.Tensors)
            Assert.False(t.Name.StartsWith("audio."), $"leaked audio tensor: {t.Name}");
    }

    [Fact]
    public void Adapter_PassesEngineArchitectureValidation()
    {
        string? path = FindRepoFile("models/qwen3-asr-0.6b-q4_k.gguf");
        Assert.SkipUnless(path != null, "models/qwen3-asr-0.6b-q4_k.gguf not found");

        using var inner = GgufModel.Open(path!);
        var adapter = new QwenAsrLlmTensorSource(inner);

        Assert.True(ModelCompatibility.IsTextGenerationArchitectureSupported(
            (string)adapter.Metadata["general.architecture"]));
    }

    /// <summary>
    /// The real end-to-end claim: construct an actual <see cref="ForwardPass"/> (the same
    /// engine class every text-generation GGUF in this repo runs through) from the adapter and
    /// run a real prefill against real Q4_K/Q8_0-quantized checkpoint weights. This is the first
    /// time this adapter has been exercised through the Engine rather than just inspected --
    /// see docs/audio-review-progress.md's QwenASR section, "next iteration" item (1).
    /// </summary>
    [Fact]
    public void Adapter_RunsRealForwardPass_ProducesFiniteNonDegenerateLogits()
    {
        string? path = FindRepoFile("models/qwen3-asr-0.6b-q4_k.gguf");
        Assert.SkipUnless(path != null, "models/qwen3-asr-0.6b-q4_k.gguf not found");

        using var inner = GgufModel.Open(path!);
        var adapter = new QwenAsrLlmTensorSource(inner);

        var hp = ModelHyperparams.FromGgufMetadata(adapter.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(adapter, backend, hp);

        // Arbitrary small, well-within-vocab token ids -- this is a structural/sanity check
        // (does the real Qwen3 GQA+QK-norm+RoPE+SwiGLU forward pass run this checkpoint's
        // weights without crashing and produce a sane distribution), NOT a golden-transcript
        // test -- no tokenizer/prompt-formatting/audio-injection exists yet to make a real
        // prompt meaningful.
        var prompt = new int[] { 785, 3974, 13876, 25, 1863 };
        var logits = fwd.Prefill(prompt).ToArray();

        int vocabSize = System.Convert.ToInt32(adapter.Metadata["qwen3.vocab_size"]);
        Assert.Equal(vocabSize, logits.Length);

        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        foreach (var v in logits)
        {
            Assert.False(float.IsNaN(v) || float.IsInfinity(v), "logit is NaN/Inf");
            if (v < min) min = v;
            if (v > max) max = v;
        }
        // A real forward pass through real weights should produce a spread of logit values,
        // not a degenerate constant (which would indicate the adapter fed garbage/zeroed
        // tensors through and the engine silently produced a flat, meaningless distribution).
        Assert.True(max - min > 1.0f, $"logits look degenerate: min={min}, max={max}");
    }

    /// <summary>
    /// The real audio-conditioning claim, mirroring CosyVoiceLlmTensorSourceTests's
    /// SpeechGenerationMode test: builds a synthetic combined embedding table (real dequantized
    /// text rows + synthetic "audio frame" rows standing in for the AuT encoder's real output),
    /// prefills a sequence mixing real text token ids with audio-frame ids offset via
    /// AudioTokenIdOffset, and confirms the real Qwen3 forward pass runs end-to-end and produces
    /// finite, non-degenerate logits over the real (unswapped) text vocabulary.
    /// </summary>
    [Fact]
    public void AudioConditioning_RunsRealForwardPass_ProducesFiniteNonDegenerateLogits()
    {
        string? path = FindRepoFile("models/qwen3-asr-0.6b-q4_k.gguf");
        Assert.SkipUnless(path != null, "models/qwen3-asr-0.6b-q4_k.gguf not found");

        using var inner = GgufModel.Open(path!);
        var adapter = new QwenAsrLlmTensorSource(inner);

        int hiddenDim = System.Convert.ToInt32(adapter.Metadata["qwen3.embedding_length"]);
        int textVocab = System.Convert.ToInt32(adapter.Metadata["qwen3.vocab_size"]);
        const int numAudioTokens = 5;
        var audioEmbeddings = new float[numAudioTokens * hiddenDim];
        var rng = new System.Random(7);
        for (int i = 0; i < audioEmbeddings.Length; i++) audioEmbeddings[i] = (float)(rng.NextDouble() * 0.1 - 0.05);

        adapter.EnableAudioConditioning(audioEmbeddings, numAudioTokens);

        Assert.Equal(textVocab, adapter.AudioTokenIdOffset);
        var embedTensor = adapter.FindTensor("token_embd.weight");
        Assert.NotNull(embedTensor);
        Assert.Equal(textVocab + numAudioTokens, embedTensor!.Value.Dimensions[1]);

        var hp = ModelHyperparams.FromGgufMetadata(adapter.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(adapter, backend, hp);

        // Two real text tokens, then two "audio frame" ids offset into the synthetic table.
        var prompt = new int[] { 785, 3974, adapter.AudioTokenIdOffset, adapter.AudioTokenIdOffset + 1 };
        var logits = fwd.Prefill(prompt).ToArray();

        Assert.Equal(textVocab, logits.Length);
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
