
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Same hypothesis test as <see cref="QwenTtsTalkerForwardPassTests"/>, for the Code Predictor:
/// confirms its real 5-layer transformer (same shape family as the Talker -- GQA, per-head
/// QK-RMSNorm, NEOX RoPE, SwiGLU -- confirmed via `list-tensors`) also runs through this
/// codebase's existing, UNMODIFIED `ForwardPass` engine via
/// <see cref="QwenTtsCodePredictorTensorSource"/>. Construction/shape/finite smoke test only, not
/// a numerical correctness claim -- see docs/audio-review-progress.md's QwenTTS entries.
/// </summary>
public sealed class QwenTtsCodePredictorForwardPassTests : HeavyTestBase
{
    private static string? FindModelPath(string fileName)
    {
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
    public void ForwardPass_ConstructsAndRuns_AgainstRealQwenTtsCodePredictorGguf()
    {
        string? modelPath = FindModelPath("qwen-talker-0.6b-base-Q8_0.gguf");
        Assert.SkipUnless(modelPath != null, "models/qwen-talker-0.6b-base-Q8_0.gguf not found");

        using var model = GgufModel.Open(modelPath!);
        var tensorSource = new QwenTtsCodePredictorTensorSource(model, numLayers: 5);
        var hp = ModelHyperparams.FromGgufMetadata(tensorSource.Metadata, tensorSource);

        Assert.Equal(5, hp.NumLayers);
        Assert.Equal(1024, hp.EmbeddingDim);
        Assert.Equal(16, hp.NumHeads);
        Assert.Equal(8, hp.NumKvHeads);
        Assert.Equal(128, hp.HeadDim);
        Assert.True(hp.IsNeoxRope, "Code Predictor RoPE should resolve to NEOX (same qwen3-tts architecture as the Talker)");

        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(tensorSource, backend, hp, maxContextLength: 32);

        var codecEmbd0 = tensorSource.FindTensor("token_embd.weight");
        Assert.NotNull(codecEmbd0);
        var embdBytes = tensorSource.GetTensorData(codecEmbd0!.Value);
        int bytesPerRow = (hp.EmbeddingDim / 32) * 34; // real Q8_0 block size: 34 bytes per 32-element block
        var row0 = new float[hp.EmbeddingDim];
        Dequantize.ToFloat32(embdBytes[..bytesPerRow], row0, codecEmbd0.Value.DType, hp.EmbeddingDim);

        var logits = fwd.ForwardEmbedding(row0, position: 0);

        Assert.True(logits.Length > 0, "Code Predictor produced no logits");
        foreach (var v in logits)
            Assert.True(float.IsFinite(v), "Code Predictor logits contained a non-finite value");
    }
}
