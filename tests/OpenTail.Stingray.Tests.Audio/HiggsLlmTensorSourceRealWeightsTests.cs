using OpenTail.Stingray.Audio.HiggsAudio;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real-weight smoke tests for <see cref="HiggsLlmTensorSource"/>: tensor shape
/// resolution against the real checkpoint (cheap, no forward pass), and a live
/// `ForwardPass.Prefill` if the environment has enough RAM for this 4B-class model's
/// FP32-dequantized footprint.</summary>
public sealed class HiggsLlmTensorSourceRealWeightsTests : HeavyTestBase
{
    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    // Real config.json numbers for the `lm` sub-config (dumped 2026-09-07, not guessed).
    private const int NumLayers = 36, HiddenDim = 2560, NumHeads = 32, NumKvHeads = 8, HeadDim = 128;
    private const int FfDim = 9728, VocabSize = 151936, NumCodebooks = 8, AudioVocabSize = 1026;
    private const float RopeTheta = 1_000_000f, RmsNormEps = 1e-6f;

    [Fact]
    public void FindTensor_OnRealCheckpoint_ResolvesRealShapes()
    {
        string? path = FindRepoFile("models/_models/higgs_audio_tts/Higgs-Audio-v3-TTS-4B-GGUF/higgs-audio-v3-tts-4b-q8_0.gguf");
        Assert.SkipUnless(path != null, "higgs-audio-v3-tts-4b-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        using var llm = new HiggsLlmTensorSource(source, NumLayers, HiddenDim, NumHeads, NumKvHeads, HeadDim, FfDim, VocabSize, RopeTheta, RmsNormEps, NumCodebooks, AudioVocabSize);

        var tokenEmbd = llm.FindTensor("token_embd.weight");
        Assert.NotNull(tokenEmbd);
        Assert.Equal([HiddenDim, VocabSize], tokenEmbd!.Value.Dimensions);

        var outputNorm = llm.FindTensor("output_norm.weight");
        Assert.NotNull(outputNorm);
        Assert.Equal([HiddenDim], outputNorm!.Value.Dimensions);

        var layer0Q = llm.FindTensor("blk.0.attn_q.weight");
        Assert.NotNull(layer0Q);
        Assert.Equal([HiddenDim, NumHeads * HeadDim], layer0Q!.Value.Dimensions);

        var layer0QNorm = llm.FindTensor("blk.0.attn_q_norm.weight");
        Assert.NotNull(layer0QNorm);
        Assert.Equal([HeadDim], layer0QNorm!.Value.Dimensions);

        var lastLayerDown = llm.FindTensor($"blk.{NumLayers - 1}.ffn_down.weight");
        Assert.NotNull(lastLayerDown);
        Assert.Equal([FfDim, HiddenDim], lastLayerDown!.Value.Dimensions);

        // No separate lm_head -- tied embeddings.
        Assert.Null(llm.FindTensor("output.weight"));

        // Real, separate modality-embedding table (audio-codebook tokens).
        var modality = llm.ModalityEmbeddingWeight;
        Assert.Equal(NumCodebooks * AudioVocabSize * HiddenDim, modality.Length);
        Assert.All(modality, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void Prefill_OnRealCheckpoint_ProducesFiniteLogits()
    {
        string? path = FindRepoFile("models/_models/higgs_audio_tts/Higgs-Audio-v3-TTS-4B-GGUF/higgs-audio-v3-tts-4b-q8_0.gguf");
        Assert.SkipUnless(path != null, "higgs-audio-v3-tts-4b-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        using var llm = new HiggsLlmTensorSource(source, NumLayers, HiddenDim, NumHeads, NumKvHeads, HeadDim, FfDim, VocabSize, RopeTheta, RmsNormEps, NumCodebooks, AudioVocabSize);

        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        var prompt = new[] { 1, 100, 200, 300 };
        var logits = fwd.Prefill(prompt).ToArray();
        Assert.Equal(VocabSize, logits.Length);
        Assert.All(logits, v => Assert.True(float.IsFinite(v)));
    }
}
