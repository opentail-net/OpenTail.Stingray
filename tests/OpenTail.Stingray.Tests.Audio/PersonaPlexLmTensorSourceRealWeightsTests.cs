using OpenTail.Stingray.Audio.PersonaPlex;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real-weight smoke tests for <see cref="PersonaPlexLmTensorSource"/>: tensor shape
/// resolution (cheap) and a live `ForwardPass.Prefill` if the environment has enough RAM.</summary>
public sealed class PersonaPlexLmTensorSourceRealWeightsTests : HeavyTestBase
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

    // Real numbers confirmed from the checkpoint's own dumped tensor shapes (2026-09-07, not
    // guessed): hiddenDim/numLayers/ffDim/textVocabSize/lmCodebooks/audioCodebookSize all
    // verified against real tensor dims. ropeTheta/rmsNormEps are NOT independently confirmed
    // (config.json carries only model_type/version for this checkpoint) -- kept at the real
    // reference's own hardcoded defaults (assets.h) since no other real source exists.
    private const int NumLayers = 32, HiddenDim = 4096, NumHeads = 32, HeadDim = 128;
    private const int FfDim = 11264, TextVocabSize = 32000, LmCodebooks = 16, AudioCodebookSize = 2048;
    private const float RopeTheta = 10000f, RmsNormEps = 1e-8f;

    [Fact]
    public void FindTensor_OnRealCheckpoint_ResolvesRealShapes()
    {
        string? path = FindRepoFile("models/_models/personaplex/PersonaPlex-GGUF/personaplex-7b-v1-q8_0.gguf");
        Assert.SkipUnless(path != null, "personaplex-7b-v1-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        using var llm = new PersonaPlexLmTensorSource(source, NumLayers, HiddenDim, NumHeads, HeadDim, FfDim, TextVocabSize, LmCodebooks, AudioCodebookSize, RopeTheta, RmsNormEps);

        var tokenEmbd = llm.FindTensor("token_embd.weight");
        Assert.NotNull(tokenEmbd);
        Assert.Equal([HiddenDim, TextVocabSize + 1], tokenEmbd!.Value.Dimensions);

        var outputWeight = llm.FindTensor("output.weight");
        Assert.NotNull(outputWeight);
        Assert.Equal([HiddenDim, TextVocabSize], outputWeight!.Value.Dimensions);

        var outputNorm = llm.FindTensor("output_norm.weight");
        Assert.NotNull(outputNorm);
        Assert.Equal([HiddenDim], outputNorm!.Value.Dimensions);

        int qkvOut = NumHeads * HeadDim;
        var layer0Q = llm.FindTensor("blk.0.attn_q.weight");
        Assert.NotNull(layer0Q);
        Assert.Equal([HiddenDim, qkvOut], layer0Q!.Value.Dimensions);
        var layer0K = llm.FindTensor("blk.0.attn_k.weight");
        Assert.NotNull(layer0K);
        Assert.Equal([HiddenDim, qkvOut], layer0K!.Value.Dimensions);
        var layer0V = llm.FindTensor("blk.0.attn_v.weight");
        Assert.NotNull(layer0V);
        Assert.Equal([HiddenDim, qkvOut], layer0V!.Value.Dimensions);
        var layer0Out = llm.FindTensor("blk.0.attn_output.weight");
        Assert.NotNull(layer0Out);
        Assert.Equal([qkvOut, HiddenDim], layer0Out!.Value.Dimensions);

        var layer0Gate = llm.FindTensor("blk.0.ffn_gate.weight");
        Assert.NotNull(layer0Gate);
        Assert.Equal([HiddenDim, FfDim], layer0Gate!.Value.Dimensions);
        var layer0Up = llm.FindTensor("blk.0.ffn_up.weight");
        Assert.NotNull(layer0Up);
        Assert.Equal([HiddenDim, FfDim], layer0Up!.Value.Dimensions);
        var layer0Down = llm.FindTensor("blk.0.ffn_down.weight");
        Assert.NotNull(layer0Down);
        Assert.Equal([FfDim, HiddenDim], layer0Down!.Value.Dimensions);

        var lastLayerNorm = llm.FindTensor($"blk.{NumLayers - 1}.attn_norm.weight");
        Assert.NotNull(lastLayerNorm);
        Assert.Equal([HiddenDim], lastLayerNorm!.Value.Dimensions);

        // No per-head q/k norm (use_qk_norm=false in the real reference).
        Assert.Null(llm.FindTensor("blk.0.attn_q_norm.weight"));

        // Real, fully separate per-codebook audio embedding tables.
        var audioEmbed0 = llm.AudioEmbeddingWeight(0);
        Assert.Equal((AudioCodebookSize + 1) * HiddenDim, audioEmbed0.Length);
        Assert.All(audioEmbed0, v => Assert.True(float.IsFinite(v)));
        var audioEmbed15 = llm.AudioEmbeddingWeight(15);
        Assert.Equal((AudioCodebookSize + 1) * HiddenDim, audioEmbed15.Length);
    }

    [Fact]
    public void Prefill_OnRealCheckpoint_ProducesFiniteLogits()
    {
        string? path = FindRepoFile("models/_models/personaplex/PersonaPlex-GGUF/personaplex-7b-v1-q8_0.gguf");
        Assert.SkipUnless(path != null, "personaplex-7b-v1-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        using var llm = new PersonaPlexLmTensorSource(source, NumLayers, HiddenDim, NumHeads, HeadDim, FfDim, TextVocabSize, LmCodebooks, AudioCodebookSize, RopeTheta, RmsNormEps);

        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        var prompt = new[] { 1, 100, 200, 300 };
        var logits = fwd.Prefill(prompt).ToArray();
        Assert.Equal(TextVocabSize, logits.Length);
        Assert.All(logits, v => Assert.True(float.IsFinite(v)));
    }
}
