using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Audio.VoxCpm2;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real, live end-to-end smoke test for VoxCPM2's MiniCPM backbone: real checkpoint, real
/// `ForwardPass.Prefill` (token ids) followed by a real `ForwardPass.ForwardEmbedding` (a
/// precomputed embedding vector, standing in for a real DiT-generated patch feature) chained
/// together, confirming `LastHidden` -- the mechanism VoxCPM2's downstream DiT/FSQ heads would
/// consume -- is populated and finite after each. This model (~1.5B) is far smaller than
/// VibeVoice ASR's 7B-class checkpoint, so a live run is attempted here (unlike VibeVoice ASR,
/// where the FP32-dequantized memory footprint OOM'd this environment).
/// </summary>
public sealed class VoxCpm2LlmForwardEmbeddingRealWeightsTests : HeavyTestBase
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

    private const int NumLayers = 28, HiddenDim = 2048, NumHeads = 16, NumKvHeads = 2, HeadDim = 128;
    private const int FfDim = 6144, VocabSize = 73448;
    private const float RopeTheta = 10000f, RmsNormEps = 1e-5f;

    [Fact]
    public void PrefillThenForwardEmbedding_OnRealCheckpoint_ProducesFiniteHiddenStates()
    {
        string? path = FindRepoFile("models/_models/voxcpm2/VoxCPM2-GGUF/voxcpm2-q8_0.gguf");
        Assert.SkipUnless(path != null, "voxcpm2-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        var llm = new VoxCpm2LlmTensorSource(source, NumLayers, HiddenDim, NumHeads, NumKvHeads, HeadDim, FfDim, VocabSize, RopeTheta, RmsNormEps, embeddingScale: 1f, residualScale: 1f, logitScale: 1f);

        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        // A few real, in-range token ids (BOS + arbitrary small ids) as a stand-in prompt.
        var prompt = new[] { 1, 100, 200, 300 };
        var logits = fwd.Prefill(prompt).ToArray();
        Assert.Equal(VocabSize, logits.Length);
        Assert.All(logits, v => Assert.True(float.IsFinite(v)));

        Assert.True(fwd.SupportsEmbeddingInput);
        var lastHiddenAfterPrefill = fwd.LastHidden.ToArray();
        Assert.Equal(HiddenDim, lastHiddenAfterPrefill.Length);
        Assert.All(lastHiddenAfterPrefill, v => Assert.True(float.IsFinite(v)));

        // A real precomputed embedding (stand-in for a real DiT-generated patch feature) fed at
        // the next position via ForwardEmbedding -- exactly VoxCPM2's real per-step mechanism.
        var rng = new Random(5);
        var embedding = new float[HiddenDim];
        for (int i = 0; i < embedding.Length; i++) embedding[i] = (float)(rng.NextDouble() * 0.1 - 0.05);

        var logitsFromEmbedding = fwd.ForwardEmbedding(embedding, position: prompt.Length).ToArray();
        Assert.Equal(VocabSize, logitsFromEmbedding.Length);
        Assert.All(logitsFromEmbedding, v => Assert.True(float.IsFinite(v)));

        var lastHiddenAfterEmbedding = fwd.LastHidden.ToArray();
        Assert.Equal(HiddenDim, lastHiddenAfterEmbedding.Length);
        Assert.All(lastHiddenAfterEmbedding, v => Assert.True(float.IsFinite(v)));
    }
}
