
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for the Fish Speech slow-AR trunk's layer 0 -- compares
/// against `scratch-llamacpp-ref/fish_speech_partial_golden.py`, which fetches ONLY the real
/// layer-0 weights + a few embedding rows directly from the real `fishaudio/s2-pro` HF
/// safetensors file via byte-range HTTP requests (no full 9.1GB download), and computes the
/// real math directly in numpy. This is the test that validates this fire's two real bug fixes
/// (head_dim=128, interleaved RoPE for the fast-AR -- this test exercises the slow-AR's
/// ForwardPass-driven trunk, confirming the head_dim fix and the pre-existing-correct NORM/
/// interleaved RoPE convention both hold under real weights).
///
/// Note: the local GGUF is Q4_K_M-quantized (not the same bytes as the real BF16 safetensors
/// oracle), so exact bit-parity isn't expected -- but architecture/math correctness should still
/// clear the same >0.99 cosine-similarity bar used everywhere else in this doc, since Q4_K_M is
/// a high-fidelity quantization.
/// </summary>
public sealed class FishSpeechSlowArTests : HeavyTestBase
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
    public void Layer0Output_RealWeights_MatchesGoldenOracle()
    {
        string? modelPath = FindModelPath("s2-pro-q4_k_m.gguf");
        Assert.SkipUnless(modelPath != null, "models/s2-pro-q4_k_m.gguf not found");

        string? idsPath = FindRepoFile("scratch-llamacpp-ref/fishspeech_golden_token_ids.txt");
        string? outPath = FindRepoFile("scratch-llamacpp-ref/fishspeech_golden_layer0_output.txt");
        Assert.SkipUnless(idsPath != null && outPath != null,
            "golden Fish Speech layer-0 files not found (re-run scratch-llamacpp-ref/fish_speech_partial_golden.py)");

        var idsCsv = File.ReadAllText(idsPath!).Trim().Split(',');
        var tokenIds = new int[idsCsv.Length];
        for (int i = 0; i < idsCsv.Length; i++) tokenIds[i] = int.Parse(idsCsv[i]);

        var lines = File.ReadAllText(outPath!).Split('\n');
        var dims = lines[0].Trim().Split(',');
        int goldenT = int.Parse(dims[0]);
        int goldenDim = int.Parse(dims[1]);
        var goldenParts = lines[1].Trim().Split(',');
        Assert.Equal(goldenT * goldenDim, goldenParts.Length);
        var golden = new float[goldenT * goldenDim];
        for (int i = 0; i < golden.Length; i++) golden[i] = float.Parse(goldenParts[i]);

        using var model = GgufModel.Open(modelPath!);
        var source = new FishSpeechTensorSource(model, numLayers: 36);
        var hp = ModelHyperparams.FromGgufMetadata(source.Metadata, source);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(source, backend, hp, maxContextLength: 512);
        fwd.EnableHiddenTaps([0]); // layer 0's output

        using var weights = new FishSpeechWeights(modelPath!);

        fwd.ResetCache();
        for (int pos = 0; pos < tokenIds.Length; pos++)
        {
            var emb = new float[weights.EmbeddingDim];
            Array.Copy(weights.Embeddings, (long)tokenIds[pos] * weights.EmbeddingDim, emb, 0, weights.EmbeddingDim);
            fwd.ForwardEmbedding(emb, pos);
        }

        double dot = 0, normA = 0, normB = 0;
        for (int t = 0; t < goldenT; t++)
        {
            var tap = fwd.HiddenTapsAt(t);
            Assert.Equal(goldenDim, tap.Length);
            for (int d = 0; d < goldenDim; d++)
            {
                float a = tap[d];
                float b = golden[t * goldenDim + d];
                dot += a * b;
                normA += a * a;
                normB += b * b;
            }
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.99, $"cosine similarity {cosine} too low vs golden Fish Speech layer-0 output");
    }
}
