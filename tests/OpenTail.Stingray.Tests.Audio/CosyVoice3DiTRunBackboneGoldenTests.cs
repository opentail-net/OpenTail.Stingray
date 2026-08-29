
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for <see cref="CosyVoice3DiTModel.RunBackbone"/> -- the
/// full real 22-layer transformer backbone (AdaLN-modulated attention+FFN blocks, RoPE,
/// timestep embedding, final norm_out/proj_out). Real oracle
/// (`scratch-llamacpp-ref/cosyvoice3_dit_runbackbone_golden.py`) transcribes the exact same math
/// already used by this codebase's real, golden-verified F5-TTS `F5Kernels`/`F5RotaryEmbedding`
/// (interleaved-pairs RoPE, non-causal softmax attention, AdaLN scale/shift/gate split) against
/// CosyVoice3's own real GGUF weights across all 22 real layers.
/// </summary>
public sealed class CosyVoice3DiTRunBackboneGoldenTests : HeavyTestBase
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
    public void RunBackbone_RealWeights_MatchesGoldenOracle()
    {
        string? modelPath = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        Assert.SkipUnless(modelPath != null, "models/cosyvoice3/CosyVoice3-2512_F16.gguf not found");

        string? inputPath = FindRepoFile("scratch-llamacpp-ref/cosyvoice3_dit_runbackbone_golden_input.txt");
        string? outputPath = FindRepoFile("scratch-llamacpp-ref/cosyvoice3_dit_runbackbone_golden_output.txt");
        Assert.SkipUnless(inputPath != null && outputPath != null, "golden DiT RunBackbone fixture not found");

        var inLines = File.ReadAllText(inputPath!).Split('\n');
        int t = int.Parse(inLines[0].Trim());
        float timestep = float.Parse(inLines[1].Trim());
        var hIn = Array.ConvertAll(inLines[2].Trim().Split(','), float.Parse);

        var outLines = File.ReadAllText(outputPath!).Split('\n');
        var dims = outLines[0].Trim().Split(',');
        int goldenT = int.Parse(dims[0]);
        int goldenDim = int.Parse(dims[1]);
        var golden = Array.ConvertAll(outLines[1].Trim().Split(','), float.Parse);

        Assert.Equal(t, goldenT);
        Assert.Equal(CosyVoice3DiTWeights.MelDim, goldenDim);

        using var model = GgufModel.Open(modelPath!);
        var w = new CosyVoice3DiTWeights(model);

        var velocity = CosyVoice3DiTModel.RunBackbone(w, hIn, timestep, t);

        Assert.Equal(golden.Length, velocity.Length);

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < velocity.Length; i++)
        {
            dot += velocity[i] * golden[i];
            normA += velocity[i] * velocity[i];
            normB += golden[i] * golden[i];
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.999, $"cosine similarity {cosine} too low vs golden RunBackbone output");
    }
}
