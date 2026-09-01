using OpenTail.Stingray.Diffusion.LTXVideo;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Transformer numeric parity check at the token count a REAL 256x256 generation actually uses
/// (H=8, W=8 patches -> 64 tokens), distinct from <see cref="LtxVideoGoldenParityTests"/>'s H=4/W=4
/// (32-token) scenario. Written 2026-09-01 while investigating a real end-to-end generation
/// producing heavy salt-and-pepper corruption despite every existing golden test (tiny-scale
/// transformer, VAE at F=1/F=2, multi-step scheduler+CFG trajectory) passing at or near machine
/// precision -- testing the hypothesis that a scale-dependent bug (RoPE continuous-coordinate grid,
/// attention numerics with more tokens) only shows up at real generation size.
/// </summary>
public sealed class LtxVideoRealScaleGoldenTests
{
    private const string ModelFileName = "ltx-video-2b-v0.9.1.safetensors";

    private static string? FindModelPath(string fileName)
    {
        string[] absoluteCandidates =
        {
            $@"C:\Git-Public\OpenTail.Stingray\models\{fileName}",
            $@"C:\p\opentail-llm\models\{fileName}",
            $@"E:\models\{fileName}",
        };
        foreach (var p in absoluteCandidates)
            if (File.Exists(p)) return p;

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

    private static string? FindGoldenDir()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "tests", "OpenTail.Stingray.Tests.Diffusion", "TestData", "LtxRealScaleGolden");
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static float[] ReadBin(string dir, string name)
    {
        var bytes = File.ReadAllBytes(Path.Combine(dir, name + ".bin"));
        var arr = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, arr, 0, bytes.Length);
        return arr;
    }

    private static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        return (float)(dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12));
    }

    [Fact]
    public void LtxVideoModel_MatchesRealReference_AtRealGenerationScale_64Tokens()
    {
        string? modelPath = FindModelPath(ModelFileName);
        string? goldenDir = FindGoldenDir();
        if (modelPath is null || goldenDir is null) return;

        using var loader = SafetensorsLoader.Open(modelPath);
        var model = new LtxVideoModel(loader);

        var latents = ReadBin(goldenDir, "latents"); // [64,128]
        var caption = ReadBin(goldenDir, "caption"); // [16,4096]
        var goldenOutput = ReadBin(goldenDir, "output"); // [64,128]

        int numFrames = 1, patchH = 8, patchW = 8;
        float timestep = 999f;

        var output = model.Forward(latents, timestep, caption, numFrames, patchH, patchW);

        Assert.Equal(goldenOutput.Length, output.Length);
        float cos = CosineSimilarity(output, goldenOutput);
        Assert.True(cos > 0.999999f, $"real-scale (64 token) full-forward cosine-sim too low: {cos}");
    }
}
