using OpenTail.Stingray.Diffusion.LTXVideo;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Multi-frame (F=2) golden parity check for <see cref="LtxVaeDecoder"/>, distinct from the
/// existing F=1 <see cref="LtxVaeDecoderGoldenParityTests"/> -- with F=1, every compress_all
/// stage's "drop the first upsampled frame" trim is trivially self-correcting (newF=2, trimmedF=1
/// regardless of whether the trim index is actually right), so F=1 alone cannot catch a real
/// multi-frame temporal-trim bug. Investigating the reported "LTX-Video output is visually wrong"
/// finding (2026-09-01) starting from the one real path (F&gt;1 decode) the existing golden suite
/// never actually exercised.
/// </summary>
public sealed class LtxVaeDecoderMultiFrameGoldenTests
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
            var p = Path.Combine(dir, "tests", "OpenTail.Stingray.Tests.Diffusion", "TestData", "LtxVaeGoldenMultiFrame");
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
    public void LtxVaeDecoder_MultiFrame_MatchesRealReference()
    {
        string? modelPath = FindModelPath(ModelFileName);
        string? goldenDir = FindGoldenDir();
        if (modelPath is null || goldenDir is null) return;

        using var loader = SafetensorsLoader.Open(modelPath);
        var decoder = new LtxVaeDecoder(loader);

        var latents = ReadBin(goldenDir, "latents"); // [128,2,2,2]
        var goldenOutput = ReadBin(goldenDir, "output"); // [3,9,64,64]

        var output = decoder.Decode(latents, decodeTimestep: 0f, f: 2, h: 2, w: 2, injectNoise: false);

        Assert.Equal(goldenOutput.Length, output.Length);

        float cos = CosineSimilarity(output, goldenOutput);
        Assert.True(cos > 0.999f, $"Multi-frame VAE decode cosine-sim too low: {cos}");
    }
}
