using OpenTail.Stingray.Diffusion.LTXVideo;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Numeric parity check of <see cref="LtxVaeDecoder"/> against a golden decode dumped from the
/// REAL, official Lightricks `ltx-video` PyPI package's `CausalVideoAutoencoder`, loaded with the
/// real `ltx-video-2b-v0.9.1.safetensors` checkpoint weights and run through its actual
/// `decode(latents, timestep=0)` path (F=1, H=2, W=2 latent -> [3,1,64,64] pixel output). This is
/// the real native source the checkpoint was trained with (not diffusers, which does not match this
/// checkpoint's VAE architecture -- see docs/055-ltx-video-implementation-plan.md).
/// </summary>
public sealed class LtxVaeDecoderGoldenParityTests
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
            var p = Path.Combine(dir, "tests", "OpenTail.Stingray.Tests.Diffusion", "TestData", "LtxVaeGolden");
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
    public void LtxVaeDecoder_MatchesRealLtxVideoPackageReference()
    {
        string? modelPath = FindModelPath(ModelFileName);
        string? goldenDir = FindGoldenDir();
        if (modelPath is null || goldenDir is null) return; // skip: needs local checkpoint + fixtures

        using var loader = SafetensorsLoader.Open(modelPath);
        var decoder = new LtxVaeDecoder(loader);

        var latents = ReadBin(goldenDir, "latents"); // [128,1,2,2]
        var goldenOutput = ReadBin(goldenDir, "output"); // [3,1,64,64]

        var output = decoder.Decode(latents, decodeTimestep: 0f, f: 1, h: 2, w: 2, injectNoise: false);

        Assert.Equal(goldenOutput.Length, output.Length);

        // Observed near machine-precision match (>0.999999) with noise injection disabled on both
        // sides -- kept at 0.999 (not tighter) to tolerate float32 accumulation-order differences
        // across the 7-stage decoder without becoming a flaky test.
        float cos = CosineSimilarity(output, goldenOutput);
        Assert.True(cos > 0.999f, $"VAE decode cosine-sim too low: {cos}");
    }
}
