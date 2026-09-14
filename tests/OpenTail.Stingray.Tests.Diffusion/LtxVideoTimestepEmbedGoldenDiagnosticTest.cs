using OpenTail.Stingray.Diffusion.LTXVideo;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Diagnostic test (docs/077, 2026-09-14): the existing golden fixture set
/// (TestData/LtxGolden/manifest.json) includes "embedded_timestep" [1,1,2048] and "temb"
/// [1,1,12288] dumps that <see cref="LtxVideoGoldenParityTests"/> never actually asserts against
/// (only rope/proj_in/caption_proj/block0/full_out are checked there) -- <see cref="LtxVideoModel"/>
/// already exposes both via <c>LastEmbeddedTimestep</c>/<c>LastTimestepProj</c>, so this closes that
/// verification gap to help bisect the block0 divergence (cosine-sim 0.9893641, see docs/077's
/// 2026-09-14 update) into either "before AdaLN" (timestep embedding branch) or "inside the block
/// itself" (attention/FFN math).
/// </summary>
public sealed class LtxVideoTimestepEmbedGoldenDiagnosticTest
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
            var p = Path.Combine(dir, "tests", "OpenTail.Stingray.Tests.Diffusion", "TestData", "LtxGolden");
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

    private static float MaxAbsDiff(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float max = 0f;
        for (int i = 0; i < a.Length; i++)
            max = MathF.Max(max, MathF.Abs(a[i] - b[i]));
        return max;
    }

    [Fact]
    public void TimestepEmbedding_And_AdaLNProjection_MatchGoldenReference()
    {
        string? modelPath = FindModelPath(ModelFileName);
        string? goldenDir = FindGoldenDir();
        if (modelPath is null || goldenDir is null) return; // skip: needs local checkpoint + fixtures

        using var loader = SafetensorsLoader.Open(modelPath);
        var model = new LtxVideoModel(loader);

        var latents = ReadBin(goldenDir, "latents");
        var caption = ReadBin(goldenDir, "caption");
        var goldenEmbeddedTimestep = ReadBin(goldenDir, "embedded_timestep");
        var goldenTemb = ReadBin(goldenDir, "temb");

        int numFrames = 2, patchH = 4, patchW = 4;
        float timestep = 500f;

        _ = model.Forward(latents, timestep, caption, numFrames, patchH, patchW);

        float embCos = CosineSimilarity(model.LastEmbeddedTimestep!, goldenEmbeddedTimestep);
        float embMaxDiff = MaxAbsDiff(model.LastEmbeddedTimestep!, goldenEmbeddedTimestep);
        Console.WriteLine($"[LTX embedded_timestep] cosine-sim={embCos:F7} maxAbsDiff={embMaxDiff:F6}");

        float tembCos = CosineSimilarity(model.LastTimestepProj!, goldenTemb);
        float tembMaxDiff = MaxAbsDiff(model.LastTimestepProj!, goldenTemb);
        Console.WriteLine($"[LTX temb/timestepProj] cosine-sim={tembCos:F7} maxAbsDiff={tembMaxDiff:F6}");

        Assert.True(embCos > 0.9999f, $"embedded_timestep cosine-sim too low: {embCos}");
        Assert.True(tembCos > 0.9999f, $"temb (adaln_single.linear out) cosine-sim too low: {tembCos}");
    }
}
