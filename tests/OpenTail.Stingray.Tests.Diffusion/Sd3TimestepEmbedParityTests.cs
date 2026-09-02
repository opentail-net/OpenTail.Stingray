using System.Runtime.InteropServices;
using System.Text.Json;
using OpenTail.Stingray.Diffusion.SD3;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Parity of <see cref="MMDiTModel.ComputeTimeAndPooledEmbedding"/> against a numpy reference
/// (scripts/sd3_timestep_embed_ref.py), which faithfully reimplements
/// examples/diffusers/src/diffusers/models/embeddings.py's CombinedTimestepTextProjEmbeddings.
/// This isolates the SINGLE conditioning vector every joint block's AdaLN modulation depends on --
/// a bug here would corrupt every block uniformly, unlike a localized per-block bug, and is the
/// first stage worth golden-verifying per docs/057-sd35-performance-handoff.md's "RESOLVED" note
/// (SD3.5 output is still pure noise at 20 steps; 5 previously-fixed bugs were reasoned from
/// source, never numerically golden-verified). Does NOT require the CLIP/T5 text encoders or VAE --
/// only the small t_embedder/y_embedder MLP weights already present in the DiT GGUF -- so this is
/// far cheaper to generate/run than a full pipeline comparison.
///
/// Requires a real SD3.5-medium DiT GGUF (STINGRAY_SD3_DIT_PATH env var) -- not vendored per this
/// project's convention (docs/057's "How to reproduce" section has the exact download command:
/// city96/stable-diffusion-3.5-medium-gguf, sd3.5_medium-Q8_0.gguf). Skips (returns) if unset or
/// the fixture from scripts/sd3_timestep_embed_ref.py hasn't been generated.
/// </summary>
public class Sd3TimestepEmbedParityTests
{
    private static float[] ReadF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        return MemoryMarshal.Cast<byte, float>(bytes).ToArray();
    }

    private static string? FindFixtureDir()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8 && dir is not null; i++)
        {
            var p = Path.Combine(dir, "tests", "fixtures", "sd3_timestep_embed");
            if (Directory.Exists(p)) return p;
            dir = Directory.GetParent(dir)?.FullName;
        }
        return null;
    }

    [Fact]
    public void ComputeTimeAndPooledEmbedding_MatchesNumpyReference()
    {
        string? ditPath = Environment.GetEnvironmentVariable("STINGRAY_SD3_DIT_PATH");
        string? fx = FindFixtureDir();
        if (ditPath is null || !File.Exists(ditPath) || fx is null) return;

        var outPath = Path.Combine(fx, "conditioning.f32");
        var metaPath = Path.Combine(fx, "meta.json");
        if (!File.Exists(outPath) || !File.Exists(metaPath)) return;

        using var doc = JsonDocument.Parse(File.ReadAllText(metaPath));
        float timestep = (float)doc.RootElement.GetProperty("timestep").GetDouble();
        int hiddenSize = doc.RootElement.GetProperty("hiddenSize").GetInt32();
        int admInChannels = doc.RootElement.GetProperty("admInChannels").GetInt32();
        var pooledEl = doc.RootElement.GetProperty("pooled");
        var pooled = new float[admInChannels];
        int idx = 0;
        foreach (var e in pooledEl.EnumerateArray()) pooled[idx++] = (float)e.GetDouble();

        float[] golden = ReadF32(outPath);
        Assert.Equal(hiddenSize, golden.Length);

        IWeightLoader loader = ditPath.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
            ? GgufWeightLoader.Open(ditPath)
            : SafetensorsLoader.Open(ditPath);
        using var mmdit = new MMDiTModel(loader, prefix: "", hiddenSize: hiddenSize, admInChannels: admInChannels);

        var got = mmdit.ComputeTimeAndPooledEmbedding(timestep, pooled);
        Assert.Equal(hiddenSize, got.Length);

        double maxAbs = 0, sumAbs = 0, dot = 0, na = 0, nb = 0;
        for (int i = 0; i < hiddenSize; i++)
        {
            double a = got[i], b = golden[i];
            double d = Math.Abs(a - b);
            if (d > maxAbs) maxAbs = d;
            sumAbs += d;
            dot += a * b; na += a * a; nb += b * b;
        }
        double meanAbs = sumAbs / hiddenSize;
        double cos = dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);

        Assert.True(cos > 0.999,
            $"cosine {cos:F6} too low (meanAbs={meanAbs:E3}, maxAbs={maxAbs:E3}) -- the AdaLN " +
            $"conditioning vector every joint block reads is wrong, which would explain " +
            $"total-noise output uniformly across all blocks.");
        Assert.True(meanAbs < 1e-2,
            $"meanAbs {meanAbs:E3} too high (cos={cos:F6}, maxAbs={maxAbs:E3})");
    }
}
