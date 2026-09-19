using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Diffusion;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Golden-parity check for FLUX.1's conditioning vector <c>vec</c>
/// (<see cref="FluxDiT.ComputeVec"/>) -- the single input every double/single block's AdaLN
/// modulation derives from, and this project's own repeated bug class this session (a wrong scale
/// or SiLU-ordering error here silently corrupts every block's conditioning uniformly, producing
/// "structured but not coherent" or noise output depending on severity -- see docs/088's FLUX.2
/// timestep-blindness bug and SD3.5's own analogous
/// <c>MMDiTModel.ComputeTimeAndPooledEmbedding</c> golden check for precedent). FLUX.1 never had
/// this check; this closes that gap.
///
/// Reference generated via <c>scripts/flux1_vec_embed_ref.py</c>, which reads the real
/// <c>time_in</c>/<c>vector_in</c> MLP weights directly out of the real checkpoint (no full
/// diffusers pipeline load) and reimplements the exact real formula confirmed against
/// <c>examples/diffusers</c>'s <c>CombinedTimestepTextProjEmbeddings</c>. The pooled-CLIP input is
/// a synthetic deterministic vector (seed 42) -- this isolates the embedding-MLP math only, not
/// the CLIP text encoder, same deliberate scope limit as SD3.5's own analogous script.
/// </summary>
public sealed class Flux1VecEmbedGoldenTests
{
    private const string ModelFileName = "flux1-schnell-Q4_K_S.gguf";

    private static string? FindModelPath()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", "flux1-schnell", ModelFileName);
            if (File.Exists(p)) return p;
            var p2 = Path.Combine(dir, "models", "_models", ModelFileName);
            if (File.Exists(p2)) return p2;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static string? FindFixtureDir()
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "tests", "fixtures", "flux1_vec_embed");
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void ComputeVec_MatchesRealReferenceFormula()
    {
        string? modelPath = FindModelPath();
        string? fixtureDir = FindFixtureDir();
        Assert.SkipUnless(modelPath != null, "flux1-schnell-Q4_K_S.gguf not found");
        Assert.SkipUnless(fixtureDir != null, "tests/fixtures/flux1_vec_embed not generated -- run scripts/flux1_vec_embed_ref.py first");

        var meta = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(fixtureDir!, "meta.json")));
        float timestep = meta.RootElement.GetProperty("timestep").GetSingle();
        int vecDim = meta.RootElement.GetProperty("vecDim").GetInt32();
        var pooledEl = meta.RootElement.GetProperty("pooled");
        var pooled = new float[vecDim];
        int idx = 0;
        foreach (var v in pooledEl.EnumerateArray()) pooled[idx++] = v.GetSingle();

        var refBytes = File.ReadAllBytes(Path.Combine(fixtureDir!, "vec.f32"));
        var refVec = new float[refBytes.Length / 4];
        Buffer.BlockCopy(refBytes, 0, refVec, 0, refBytes.Length);

        var model = GgufModel.Open(modelPath!);
        var p = FluxParams.FromMetadata(model.Metadata);
        using var backend = new CpuBackend();
        using var dit = new FluxDiT(model, p, backend);

        // FLUX.1-schnell has no guidance_in -- guidance value is unused when p.HasGuidanceIn is
        // false, pass a dummy value.
        var vec = dit.ComputeVec(timestep, pooled, guidance: 1.0f);

        Assert.Equal(refVec.Length, vec.Length);

        double sumSq = 0, dot = 0, refSumSq = 0;
        for (int i = 0; i < vec.Length; i++)
        {
            dot += (double)vec[i] * refVec[i];
            sumSq += (double)vec[i] * vec[i];
            refSumSq += (double)refVec[i] * refVec[i];
        }
        double cosine = dot / (Math.Sqrt(sumSq) * Math.Sqrt(refSumSq) + 1e-12);
        double maxDiff = 0;
        for (int i = 0; i < vec.Length; i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(vec[i] - refVec[i]));

        Console.WriteLine($"[Flux1 vec golden] cosine={cosine:F9} maxDiff={maxDiff:E4}");
        Assert.True(cosine > 0.9999, $"cosine similarity too low: {cosine}");
        Assert.True(maxDiff < 1e-2, $"maxDiff too high: {maxDiff}");
    }
}
