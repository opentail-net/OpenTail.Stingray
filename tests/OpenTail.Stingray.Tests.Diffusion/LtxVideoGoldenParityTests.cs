using OpenTail.Stingray.Diffusion.LTXVideo;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Numeric parity check of <see cref="LtxVideoModel"/> against golden intermediate tensors dumped
/// from HuggingFace `diffusers`' real `LTXVideoTransformer3DModel`, loaded with the REAL
/// `ltx-video-2b-v0.9.1.safetensors` checkpoint weights (see
/// <c>scripts</c>-adjacent dump script referenced in docs/055-ltx-video-implementation-plan.md --
/// generated via a one-off Python script run against the actual local checkpoint, not synthetic).
/// This is the numeric verification step 1-4 of the plan's build order call for -- distinct from
/// <see cref="LtxVideoTests"/>'s synthetic structural/shape tests and
/// <see cref="LtxVideoRealWeightsTests"/>'s config-detection check.
/// </summary>
public sealed class LtxVideoGoldenParityTests
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

    private static float MaxAbsDiff(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        Assert.Equal(a.Length, b.Length);
        float max = 0f;
        for (int i = 0; i < a.Length; i++)
            max = MathF.Max(max, MathF.Abs(a[i] - b[i]));
        return max;
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
    public void LtxVideoModel_MatchesRealDiffusersReference_OnRope_Patchify_CaptionProj_Block0_FullOutput()
    {
        string? modelPath = FindModelPath(ModelFileName);
        string? goldenDir = FindGoldenDir();
        if (modelPath is null || goldenDir is null) return; // skip: needs local checkpoint + fixtures

        using var loader = SafetensorsLoader.Open(modelPath);
        var model = new LtxVideoModel(loader);

        // Golden dump was generated with numFrames=2, H=4, W=4 (32 tokens), 16 caption tokens,
        // timestep=500, from a fixed torch.manual_seed(42) latent/caption draw -- see
        // ltx_ref_bin/manifest.json for shapes.
        var latents = ReadBin(goldenDir, "latents");
        var caption = ReadBin(goldenDir, "caption");
        var goldenRopeCos = ReadBin(goldenDir, "rope_cos");
        var goldenRopeSin = ReadBin(goldenDir, "rope_sin");
        var goldenProjIn = ReadBin(goldenDir, "proj_in_out");
        var goldenCaptionProj = ReadBin(goldenDir, "caption_proj_out");
        var goldenBlock0 = ReadBin(goldenDir, "block0_out");
        var goldenFullOut = ReadBin(goldenDir, "full_out");

        int numFrames = 2, patchH = 4, patchW = 4;
        float timestep = 500f;

        var output = model.Forward(latents, timestep, caption, numFrames, patchH, patchW);

        // RoPE: exact formula match expected (near machine precision).
        Assert.True(CosineSimilarity(model.LastRopeCos!, goldenRopeCos) > 0.9999f,
            $"RoPE cos cosine-sim too low: {CosineSimilarity(model.LastRopeCos!, goldenRopeCos)}");
        Assert.True(CosineSimilarity(model.LastRopeSin!, goldenRopeSin) > 0.999f,
            $"RoPE sin cosine-sim too low: {CosineSimilarity(model.LastRopeSin!, goldenRopeSin)}");
        // NOT machine-precision: rotation angles reach ~theta*pi/2 (~15708 rad) at the highest
        // frequency index, where float32 argument-reduction noise in cos/sin genuinely differs by
        // this much between MathF and torch's `pow`/`cos` even for an exactly-equivalent formula --
        // cosine similarity (checked above) is the real correctness signal here, not this bound.
        Assert.True(MaxAbsDiff(model.LastRopeCos!, goldenRopeCos) < 0.02f,
            $"RoPE cos max-abs-diff too high: {MaxAbsDiff(model.LastRopeCos!, goldenRopeCos)}");

        // patchify_proj (real Linear, should match to float32 precision modulo BF16 checkpoint
        // rounding -- weights are stored BF16 in the real checkpoint).
        Assert.True(CosineSimilarity(model.LastProjInOut!, goldenProjIn) > 0.999f,
            $"proj_in cosine-sim too low: {CosineSimilarity(model.LastProjInOut!, goldenProjIn)}");

        // caption_projection (Linear -> GELU -> Linear).
        Assert.True(CosineSimilarity(model.LastCaptionProjOut!, goldenCaptionProj) > 0.999f,
            $"caption_projection cosine-sim too low: {CosineSimilarity(model.LastCaptionProjOut!, goldenCaptionProj)}");

        // Block 0 full forward (self-attn+RoPE, cross-attn, FFN, AdaLN modulation all combined --
        // the single highest-value numeric check per the plan's own build order).
        float block0Cos = CosineSimilarity(model.LastBlock0Out!, goldenBlock0);
        Assert.True(block0Cos > 0.99f, $"block0 cosine-sim too low: {block0Cos}");

        // Full 28-block forward + final projection.
        float fullCos = CosineSimilarity(output, goldenFullOut);
        Assert.True(fullCos > 0.95f, $"full-forward cosine-sim too low: {fullCos}");
    }
}
