using OpenTail.Stingray.Diffusion.Flux2;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Stage 1 of the FLUX.2 speed investigation (docs/087/088, 2026-09-19, ChatGPT-guided): measures
/// whether a single forward pass is dequantization-bound or matmul-bound, WITHOUT running a full
/// multi-step generation (which currently takes 30+ minutes and was killed as unusable for
/// iteration). Set <c>STINGRAY_FLUX2_PERF_TRACE=1</c> to enable the trace and print the split.
/// </summary>
public sealed class Flux2PerfTraceSingleForwardTests
{
    private static string? FindModelPath(string fileName)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", "_models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Flux2DiT_SinglerForwardPass_512Resolution_PerfTrace()
    {
        string? ditPath = FindModelPath("flux2-dev-Q4_K_S.gguf");
        Assert.SkipUnless(ditPath != null, "FLUX.2 DiT checkpoint not found");

        Environment.SetEnvironmentVariable("STINGRAY_FLUX2_PERF_TRACE", "1");
        Flux2DiT.PerfTrace.Reset();

        using var weights = GgufWeightLoader.Open(ditPath!);
        var p = new Flux2Params();
        using var dit = new Flux2DiT(weights, p);

        // 512x512 -> patchH=patchW=32 -> 1024 image tokens, matching the real production resolution
        // regime this session found the model expects (limit_pixels = 1024**2 in sampling.py).
        int patchW = 32, patchH = 32;
        int nTargetTokens = patchW * patchH;
        int inChannels = p.InChannels;

        var rng = new Random(42);
        var targetLatent = new float[nTargetTokens * inChannels];
        for (int i = 0; i < targetLatent.Length; i++) targetLatent[i] = (float)(rng.NextDouble() - 0.5);

        var targetPositions = new int[nTargetTokens * 4];
        int idx = 0;
        for (int y = 0; y < patchH; y++)
            for (int x = 0; x < patchW; x++)
            {
                targetPositions[idx * 4 + 1] = y;
                targetPositions[idx * 4 + 2] = x;
                idx++;
            }

        // Small synthetic text context (real conditioning isn't needed to measure dequant-vs-matmul
        // split -- this is a perf trace, not a coherence check).
        int nTxt = 64;
        var txtEmbeds = new float[nTxt * p.ContextInDim];
        for (int i = 0; i < txtEmbeds.Length; i++) txtEmbeds[i] = (float)(rng.NextDouble() - 0.5) * 0.1f;
        var txtPositions = new int[nTxt * 4];
        for (int i = 0; i < nTxt; i++) txtPositions[i * 4 + 3] = i;

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var v = dit.Forward(targetLatent, targetPositions, null, null, txtEmbeds, txtPositions,
            Array.Empty<float>(), 500.0f, 3.5f);
        sw.Stop();

        Console.WriteLine($"[Flux2 single forward, 512x512] wall clock: {sw.Elapsed.TotalSeconds:F2}s");
        Flux2DiT.PerfTrace.Report("single forward @ 512x512", dit.WeightCache);

        Assert.True(v.Length > 0);
        Assert.All(v, x => Assert.True(float.IsFinite(x)));
    }
}
