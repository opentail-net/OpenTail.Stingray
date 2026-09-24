using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Real-weight, real-scale bisection for Z-Image-Turbo's GPU residency bug (docs/094 Phase 7):
/// the existing small-scale synthetic parity test (dim=384, nHeads=3, t=24) passes, but the real
/// end-to-end run (dim=3840, nHeads=30, real t) produces pure noise. Loads the REAL checkpoint
/// (`z_image_turbo-Q4_0.gguf`) at REAL dimensions and runs the 30-block main loop through both
/// <see cref="ZImageDiT.RunMainLayersCpuForTest"/> and <see cref="ZImageDiT.RunMainLayersGpuForTest"/>
/// with per-block capture, to find the first block where they diverge -- same technique already
/// proven for Qwen Image's own GPU bug this session.
/// </summary>
public sealed class ZImageGpuRealScaleBisectTests
{
    private readonly ITestOutputHelper _output;

    public ZImageGpuRealScaleBisectTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private static string? FindModelPath(string fileName)
    {
        string[] absoluteCandidates =
        {
            $@"C:\Git-Public\OpenTail.Stingray\models\_models\{fileName}",
            $@"C:\Git-Public\OpenTail.Stingray\models\{fileName}",
        };
        foreach (var p in absoluteCandidates)
        {
            if (File.Exists(p)) return p;
        }
        return null;
    }

    [Fact]
    public void Block0Parity_RealScale()
    {
        string? ditPath = FindModelPath("z_image_turbo-Q4_0.gguf");
        if (ditPath is null) return;

        var p = new ZImageParams();
        int dim = p.Dim;
        using var cpuWeights = GgufWeightLoader.Open(ditPath);
        using var cpuDit = new ZImageDiT(cpuWeights, p);
        using var vulkan = new VulkanBackend();
        using var gpuWeights = GgufWeightLoader.Open(ditPath);
        using var gpuDit = new ZImageDiT(gpuWeights, p, vulkan);

        const int nImg = 256;
        const int nTxt = 64;
        int nTok = nImg + nTxt;

        var rng = new Random(42);
        var x = new float[nTok * dim];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var adaln = new float[p.AdalnEmbedDim];
        for (int i = 0; i < adaln.Length; i++) adaln[i] = (float)(rng.NextDouble() * 0.5 - 0.25);

        var posIds = new int[nTok * 3];
        for (int i = 0; i < nTok; i++) { posIds[i * 3] = i; posIds[i * 3 + 1] = 0; posIds[i * 3 + 2] = 0; }
        var rope = new ZImageRoPE(p);
        var freqs = rope.BuildFreqs(posIds, nTok);

        var cpuX = (float[])x.Clone();
        cpuDit.ApplyBlockForTest("layers.0", cpuX, nTok, freqs, adaln, true);

        var gpuX = (float[])x.Clone();
        gpuDit.ApplyBlockForTest("layers.0", gpuX, nTok, freqs, adaln, true);

        double dot = 0, normA = 0, normB = 0;
        float maxDiff = 0;
        for (int i = 0; i < cpuX.Length; i++)
        {
            float diff = MathF.Abs(cpuX[i] - gpuX[i]);
            if (diff > maxDiff) maxDiff = diff;
            dot += (double)cpuX[i] * gpuX[i];
            normA += (double)cpuX[i] * cpuX[i];
            normB += (double)gpuX[i] * gpuX[i];
        }
        double cosSim = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Console.WriteLine($"[Block0Parity] cosSim={cosSim:F6} maxDiff={maxDiff:F4}");
        Assert.True(cosSim > 0.999, $"Block 0 parity failed: cosSim={cosSim:F6} maxDiff={maxDiff:F4}");
    }

    [Fact]
    public void FindFirstDivergentMainBlock_RealWeights_RealScale()
    {
        string? ditPath = FindModelPath("z_image_turbo-Q4_0.gguf");
        if (ditPath is null)
        {
            _output.WriteLine("[ZImageGpuRealScaleBisectTests] Checkpoint missing, skipping."); Console.WriteLine("[ZImageGpuRealScaleBisectTests] Checkpoint missing, skipping.");
            return;
        }

        var p = new ZImageParams();
        int dim = p.Dim;
        int headDim = p.HeadDim;
        int halfHead = headDim / 2;

        using var cpuWeights = GgufWeightLoader.Open(ditPath);
        using var cpuDit = new ZImageDiT(cpuWeights, p);
        using var vulkan = new VulkanBackend();
        using var gpuWeights = GgufWeightLoader.Open(ditPath);
        using var gpuDit = new ZImageDiT(gpuWeights, p, vulkan);

        // Real-scale (256x256-equivalent) but synthetic activation/RoPE input, matching the exact
        // technique already used for Qwen Image's own real-weight bisection this session: real
        // trained weights + a plausible-magnitude synthetic activation, sufficient to exercise the
        // real GPU dispatch/shader code at real dimensions without needing the full text-encoder/
        // VAE pipeline wired up just for this diagnostic.
        const int nImg = 256; // 16x16 patches, a real 256x256-equivalent token count
        const int nTxt = 64;
        int nTok = nImg + nTxt;

        var rng = new Random(42);
        var x = new float[nTok * dim];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var adaln = new float[p.AdalnEmbedDim];
        for (int i = 0; i < adaln.Length; i++) adaln[i] = (float)(rng.NextDouble() * 0.5 - 0.25);

        // Real RoPE freqs for a simple sequential position assignment (matches ZImageRoPE's own
        // real interleaved (cos,sin)-pair-per-token output format).
        var posIds = new int[nTok * 3];
        for (int i = 0; i < nTok; i++) { posIds[i * 3] = i; posIds[i * 3 + 1] = 0; posIds[i * 3 + 2] = 0; }
        var rope = new ZImageRoPE(p);
        var freqs = rope.BuildFreqs(posIds, nTok);

        var cpuBlocks = new Dictionary<int, float[]>();
        var gpuBlocks = new Dictionary<int, float[]>();
        cpuDit.OnMainBlockOutputCpu = (l, data) => cpuBlocks[l] = data;
        gpuDit.OnMainBlockOutputGpu = (l, data) => gpuBlocks[l] = (float[])data.Clone();

        var cpuX = (float[])x.Clone();
        _output.WriteLine("[Bisect] Running CPU main-block loop..."); Console.WriteLine("[Bisect] Running CPU main-block loop...");
        cpuDit.RunMainLayersCpuForTest(cpuX, nTok, freqs, adaln);

        var gpuX = (float[])x.Clone();
        _output.WriteLine("[Bisect] Running GPU main-block loop..."); Console.WriteLine("[Bisect] Running GPU main-block loop...");
        gpuDit.RunMainLayersGpuForTest(vulkan, gpuX, nTok, freqs, adaln);

        Assert.True(cpuBlocks.Count > 0, "CPU per-block hook never fired");
        Assert.True(gpuBlocks.Count > 0, "GPU per-block hook never fired");

        int firstDivergent = -1;
        for (int l = 0; l < p.NLayers; l++)
        {
            if (!cpuBlocks.TryGetValue(l, out var cpuOut) || !gpuBlocks.TryGetValue(l, out var gpuOut))
            {
                string skip = $"[Bisect] Block {l}: missing capture on one side, skipping";
                _output.WriteLine(skip); Console.WriteLine(skip);
                continue;
            }

            double dot = 0, normA = 0, normB = 0;
            float maxDiff = 0;
            bool cpuNonFinite = false, gpuNonFinite = false;
            for (int i = 0; i < cpuOut.Length; i++)
            {
                if (!float.IsFinite(cpuOut[i])) cpuNonFinite = true;
                if (!float.IsFinite(gpuOut[i])) gpuNonFinite = true;
                float diff = MathF.Abs(cpuOut[i] - gpuOut[i]);
                if (diff > maxDiff) maxDiff = diff;
                dot += (double)cpuOut[i] * gpuOut[i];
                normA += (double)cpuOut[i] * cpuOut[i];
                normB += (double)gpuOut[i] * gpuOut[i];
            }
            double cosSim = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));

            string flag = "";
            if (cosSim < 0.999 && firstDivergent < 0)
            {
                firstDivergent = l;
                flag = "  <-- FIRST DIVERGENCE";
            }

            string m = $"[Bisect] Block {l}: cosine={cosSim:F6} maxDiff={maxDiff:E4} cpuNonFinite={cpuNonFinite} gpuNonFinite={gpuNonFinite}{flag}";
            _output.WriteLine(m); Console.WriteLine(m);
        }

        string conclusion = firstDivergent < 0
            ? "[Bisect] No block crossed the 0.999 cosine threshold -- CPU and GPU stayed in agreement across all 30 real-scale blocks."
            : $"[Bisect] CONCLUSION: first meaningful divergence at block {firstDivergent} of {p.NLayers}.";
        _output.WriteLine(conclusion); Console.WriteLine(conclusion);
    }
}
