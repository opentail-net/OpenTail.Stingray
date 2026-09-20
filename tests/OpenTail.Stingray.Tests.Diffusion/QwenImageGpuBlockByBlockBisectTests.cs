using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.QwenImage;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Stage-by-stage GPU-vs-CPU intermediate tensor dump for Qwen Image's known GPU correctness bug
/// (docs/094 Phase 2: cosine 0.990/maxDiff 0.218 overall, 8 op-level candidates individually ruled
/// out via code review without finding the cause). Same technique that found Z-Image's sign-
/// convention bug and FLUX's T5-padding bug -- compare the [img] hidden state after EVERY block on
/// both paths and report the first block where the two diverge meaningfully, instead of continuing
/// to guess at op-level candidates.
/// </summary>
public sealed class QwenImageGpuBlockByBlockBisectTests
{
    private readonly ITestOutputHelper _output;

    public QwenImageGpuBlockByBlockBisectTests(ITestOutputHelper output)
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
    public void FindFirstDivergentBlock()
    {
        string? modelPath = FindModelPath("qwen-image-Q3_K_S.gguf");
        if (modelPath is null)
        {
            _output.WriteLine("[QwenImageGpuBlockByBlockBisectTests] Checkpoint missing, skipping."); Console.WriteLine("[QwenImageGpuBlockByBlockBisectTests] Checkpoint missing, skipping.");
            return;
        }

        using var cpuWeights = GgufWeightLoader.Open(modelPath);
        using var cpuModel = new QwenImageModel(cpuWeights);
        using var vulkan = new VulkanBackend();
        using var gpuWeights = GgufWeightLoader.Open(modelPath);
        using var gpuModel = new QwenImageModel(gpuWeights, backend: vulkan);

        // Realistic-scale inputs (2026-09-20 revision): the original uniform [-1,1] latent /
        // [0,0.1] text-context / timestep=1000 (a trajectory BOUNDARY value) combination drove
        // activations to 2.6-44 MILLION magnitude by the earlier bisection run -- orders of
        // magnitude past a healthy diffusion hidden state, which chaotically amplifies ordinary
        // FP16-vs-FP32 rounding differences and produces a divergence signal dominated by that
        // artifact rather than the real bug. Use a proper unit-Gaussian latent (matching
        // PackLatents' real expected input distribution) and a realistic MID-trajectory timestep.
        const int latH = 32, latW = 32, latC = 16;
        var rng = new Random(42);
        var latent = new float[latC * latH * latW];
        for (int i = 0; i < latent.Length - 1; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double radius = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;
            latent[i] = (float)(radius * Math.Cos(theta));
            latent[i + 1] = (float)(radius * Math.Sin(theta));
        }

        int seqLen = 8;
        var textContext = new float[seqLen * QwenImageModel.ContextDim];
        for (int i = 0; i < textContext.Length; i++) textContext[i] = (float)(rng.NextDouble() * 2 - 1) * 0.02f;

        var cpuBlocks = new Dictionary<int, float[]>();
        var gpuBlocks = new Dictionary<int, float[]>();
        cpuModel.OnBlockOutputCpu = (b, data) => cpuBlocks[b] = (float[])data.Clone();
        gpuModel.OnBlockOutputGpu = (b, data) => gpuBlocks[b] = (float[])data.Clone();

        _output.WriteLine("[Bisect] Running CPU forward with per-block capture..."); Console.WriteLine("[Bisect] Running CPU forward with per-block capture...");
        cpuModel.Forward(latent, 500f, textContext, latH, latW);

        _output.WriteLine("[Bisect] Running GPU forward with per-block capture..."); Console.WriteLine("[Bisect] Running GPU forward with per-block capture...");
        gpuModel.Forward(latent, 500f, textContext, latH, latW);

        Assert.True(cpuBlocks.Count > 0, "CPU per-block hook never fired");
        Assert.True(gpuBlocks.Count > 0, "GPU per-block hook never fired");

        int firstDivergent = -1;
        for (int b = 0; b < cpuModel.NumLayers; b++)
        {
            if (!cpuBlocks.TryGetValue(b, out var cpuOut) || !gpuBlocks.TryGetValue(b, out var gpuOut))
            {
                _output.WriteLine($"[Bisect] Block {b}: missing capture on one side, skipping"); Console.WriteLine($"[Bisect] Block {b}: missing capture on one side, skipping");
                continue;
            }

            double dot = 0, normA = 0, normB = 0;
            float maxDiff = 0;
            for (int i = 0; i < cpuOut.Length; i++)
            {
                float diff = MathF.Abs(cpuOut[i] - gpuOut[i]);
                if (diff > maxDiff) maxDiff = diff;
                dot += (double)cpuOut[i] * gpuOut[i];
                normA += (double)cpuOut[i] * cpuOut[i];
                normB += (double)gpuOut[i] * gpuOut[i];
            }
            double cosSim = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));

            bool cpuHasNonFinite = Array.Exists(cpuOut, v => !float.IsFinite(v));
            bool gpuHasNonFinite = Array.Exists(gpuOut, v => !float.IsFinite(v));
            float cpuMaxAbs = 0, gpuMaxAbs = 0;
            foreach (var v in cpuOut) { if (float.IsFinite(v) && MathF.Abs(v) > cpuMaxAbs) cpuMaxAbs = MathF.Abs(v); }
            foreach (var v in gpuOut) { if (float.IsFinite(v) && MathF.Abs(v) > gpuMaxAbs) gpuMaxAbs = MathF.Abs(v); }
            if (cpuHasNonFinite || gpuHasNonFinite || cpuMaxAbs > 60000 || gpuMaxAbs > 60000)
            {
                string m = $"[Bisect] Block {b}: NaN/Inf check -- cpuNonFinite={cpuHasNonFinite} gpuNonFinite={gpuHasNonFinite} cpuMaxAbs={cpuMaxAbs:E3} gpuMaxAbs={gpuMaxAbs:E3}";
                _output.WriteLine(m); Console.WriteLine(m);
            }

            string flag = "";
            if (cosSim < 0.999 && firstDivergent < 0)
            {
                firstDivergent = b;
                flag = "  <-- FIRST DIVERGENCE";
            }

            // Only print every 5th block plus the divergence point, to keep output readable across
            // 60 blocks.
            if (b % 5 == 0 || b == cpuModel.NumLayers - 1 || flag.Length > 0)
            {
                _output.WriteLine($"[Bisect] Block {b}: cosine={cosSim:F6} maxDiff={maxDiff:E4}{flag}"); Console.WriteLine($"[Bisect] Block {b}: cosine={cosSim:F6} maxDiff={maxDiff:E4}{flag}");
            }
        }

        if (firstDivergent < 0)
        {
            _output.WriteLine("[Bisect] No block crossed the 0.999 cosine threshold -- divergence (if any) accumulates gradually rather than starting at one identifiable block."); Console.WriteLine("[Bisect] No block crossed the 0.999 cosine threshold -- divergence (if any) accumulates gradually rather than starting at one identifiable block.");
        }
        else
        {
            _output.WriteLine($"[Bisect] CONCLUSION: first meaningful divergence at block {firstDivergent} of {cpuModel.NumLayers}."); Console.WriteLine($"[Bisect] CONCLUSION: first meaningful divergence at block {firstDivergent} of {cpuModel.NumLayers}.");
        }
    }
}
