using System.Diagnostics;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Direct GPU-vs-CPU unit test for the modulation Sgemm itself (docs/094 Phase 1, 2026-09-20
/// bisection follow-up #2): <see cref="Sd3AdaLNModulateGpuCpuParityTests"/> just proved the
/// `AdaLNModulate` shader's own LayerNorm/affine math is exact (maxDiff ~1e-6) on identical
/// hand-fed inputs. But the block-0 bisection found `ws.ImgMod`/`ws.TxtMod` (the modulation
/// VECTOR fed INTO AdaLNModulate, produced by `Sgemm(ws.ImgMod, tVecGpu, bw.ImgModWeight, 1,
/// HiddenSize, imgModChunks*HiddenSize)`) already diverges before AdaLNModulate ever runs. This
/// isolates that Sgemm call at the EXACT problematic shape (M=1, K=1536, N=13824 -- SD3.5-medium's
/// real `x_block.adaLN_modulation.1` dimensions for a dual-attention block) with a synthetic
/// random weight matrix and input vector, comparing the GPU M=1 matvec path against a naive CPU
/// dot product -- no quantization, no real checkpoint, no AdaLN, isolating the GEMM math alone.
/// </summary>
public sealed class Sd3ModulationSgemmGpuCpuParityTests
{
    private readonly ITestOutputHelper _output;

    public Sd3ModulationSgemmGpuCpuParityTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void Sgemm_M1_LargeN_GpuMatchesCpu()
    {
        using var backend = new VulkanBackend();

        const int K = 1536;   // SD3.5-medium's real HiddenSize
        const int N = 13824;  // SD3.5-medium's real x_block.adaLN_modulation.1 output dim (9*1536, dual-attn block)

        var rng = new Random(42);
        var x = new float[K];
        for (int i = 0; i < K; i++) x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        // Weight stored row-major [N, K] (PyTorch nn.Linear convention: out_features x in_features),
        // matching MMDiTGpuWeights' TensorShape.D2(N, K) upload convention.
        var w = new float[N * K];
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 0.1 - 0.05);

        // CPU timing: several reps of the dot-product loop, discard the first (JIT warmup).
        var cpuOut = new float[N];
        const int cpuReps = 5;
        double cpuMsTotal = 0;
        for (int rep = 0; rep < cpuReps; rep++)
        {
            var sw = Stopwatch.StartNew();
            for (int n = 0; n < N; n++)
            {
                float acc = 0f;
                int rowOff = n * K;
                for (int k = 0; k < K; k++) acc += x[k] * w[rowOff + k];
                cpuOut[n] = acc;
            }
            sw.Stop();
            if (rep > 0) cpuMsTotal += sw.Elapsed.TotalMilliseconds; // discard rep 0 (JIT warmup)
        }
        double cpuMsAvg = cpuMsTotal / (cpuReps - 1);

        var xGpu = backend.Upload(x, TensorShape.D1(K), exact: true);
        var wGpu = backend.Upload(w, TensorShape.D2(N, K), exact: true);
        var outGpu = backend.Allocate(TensorShape.D1(N));

        // GPU timing: warm up the pipeline/shader compile once (untimed), then time several reps.
        backend.Sgemm(outGpu, xGpu, wGpu, M: 1, K: K, N: N);
        var gpuOut = new float[N];
        backend.Download(outGpu, gpuOut); // forces sync after the warmup dispatch

        const int gpuReps = 5;
        var swGpuTotal = Stopwatch.StartNew();
        for (int rep = 0; rep < gpuReps; rep++)
            backend.Sgemm(outGpu, xGpu, wGpu, M: 1, K: K, N: N);
        backend.Download(outGpu, gpuOut); // forces sync so the timer includes real completion, not just dispatch
        swGpuTotal.Stop();
        double gpuMsAvg = swGpuTotal.Elapsed.TotalMilliseconds / gpuReps;

        backend.Free(xGpu);
        backend.Free(wGpu);
        backend.Free(outGpu);

        string perfMsg = $"[Sd3SgemmParity] perf (M=1,K={K},N={N}): CPU {cpuMsAvg:F4} ms/call (avg of {cpuReps - 1} reps) | GPU {gpuMsAvg:F4} ms/call (avg of {gpuReps} reps, sync'd via Download) | ratio: GPU is {(cpuMsAvg / gpuMsAvg):F2}x {(gpuMsAvg < cpuMsAvg ? "faster" : "SLOWER")} than CPU for this single call shape";
        _output.WriteLine(perfMsg);
        Console.WriteLine(perfMsg);

        double dot = 0, na = 0, nb = 0;
        float maxDiff = 0;
        int worstIdx = 0;
        for (int i = 0; i < N; i++)
        {
            float diff = MathF.Abs(cpuOut[i] - gpuOut[i]);
            if (diff > maxDiff) { maxDiff = diff; worstIdx = i; }
            dot += cpuOut[i] * gpuOut[i];
            na += cpuOut[i] * cpuOut[i];
            nb += gpuOut[i] * gpuOut[i];
        }
        double cosSim = dot / (Math.Sqrt(na) * Math.Sqrt(nb));

        string msg = $"[Sd3SgemmParity] M=1,K={K},N={N}: cosine={cosSim:F6} maxDiff={maxDiff:E6} worst: cpu={cpuOut[worstIdx]:F6} gpu={gpuOut[worstIdx]:F6} (n={worstIdx})";
        _output.WriteLine(msg);
        Console.WriteLine(msg);

        Assert.True(maxDiff < 1e-3f, $"Sgemm(M=1,K={K},N={N}) GPU output diverges from a naive CPU dot product by {maxDiff:E6} on synthetic random inputs -- a real bug in the M=1 matvec GPU path at this shape.");
    }
}
