using System.Diagnostics;
using OpenTail.Stingray.Engine;
using OpenTail.Stingray.Engine.Verification;

namespace OpenTail.Stingray.Tests.Vulkan;

/// <summary>
/// Gate measurement for the batched-MoE prefill plan (docs/2-coverage/2026-10-03-batched-moe-prefill-plan.md, Phase 2):
/// "proceed only if GPU attention for the chunk beats CPU batched attention on the target hardware". Times two Vulkan
/// kernels on a 16-token chunk against <c>ForwardPass.ComputeBatchedCausalAttention</c> on the same data:
/// <c>AttentionBatched</c> (the plain kernel) and <c>AttentionBatchedFlash</c>, which <c>GpuForwardPass</c> uses for a chunk
/// of 16 or fewer tokens (the production choice). Dispatch plus fence is timed, and so is a round trip that also uploads Q
/// and downloads the output. The OLMoE head shape is read from its GGUF. The test asserts that the outputs agree, not that
/// either is faster: the decision is read from the printed numbers and recorded in the plan. Integrated GPU only (CLAUDE.md
/// rule 13).
/// </summary>
public sealed class PrefillAttentionGateTests : HeavyTestBase
{
    private const string Olmoe = "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf";
    private const int Chunk = 16;
    private const int Repeats = 5;

    // The kernels require basePos + numQueries <= 4096 (AttentionBatched), so 4080 is the largest position with a full chunk.
    private static readonly int[] BasePositions = [0, 256, 1024, 2032, 3584, 4080];

    [Fact]
    public unsafe void GpuBatchedAttention_AgreesWithCpuBatchedAttention_AndTheTimingsArePrinted()
    {
        string? path = ModelLocator.FindDeep(Olmoe);
        Assert.SkipWhen(path is null, $"{Olmoe} not present");
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");
        using var _gpu = gpu;

        using var model = GgufModel.Open(path!);
        var hp = ArchitectureModelResolver.ResolveHyperparams(model);
        int numHeads = hp.NumHeads, numKvHeads = hp.NumKvHeads, headDim = hp.HeadDim;
        int qDim = numHeads * headDim, kvDim = numKvHeads * headDim;
        float scale = 1f / MathF.Sqrt(headDim);   // the Vulkan kernels have no scale argument: they use 1/sqrt(headDim)
        Console.WriteLine($"[attn-gate] heads {numHeads}/{numKvHeads}, headDim {headDim}, chunk {Chunk}, repeats {Repeats} (median)");
        Console.WriteLine("[attn-gate] basePos  cpu_ms  gpu_plain_ms  gpu_flash_ms  gpu_flash_roundtrip_ms  flash_speedup  max_rel_err");

        var rng = new Random(20260606);
        foreach (int basePos in BasePositions)
        {
            int kvLen = basePos + Chunk;
            var qHost = RandomFloats(rng, Chunk * qDim);
            var kHost = RandomFloats(rng, kvLen * kvDim);
            var vHost = RandomFloats(rng, kvLen * kvDim);

            // CPU reference and timing: the same kernel the CPU prefill runs. A plain loop, because fixed pointers cannot be
            // captured by a lambda. Two warm-up runs, then Repeats measured runs.
            var cpuOut = new float[Chunk * qDim];
            double cpuMs;
            fixed (float* qPtr = qHost)
            fixed (float* kPtr = kHost)
            fixed (float* vPtr = vHost)
            fixed (float* outPtr = cpuOut)
            {
                var times = new double[Repeats + 2];
                for (int i = 0; i < times.Length; i++)
                {
                    long t0 = Stopwatch.GetTimestamp();
                    OpenTail.Stingray.Engine.ForwardPass.ComputeBatchedCausalAttention(
                        qPtr, kPtr, vPtr, outPtr, N: Chunk, startPos: basePos, numHeads, numKvHeads, headDim, scale);
                    times[i] = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
                }
                cpuMs = MedianOfMeasured(times);
            }

            // GPU: K, V and Q stay resident for the kernel timings; the round trip also uploads Q and downloads the output.
            var gq = gpu!.Upload(qHost, TensorShape.D1(qHost.Length));
            var gk = gpu.Upload(kHost, TensorShape.D1(kHost.Length));
            var gv = gpu.Upload(vHost, TensorShape.D1(vHost.Length));
            var gPlainOut = gpu.Allocate(TensorShape.D1(Chunk * qDim));
            var gFlashOut = gpu.Allocate(TensorShape.D1(Chunk * qDim));
            var plainHost = new float[Chunk * qDim];
            var flashHost = new float[Chunk * qDim];

            double plainMs = MedianMs(() =>
            {
                gpu.AttentionBatched(gq, gk, gv, gPlainOut, (uint)numHeads, (uint)numKvHeads, (uint)headDim,
                    (uint)basePos, Chunk, (uint)kvLen);
                gpu.Synchronize();
            });
            double flashMs = MedianMs(() =>
            {
                gpu.AttentionBatchedFlash(gq, gk, gv, gFlashOut, (uint)numHeads, (uint)numKvHeads, (uint)headDim,
                    (uint)basePos, Chunk, (uint)kvLen);
                gpu.Synchronize();
            });
            double flashRoundTripMs = MedianMs(() =>
            {
                var q2 = gpu.Upload(qHost, TensorShape.D1(qHost.Length));
                gpu.AttentionBatchedFlash(q2, gk, gv, gFlashOut, (uint)numHeads, (uint)numKvHeads, (uint)headDim,
                    (uint)basePos, Chunk, (uint)kvLen);
                gpu.Synchronize();
                gpu.Download(gFlashOut, flashHost);
                gpu.Free(q2);
            });

            gpu.Download(gPlainOut, plainHost);
            gpu.Download(gFlashOut, flashHost);
            double relPlain = RelativeError(plainHost, cpuOut), relFlash = RelativeError(flashHost, cpuOut);

            gpu.Free(gq); gpu.Free(gk); gpu.Free(gv); gpu.Free(gPlainOut); gpu.Free(gFlashOut);

            Console.WriteLine(FormattableString.Invariant(
                $"[attn-gate] {basePos,7}  {cpuMs,6:F3}  {plainMs,12:F3}  {flashMs,12:F3}  {flashRoundTripMs,23:F3}  {cpuMs / flashMs,13:F2}  {Math.Max(relPlain, relFlash),11:E2}"));
            Assert.True(relPlain < 1e-3, $"plain GPU attention disagrees with the CPU at basePos {basePos}: {relPlain:E2}");
            Assert.True(relFlash < 1e-3, $"flash GPU attention disagrees with the CPU at basePos {basePos}: {relFlash:E2}");
        }
    }

    private static double RelativeError(float[] gpu, float[] cpu)
    {
        double maxAbs = 0, refAbs = 1e-12;
        for (int i = 0; i < cpu.Length; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(gpu[i] - cpu[i]));
            refAbs = Math.Max(refAbs, Math.Abs(cpu[i]));
        }
        return maxAbs / refAbs;
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var a = new float[count];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    /// <summary>Median of the measured runs: <paramref name="times"/> holds two warm-up runs first, then <see cref="Repeats"/> runs.</summary>
    private static double MedianOfMeasured(double[] times)
    {
        var measured = times[2..];
        Array.Sort(measured);
        return measured[measured.Length / 2];
    }

    /// <summary>Median of <see cref="Repeats"/> timed runs after two warm-up runs.</summary>
    private static double MedianMs(Action action)
    {
        action();
        action();
        var times = new double[Repeats];
        for (int i = 0; i < Repeats; i++)
        {
            long t0 = Stopwatch.GetTimestamp();
            action();
            times[i] = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        }
        Array.Sort(times);
        return times[Repeats / 2];
    }
}
