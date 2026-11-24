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
///
/// NOTE (2026-11-24, iteration 41): <c>ComputeBatchedCausalAttention</c> is not the production CPU prefill attention (it has no
/// production caller). The production-path comparison is <c>GpuWholePromptAttention_IsTimedPerLayer_AndAgreesWithTheReference</c>,
/// and its numbers, with the round trip, are in docs/2-coverage/2026-10-03-batched-moe-prefill-plan.md.
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

    /// <summary>
    /// Whole-prompt attention for one layer, GPU against the production CPU prefill (the per-layer numbers are in the plan, from
    /// <c>STINGRAY_PROFILE_PREFILL=1</c>): one <c>AttentionFlashPrefillGqa</c> dispatch over the whole prompt, and the same prompt
    /// as 16-token <c>AttentionBatchedFlash</c> chunks, each at its own position (the accounting Phase 2 would pay). The
    /// reference is <c>ForwardPass.ComputeBatchedCausalAttention</c>, which is correct but not the production kernel; it is used
    /// only for parity.
    /// </summary>
    [Fact]
    public unsafe void GpuWholePromptAttention_IsTimedPerLayer_AndAgreesWithTheReference()
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
        float scale = 1f / MathF.Sqrt(headDim);
        Console.WriteLine("[attn-whole] prompt_tokens  gpu_whole_prompt_per_layer_ms  gpu_chunked16_per_layer_ms  rel_err_whole_vs_reference");

        var rng = new Random(20260607);
        foreach (int n in new[] { 1224, 2014 })
        {
            var qHost = RandomFloats(rng, n * qDim);
            var kHost = RandomFloats(rng, n * kvDim);
            var vHost = RandomFloats(rng, n * kvDim);
            var gq = gpu!.Upload(qHost, TensorShape.D1(qHost.Length));
            var gk = gpu.Upload(kHost, TensorShape.D1(kHost.Length));
            var gv = gpu.Upload(vHost, TensorShape.D1(vHost.Length));
            var gWhole = gpu.Allocate(TensorShape.D1(n * qDim));

            double wholeMs = MedianMs(() =>
            {
                gpu.AttentionFlashPrefillGqa(gq, gk, gv, gWhole, (uint)numHeads, (uint)numKvHeads, (uint)headDim, 0, n);
                gpu.Synchronize();
            });
            var wholeHost = new float[n * qDim];
            gpu.Download(gWhole, wholeHost);

            // The chunked accounting: one tensor per 16-token chunk, uploaded before timing, so the timed work is the dispatches.
            int chunks = (n + Chunk - 1) / Chunk;
            var qChunks = new Tensor[chunks];
            var oChunks = new Tensor[chunks];
            for (int c = 0; c < chunks; c++)
            {
                int rows = Math.Min(Chunk, n - c * Chunk);
                qChunks[c] = gpu.Upload(qHost.AsSpan(c * Chunk * qDim, rows * qDim), TensorShape.D1(rows * qDim));
                oChunks[c] = gpu.Allocate(TensorShape.D1(rows * qDim));
            }
            double chunkedMs = MedianMs(() =>
            {
                for (int c = 0; c < chunks; c++)
                {
                    int rows = Math.Min(Chunk, n - c * Chunk);
                    gpu.AttentionBatchedFlash(qChunks[c], gk, gv, oChunks[c], (uint)numHeads, (uint)numKvHeads, (uint)headDim,
                        (uint)(c * Chunk), rows, (uint)n);
                    gpu.Synchronize();
                }
            });

            var reference = new float[n * qDim];
            fixed (float* qPtr = qHost)
            fixed (float* kPtr = kHost)
            fixed (float* vPtr = vHost)
            fixed (float* outPtr = reference)
                OpenTail.Stingray.Engine.ForwardPass.ComputeBatchedCausalAttention(
                    qPtr, kPtr, vPtr, outPtr, N: n, startPos: 0, numHeads, numKvHeads, headDim, scale);
            double rel = RelativeError(wholeHost, reference);

            Console.WriteLine(FormattableString.Invariant(
                $"[attn-whole] {n,13}  {wholeMs,27:F3}  {chunkedMs,26:F3}  {rel,25:E2}"));
            Assert.True(rel < 1e-3, $"whole-prompt GPU attention disagrees with the reference at {n} tokens: {rel:E2}");

            gpu.Free(gq); gpu.Free(gk); gpu.Free(gv); gpu.Free(gWhole);
            foreach (var t in qChunks) gpu.Free(t);
            foreach (var t in oChunks) gpu.Free(t);
        }
    }

    /// <summary>
    /// The host round trip a batched trunk pays per layer for its activations: an <c>[N x hidden]</c> fp32 block uploaded to the GPU
    /// and downloaded again (the host MoE step sits between the two), timed with <c>Upload</c> and <c>Download</c> as the backend
    /// has them. Each upload allocates, so the time is an upper bound for a reused buffer; the figure is recorded in the plan.
    /// </summary>
    [Fact]
    public void ActivationRoundTrip_IsTimedForWholePromptAndChunkBlocks()
    {
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");
        using var _gpu = gpu;
        const int hidden = 2048;   // OLMoE embedding width
        Console.WriteLine("[attn-roundtrip] rows  block_mib  upload_download_ms (median of 5, after 2 warm-ups)");
        var rng = new Random(20260608);
        foreach (int rows in new[] { Chunk, 1224, 2014 })
        {
            var block = RandomFloats(rng, rows * hidden);
            var back = new float[rows * hidden];
            double ms = MedianMs(() =>
            {
                var t = gpu!.Upload(block, TensorShape.D1(block.Length));
                gpu.Download(t, back);
                gpu.Free(t);
            });
            Console.WriteLine(FormattableString.Invariant($"[attn-roundtrip] {rows,5}  {block.Length * 4.0 / (1 << 20),9:F2}  {ms,19:F3}"));
            Assert.Equal(block, back);   // every element made the trip unchanged
        }
    }

    /// <summary>
    /// The reused-buffer round trip a whole-prompt batched trunk would pay per layer (Phase 2b): the host writes an
    /// <c>[N x 2048]</c> block into a pinned buffer, copies it to a device buffer, copies it back to a second pinned buffer, and
    /// the host reads it. Both pinned buffers and the device buffer are allocated once, outside the timing; the time includes both
    /// device copies and their synchronisation. Recorded next to the allocating figure in the plan.
    /// </summary>
    [Fact]
    public unsafe void PinnedActivationRoundTrip_WithDeviceCopies_IsTimed()
    {
        VulkanBackend? gpu;
        try { gpu = new VulkanBackend(); } catch { gpu = null; }
        Assert.SkipWhen(gpu is null, "no Vulkan device available on this host");
        using var _gpu = gpu;
        const int hidden = 2048;
        Console.WriteLine("[attn-pinned] rows  block_mib  pinned_round_trip_ms (host write, copy to device, copy back, host read; median of 5)");
        var rng = new Random(20260609);
        foreach (int rows in new[] { 1224, 2014 })
        {
            int count = rows * hidden;
            var block = RandomFloats(rng, count);
            var back = new float[count];
            var pinIn = gpu!.AllocatePinned(TensorShape.D1(count));
            var pinOut = gpu.AllocatePinned(TensorShape.D1(count));
            var devBuf = gpu.Allocate(TensorShape.D1(count));

            // The four steps are timed separately, so a slow step is named rather than averaged into the total.
            double writeMs = MedianMs(() => gpu.WritePinned(pinIn, block));
            double copyInMs = MedianMs(() =>
            {
                gpu.BeginRecord();
                gpu.RecordComputeCopy(devBuf, pinIn);
                gpu.RecordComputeToHostBarrier();
                gpu.EndRecordAndSubmit();
                gpu.Synchronize();
            });
            double copyOutMs = MedianMs(() =>
            {
                gpu.BeginRecord();
                gpu.RecordComputeCopy(pinOut, devBuf);
                gpu.RecordComputeToHostBarrier();
                gpu.EndRecordAndSubmit();
                gpu.Synchronize();
            });
            double readMs = MedianMs(() =>
            {
                float* p = gpu.MapPinned(pinOut);
                new ReadOnlySpan<float>(p, count).CopyTo(back);
                gpu.UnmapPinned(pinOut);
            });
            Assert.Equal(block, back);   // every element made the trip unchanged
            Console.WriteLine(FormattableString.Invariant(
                $"[attn-pinned] {rows,5}  {count * 4.0 / (1 << 20),9:F2}  write {writeMs,9:F3}  copy_in {copyInMs,9:F3}  copy_out {copyOutMs,9:F3}  host_read {readMs,9:F3}  total {writeMs + copyInMs + copyOutMs + readMs,9:F3}"));

            // The path a batched trunk would use instead: device to ordinary (cached) host memory with Download, and back with Upload.
            double downloadMs = MedianMs(() => gpu.Download(devBuf, back));
            double uploadMs = MedianMs(() =>
            {
                var t = gpu.Upload(block, TensorShape.D1(count));
                gpu.Free(t);
            });
            Console.WriteLine(FormattableString.Invariant(
                $"[attn-cached] {rows,5}  download {downloadMs,9:F3}  upload_allocating {uploadMs,9:F3}  total {downloadMs + uploadMs,9:F3}"));

            gpu.Free(pinIn); gpu.Free(pinOut); gpu.Free(devBuf);
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
