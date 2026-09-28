using System.Runtime.InteropServices;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Routed-expert stage of batched MoE prefill, shared by <see cref="ForwardPass"/> and <see cref="HybridGdnForwardPass"/>.
/// The (token, slot) pairs arrive bucketed by expert (CSR: <c>expStart</c>, <c>pairOf</c>); each used expert gathers its
/// tokens and runs gate/up, SiLU·mul and down as batched matmuls, then writes its unweighted outputs to the pairs' rows of
/// <c>partialOut</c> (the caller reduces them per token in top-k slot order).
///
/// <para>Experts run in parallel with per-worker buffers. With ~16 tokens per expert (Qwen3.6-35B-A3B: 515 tokens x 8 of
/// 256 experts) each matmul is too small to pay its own fork/join, and the serial expert loop spent about half its time
/// there (2026-09-28: 5645 -> ~2860 ms per 515-token prefill). Every expert runs the same kernels on the same operands as
/// the serial loop did and writes disjoint rows, so the output is unchanged.</para>
/// </summary>
internal static unsafe class MoeBatchedExperts
{
    /// <param name="expStart">CSR offsets, <paramref name="numExperts"/> + 1 entries.</param>
    /// <param name="pairOf">Per bucket entry, the pair index <c>token * numActive + slot</c>.</param>
    /// <param name="inputScale">Optional per-pair scale applied to the gate and up rows before SiLU·mul (Llama-4 sigmoid
    /// gating scales the FFN input, not its output); null for none.</param>
    public static void Run(
        int numExperts, int* expStart, int* pairOf, int numActive,
        float* normIn, int embDim, int expertDim,
        byte* gateBase, DType gateDt, int bprGate,
        byte* upBase, DType upDt, int bprUp,
        byte* downBase, DType downDt, int bprDown,
        float* inputScale, float* partialOut)
    {
        int maxCnt = 0;
        for (int x = 0; x < numExperts; x++) maxCnt = Math.Max(maxCnt, expStart[x + 1] - expStart[x]);
        if (maxCnt == 0) return;
        long perWorker = (long)maxCnt * (Math.Max(embDim, expertDim) + 2L * expertDim + embDim);

        Parallel.For(0, numExperts, s_opts,
            () => (nint)NativeMemory.Alloc((nuint)(perWorker * sizeof(float))),
            (x, _, buf) =>
            {
                int p0 = expStart[x], cnt = expStart[x + 1] - p0;
                if (cnt == 0) return buf;
                float* gathered = (float*)buf;
                float* gate = gathered + (long)maxCnt * Math.Max(embDim, expertDim);
                float* up = gate + (long)maxCnt * expertDim;
                float* down = up + (long)maxCnt * expertDim;

                for (int i = 0; i < cnt; i++)
                    new ReadOnlySpan<float>(normIn + (long)(pairOf[p0 + i] / numActive) * embDim, embDim)
                        .CopyTo(new Span<float>(gathered + (long)i * embDim, embDim));

                SimdKernels.MatMulBatched(gate, gateBase + (long)x * expertDim * bprGate, gathered,
                    cnt, expertDim, embDim, gateDt, allowQ8: true);
                SimdKernels.MatMulBatched(up, upBase + (long)x * expertDim * bprUp, gathered,
                    cnt, expertDim, embDim, upDt, allowQ8: true);

                if (inputScale is not null)
                    for (int i = 0; i < cnt; i++)
                    {
                        float w = inputScale[pairOf[p0 + i]];
                        SimdKernels.ScaleInPlace(gate + (long)i * expertDim, w, expertDim);
                        SimdKernels.ScaleInPlace(up + (long)i * expertDim, w, expertDim);
                    }

                // The bucket's rows are contiguous, so one SiLuMul covers the whole batch.
                SimdKernels.SiLuMul(gate, up, cnt * expertDim);
                SimdKernels.MatMulBatched(down, downBase + (long)x * embDim * bprDown, gate,
                    cnt, embDim, expertDim, downDt, allowQ8: true);

                for (int i = 0; i < cnt; i++)
                    new ReadOnlySpan<float>(down + (long)i * embDim, embDim)
                        .CopyTo(new Span<float>(partialOut + (long)pairOf[p0 + i] * embDim, embDim));
                return buf;
            },
            buf => NativeMemory.Free((void*)buf));
    }

    // Pinned to ProcessorCount so back-to-back per-layer sweeps don't grow the thread pool.
    private static readonly ParallelOptions s_opts = new() { MaxDegreeOfParallelism = Environment.ProcessorCount };
}
