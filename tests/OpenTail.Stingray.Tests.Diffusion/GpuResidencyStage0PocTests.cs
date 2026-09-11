using OpenTail.Stingray.Core;
using OpenTail.Stingray.Vulkan;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Stage 0 of docs/067-sdxl-unet-gpu-residency-plan.md: proves the actual mechanism the whole
/// residency rewrite depends on, before writing any SDXL-specific code. Question: can a GPU
/// tensor produced by one dispatch be consumed directly by a second, dependent dispatch recorded
/// into the SAME `BeginRecord()`/`EndRecordAndSubmit()` session, with NO `Upload`/`Download` call
/// in between -- only a `RecordBarrier()` -- and still produce a numerically correct result?
///
/// Real finding (2026-09-12): `Sgemm` already routes through `DispatchOrRecord`, which already
/// checks `_recording` and records into `_transferCmd` instead of dispatching+waiting immediately
/// -- this mechanism already exists at the compute-dispatch level. The actual gap the plan
/// identified is narrower than "does recording work at all": it's specifically that `Upload`/
/// `Download` always do their own immediate `SubmitAndWait`, never checking `_recording`. This
/// test proves the compute-chaining half works; it deliberately does NOT attempt to record an
/// Upload/Download inside the session (that remains the open Stage-0 fallback decision in the
/// plan if a future stage needs it).
/// </summary>
public sealed class GpuResidencyStage0PocTests
{
    [Fact]
    public void TwoChainedSgemmDispatches_RecordedInOneSession_MatchImmediatePerOpBaseline()
    {
        const int M = 4, K = 8, K2 = 8, N = 4;
        var rng = new Random(42);
        float[] x = new float[M * K];
        float[] w1 = new float[K2 * K]; // [K2 rows, K cols] -- Sgemm's B is [N,K] row-major
        float[] w2 = new float[N * K2];
        foreach (ref var v in x.AsSpan()) v = (float)(rng.NextDouble() * 2 - 1);
        foreach (ref var v in w1.AsSpan()) v = (float)(rng.NextDouble() * 2 - 1);
        foreach (ref var v in w2.AsSpan()) v = (float)(rng.NextDouble() * 2 - 1);

        // Real CPU reference: y = x @ w1^T [M,K2], z = y @ w2^T [M,N]
        float[] yRef = new float[M * K2];
        for (int m = 0; m < M; m++)
            for (int n = 0; n < K2; n++)
            {
                float acc = 0f;
                for (int k = 0; k < K; k++) acc += x[m * K + k] * w1[n * K + k];
                yRef[m * K2 + n] = acc;
            }
        float[] zRef = new float[M * N];
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
            {
                float acc = 0f;
                for (int k = 0; k < K2; k++) acc += yRef[m * K2 + k] * w2[n * K2 + k];
                zRef[m * N + n] = acc;
            }

        using var backend = new VulkanBackend();
        var xGpu = backend.Upload(x, TensorShape.D1(x.Length));
        var w1Gpu = backend.Upload(w1, TensorShape.D1(w1.Length));
        var w2Gpu = backend.Upload(w2, TensorShape.D1(w2.Length));
        var yGpu = backend.Allocate(TensorShape.D1(M * K2));
        var zGpu = backend.Allocate(TensorShape.D1(M * N));

        try
        {
            // The actual proof: BeginRecord -> two dependent Sgemm dispatches, `yGpu` consumed
            // directly by the second dispatch as a Tensor handle, no Upload/Download of `yGpu`
            // at any point -- only a RecordBarrier between them.
            backend.BeginRecord();
            backend.Sgemm(yGpu, xGpu, w1Gpu, M, K, K2);
            backend.RecordBarrier();
            backend.Sgemm(zGpu, yGpu, w2Gpu, M, K2, N);
            backend.EndRecordAndSubmit();

            var zResult = new float[M * N];
            backend.Download(zGpu, zResult);

            float maxAbsDiff = 0f;
            for (int i = 0; i < zResult.Length; i++)
                maxAbsDiff = Math.Max(maxAbsDiff, Math.Abs(zResult[i] - zRef[i]));

            Assert.True(maxAbsDiff < 1e-3f,
                $"Recorded chained-dispatch result diverges from the CPU reference by {maxAbsDiff} -- the residency mechanism itself may be broken, not just a precision artifact.");
        }
        finally
        {
            backend.Free(xGpu);
            backend.Free(w1Gpu);
            backend.Free(w2Gpu);
            backend.Free(yGpu);
            backend.Free(zGpu);
        }
    }
}
