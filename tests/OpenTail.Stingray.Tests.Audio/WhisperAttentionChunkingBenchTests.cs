using System;
using System.Diagnostics;
using System.IO;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// TEMPORARY, throwaway A/B/C bench for the attention-parallelization-granularity hypothesis (see
/// docs/audio-review-progress.md's encoder operator-breakdown entry): the encoder's attention math
/// (scores+softmax+weighted-sum, separate from the Q/K/V/Out linear projections) is 22-32% of
/// encoder time and currently parallelizes over `Parallel.For(0, numHeads)` -- one work item per
/// head. Tests whether splitting each head into smaller query-position chunks (finer scheduling
/// granularity for the TPL, per external advice) actually helps, before committing to a rewrite.
/// Pure repartitioning of the SAME math (no algorithm change, no precision change) at realistic
/// shapes (Medium: 16 heads/64 headDim, Large-v3: 20 heads/64 headDim, both at 1500 frames).
/// Not part of the permanent suite, delete after use.
/// </summary>
public sealed class WhisperAttentionChunkingBenchTests : HeavyTestBase
{
    private static readonly (string Label, int Frames, int Heads, int HeadDim)[] Shapes =
    [
        ("Small (12 heads, evenly divides 12 threads)", 1500, 12, 64),
        ("Medium (16 heads)", 1500, 16, 64),
        ("Large-v3 (20 heads)", 1500, 20, 64),
    ];

    /// <summary>
    /// TEMPORARY quick perf-first check (measure before correctness, per explicit instruction):
    /// blocked/tiled attention with online (streaming) softmax over K/V blocks, vs. the current
    /// committed per-row-chunked baseline, at Large-v3's shape (the case that didn't benefit from
    /// finer chunking). Hypothesis: re-scanning the full K/V per head (768KB at headDim=64) once
    /// per query row (1500x) might not be cache-resident the way this session's prior CFM/Q8_0
    /// "should be memory-bound" guesses assumed -- but given both of those were WRONG in this
    /// exact codebase (working sets turned out to already be cache-resident), this is explicitly
    /// a "measure first, only bother with correctness if it actually wins" check.
    /// </summary>
    [Fact]
    public void Compare_ChunkedBaseline_Vs_TiledStreamingSoftmax_PerfOnly()
    {
        var report = new System.Text.StringBuilder();
        var rng = new Random(7);
        const int frames = 1500, heads = 20, headDim = 64;
        int dModel = heads * headDim;
        var q = new float[frames * dModel];
        var k = new float[frames * dModel];
        var v = new float[frames * dModel];
        for (int i = 0; i < q.Length; i++) { q[i] = (float)(rng.NextDouble() * 2 - 1); k[i] = (float)(rng.NextDouble() * 2 - 1); v[i] = (float)(rng.NextDouble() * 2 - 1); }
        float scale = 1f / MathF.Sqrt(headDim);

        var outChunked = new float[frames * dModel];
        var outTiled = new float[frames * dModel];

        var outTiled128 = new float[frames * dModel];
        var outTiled256 = new float[frames * dModel];
        double chunked = TimeBest(() => AttentionChunked(q, k, v, frames, dModel, heads, headDim, scale, outChunked, chunkSize: CeilDiv(frames, 4)));
        double tiled64 = TimeBest(() => AttentionTiledStreaming(q, k, v, frames, dModel, heads, headDim, scale, outTiled, kBlockSize: 64));
        double tiled128 = TimeBest(() => AttentionTiledStreaming(q, k, v, frames, dModel, heads, headDim, scale, outTiled128, kBlockSize: 128));
        double tiled256 = TimeBest(() => AttentionTiledStreaming(q, k, v, frames, dModel, heads, headDim, scale, outTiled256, kBlockSize: 256));

        report.AppendLine($"ProcessorCount={Environment.ProcessorCount}");
        report.AppendLine($"Large-v3 shape (20 heads, 64 headDim, 1500 frames): chunked(4/head)={chunked:F1}ms  " +
                           $"tiledStreaming(kBlock=64)={tiled64:F1}ms ({Ratio(tiled64, chunked)})  " +
                           $"tiledStreaming(kBlock=128)={tiled128:F1}ms ({Ratio(tiled128, chunked)})  " +
                           $"tiledStreaming(kBlock=256)={tiled256:F1}ms ({Ratio(tiled256, chunked)})");

        Console.Error.WriteLine(report.ToString());
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "whisper_attention_tiling_perf_only.txt"), report.ToString());

        // Perf-only: no correctness assertion here by design (see doc comment / explicit instruction).
    }

    /// <summary>
    /// Blocked attention with online (streaming, numerically-stable) softmax: processes K/V in
    /// blocks, maintaining a running max/sum/weighted-output accumulator per query row instead of
    /// materializing the full per-query score row before normalizing. One query at a time (same
    /// granularity as the chunked baseline), parallelized identically over (head, query-chunk).
    /// </summary>
    private static void AttentionTiledStreaming(float[] q, float[] k, float[] v, int frames, int dModel, int heads, int headDim, float scale, float[] output, int kBlockSize)
    {
        int chunkSize = CeilDiv(frames, 4);
        int chunksPerHead = CeilDiv(frames, chunkSize);
        int totalWorkItems = heads * chunksPerHead;

        Parallel.For(0, totalWorkItems, w =>
        {
            int h = w / chunksPerHead;
            int chunk = w % chunksPerHead;
            int headOff = h * headDim;
            int qStart = chunk * chunkSize;
            int qEnd = Math.Min(qStart + chunkSize, frames);

            var accum = new float[headDim];
            var tileScores = new float[kBlockSize];

            for (int i = qStart; i < qEnd; i++)
            {
                var querySpan = q.AsSpan(i * dModel + headOff, headDim);
                Array.Clear(accum);
                float runningMax = float.NegativeInfinity;
                float runningSum = 0f;

                for (int kStart = 0; kStart < frames; kStart += kBlockSize)
                {
                    int kEnd = Math.Min(kStart + kBlockSize, frames);
                    int blockLen = kEnd - kStart;

                    float blockMax = float.NegativeInfinity;
                    for (int jj = 0; jj < blockLen; jj++)
                    {
                        float s = TensorPrimitives.Dot(querySpan, k.AsSpan((kStart + jj) * dModel + headOff, headDim)) * scale;
                        tileScores[jj] = s;
                        if (s > blockMax) blockMax = s;
                    }

                    float newMax = MathF.Max(runningMax, blockMax);
                    float correction = runningMax == float.NegativeInfinity ? 0f : MathF.Exp(runningMax - newMax);

                    float blockSum = 0f;
                    for (int jj = 0; jj < blockLen; jj++)
                    {
                        float e = MathF.Exp(tileScores[jj] - newMax);
                        tileScores[jj] = e;
                        blockSum += e;
                    }

                    for (int d = 0; d < headDim; d++) accum[d] *= correction;
                    for (int jj = 0; jj < blockLen; jj++)
                    {
                        var vRow = v.AsSpan((kStart + jj) * dModel + headOff, headDim);
                        TensorPrimitives.MultiplyAdd(vRow, tileScores[jj], accum, accum);
                    }

                    runningSum = runningSum * correction + blockSum;
                    runningMax = newMax;
                }

                var outSpan = output.AsSpan(i * dModel + headOff, headDim);
                float invSum = 1f / runningSum;
                for (int d = 0; d < headDim; d++) outSpan[d] = accum[d] * invSum;
            }
        });
    }

    [Fact]
    public void Compare_HeadOnly_Vs_4ChunksPerHead_Vs_8ChunksPerHead()
    {
        var report = new System.Text.StringBuilder();
        report.AppendLine($"ProcessorCount={Environment.ProcessorCount}");
        var rng = new Random(42);

        foreach (var (label, frames, heads, headDim) in Shapes)
        {
            int dModel = heads * headDim;
            var q = new float[frames * dModel];
            var k = new float[frames * dModel];
            var v = new float[frames * dModel];
            for (int i = 0; i < q.Length; i++) { q[i] = (float)(rng.NextDouble() * 2 - 1); k[i] = (float)(rng.NextDouble() * 2 - 1); v[i] = (float)(rng.NextDouble() * 2 - 1); }
            float scale = 1f / MathF.Sqrt(headDim);

            var outBaseline = new float[frames * dModel];
            var out4 = new float[frames * dModel];
            var out8 = new float[frames * dModel];
            var out16 = new float[frames * dModel];

            double baseline = TimeBest(() => AttentionHeadOnly(q, k, v, frames, dModel, heads, headDim, scale, outBaseline));
            double c4 = TimeBest(() => AttentionChunked(q, k, v, frames, dModel, heads, headDim, scale, out4, chunkSize: CeilDiv(frames, 4)));
            double c8 = TimeBest(() => AttentionChunked(q, k, v, frames, dModel, heads, headDim, scale, out8, chunkSize: CeilDiv(frames, 8)));
            double c16 = TimeBest(() => AttentionChunked(q, k, v, frames, dModel, heads, headDim, scale, out16, chunkSize: CeilDiv(frames, 16)));

            AssertClose(outBaseline, out4, label + " 4-chunk vs baseline");
            AssertClose(outBaseline, out8, label + " 8-chunk vs baseline");
            AssertClose(outBaseline, out16, label + " 16-chunk vs baseline");

            report.AppendLine($"{label}: baseline({heads}tasks)={baseline:F1}ms  4chunks/head({heads * 4}tasks)={c4:F1}ms({Ratio(c4, baseline)})  " +
                               $"8chunks/head({heads * 8}tasks)={c8:F1}ms({Ratio(c8, baseline)})  16chunks/head({heads * 16}tasks)={c16:F1}ms({Ratio(c16, baseline)})");
        }

        var reportText = report.ToString();
        Console.Error.WriteLine(reportText);
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "whisper_attention_chunking_result.txt"), reportText);
    }

    private static int CeilDiv(int a, int b) => (a + b - 1) / b;
    private static string Ratio(double variant, double baseline) => $"{variant / baseline:F2}x";

    /// <summary>Current production parallelization: one Parallel.For work item per head (copied from WhisperEncoder.ComputeMultiHeadSelfAttentionReal, not modified).</summary>
    private static void AttentionHeadOnly(float[] q, float[] k, float[] v, int frames, int dModel, int heads, int headDim, float scale, float[] output)
    {
        Parallel.For(0, heads, h =>
        {
            int headOff = h * headDim;
            var scores = new float[frames];
            for (int i = 0; i < frames; i++)
            {
                var querySpan = q.AsSpan(i * dModel + headOff, headDim);
                for (int j = 0; j < frames; j++)
                    scores[j] = TensorPrimitives.Dot(querySpan, k.AsSpan(j * dModel + headOff, headDim)) * scale;
                TensorPrimitives.SoftMax(scores.AsSpan(0, frames), scores.AsSpan(0, frames));
                var weighted = output.AsSpan(i * dModel + headOff, headDim);
                weighted.Clear();
                for (int j = 0; j < frames; j++)
                    TensorPrimitives.MultiplyAdd(v.AsSpan(j * dModel + headOff, headDim), scores[j], weighted, weighted);
            }
        });
    }

    /// <summary>Same exact math, but partitioned over (head, contiguous query-chunk) instead of head alone -- each work item owns a contiguous range of query rows within one head, so no synchronization/reduction is needed across chunks.</summary>
    private static void AttentionChunked(float[] q, float[] k, float[] v, int frames, int dModel, int heads, int headDim, float scale, float[] output, int chunkSize)
    {
        int chunksPerHead = CeilDiv(frames, chunkSize);
        int totalWorkItems = heads * chunksPerHead;

        Parallel.For(0, totalWorkItems, w =>
        {
            int h = w / chunksPerHead;
            int chunk = w % chunksPerHead;
            int headOff = h * headDim;
            int qStart = chunk * chunkSize;
            int qEnd = Math.Min(qStart + chunkSize, frames);

            var scores = new float[frames];
            for (int i = qStart; i < qEnd; i++)
            {
                var querySpan = q.AsSpan(i * dModel + headOff, headDim);
                for (int j = 0; j < frames; j++)
                    scores[j] = TensorPrimitives.Dot(querySpan, k.AsSpan(j * dModel + headOff, headDim)) * scale;
                TensorPrimitives.SoftMax(scores.AsSpan(0, frames), scores.AsSpan(0, frames));
                var weighted = output.AsSpan(i * dModel + headOff, headDim);
                weighted.Clear();
                for (int j = 0; j < frames; j++)
                    TensorPrimitives.MultiplyAdd(v.AsSpan(j * dModel + headOff, headDim), scores[j], weighted, weighted);
            }
        });
    }

    private static double TimeBest(Action run)
    {
        run(); // warmup
        const int n = 5;
        double best = double.MaxValue;
        for (int i = 0; i < n; i++)
        {
            var sw = Stopwatch.StartNew();
            run();
            sw.Stop();
            if (sw.Elapsed.TotalMilliseconds < best) best = sw.Elapsed.TotalMilliseconds;
        }
        return best;
    }

    private static void AssertClose(float[] expected, float[] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float tol = 1e-3f + 1e-4f * MathF.Abs(expected[i]);
            if (diff > tol)
                Assert.Fail($"{label}: mismatch at index {i}: expected {expected[i]}, got {actual[i]}");
        }
    }
}
