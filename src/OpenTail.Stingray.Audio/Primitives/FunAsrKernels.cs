using System;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>
/// Shared math for FunASR's SAN-M encoder (<c>FunASR/FunAsrEncoder.cs</c>) and decoder
/// (<c>FunASR/FunAsrRealDecoder.cs</c>) -- extracted after both were independently ported and
/// golden-verified (see docs/audio-review-progress.md's FunASR section), following the same
/// DRY-after-verification pattern already used for <see cref="S3GenConformerKernels"/>. Both
/// pipelines had copy-pasted identical <c>Linear</c>/<c>LinearNoBias</c>/<c>LayerNorm</c>/
/// <c>SoftmaxInPlace</c> helpers, and the FSMN depthwise-conv memory term (encoder's self-
/// attention branch, decoder's FSMN-only self-attention) is bit-for-bit the same algorithm in
/// both: symmetric-padded per-channel Conv1d, residual = the conv's own input.
///
/// <para><b>Performance pass, same fire</b>: the original per-pipeline <c>Linear</c> looped
/// output channels with a scalar <c>SimdKernels.DotF32</c> call each -- missing
/// <see cref="SimdKernels.MatVecF32"/>'s own internal <c>Parallel.For</c> over output rows
/// (kicks in at outDim &gt;= 64, which every Linear call in this pipeline hits: QKV projections
/// are 512-&gt;1536, FFN is 512-&gt;2048, vocab projection is 512-&gt;8404). Routing through
/// <see cref="SimdKernels.MatVecF32"/> here picks that up for free. Also parallelizes multi-head
/// attention over heads (<c>Parallel.For(0, heads, ...)</c>) and the FSMN conv over channels,
/// matching the per-head/per-channel parallelization convention already used by
/// <c>WhisperEncoder.cs</c>.</para>
/// </summary>
public static class FunAsrKernels
{
    /// <summary>weight @ input + bias, output dim rows. Routes through <see cref="SimdKernels.MatVecF32"/> so large output dims (QKV/FFN/vocab projections) get its internal per-row parallelization.</summary>
    public static unsafe float[] Linear(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* wp = weight, xp = input, op = output)
        {
            SimdKernels.MatVecF32(op, wp, xp, outDim, inDim);
        }
        TensorPrimitives.Add((ReadOnlySpan<float>)output, bias, output);
        return output;
    }

    /// <summary>Same as <see cref="Linear"/> but with no bias term (used by the decoder FFN's second projection, which has no bias in the real checkpoint).</summary>
    public static unsafe float[] LinearNoBias(float[] input, float[] weight, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* wp = weight, xp = input, op = output)
        {
            SimdKernels.MatVecF32(op, wp, xp, outDim, inDim);
        }
        return output;
    }

    public static float[] LayerNorm(float[] x, float[] weight, float[] bias, float eps = 1e-12f)
    {
        int n = x.Length;
        float mean = TensorPrimitives.Sum((ReadOnlySpan<float>)x) / n;
        float variance = 0f;
        for (int i = 0; i < n; i++) { float d = x[i] - mean; variance += d * d; }
        variance /= n;
        float invStd = 1f / MathF.Sqrt(variance + eps);

        var output = new float[n];
        for (int i = 0; i < n; i++)
            output[i] = (x[i] - mean) * invStd * weight[i] + bias[i];
        return output;
    }

    public static void SoftmaxInPlace(float[] scores)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < scores.Length; i++) if (scores[i] > max) max = scores[i];
        float sum = 0f;
        for (int i = 0; i < scores.Length; i++)
        {
            float e = MathF.Exp(scores[i] - max);
            scores[i] = e;
            sum += e;
        }
        float invSum = 1f / sum;
        for (int i = 0; i < scores.Length; i++) scores[i] *= invSum;
    }

    /// <summary>Real FSMN depthwise (per-channel) Conv1d memory term: symmetric pad, residual add of the conv's OWN input (not a caller-supplied different residual) -- shared verbatim by the encoder's `forward_fsmn` self-attention branch and the decoder's FSMN-only self-attention. Parallelized over channels.</summary>
    public static float[][] FsmnDepthwiseConv(float[][] x, float[] fsmnWeight, int kernel)
    {
        int t = x.Length;
        int c = x[0].Length;
        int left = (kernel - 1) / 2;

        var output = new float[t][];
        for (int ti = 0; ti < t; ti++) output[ti] = new float[c];

        Parallel.For(0, c, ch =>
        {
            int wBase = ch * kernel;
            for (int ti = 0; ti < t; ti++)
            {
                float sum = 0f;
                for (int kk = 0; kk < kernel; kk++)
                {
                    int srcT = ti - left + kk;
                    if ((uint)srcT < (uint)t) sum += x[srcT][ch] * fsmnWeight[wBase + kk];
                }
                output[ti][ch] = sum + x[ti][ch];
            }
        });
        return output;
    }

    /// <summary>Standard scaled-dot-product multi-head attention core (query length may differ from key/value length, e.g. decoder cross-attention). Parallelized over heads.</summary>
    public static float[][] MultiHeadAttention(float[][] q, float[][] k, float[][] v, int heads)
    {
        int tq = q.Length;
        int tk = k.Length;
        int nFeat = q[0].Length;
        int dK = nFeat / heads;
        float scale = MathF.Pow(dK, -0.5f);

        var context = new float[tq][];
        for (int i = 0; i < tq; i++) context[i] = new float[nFeat];

        Parallel.For(0, heads, h =>
        {
            int off = h * dK;
            var scores = new float[tk];
            for (int i = 0; i < tq; i++)
            {
                for (int j = 0; j < tk; j++)
                    scores[j] = TensorPrimitives.Dot(q[i].AsSpan(off, dK), k[j].AsSpan(off, dK)) * scale;
                SoftmaxInPlace(scores);

                var ctxSpan = context[i].AsSpan(off, dK);
                for (int j = 0; j < tk; j++)
                    TensorPrimitives.MultiplyAdd(v[j].AsSpan(off, dK), scores[j], ctxSpan, ctxSpan);
            }
        });
        return context;
    }
}
