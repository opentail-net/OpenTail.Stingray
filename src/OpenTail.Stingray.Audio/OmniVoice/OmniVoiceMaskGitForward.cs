using System.Numerics.Tensors;
using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.OmniVoice;

/// <summary>
/// Real, self-contained (non-`IForwardPass`) forward pass for OmniVoice's own MaskGIT transformer,
/// ported from `generator.cpp`'s `decoder_layer` (not guessed): standard Qwen3-style GQA + RoPE-
/// NEOX + per-head q/k RMSNorm + bias-free SwiGLU MLP, but FULLY NON-CAUSAL -- every position
/// attends to every other position with no masking at all (confirmed real: the reference's own
/// `attention_mask` zeroes the WHOLE valid `[0,len)x[0,len)` block, no triangular restriction),
/// so this implementation simply omits any masking rather than replicating the reference's
/// padding-capacity scheme (this port has no fixed-capacity graph to pad).
///
/// <para>Layout: every activation is one flat token-major buffer `[seqLen, dim]`, and every
/// projection is ONE batched GEMM over all tokens (<see cref="DenseKernels.LinearBatchedNoBias"/>),
/// not a mat-vec per token (which re-streamed each weight matrix seqLen times; 2026-09-25 perf pass).</para>
/// </summary>
public static class OmniVoiceMaskGitForward
{
    private const int Hidden = OmniVoiceMaskGitWeights.HiddenDim;
    private const int NumHeads = OmniVoiceMaskGitWeights.NumHeads;
    private const int NumKvHeads = OmniVoiceMaskGitWeights.NumKvHeads;
    private const int HeadDim = OmniVoiceMaskGitWeights.HeadDim;
    private const int QDim = NumHeads * HeadDim;
    private const int KvDim = NumKvHeads * HeadDim;
    private const int FfDim = OmniVoiceMaskGitWeights.FfDim;

    /// <summary>Runs the full `NumLayers`-deep transformer over one flat `[seqLen, HiddenDim]`
    /// embedding buffer (frame-major), returning the FINAL RMSNorm'd hidden states in the same
    /// layout. Positions are real sequential `0..seqLen-1` (matches the reference's own
    /// `position_host[i]=i`).</summary>
    public static float[] Forward(OmniVoiceMaskGitWeights w, float[] embeddings, int seqLen)
    {
        var ws = new Workspace(seqLen);
        var hidden = (float[])embeddings.Clone();
        for (int l = 0; l < OmniVoiceMaskGitWeights.NumLayers; l++)
            Layer(w.Layers[l], hidden, seqLen, ws);

        var output = new float[seqLen * Hidden];
        RmsNormRows(hidden, output, w.FinalNorm, seqLen, Hidden);
        return output;
    }

    private sealed class Workspace(int seqLen)
    {
        public readonly float[] XNorm = new float[seqLen * Hidden];
        public readonly float[] Q = new float[seqLen * QDim];
        public readonly float[] K = new float[seqLen * KvDim];
        public readonly float[] V = new float[seqLen * KvDim];
        public readonly float[] Context = new float[seqLen * QDim];
        public readonly float[] Proj = new float[seqLen * Hidden];
        public readonly float[] Gate = new float[seqLen * FfDim];
        public readonly float[] Up = new float[seqLen * FfDim];
        public readonly (float[] Cos, float[] Sin) Rope = BuildRopeTable(seqLen);
    }

    /// <summary>One decoder layer, updating <paramref name="x"/> ([seqLen, Hidden]) in place.</summary>
    private static void Layer(OmniVoiceMaskGitLayerWeights w, float[] x, int seqLen, Workspace ws)
    {
        const int kvRepeats = NumHeads / NumKvHeads;

        RmsNormRows(x, ws.XNorm, w.InputNorm, seqLen, Hidden);
        DenseKernels.LinearBatchedNoBias(ws.XNorm, w.QProj, ws.Q, seqLen, Hidden, QDim);
        DenseKernels.LinearBatchedNoBias(ws.XNorm, w.KProj, ws.K, seqLen, Hidden, KvDim);
        DenseKernels.LinearBatchedNoBias(ws.XNorm, w.VProj, ws.V, seqLen, Hidden, KvDim);

        // Real per-head q/k RMSNorm + RoPE-NEOX, in place, per token.
        var (cos, sin) = ws.Rope;
        Parallel.For(0, seqLen, t =>
        {
            for (int h = 0; h < NumHeads; h++)
                NormRopeHead(ws.Q.AsSpan(t * QDim + h * HeadDim, HeadDim), w.QNorm, cos, sin, t);
            for (int h = 0; h < NumKvHeads; h++)
                NormRopeHead(ws.K.AsSpan(t * KvDim + h * HeadDim, HeadDim), w.KNorm, cos, sin, t);
        });

        // Real full (non-causal) scaled-dot-product attention, GQA-repeated kv, parallel over
        // (head, query-chunk) so all cores stay busy even with few heads.
        float scale = 1f / MathF.Sqrt(HeadDim);
        const int chunk = 16;
        int qChunks = (seqLen + chunk - 1) / chunk;
        var q = ws.Q; var k = ws.K; var v = ws.V; var context = ws.Context;
        Parallel.For(0, NumHeads * qChunks, () => new float[seqLen], (item, _, scores) =>
        {
            int h = item / qChunks, t0 = item % qChunks * chunk, t1 = Math.Min(seqLen, t0 + chunk);
            int kvOff = h / kvRepeats * HeadDim;
            for (int tq = t0; tq < t1; tq++)
            {
                var qv = new ReadOnlySpan<float>(q, tq * QDim + h * HeadDim, HeadDim);
                float maxScore = float.NegativeInfinity;
                for (int tk = 0; tk < seqLen; tk++)
                {
                    float dot = TensorPrimitives.Dot(qv, new ReadOnlySpan<float>(k, tk * KvDim + kvOff, HeadDim)) * scale;
                    scores[tk] = dot;
                    if (dot > maxScore) maxScore = dot;
                }
                double sum = 0;
                for (int tk = 0; tk < seqLen; tk++)
                {
                    float e = MathF.Exp(scores[tk] - maxScore);
                    scores[tk] = e;
                    sum += e;
                }
                var outHead = context.AsSpan(tq * QDim + h * HeadDim, HeadDim);
                outHead.Clear();
                for (int tk = 0; tk < seqLen; tk++)
                {
                    float p = (float)(scores[tk] / sum);
                    if (p == 0f) continue;
                    TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(v, tk * KvDim + kvOff, HeadDim), p, outHead, outHead);
                }
            }
            return scores;
        }, static _ => { });

        DenseKernels.LinearBatchedNoBias(ws.Context, w.OProj, ws.Proj, seqLen, QDim, Hidden);
        TensorPrimitives.Add(x, ws.Proj, x);

        RmsNormRows(x, ws.XNorm, w.PostNorm, seqLen, Hidden);
        DenseKernels.LinearBatchedNoBias(ws.XNorm, w.GateProj, ws.Gate, seqLen, Hidden, FfDim);
        DenseKernels.LinearBatchedNoBias(ws.XNorm, w.UpProj, ws.Up, seqLen, Hidden, FfDim);
        DenseKernels.SiluInPlace(ws.Gate);
        TensorPrimitives.Multiply(ws.Gate, ws.Up, ws.Gate);
        DenseKernels.LinearBatchedNoBias(ws.Gate, w.DownProj, ws.Proj, seqLen, FfDim, Hidden);
        TensorPrimitives.Add(x, ws.Proj, x);
    }

    /// <summary>Real RoPE-NEOX (rotate-half convention, matching `GGML_ROPE_TYPE_NEOX`): pairs
    /// `(i, i+headDim/2)` for `i` in `[0, headDim/2)`, `theta_i = ropeTheta^(-2i/headDim)`.
    /// Table `[seqLen, headDim/2]`, computed once per forward in double precision.</summary>
    private static (float[] Cos, float[] Sin) BuildRopeTable(int seqLen)
    {
        const int half = HeadDim / 2;
        var cos = new float[seqLen * half];
        var sin = new float[seqLen * half];
        for (int i = 0; i < half; i++)
        {
            double theta = Math.Pow(OmniVoiceMaskGitWeights.RopeTheta, -2.0 * i / HeadDim);
            for (int t = 0; t < seqLen; t++)
            {
                double angle = t * theta;
                cos[t * half + i] = (float)Math.Cos(angle);
                sin[t * half + i] = (float)Math.Sin(angle);
            }
        }
        return (cos, sin);
    }

    private static void NormRopeHead(Span<float> head, float[] normWeight, float[] cos, float[] sin, int position)
    {
        RmsNormInPlace(head, normWeight);
        const int half = HeadDim / 2;
        int off = position * half;
        for (int i = 0; i < half; i++)
        {
            float c = cos[off + i], s = sin[off + i];
            float a = head[i], b = head[i + half];
            head[i] = a * c - b * s;
            head[i + half] = a * s + b * c;
        }
    }

    private static void RmsNormRows(float[] src, float[] dst, float[] weight, int rows, int dim)
    {
        Parallel.For(0, rows, r =>
        {
            var x = new ReadOnlySpan<float>(src, r * dim, dim);
            var y = dst.AsSpan(r * dim, dim);
            double sumSq = TensorPrimitives.SumOfSquares(x);
            float invRms = (float)(1.0 / Math.Sqrt(sumSq / dim + OmniVoiceMaskGitWeights.RmsNormEps));
            TensorPrimitives.Multiply(x, invRms, y);
            TensorPrimitives.Multiply(y, weight, y);
        });
    }

    private static void RmsNormInPlace(Span<float> x, float[] weight)
    {
        double sumSq = TensorPrimitives.SumOfSquares((ReadOnlySpan<float>)x);
        float invRms = (float)(1.0 / Math.Sqrt(sumSq / x.Length + OmniVoiceMaskGitWeights.RmsNormEps));
        TensorPrimitives.Multiply(x, invRms, x);
        TensorPrimitives.Multiply(x, weight, x);
    }
}
