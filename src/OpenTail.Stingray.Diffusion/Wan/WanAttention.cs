using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace OpenTail.Stingray.Diffusion.Wan;

/// <summary>
/// High-performance cache-friendly MultiHeadAttention kernel for Wan 2.1 DiT.
/// Transposes input to head-contiguous memory [heads, seq, headDim] to eliminate 6KB memory strides
/// and keep inner token dot products strictly within sequential L1/L2 cache lines.
/// </summary>
public static class WanAttention
{
    public static unsafe void TiledMultiHeadAttention(
        ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v, Span<float> output,
        int qSeq, int kvSeq, int numHeads, int headDim)
    {
        int totalQ = qSeq * numHeads * headDim;
        int totalKV = kvSeq * numHeads * headDim;

        float[] qTrans = ArrayPool<float>.Shared.Rent(totalQ);
        float[] kTrans = ArrayPool<float>.Shared.Rent(totalKV);
        float[] vTrans = ArrayPool<float>.Shared.Rent(totalKV);
        float[] outTrans = ArrayPool<float>.Shared.Rent(totalQ);

        try
        {
            fixed (float* pQ = q, pK = k, pV = v, pOut = output,
                          pQT = qTrans, pKT = kTrans, pVT = vTrans, pOT = outTrans)
            {
                TransposeToHeadContiguous(pQ, pQT, qSeq, numHeads, headDim);
                TransposeToHeadContiguous(pK, pKT, kvSeq, numHeads, headDim);
                TransposeToHeadContiguous(pV, pVT, kvSeq, numHeads, headDim);

                float scale = 1.0f / MathF.Sqrt(headDim);
                nint qtAddr = (nint)pQT;
                nint ktAddr = (nint)pKT;
                nint vtAddr = (nint)pVT;
                nint otAddr = (nint)pOT;

                Parallel.For(0, numHeads, h =>
                {
                    float* qHead = (float*)qtAddr + (long)h * qSeq * headDim;
                    float* kHead = (float*)ktAddr + (long)h * kvSeq * headDim;
                    float* vHead = (float*)vtAddr + (long)h * kvSeq * headDim;
                    float* oHead = (float*)otAddr + (long)h * qSeq * headDim;

                    if (Avx2.IsSupported && Fma.IsSupported && headDim == 128)
                    {
                        ComputeHeadAttentionAvx2_Dim128(qHead, kHead, vHead, oHead, qSeq, kvSeq, scale);
                    }
                    else if (Avx2.IsSupported && Fma.IsSupported && (headDim % 8 == 0))
                    {
                        ComputeHeadAttentionAvx2_Generic(qHead, kHead, vHead, oHead, qSeq, kvSeq, headDim, scale);
                    }
                    else
                    {
                        ComputeHeadAttentionScalar(qHead, kHead, vHead, oHead, qSeq, kvSeq, headDim, scale);
                    }
                });

                TransposeFromHeadContiguous(pOT, pOut, qSeq, numHeads, headDim);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(qTrans);
            ArrayPool<float>.Shared.Return(kTrans);
            ArrayPool<float>.Shared.Return(vTrans);
            ArrayPool<float>.Shared.Return(outTrans);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void ComputeHeadAttentionAvx2_Dim128(
        float* qHead, float* kHead, float* vHead, float* oHead,
        int qSeq, int kvSeq, float scale)
    {
        const int headDim = 128;
        float* scores;
        float* heapScores = null;
        if (kvSeq <= 4096)
        {
            float* stackBuf = stackalloc float[kvSeq];
            scores = stackBuf;
        }
        else
        {
            heapScores = (float*)System.Runtime.InteropServices.NativeMemory.Alloc((nuint)kvSeq, sizeof(float));
            scores = heapScores;
        }

        try
        {
            var vScale = Vector256.Create(scale);

            for (int i = 0; i < qSeq; i++)
            {
                float* qRow = qHead + (long)i * headDim;

                // Load Q row into 16 YMM registers
                Vector256<float> q0 = Avx.LoadVector256(qRow + 0);
                Vector256<float> q1 = Avx.LoadVector256(qRow + 8);
                Vector256<float> q2 = Avx.LoadVector256(qRow + 16);
                Vector256<float> q3 = Avx.LoadVector256(qRow + 24);
                Vector256<float> q4 = Avx.LoadVector256(qRow + 32);
                Vector256<float> q5 = Avx.LoadVector256(qRow + 40);
                Vector256<float> q6 = Avx.LoadVector256(qRow + 48);
                Vector256<float> q7 = Avx.LoadVector256(qRow + 56);
                Vector256<float> q8 = Avx.LoadVector256(qRow + 64);
                Vector256<float> q9 = Avx.LoadVector256(qRow + 72);
                Vector256<float> q10 = Avx.LoadVector256(qRow + 80);
                Vector256<float> q11 = Avx.LoadVector256(qRow + 88);
                Vector256<float> q12 = Avx.LoadVector256(qRow + 96);
                Vector256<float> q13 = Avx.LoadVector256(qRow + 104);
                Vector256<float> q14 = Avx.LoadVector256(qRow + 112);
                Vector256<float> q15 = Avx.LoadVector256(qRow + 120);

                float maxScore = float.NegativeInfinity;

                for (int j = 0; j < kvSeq; j++)
                {
                    float* kRow = kHead + (long)j * headDim;

                    Vector256<float> dot0 = Avx.Multiply(q0, Avx.LoadVector256(kRow + 0));
                    dot0 = Fma.MultiplyAdd(q1, Avx.LoadVector256(kRow + 8), dot0);
                    dot0 = Fma.MultiplyAdd(q2, Avx.LoadVector256(kRow + 16), dot0);
                    dot0 = Fma.MultiplyAdd(q3, Avx.LoadVector256(kRow + 24), dot0);
                    dot0 = Fma.MultiplyAdd(q4, Avx.LoadVector256(kRow + 32), dot0);
                    dot0 = Fma.MultiplyAdd(q5, Avx.LoadVector256(kRow + 40), dot0);
                    dot0 = Fma.MultiplyAdd(q6, Avx.LoadVector256(kRow + 48), dot0);
                    dot0 = Fma.MultiplyAdd(q7, Avx.LoadVector256(kRow + 56), dot0);

                    Vector256<float> dot1 = Avx.Multiply(q8, Avx.LoadVector256(kRow + 64));
                    dot1 = Fma.MultiplyAdd(q9, Avx.LoadVector256(kRow + 72), dot1);
                    dot1 = Fma.MultiplyAdd(q10, Avx.LoadVector256(kRow + 80), dot1);
                    dot1 = Fma.MultiplyAdd(q11, Avx.LoadVector256(kRow + 88), dot1);
                    dot1 = Fma.MultiplyAdd(q12, Avx.LoadVector256(kRow + 96), dot1);
                    dot1 = Fma.MultiplyAdd(q13, Avx.LoadVector256(kRow + 104), dot1);
                    dot1 = Fma.MultiplyAdd(q14, Avx.LoadVector256(kRow + 112), dot1);
                    dot1 = Fma.MultiplyAdd(q15, Avx.LoadVector256(kRow + 120), dot1);

                    Vector256<float> dotTot = Avx.Add(dot0, dot1);
                    // Horizontal sum across 8 floats in dotTot
                    Vector128<float> low = dotTot.GetLower();
                    Vector128<float> high = dotTot.GetUpper();
                    Vector128<float> sum128 = Sse.Add(low, high);
                    sum128 = Sse3.HorizontalAdd(sum128, sum128);
                    sum128 = Sse3.HorizontalAdd(sum128, sum128);
                    float dot = sum128.ToScalar() * scale;

                    scores[j] = dot;
                    if (dot > maxScore) maxScore = dot;
                }

                float sumExp = 0f;
                for (int j = 0; j < kvSeq; j++)
                {
                    float e = MathF.Exp(scores[j] - maxScore);
                    scores[j] = e;
                    sumExp += e;
                }
                float invSum = 1f / sumExp;

                // FlashAttention-style register accumulation for O row (16 YMM registers)
                Vector256<float> o0 = Vector256<float>.Zero;
                Vector256<float> o1 = Vector256<float>.Zero;
                Vector256<float> o2 = Vector256<float>.Zero;
                Vector256<float> o3 = Vector256<float>.Zero;
                Vector256<float> o4 = Vector256<float>.Zero;
                Vector256<float> o5 = Vector256<float>.Zero;
                Vector256<float> o6 = Vector256<float>.Zero;
                Vector256<float> o7 = Vector256<float>.Zero;
                Vector256<float> o8 = Vector256<float>.Zero;
                Vector256<float> o9 = Vector256<float>.Zero;
                Vector256<float> o10 = Vector256<float>.Zero;
                Vector256<float> o11 = Vector256<float>.Zero;
                Vector256<float> o12 = Vector256<float>.Zero;
                Vector256<float> o13 = Vector256<float>.Zero;
                Vector256<float> o14 = Vector256<float>.Zero;
                Vector256<float> o15 = Vector256<float>.Zero;

                for (int j = 0; j < kvSeq; j++)
                {
                    float s = scores[j] * invSum;
                    if (s == 0f) continue;
                    var vs = Vector256.Create(s);
                    float* vRow = vHead + (long)j * headDim;

                    o0 = Fma.MultiplyAdd(vs, Avx.LoadVector256(vRow + 0), o0);
                    o1 = Fma.MultiplyAdd(vs, Avx.LoadVector256(vRow + 8), o1);
                    o2 = Fma.MultiplyAdd(vs, Avx.LoadVector256(vRow + 16), o2);
                    o3 = Fma.MultiplyAdd(vs, Avx.LoadVector256(vRow + 24), o3);
                    o4 = Fma.MultiplyAdd(vs, Avx.LoadVector256(vRow + 32), o4);
                    o5 = Fma.MultiplyAdd(vs, Avx.LoadVector256(vRow + 40), o5);
                    o6 = Fma.MultiplyAdd(vs, Avx.LoadVector256(vRow + 48), o6);
                    o7 = Fma.MultiplyAdd(vs, Avx.LoadVector256(vRow + 56), o7);
                    o8 = Fma.MultiplyAdd(vs, Avx.LoadVector256(vRow + 64), o8);
                    o9 = Fma.MultiplyAdd(vs, Avx.LoadVector256(vRow + 72), o9);
                    o10 = Fma.MultiplyAdd(vs, Avx.LoadVector256(vRow + 80), o10);
                    o11 = Fma.MultiplyAdd(vs, Avx.LoadVector256(vRow + 88), o11);
                    o12 = Fma.MultiplyAdd(vs, Avx.LoadVector256(vRow + 96), o12);
                    o13 = Fma.MultiplyAdd(vs, Avx.LoadVector256(vRow + 104), o13);
                    o14 = Fma.MultiplyAdd(vs, Avx.LoadVector256(vRow + 112), o14);
                    o15 = Fma.MultiplyAdd(vs, Avx.LoadVector256(vRow + 120), o15);
                }

                float* oRow = oHead + (long)i * headDim;
                Avx.Store(oRow + 0, o0);
                Avx.Store(oRow + 8, o1);
                Avx.Store(oRow + 16, o2);
                Avx.Store(oRow + 24, o3);
                Avx.Store(oRow + 32, o4);
                Avx.Store(oRow + 40, o5);
                Avx.Store(oRow + 48, o6);
                Avx.Store(oRow + 56, o7);
                Avx.Store(oRow + 64, o8);
                Avx.Store(oRow + 72, o9);
                Avx.Store(oRow + 80, o10);
                Avx.Store(oRow + 88, o11);
                Avx.Store(oRow + 96, o12);
                Avx.Store(oRow + 104, o13);
                Avx.Store(oRow + 112, o14);
                Avx.Store(oRow + 120, o15);
            }
        }
        finally
        {
            if (heapScores != null)
                System.Runtime.InteropServices.NativeMemory.Free(heapScores);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void ComputeHeadAttentionAvx2_Generic(
        float* qHead, float* kHead, float* vHead, float* oHead,
        int qSeq, int kvSeq, int headDim, float scale)
    {
        float* scores;
        float* heapScores = null;
        if (kvSeq <= 4096)
        {
            float* stackBuf = stackalloc float[kvSeq];
            scores = stackBuf;
        }
        else
        {
            heapScores = (float*)System.Runtime.InteropServices.NativeMemory.Alloc((nuint)kvSeq, sizeof(float));
            scores = heapScores;
        }

        try
        {
            int numVecs = headDim / 8;

            for (int i = 0; i < qSeq; i++)
            {
                float* qRow = qHead + (long)i * headDim;
                float maxScore = float.NegativeInfinity;

                for (int j = 0; j < kvSeq; j++)
                {
                    float* kRow = kHead + (long)j * headDim;
                    Vector256<float> dotVec = Vector256<float>.Zero;

                    for (int v = 0; v < numVecs; v++)
                    {
                        dotVec = Fma.MultiplyAdd(
                            Avx.LoadVector256(qRow + v * 8),
                            Avx.LoadVector256(kRow + v * 8),
                            dotVec);
                    }

                    Vector128<float> low = dotVec.GetLower();
                    Vector128<float> high = dotVec.GetUpper();
                    Vector128<float> sum128 = Sse.Add(low, high);
                    sum128 = Sse3.HorizontalAdd(sum128, sum128);
                    sum128 = Sse3.HorizontalAdd(sum128, sum128);
                    float dot = sum128.ToScalar() * scale;

                    scores[j] = dot;
                    if (dot > maxScore) maxScore = dot;
                }

                float sumExp = 0f;
                for (int j = 0; j < kvSeq; j++)
                {
                    float e = MathF.Exp(scores[j] - maxScore);
                    scores[j] = e;
                    sumExp += e;
                }
                float invSum = 1f / sumExp;

                float* oRow = oHead + (long)i * headDim;
                for (int v = 0; v < numVecs; v++)
                {
                    Avx.Store(oRow + v * 8, Vector256<float>.Zero);
                }

                for (int j = 0; j < kvSeq; j++)
                {
                    float s = scores[j] * invSum;
                    if (s == 0f) continue;
                    var vs = Vector256.Create(s);
                    float* vRow = vHead + (long)j * headDim;

                    for (int v = 0; v < numVecs; v++)
                    {
                        var existing = Avx.LoadVector256(oRow + v * 8);
                        var vVal = Avx.LoadVector256(vRow + v * 8);
                        Avx.Store(oRow + v * 8, Fma.MultiplyAdd(vs, vVal, existing));
                    }
                }
            }
        }
        finally
        {
            if (heapScores != null)
                System.Runtime.InteropServices.NativeMemory.Free(heapScores);
        }
    }

    private static unsafe void ComputeHeadAttentionScalar(
        float* qHead, float* kHead, float* vHead, float* oHead,
        int qSeq, int kvSeq, int headDim, float scale)
    {
        float* scores;
        float* heapScores = null;
        if (kvSeq <= 4096)
        {
            float* stackBuf = stackalloc float[kvSeq];
            scores = stackBuf;
        }
        else
        {
            heapScores = (float*)System.Runtime.InteropServices.NativeMemory.Alloc((nuint)kvSeq, sizeof(float));
            scores = heapScores;
        }

        try
        {
            for (int i = 0; i < qSeq; i++)
            {
                float* qRow = qHead + (long)i * headDim;
                var qSpan = new ReadOnlySpan<float>(qRow, headDim);
                float maxScore = float.NegativeInfinity;

                for (int j = 0; j < kvSeq; j++)
                {
                    float* kRow = kHead + (long)j * headDim;
                    var kSpan = new ReadOnlySpan<float>(kRow, headDim);
                    float dot = TensorPrimitives.Dot(qSpan, kSpan) * scale;
                    scores[j] = dot;
                    if (dot > maxScore) maxScore = dot;
                }

                float sumExp = 0f;
                for (int j = 0; j < kvSeq; j++)
                {
                    scores[j] = MathF.Exp(scores[j] - maxScore);
                    sumExp += scores[j];
                }
                float invSum = 1f / sumExp;

                float* oRow = oHead + (long)i * headDim;
                var outSpan = new Span<float>(oRow, headDim);
                outSpan.Clear();

                for (int j = 0; j < kvSeq; j++)
                {
                    float s = scores[j] * invSum;
                    if (s == 0f) continue;
                    float* vRow = vHead + (long)j * headDim;
                    var vSpan = new ReadOnlySpan<float>(vRow, headDim);
                    TensorPrimitives.MultiplyAdd(vSpan, s, outSpan, outSpan);
                }
            }
        }
        finally
        {
            if (heapScores != null)
                System.Runtime.InteropServices.NativeMemory.Free(heapScores);
        }
    }

    public static unsafe void TransposeToHeadContiguous(float* src, float* dst, int seq, int heads, int headDim)
    {
        int dim = heads * headDim;
        nint srcAddr = (nint)src;
        nint dstAddr = (nint)dst;
        Parallel.For(0, heads, h =>
        {
            float* pSrc = (float*)srcAddr;
            float* pDst = (float*)dstAddr;
            for (int t = 0; t < seq; t++)
            {
                float* srcToken = pSrc + (long)t * dim + (long)h * headDim;
                float* dstHead = pDst + ((long)h * seq + t) * headDim;
                for (int d = 0; d < headDim; d++)
                    dstHead[d] = srcToken[d];
            }
        });
    }

    public static unsafe void TransposeFromHeadContiguous(float* src, float* dst, int seq, int heads, int headDim)
    {
        int dim = heads * headDim;
        nint srcAddr = (nint)src;
        nint dstAddr = (nint)dst;
        Parallel.For(0, heads, h =>
        {
            float* pSrc = (float*)srcAddr;
            float* pDst = (float*)dstAddr;
            for (int t = 0; t < seq; t++)
            {
                float* srcHead = pSrc + ((long)h * seq + t) * headDim;
                float* dstToken = pDst + (long)t * dim + (long)h * headDim;
                for (int d = 0; d < headDim; d++)
                    dstToken[d] = srcHead[d];
            }
        });
    }
}
