using System.Numerics.Tensors;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OpenTail.Stingray.Audio.Primitives;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.VoxCpm2;

/// <summary>
/// Shared bidirectional (non-causal) MiniCPM transformer stack, factored out of
/// <see cref="VoxCpm2LocalEncoder"/> since VoxCPM2's local encoder AND its DiT estimator's
/// decoder (`generator.cpp`'s `VoxCPM2DiTEstimatorRuntime`) both call the SAME real
/// `minicpm_transformer(..., is_causal=false)` helper over the SAME real config shape
/// (`hidden_dim=1024`, `ffn_dim=4096`, `num_heads=16`, `num_layers=12`, `kv_channels=128`,
/// `num_key_value_heads=2` -- confirmed identical for both `encoder_config` and `dit_config` in
/// the real checkpoint's `config.json`), just with different learned weights and different input
/// sequence construction. Real NEOX+longrope RoPE (see <see cref="VoxCpm2LocalEncoder"/>'s doc
/// comment for the full derivation) is shared too.
/// </summary>
public static class VoxCpm2MiniCpmBidirectionalStack
{
    public const int HiddenDim = 1024;
    public const int NumHeads = 16;
    public const int NumKvHeads = 2;
    public const int HeadDim = 128;
    public const int FfnDim = 4096;
    public const float RopeTheta = 10000f;
    public const float RmsNormEps = 1e-5f;

    public static unsafe float[][] Run(float[][] hidden, VoxCpm2MiniCpmLayerWeights[] layers, float[] finalNorm, int seqLen)
    {
        int halfDim = HeadDim / 2;
        var cos = new float[seqLen * halfDim];
        var sin = new float[seqLen * halfDim];
        fixed (float* cosPtr = cos, sinPtr = sin, freqPtr = VoxCpm2LocalEncoder.RopeShortFactor)
            SimdKernels.BuildRopeTable(cosPtr, sinPtr, seqLen, HeadDim, RopeTheta, freqPtr);

        int qOut = NumHeads * HeadDim;
        int kvOut = NumKvHeads * HeadDim;
        int kvRepeats = NumHeads / NumKvHeads;
        float scale = 1f / MathF.Sqrt(HeadDim);

        var current = new float[seqLen * HiddenDim];
        for (int t = 0; t < seqLen; t++)
            Array.Copy(hidden[t], 0, current, t * HiddenDim, HiddenDim);

        var normed = new float[seqLen * HiddenDim];
        var q = new float[seqLen * qOut];
        var k = new float[seqLen * kvOut];
        var v = new float[seqLen * kvOut];
        var context = new float[seqLen * qOut];
        var attnOut = new float[seqLen * HiddenDim];
        var ffnNormed = new float[seqLen * HiddenDim];
        var gate = new float[seqLen * FfnDim];
        var up = new float[seqLen * FfnDim];
        var down = new float[seqLen * HiddenDim];
        float* scores = stackalloc float[seqLen];

        fixed (float* cosPtr = cos, sinPtr = sin,
               currentPtr = current, normedPtr = normed,
               qPtr = q, kPtr = k, vPtr = v,
               contextPtr = context, attnOutPtr = attnOut,
               ffnNormedPtr = ffnNormed, gatePtr = gate, upPtr = up, downPtr = down,
               finalNormPtr = finalNorm)
        {
            foreach (var layer in layers)
            {
                fixed (float* inNormPtr = layer.InputNorm,
                       qWPtr = layer.QProjWeight, kWPtr = layer.KProjWeight, vWPtr = layer.VProjWeight,
                       oWPtr = layer.OProjWeight, postNormPtr = layer.PostNorm,
                       gateWPtr = layer.GateProjWeight, upWPtr = layer.UpProjWeight, downWPtr = layer.DownProjWeight)
                {
                    // 1. Input Norm
                    for (int t = 0; t < seqLen; t++)
                        SimdKernels.RmsNorm(normedPtr + t * HiddenDim, currentPtr + t * HiddenDim, inNormPtr, HiddenDim, RmsNormEps);

                    // 2. Q, K, V Projections (Batched GEMM)
                    SimdKernels.MatMulBatchedF32(qPtr, qWPtr, normedPtr, seqLen, qOut, HiddenDim);
                    SimdKernels.MatMulBatchedF32(kPtr, kWPtr, normedPtr, seqLen, kvOut, HiddenDim);
                    SimdKernels.MatMulBatchedF32(vPtr, vWPtr, normedPtr, seqLen, kvOut, HiddenDim);

                    // 3. RoPE
                    for (int t = 0; t < seqLen; t++)
                    {
                        ApplyRopeNeox(qPtr + t * qOut, NumHeads, HeadDim, cosPtr, sinPtr, t, halfDim);
                        ApplyRopeNeox(kPtr + t * kvOut, NumKvHeads, HeadDim, cosPtr, sinPtr, t, halfDim);
                    }

                    // 4. Attention
                    for (int ti = 0; ti < seqLen; ti++)
                    {
                        float* ctxTi = contextPtr + ti * qOut;
                        new Span<float>(ctxTi, qOut).Clear();
                        for (int h = 0; h < NumHeads; h++)
                        {
                            int hOff = h * HeadDim;
                            int kvHOff = (h / kvRepeats) * HeadDim;
                            float* qHead = qPtr + ti * qOut + hOff;
                            for (int tj = 0; tj < seqLen; tj++)
                            {
                                float* kHead = kPtr + tj * kvOut + kvHOff;
                                scores[tj] = SimdKernels.DotF32(qHead, kHead, HeadDim) * scale;
                            }
                            DenseKernels.SoftmaxInPlace(new Span<float>(scores, seqLen));

                            float* ctxHead = ctxTi + hOff;
                            for (int tj = 0; tj < seqLen; tj++)
                            {
                                float p = scores[tj];
                                float* vHead = vPtr + tj * kvOut + kvHOff;
                                if (Avx2.IsSupported && Fma.IsSupported)
                                {
                                    var vp = Vector256.Create(p);
                                    for (int d = 0; d < HeadDim; d += 8)
                                    {
                                        var vc = Avx.LoadVector256(ctxHead + d);
                                        var vv = Avx.LoadVector256(vHead + d);
                                        vc = Fma.MultiplyAdd(vp, vv, vc);
                                        Avx.Store(ctxHead + d, vc);
                                    }
                                }
                                else
                                {
                                    for (int d = 0; d < HeadDim; d++)
                                        ctxHead[d] += p * vHead[d];
                                }
                            }
                        }
                    }

                    // 5. O Projection (Batched GEMM)
                    SimdKernels.MatMulBatchedF32(attnOutPtr, oWPtr, contextPtr, seqLen, HiddenDim, qOut);

                    // 6. Residual Add
                    TensorPrimitives.Add(new ReadOnlySpan<float>(currentPtr, seqLen * HiddenDim),
                                         new ReadOnlySpan<float>(attnOutPtr, seqLen * HiddenDim),
                                         new Span<float>(currentPtr, seqLen * HiddenDim));

                    // 7. Post Norm
                    for (int t = 0; t < seqLen; t++)
                        SimdKernels.RmsNorm(ffnNormedPtr + t * HiddenDim, currentPtr + t * HiddenDim, postNormPtr, HiddenDim, RmsNormEps);

                    // 8. Gate & Up Projections (Batched GEMM)
                    SimdKernels.MatMulBatchedF32(gatePtr, gateWPtr, ffnNormedPtr, seqLen, FfnDim, HiddenDim);
                    SimdKernels.MatMulBatchedF32(upPtr, upWPtr, ffnNormedPtr, seqLen, FfnDim, HiddenDim);

                    // 9. Fused SiLU(gate) * up
                    SimdKernels.SiLuMul(gatePtr, upPtr, seqLen * FfnDim);

                    // 10. Down Projection (Batched GEMM)
                    SimdKernels.MatMulBatchedF32(downPtr, downWPtr, gatePtr, seqLen, HiddenDim, FfnDim);

                    // 11. Residual Add
                    TensorPrimitives.Add(new ReadOnlySpan<float>(currentPtr, seqLen * HiddenDim),
                                         new ReadOnlySpan<float>(downPtr, seqLen * HiddenDim),
                                         new Span<float>(currentPtr, seqLen * HiddenDim));
                }
            }

            // Final RmsNorm
            for (int t = 0; t < seqLen; t++)
                SimdKernels.RmsNorm(normedPtr + t * HiddenDim, currentPtr + t * HiddenDim, finalNormPtr, HiddenDim, RmsNormEps);
        }

        var output = new float[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            output[t] = new float[HiddenDim];
            Array.Copy(normed, t * HiddenDim, output[t], 0, HiddenDim);
        }
        return output;
    }

    private static unsafe void ApplyRopeNeox(float* x, int numHeads, int headDim, float* cos, float* sin, int position, int halfDim)
    {
        int cosBase = position * halfDim;
        for (int h = 0; h < numHeads; h++)
        {
            float* head = x + h * headDim;
            for (int i = 0; i < halfDim; i++)
            {
                float c = cos[cosBase + i];
                float s = sin[cosBase + i];
                float x0 = head[i];
                float x1 = head[i + halfDim];
                head[i] = x0 * c - x1 * s;
                head[i + halfDim] = x0 * s + x1 * c;
            }
        }
    }

    internal static unsafe float[][] RmsNormRows(float[][] rows, float[] weight)
    {
        var output = new float[rows.Length][];
        fixed (float* wPtr = weight)
        {
            for (int t = 0; t < rows.Length; t++)
            {
                output[t] = new float[rows[t].Length];
                fixed (float* outPtr = output[t], inPtr = rows[t])
                {
                    SimdKernels.RmsNorm(outPtr, inPtr, wPtr, rows[t].Length, RmsNormEps);
                }
            }
        }
        return output;
    }

    internal static unsafe float[] Linear(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* outPtr = output, wPtr = weight, inPtr = input)
        {
            if (bias.Length > 0)
            {
                fixed (float* bPtr = bias)
                {
                    SimdKernels.MatVecF32(outPtr, wPtr, bPtr, inPtr, outDim, inDim);
                }
            }
            else
            {
                SimdKernels.MatVecF32(outPtr, wPtr, null, inPtr, outDim, inDim);
            }
        }
        return output;
    }

    internal static void SiluInPlace(float[] x) => DenseKernels.SiluInPlace(x);
}
