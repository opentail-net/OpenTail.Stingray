using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace OpenTail.Stingray.Cpu;

/// <summary>
/// Blockwise quantization and dequantization primitives for HuggingFace / PyTorch models (BitsAndBytes NF4/FP4, AWQ, GPTQ).
/// Reference: ONNX Runtime contrib_ops/cpu/quantization (blockwise_quant_block_bnb4.h, dequantize_blockwise_bnb4.h, matmul_nbits.cc)
/// 
/// Why it's useful:
/// Enables native execution and direct matrix-vector dot products for 4-bit Safetensors checkpoints (such as BitsAndBytes
/// NF4 and AWQ/GPTQ) directly in .NET without requiring pre-conversion to GGUF format or full tensor memory decompression.
/// </summary>
public static unsafe class BlockwiseQuant
{
    /// <summary>
    /// NormalFloat4 (NF4) codebook mapping 4-bit integer nibbles [0..15] to normalized quant values.
    /// Source: onnxruntime::contrib::nf4_qaunt_map (blockwise_quant_block_bnb4.h) / bitsandbytes.
    /// </summary>
    public static readonly float[] Nf4QuantMap =
    [
        -1.0f,
        -0.6961928009986877f,
        -0.5250730514526367f,
        -0.39491748809814453f,
        -0.28444138169288635f,
        -0.18477343022823334f,
        -0.09105003625154495f,
        0.0f,
        0.07958029955625534f,
        0.16093020141124725f,
        0.24611230194568634f,
        0.33791524171829224f,
        0.44070982933044434f,
        0.5626170039176941f,
        0.7229568362236023f,
        1.0f
    ];

    /// <summary>
    /// Float4 (FP4 E2M1) codebook mapping 4-bit integer nibbles [0..15] to normalized quant values.
    /// Source: onnxruntime::contrib::fp4_qaunt_map (blockwise_quant_block_bnb4.h).
    /// </summary>
    public static readonly float[] Fp4QuantMap =
    [
        0.00000000f, 5.208333333e-03f, 0.66666667f, 1.00000000f,
        0.33333333f, 0.50000000f, 0.16666667f, 0.25000000f,
        -0.00000000f, -5.208333333e-03f, -0.66666667f, -1.00000000f,
        -0.33333333f, -0.50000000f, -0.16666667f, -0.25000000f
    ];

    /// <summary>
    /// Dequantizes BitsAndBytes NF4 blockwise packed nibbles to Float32.
    /// Source: onnxruntime::contrib::DequantizeBlockBnb4 (dequantize_blockwise_bnb4.h).
    /// </summary>
    public static void DequantizeNf4Blockwise(
        ReadOnlySpan<byte> srcPacked,
        ReadOnlySpan<float> absmaxBlocks,
        Span<float> dst,
        int numElements,
        int blockSize = 64)
    {
        int numBlocks = (numElements + blockSize - 1) / blockSize;
        fixed (byte* pSrc = srcPacked)
        fixed (float* pAbsmax = absmaxBlocks)
        fixed (float* pDst = dst)
        fixed (float* pMap = Nf4QuantMap)
        {
            var src = pSrc;
            var absmax = pAbsmax;
            var d = pDst;
            var map = pMap;
            int numAbsmax = absmaxBlocks.Length;

            Parallel.For(0, numBlocks, b =>
            {
                int blockLen = Math.Min(blockSize, numElements - b * blockSize);
                int srcOff = (b * blockSize) / 2;
                int dstOff = b * blockSize;
                float scale = b < numAbsmax ? absmax[b] : 1.0f;

                for (int i = 0; i < blockLen; i += 2)
                {
                    byte val = src[srcOff + (i >> 1)];
                    int high = val >> 4;
                    int low = val & 0x0F;

                    d[dstOff + i] = map[high] * scale;
                    if (i + 1 < blockLen)
                    {
                        d[dstOff + i + 1] = map[low] * scale;
                    }
                }
            });
        }
    }

    /// <summary>
    /// Dequantizes BitsAndBytes FP4 blockwise packed nibbles to Float32.
    /// Source: onnxruntime::contrib::DequantizeBlockBnb4 (dequantize_blockwise_bnb4.h).
    /// </summary>
    public static void DequantizeFp4Blockwise(
        ReadOnlySpan<byte> srcPacked,
        ReadOnlySpan<float> absmaxBlocks,
        Span<float> dst,
        int numElements,
        int blockSize = 64)
    {
        int numBlocks = (numElements + blockSize - 1) / blockSize;
        fixed (byte* pSrc = srcPacked)
        fixed (float* pAbsmax = absmaxBlocks)
        fixed (float* pDst = dst)
        fixed (float* pMap = Fp4QuantMap)
        {
            var src = pSrc;
            var absmax = pAbsmax;
            var d = pDst;
            var map = pMap;
            int numAbsmax = absmaxBlocks.Length;

            Parallel.For(0, numBlocks, b =>
            {
                int blockLen = Math.Min(blockSize, numElements - b * blockSize);
                int srcOff = (b * blockSize) / 2;
                int dstOff = b * blockSize;
                float scale = b < numAbsmax ? absmax[b] : 1.0f;

                for (int i = 0; i < blockLen; i += 2)
                {
                    byte val = src[srcOff + (i >> 1)];
                    int high = val >> 4;
                    int low = val & 0x0F;

                    d[dstOff + i] = map[high] * scale;
                    if (i + 1 < blockLen)
                    {
                        d[dstOff + i + 1] = map[low] * scale;
                    }
                }
            });
        }
    }

    /// <summary>
    /// Fused Matrix-Vector multiplication for BitsAndBytes NF4 weights against FP32 activation vector.
    /// Source: onnxruntime::contrib::MatMulBnb4 (matmul_bnb4.cc).
    /// Eliminates the memory bandwidth penalty of dequantizing large weight matrices to RAM during token generation.
    /// </summary>
    public static void MatVecNf4(
        ReadOnlySpan<byte> packedMatrix,
        ReadOnlySpan<float> absmaxBlocks,
        ReadOnlySpan<float> inputVector,
        Span<float> outputVector,
        int rows,
        int cols,
        int blockSize = 64)
    {
        int bytesPerRow = cols / 2;
        int blocksPerRow = cols / blockSize;

        fixed (byte* pMatrix = packedMatrix)
        fixed (float* pAbsmax = absmaxBlocks)
        fixed (float* pInput = inputVector)
        fixed (float* pOutput = outputVector)
        fixed (float* pMap = Nf4QuantMap)
        {
            var matrix = pMatrix;
            var absmax = pAbsmax;
            var input = pInput;
            var output = pOutput;
            var map = pMap;

            Parallel.For(0, rows, r =>
            {
                byte* rowPacked = matrix + (long)r * bytesPerRow;
                float* rowAbsmax = absmax + (long)r * blocksPerRow;
                float sum = 0.0f;

                for (int b = 0; b < blocksPerRow; b++)
                {
                    float scale = rowAbsmax[b];
                    int byteOff = b * (blockSize / 2);
                    int inOff = b * blockSize;
                    float blockDot = 0.0f;

                    for (int i = 0; i < blockSize; i += 2)
                    {
                        byte val = rowPacked[byteOff + (i >> 1)];
                        int high = val >> 4;
                        int low = val & 0x0F;

                        float w0 = map[high];
                        float w1 = map[low];

                        blockDot += w0 * input[inOff + i] + w1 * input[inOff + i + 1];
                    }

                    sum += blockDot * scale;
                }

                output[r] = sum;
            });
        }
    }

    /// <summary>
    /// Dequantizes flat N-Bit quantized weights (AWQ / GPTQ / RTN 4-bit) with block scaling and optional zero-points.
    /// Source: onnxruntime::contrib::DequantizeMatMulNBits (matmul_nbits.cc, mlas_qnbit.h).
    /// </summary>
    public static void DequantizeNBits4(
        ReadOnlySpan<byte> srcPacked,
        ReadOnlySpan<float> scales,
        ReadOnlySpan<byte> zeroPoints,
        Span<float> dst,
        int rows,
        int cols,
        int blockSize = 32)
    {
        int bytesPerRow = (cols + 1) / 2;
        int blocksPerRow = (cols + blockSize - 1) / blockSize;
        bool hasZeroPoints = !zeroPoints.IsEmpty;

        fixed (byte* pSrc = srcPacked)
        fixed (float* pScales = scales)
        fixed (byte* pZp = zeroPoints)
        fixed (float* pDst = dst)
        {
            var src = pSrc;
            var sc = pScales;
            var zpPtr = pZp;
            var d = pDst;

            Parallel.For(0, rows, r =>
            {
                int rowByteOff = r * bytesPerRow;
                int rowScaleOff = r * blocksPerRow;
                int rowDstOff = r * cols;

                for (int b = 0; b < blocksPerRow; b++)
                {
                    float scale = sc[rowScaleOff + b];
                    float zp = hasZeroPoints ? zpPtr[rowScaleOff + b] : 8.0f;
                    int blockDstOff = rowDstOff + b * blockSize;
                    int blockSrcOff = rowByteOff + b * (blockSize / 2);
                    int blockLen = Math.Min(blockSize, cols - b * blockSize);

                    for (int i = 0; i < blockLen; i += 2)
                    {
                        byte val = src[blockSrcOff + (i >> 1)];
                        float v0 = (val & 0x0F) - zp;
                        d[blockDstOff + i] = v0 * scale;

                        if (i + 1 < blockLen)
                        {
                            float v1 = (val >> 4) - zp;
                            d[blockDstOff + i + 1] = v1 * scale;
                        }
                    }
                }
            });
        }
    }

    /// <summary>
    /// Fused Matrix-Vector multiplication for flat 4-bit quantized weights (AWQ / GPTQ / RTN) against FP32 activation vector.
    /// Source: onnxruntime::contrib::MatMulNBits (matmul_nbits.cc).
    /// </summary>
    public static void MatVecNBits4(
        ReadOnlySpan<byte> packedMatrix,
        ReadOnlySpan<float> scales,
        ReadOnlySpan<byte> zeroPoints,
        ReadOnlySpan<float> inputVector,
        Span<float> outputVector,
        int rows,
        int cols,
        int blockSize = 32)
    {
        int bytesPerRow = (cols + 1) / 2;
        int blocksPerRow = (cols + blockSize - 1) / blockSize;
        bool hasZeroPoints = !zeroPoints.IsEmpty;

        fixed (byte* pMatrix = packedMatrix)
        fixed (float* pScales = scales)
        fixed (byte* pZp = zeroPoints)
        fixed (float* pInput = inputVector)
        fixed (float* pOutput = outputVector)
        {
            var matrix = pMatrix;
            var sc = pScales;
            var zpPtr = pZp;
            var input = pInput;
            var output = pOutput;

            Parallel.For(0, rows, r =>
            {
                byte* rowPacked = matrix + (long)r * bytesPerRow;
                float* rowScales = sc + (long)r * blocksPerRow;
                byte* rowZp = hasZeroPoints ? (zpPtr + (long)r * blocksPerRow) : null;
                float sum = 0.0f;

                for (int b = 0; b < blocksPerRow; b++)
                {
                    float scale = rowScales[b];
                    float zp = rowZp != null ? rowZp[b] : 8.0f;
                    int byteOff = b * (blockSize / 2);
                    int inOff = b * blockSize;
                    int blockLen = Math.Min(blockSize, cols - b * blockSize);

                    float blockDot = 0.0f;
                    float blockInSum = 0.0f;

                    for (int i = 0; i < blockLen; i += 2)
                    {
                        byte val = rowPacked[byteOff + (i >> 1)];
                        float q0 = (float)(val & 0x0F);
                        float in0 = input[inOff + i];
                        blockDot += q0 * in0;
                        blockInSum += in0;

                        if (i + 1 < blockLen)
                        {
                            float q1 = (float)(val >> 4);
                            float in1 = input[inOff + i + 1];
                            blockDot += q1 * in1;
                            blockInSum += in1;
                        }
                    }

                    // (dot(q, in) - zp * sum(in)) * scale
                    sum += (blockDot - zp * blockInSum) * scale;
                }

                output[r] = sum;
            });
        }
    }
}
