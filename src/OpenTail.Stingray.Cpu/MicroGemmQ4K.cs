using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace OpenTail.Stingray.Cpu;

/// <summary>
/// Small-Batch Micro-Kernel for Q4_K quantized GGUF weights.
/// Optimised for small prompt sequence batch sizes (M in [1, 16]).
/// Gated behind <c>STINGRAY_Q4K_MICRO_GEMM=1</c> (or <c>STINGRAY_CPU_MICRO_GEMM=1</c>).
/// Defaults to <b>false (disabled)</b>.
/// </summary>
public static unsafe class MicroGemmQ4K
{
    /// <summary>
    /// Attempts to execute small-batch Q4_K matrix multiplication for M &lt;= 16.
    /// Returns true if executed; false if disabled or batch size &gt; 16.
    /// </summary>
    public static bool TryMatMulQ4K(
        float* output,
        float* input,
        byte* weights,
        int batchSize,
        int rows,
        int cols)
    {
        if (!MicroGemmConfig.IsEnabled || batchSize < 1 || batchSize > 16)
            return false;

        if (!Avx2.IsSupported)
            return false;

        int blocksPerRow = cols / 256;
        if (blocksPerRow * 256 != cols)
            return false;

        // Process M rows of input against N columns of weights
        for (int m = 0; m < batchSize; m++)
        {
            float* inputRow = input + (nuint)m * (nuint)cols;
            float* outputRow = output + (nuint)m * (nuint)rows;

            for (int r = 0; r < rows; r++)
            {
                byte* wRow = weights + (nuint)r * (nuint)blocksPerRow * 144;
                float sum = 0.0f;

                for (int b = 0; b < blocksPerRow; b++)
                {
                    byte* block = wRow + b * 144;
                    float* inBlock = inputRow + b * 256;

                    // Unpack d and dmin (fp16)
                    ushort rawD = *(ushort*)block;
                    ushort rawDmin = *(ushort*)(block + 2);
                    float d = HalfToFloat(rawD);
                    float dmin = HalfToFloat(rawDmin);

                    byte* scales = block + 4;
                    byte* qs = block + 16;

                    // Process 256 elements in 8 chunks of 32
                    for (int chunk = 0; chunk < 8; chunk++)
                    {
                        byte sc = scales[chunk];
                        byte mn = scales[chunk + 8];

                        float scale = d * (sc & 63);
                        float min = dmin * (mn & 63);

                        float* inPtr = inBlock + chunk * 32;
                        byte* qPtr = qs + chunk * 16;

                        Vector256<float> acc = Vector256<float>.Zero;

                        for (int i = 0; i < 16; i++)
                        {
                            byte packed = qPtr[i];
                            int q0 = packed & 0x0F;
                            int q1 = (packed >> 4) & 0x0F;

                            float val0 = q0 * scale - min;
                            float val1 = q1 * scale - min;

                            sum += inPtr[i * 2 + 0] * val0;
                            sum += inPtr[i * 2 + 1] * val1;
                        }
                    }
                }

                outputRow[r] = sum;
            }
        }

        return true;
    }

    private static float HalfToFloat(ushort half)
    {
        uint sign = (uint)(half & 0x8000) << 16;
        uint exp  = (uint)(half & 0x7C00) >> 10;
        uint mant = (uint)(half & 0x03FF);

        if (exp == 0)
        {
            if (mant == 0)
            {
                return BitConverter.UInt32BitsToSingle(sign);
            }
            while ((mant & 0x0400) == 0)
            {
                mant <<= 1;
                exp--;
            }
            exp++;
            mant &= ~0x0400u;
        }
        else if (exp == 31)
        {
            exp = 255;
        }

        exp = exp + (127 - 15);
        mant = mant << 13;

        uint bits = sign | (exp << 23) | mant;
        return BitConverter.UInt32BitsToSingle(bits);
    }
}
