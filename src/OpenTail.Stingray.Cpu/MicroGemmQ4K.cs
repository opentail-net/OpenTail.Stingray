
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

                    // ggml's q4_K layout (dequantize_row_q4_K / get_scale_min_k4 in
                    // ggml-quants.c, mirrored by Dequantize.DequantQ4K/GetScaleMinK4): 256
                    // elements as four 64-element super-chunks. Each super-chunk consumes 32
                    // bytes of qs and TWO 6-bit scale/min pairs (spliced out of the 12-byte
                    // scales buffer, not read as 16 contiguous bytes) -- the LOW nibble of all
                    // 32 bytes forms the first 32 elements (scale/min pair j), the HIGH nibble
                    // of the SAME 32 bytes forms the next 32 elements (scale/min pair j+1). This
                    // is not the same decomposition as "32 independent 2-values-per-byte
                    // chunks" -- getting the low/high-nibble grouping wrong silently corrupts
                    // half of every super-chunk even with the scale/min splice fixed.
                    int qIdx = 0;
                    for (int outer = 0; outer < 4; outer++)
                    {
                        int j1 = outer * 2;
                        int j2 = outer * 2 + 1;

                        GetScaleMinK4(scales, j1, out byte sc1, out byte mn1);
                        GetScaleMinK4(scales, j2, out byte sc2, out byte mn2);

                        float scaleLow = d * sc1;
                        float minLow = dmin * mn1;
                        float scaleHigh = d * sc2;
                        float minHigh = dmin * mn2;

                        float* inLow = inBlock + outer * 64;
                        float* inHigh = inLow + 32;
                        byte* qPtr = qs + qIdx;

                        for (int l = 0; l < 32; l++)
                        {
                            byte packed = qPtr[l];
                            int qLow = packed & 0x0F;
                            int qHigh = packed >> 4;

                            float valLow = qLow * scaleLow - minLow;
                            float valHigh = qHigh * scaleHigh - minHigh;

                            sum += inLow[l] * valLow;
                            sum += inHigh[l] * valHigh;
                        }

                        qIdx += 32;
                    }
                }

                outputRow[r] = sum;
            }
        }

        return true;
    }

    /// <summary>
    /// Decode one 6-bit scale and min from the packed 12-byte scale/min buffer.
    /// Matches get_scale_min_k4 in ggml-quants.c (and Dequantize.GetScaleMinK4).
    /// </summary>
    private static void GetScaleMinK4(byte* scales, int j, out byte scale, out byte min)
    {
        if (j < 4)
        {
            scale = (byte)(scales[j] & 63);
            min = (byte)(scales[j + 4] & 63);
        }
        else
        {
            scale = (byte)((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4));
            min = (byte)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
        }
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
