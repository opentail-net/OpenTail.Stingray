using System;

namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>
/// Encodes a plain float32 weight matrix into the real Q8_0 block format this codebase's own
/// <see cref="OpenTail.Stingray.Cpu.SimdKernels.MatVecQ8_0"/>/<see cref="OpenTail.Stingray.Cpu.SimdKernels.DotQ8_0"/>
/// kernels already consume for GGUF Q8_0 tensors elsewhere in this engine (34 bytes per 32-element
/// block: 2-byte IEEE754 half scale + 32 signed int8 values, symmetric absmax scaling -- the
/// exact same scheme <c>SimdKernels.QuantizeRowToQ8_0</c> already uses for activations, just with
/// an fp16 scale instead of that function's fp32 scratch scale, to match the ON-WEIGHT block
/// format <see cref="OpenTail.Stingray.Cpu.SimdKernels.DotQ8_0"/> actually reads).
///
/// <para><b>Why this exists</b>: Fish Speech's fast-AR sub-network (measured this session's
/// performance pass) was found to be memory-bandwidth-bound on plain-float32 weight reads
/// (~40ms/call, perfectly linear 1-&gt;9 call scaling, ~1.58GB of FP32 weight re-read 9x/frame
/// -&gt; ~39GB/s effective bandwidth at the measured cost -- see docs/audio-review-progress.md's
/// Fish Speech performance-pass entries for the full diagnosis) and was already separately proven
/// numerically safe at Q8_0 precision (cosine ~0.9995 vs. Q4_K_M's ~0.489). Moved here (from its
/// original home in the FishSpeech namespace) once Parler-TTS's decoder needed the exact same
/// technique -- shared per this project's DRY convention (see e.g. `DenseKernels.cs`'s own doc
/// comment for the same rationale after a near-identical duplication was found elsewhere).</para>
/// </summary>
public static class Q8_0WeightQuantizer
{
    /// <summary>Bytes needed to store a [rows, cols] matrix in Q8_0 block format. <paramref name="cols"/> must be a multiple of 32.</summary>
    public static long ByteSize(int rows, int cols)
    {
        if (cols % 32 != 0) throw new ArgumentException($"Q8_0 requires cols % 32 == 0, got {cols}.");
        long bytesPerRow = (cols / 32) * 34L;
        return bytesPerRow * rows;
    }

    /// <summary>Quantizes a flat row-major [rows, cols] float32 matrix into real Q8_0 blocks, one row after another.</summary>
    public static unsafe byte[] Quantize(float[] source, int rows, int cols)
    {
        if (cols % 32 != 0) throw new ArgumentException($"Q8_0 requires cols % 32 == 0, got {cols}.");
        int numBlocksPerRow = cols / 32;
        var dst = new byte[ByteSize(rows, cols)];

        fixed (float* src = source)
        fixed (byte* dstPtr = dst)
        {
            for (int r = 0; r < rows; r++)
            {
                float* rowSrc = src + (long)r * cols;
                byte* rowDst = dstPtr + (long)r * numBlocksPerRow * 34;

                for (int b = 0; b < numBlocksPerRow; b++)
                {
                    float* x = rowSrc + b * 32;
                    byte* block = rowDst + b * 34;

                    float amax = 0f;
                    for (int i = 0; i < 32; i++)
                    {
                        float a = MathF.Abs(x[i]);
                        if (a > amax) amax = a;
                    }

                    float d = amax / 127f;
                    float id = d != 0f ? 1f / d : 0f;

                    var half = (Half)d;
                    ushort halfBits = BitConverter.HalfToUInt16Bits(half);
                    block[0] = (byte)(halfBits & 0xFF);
                    block[1] = (byte)(halfBits >> 8);

                    sbyte* qs = (sbyte*)(block + 2);
                    for (int i = 0; i < 32; i++)
                        qs[i] = (sbyte)Math.Clamp(MathF.Round(x[i] * id), -127f, 127f);
                }
            }
        }
        return dst;
    }
}
