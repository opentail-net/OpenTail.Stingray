using System;

namespace OpenTail.Stingray.Audio.FishSpeech;

/// <summary>
/// Encodes a plain float32 weight matrix into the real Q8_0 block format this codebase's own
/// <see cref="OpenTail.Stingray.Cpu.SimdKernels.MatVecQ8_0"/>/<see cref="OpenTail.Stingray.Cpu.SimdKernels.DotQ8_0"/>
/// kernels already consume for GGUF Q8_0 tensors elsewhere in this engine (34 bytes per 32-element
/// block: 2-byte IEEE754 half scale + 32 signed int8 values, symmetric absmax scaling -- the
/// exact same scheme <c>SimdKernels.QuantizeRowToQ8_0</c> already uses for activations, just with
/// an fp16 scale instead of that function's fp32 scratch scale, to match the ON-WEIGHT block
/// format <see cref="OpenTail.Stingray.Cpu.SimdKernels.DotQ8_0"/> actually reads).
///
/// <para><b>Why this exists</b>: this session's performance pass measured the Fish Speech fast-AR
/// sub-network's dominant cost as memory-bandwidth-bound plain-float32 weight reads (~40ms/call,
/// perfectly linear 1-&gt;9 call scaling, ~1.58GB of FP32 weight re-read 9x/frame -&gt; ~39GB/s
/// effective bandwidth at the measured cost -- see docs/audio-review-progress.md's Fish Speech
/// performance-pass entries for the full diagnosis). This exact sub-network was ALSO already
/// measured to fail badly at Q4_K_M precision (cosine ~0.489 vs. a real oracle) but pass cleanly
/// at Q8_0 (cosine ~0.9995) -- so Q8_0 is the only quantization level with an existing numerical
/// safety proof for this specific sub-network, not a generic assumption.</para>
/// </summary>
public static class FishSpeechQ8_0Weight
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
