using System.Runtime.CompilerServices;

namespace OpenTail.Stingray.Cpu;

/// <summary>
/// Scalar IEEE half to float for the per-block scale of every quantised dtype. <c>(float)BitConverter.UInt16BitsToHalf(bits)</c> compiles, on this runtime
/// and with AggressiveOptimization/FullOpts, to a real un-inlined <c>call System.Half:op_Explicit</c>. Inside a dot-product loop that costs a call per scale and,
/// on Windows x64 where only the low 128 bits of xmm6-15 are callee-saved, forces the JIT to split and re-insert the upper halves of every live ymm accumulator
/// around it; measured 2026-10-04 at ~20% of the Q4_K row-dot time (docs/4-performance/2026-10-04-moe-decode-gap-investigation.md, section 9).
/// This version re-biases the exponent by multiplying with 2^112 (zeros and subnormals come out exact) and sends the one case that trick gets wrong, half
/// exponent 31 (inf/NaN, never a valid scale), to the real <c>Half</c> cast out of line, so it is bit-identical to the cast for all 65536 inputs
/// (HalfConvTests checks every bit pattern). Measured: a branch-free arithmetic patch for inf/NaN was slower than this predictable test, and dropping
/// the special case altogether gains only 5-10% more.
/// </summary>
internal static unsafe class HalfConv
{
    private const float Rescale = 5.192296858534828e33f;   // 2^112

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ToFloat(ushort h)
    {
        if ((h & 0x7C00) == 0x7C00) return Special(h);
        uint bits = BitConverter.SingleToUInt32Bits(BitConverter.UInt32BitsToSingle((uint)(h & 0x7fff) << 13) * Rescale);
        return BitConverter.UInt32BitsToSingle(bits | ((uint)(h & 0x8000) << 16));
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static float Special(ushort h) => (float)BitConverter.UInt16BitsToHalf(h);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ToFloat(byte lo, byte hi) => ToFloat((ushort)(lo | (hi << 8)));

    /// <summary>The little-endian half stored at <paramref name="p"/> (one 16-bit load).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ToFloat(byte* p) => ToFloat(*(ushort*)p);
}
