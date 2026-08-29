
namespace OpenTail.Stingray.Cpu;

/// <summary>
/// P/Invoke wrapper around the prebuilt <c>native/f16c_shim.dll</c> (source: <c>native/f16c_shim.c</c>,
/// rebuild via <c>native/build.bat</c>) -- a hand-written ~20-line AVX2/F16C dot-product kernel that
/// decodes F16 weight values directly into registers inside the FMA accumulation loop via the real
/// hardware <c>VCVTPH2PS</c> instruction, matching ggml's own <c>ggml_vec_dot_f16</c> (see
/// docs/audio-review-progress.md's ggml/F16C investigation entry for the full story).
///
/// <para><b>Why native code, when this codebase generally avoids it (see
/// docs/done/openblas-elimination-findings-2026-08-20.md)</b>: unlike OpenBLAS, this isn't a
/// third-party dependency with its own versioning/build baggage -- it's ~20 lines of code this
/// project fully owns and controls. The reason it has to be native at all is that .NET's managed
/// SIMD API has no path to real hardware F16C: <c>Half</c> is not a legal <c>Vector128&lt;T&gt;</c>/
/// <c>Vector256&lt;T&gt;</c> element type in .NET 10, and both a hand-rolled AVX2 software
/// bit-manipulation conversion AND relying on the JIT's scalar <c>(float)Half</c> cast were measured
/// at 9-15x SLOWER than plain F32 on Whisper's real encoder shapes -- there is no viable
/// pure-managed path to this specific win.</para>
///
/// <para><b>Availability</b>: Windows x64 only for now (the shipped DLL is built with MSVC
/// <c>/arch:AVX2</c>). <see cref="IsAvailable"/> is computed once at process start by attempting a
/// trivial real call and catching any load/platform failure; every caller MUST check it and fall
/// back to the existing F32 path (e.g. <see cref="SimdKernels.MatVecF32"/>) when false -- never call
/// <see cref="Dot"/> unconditionally.</para>
/// </summary>
public static class F16CNative
{
    private const string LibName = "f16c_shim";

    [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
    private static extern unsafe float f16c_dot(float* input, ushort* weightF16Bits, int k);

    /// <summary>True if the native shim loaded successfully and a real call executed without error. Computed once.</summary>
    public static readonly bool IsAvailable = ProbeAvailability();

    private static unsafe bool ProbeAvailability()
    {
        if (!Avx2.IsSupported) return false;

        try
        {
            float input = 1.0f;
            ushort weightBits = BitConverter.HalfToUInt16Bits((Half)1.0f);
            float result = f16c_dot(&input, &weightBits, 1);
            // k=1 takes the scalar tail path in the native kernel (see f16c_shim.c), so this also
            // exercises that path, not just the 8-wide vector loop.
            return Math.Abs(result - 1.0f) < 1e-3f;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Real hardware F16C dot product: <c>sum(input[i] * (float)weightF16Bits[i])</c> for
    /// <paramref name="k"/> elements. Caller MUST check <see cref="IsAvailable"/> first.
    /// </summary>
    public static unsafe float Dot(float* input, ushort* weightF16Bits, int k) => f16c_dot(input, weightF16Bits, k);
}
