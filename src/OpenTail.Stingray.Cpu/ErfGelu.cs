using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace OpenTail.Stingray.Cpu;

/// <summary>
/// Exact (erf) GELU, PyTorch <c>F.gelu</c> / HF <c>hidden_act: "gelu"</c>: <c>0.5·x·(1 + erf(x/√2))</c>.
/// Not the tanh approximation (<see cref="SimdKernels.GeluInPlace"/>, HF "gelu_new"/"gelu_pytorch_tanh").
/// erf uses Abramowitz-Stegun 7.1.26 (max abs error 1.5e-7), the same form as the audio ports' copies.
/// </summary>
public static class ErfGelu
{
    public static float Apply(float x) => 0.5f * x * (1f + Erf(x * 0.70710678118654752f));

    public static void InPlace(Span<float> x)
    {
        int i = 0;
        if (Avx2.IsSupported && Fma.IsSupported && x.Length >= 8)
        {
            // Same Abramowitz-Stegun form, 8 lanes at a time; exp via SimdKernels.ExpApprox256.
            var inv = Vector256.Create(0.70710678118654752f);
            var one = Vector256.Create(1f);
            var half = Vector256.Create(0.5f);
            var p = Vector256.Create(0.3275911f);
            var a1 = Vector256.Create(0.254829592f);
            var a2 = Vector256.Create(-0.284496736f);
            var a3 = Vector256.Create(1.421413741f);
            var a4 = Vector256.Create(-1.453152027f);
            var a5 = Vector256.Create(1.061405429f);
            var signMask = Vector256.Create(-0f);
            ref float start = ref MemoryMarshal.GetReference(x);
            for (; i + 8 <= x.Length; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref start, (nuint)i);
                var z = Avx.Multiply(v, inv);
                var a = Avx.AndNot(signMask, z);
                var t = Avx.Divide(one, Fma.MultiplyAdd(p, a, one));
                var poly = Fma.MultiplyAdd(a5, t, a4);
                poly = Fma.MultiplyAdd(poly, t, a3);
                poly = Fma.MultiplyAdd(poly, t, a2);
                poly = Fma.MultiplyAdd(poly, t, a1);
                poly = Avx.Multiply(poly, t);
                var e = SimdKernels.ExpApprox256(Avx.Xor(Avx.Multiply(a, a), signMask));
                var y = Fma.MultiplyAddNegated(poly, e, one);           // 1 - poly * e
                var erf = Avx.Or(y, Avx.And(z, signMask));              // copy the sign of z
                Avx.Multiply(Avx.Multiply(half, v), Avx.Add(one, erf)).StoreUnsafe(ref start, (nuint)i);
            }
        }
        for (; i < x.Length; i++) x[i] = Apply(x[i]);
    }

    public static float Erf(float x)
    {
        float sign = x < 0 ? -1f : 1f;
        x = MathF.Abs(x);
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float t = 1f / (1f + p * x);
        float y = 1f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * MathF.Exp(-x * x);
        return sign * y;
    }
}
