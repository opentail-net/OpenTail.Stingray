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
        for (int i = 0; i < x.Length; i++) x[i] = Apply(x[i]);
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
