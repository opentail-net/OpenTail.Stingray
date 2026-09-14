namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Diagnostic test (docs/077, 2026-09-14): dumps `LtxVideoPipeline.GenerateVideo`'s real
/// `shiftedTimesteps`/`dt` schedule (re-derived here from the exact same formula, since the real
/// method doesn't expose it) for the real 512x512/256-token case at several step counts, to check
/// the leading hypothesis that more steps causing WORSE output (2-step: plausible colors, 8-step:
/// recognizable apple, 20-step: destroyed into banding) is caused by a numerical instability in the
/// `TimeShift` formula as `t -> 0`, which more steps reach.
/// </summary>
public sealed class LtxSchedulerDiagnosticTest
{
    private static float GetNormalShift(int nTokens, int minTokens = 1024, int maxTokens = 4096,
        float minShift = 0.95f, float maxShift = 2.05f)
    {
        float m = (maxShift - minShift) / (maxTokens - minTokens);
        float b = minShift - m * minTokens;
        return m * nTokens + b;
    }

    private static float TimeShift(float mu, float sigma, float t)
        => MathF.Exp(mu) / (MathF.Exp(mu) + MathF.Pow(1.0f / t - 1.0f, sigma));

    [Theory]
    [InlineData(2)]
    [InlineData(8)]
    [InlineData(20)]
    public void DumpShiftedTimestepsAndDt(int steps)
    {
        const int numLatentTokens = 16 * 16; // real 512x512 patchH=patchW=16
        float sd3Shift = GetNormalShift(numLatentTokens);
        Console.WriteLine($"[steps={steps}] sd3Shift={sd3Shift:F6}");

        var shiftedTimesteps = new float[steps + 1];
        for (int i = 0; i < steps; i++)
        {
            float tRaw = 1.0f - (float)i / steps;
            shiftedTimesteps[i] = TimeShift(sd3Shift, 1.0f, tRaw);
        }
        shiftedTimesteps[steps] = 0f;

        for (int i = 0; i < steps; i++)
        {
            float tRaw = 1.0f - (float)i / steps;
            float dt = shiftedTimesteps[i] - shiftedTimesteps[i + 1];
            Console.WriteLine($"[steps={steps}] i={i,3} tRaw={tRaw:F6} tShifted={shiftedTimesteps[i]:F6} dt={dt:F6}");
        }
    }
}
