using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Regression for the HC mix activation rounding. llama.cpp feeds Q8_0 matmuls a Q8 int8-grid activation
/// (d = amax/127 per 32-value block); the HC mix did not, which left a 1.08% error in hc_mixed-0 (Phase 0.2).
/// The rounding helper must match Stingray's own Q8_0 activation quantiser.
/// </summary>
public sealed class Qwen4ExpHcActivationTests
{
    [Fact]
    public unsafe void RoundToQ8Blocks_MatchesStingrayQ8_0Quantiser()
    {
        var rng = new Random(7);
        const int n = 256; // eight 32-value blocks
        float[] x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 4 - 2);

        byte[] scratch = new byte[(n / 32) * 36];
        fixed (float* src = x)
        fixed (byte* dst = scratch)
        {
            SimdKernels.QuantizeRowToQ8_0(src, n, dst);
        }

        float[] rounded = (float[])x.Clone();
        Qwen4ExpGatedResidual.RoundToQ8Blocks(rounded);

        int mismatches = 0;
        for (int b = 0; b < n / 32; b++)
        {
            float d = BitConverter.ToSingle(scratch, b * 36);
            for (int i = 0; i < 32; i++)
            {
                sbyte q = (sbyte)scratch[b * 36 + 4 + i];
                float expected = d * q;
                if (MathF.Abs(expected - rounded[b * 32 + i]) > 1e-6f) mismatches++;
            }
        }
        Assert.True(mismatches <= 1, $"{mismatches} of {n} values differ from the Stingray Q8_0 quantiser.");
    }

    [Fact]
    public void RoundToQ8Blocks_IsIdempotent_AndKeepsZeroBlocks()
    {
        float[] x = new float[64];
        for (int i = 0; i < 32; i++) x[i] = i * 0.1f;
        Qwen4ExpGatedResidual.RoundToQ8Blocks(x);
        float[] once = (float[])x.Clone();
        Qwen4ExpGatedResidual.RoundToQ8Blocks(x);
        for (int i = 0; i < 64; i++) Assert.Equal(once[i], x[i], 6);
        for (int i = 32; i < 64; i++) Assert.Equal(0f, x[i]);
    }
}
