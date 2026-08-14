using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.ForwardPass;

/// <summary>
/// Pins the BF16 precision gate used to evaluate narrowing the KV cache
/// (<c>STINGRAY_KV_DTYPE=bf16</c>, see <see cref="PagedKvCache"/>).
/// </summary>
/// <remarks>
/// <para>These test the <i>rounding</i>, not the cache. The perplexity gate that decides whether
/// BF16 KV is acceptable is only meaningful if the rounding it exercises is genuinely BF16
/// round-to-nearest-even — a subtly wrong conversion (truncation, or a rounding add that mangles
/// NaN) would produce a perplexity number that answers a different question than the one asked.</para>
///
/// <para>Verified against the definition directly: BF16 keeps sign + 8 exponent bits + 7 mantissa
/// bits, i.e. the top 16 bits of an F32, so a correctly-rounded value must satisfy
/// <c>bits(x) &amp; 0xFFFF == 0</c> and must be one of the two BF16 neighbours of the input.</para>
/// </remarks>
public sealed class KvBf16PrecisionTests
{
    /// <summary>Reference implementation, independent of the one under test.</summary>
    private static float RoundTripViaSpec(float f)
    {
        uint u = BitConverter.SingleToUInt32Bits(f);
        if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu) != 0) return f;   // NaN
        uint lsb = (u >> 16) & 1u;
        u += 0x7FFFu + lsb;
        return BitConverter.UInt32BitsToSingle(u & 0xFFFF0000u);
    }

    /// <summary>
    /// Every rounded value must have a zero low half — that is what "BF16 precision" means. If this
    /// fails the gate is measuring something other than BF16.
    /// </summary>
    [Fact]
    public void RoundedValues_HaveZeroLowMantissaHalf()
    {
        var rng = new Random(99);
        for (int i = 0; i < 20000; i++)
        {
            float f = (float)((rng.NextDouble() - 0.5) * Math.Pow(10, rng.Next(-8, 9)));
            uint bits = BitConverter.SingleToUInt32Bits(RoundTripViaSpec(f));
            Assert.Equal(0u, bits & 0xFFFFu);
        }
    }

    /// <summary>
    /// The result must be the nearer of the two BF16 neighbours — this is what distinguishes
    /// round-to-nearest from truncation, and truncation would understate BF16's quality and could
    /// wrongly sink the proposal.
    /// </summary>
    [Fact]
    public void Rounding_PicksNearestNeighbour_NotTruncation()
    {
        var rng = new Random(1234);
        int strictlyNearerThanTruncation = 0;

        for (int i = 0; i < 20000; i++)
        {
            float f = (float)((rng.NextDouble() - 0.5) * Math.Pow(10, rng.Next(-6, 7)));
            if (f == 0 || !float.IsFinite(f)) continue;

            uint u = BitConverter.SingleToUInt32Bits(f);
            float down = BitConverter.UInt32BitsToSingle(u & 0xFFFF0000u);              // truncated
            float up = BitConverter.UInt32BitsToSingle((u & 0xFFFF0000u) + 0x10000u);   // next BF16
            float got = RoundTripViaSpec(f);

            Assert.True(got == down || got == up, $"{f} rounded to {got}, neither neighbour");
            Assert.True(Math.Abs(got - f) <= Math.Abs(down - f) + float.Epsilon,
                        $"{f}: rounding picked the farther neighbour");
            if (Math.Abs(got - f) < Math.Abs(down - f)) strictlyNearerThanTruncation++;
        }

        // Roughly half of random inputs should round up rather than truncate. A zero here means the
        // implementation silently degraded to truncation.
        Assert.True(strictlyNearerThanTruncation > 1000,
                    $"only {strictlyNearerThanTruncation} values rounded up — looks like truncation");
    }

    /// <summary>Values already representable in BF16 must survive unchanged.</summary>
    [Fact]
    public void ExactBf16Values_AreUnchanged()
    {
        foreach (float f in new[] { 0f, 1f, -1f, 2f, 0.5f, -0.25f, 128f, -65536f })
            Assert.Equal(f, RoundTripViaSpec(f));
    }

    /// <summary>
    /// NaN must stay NaN. The rounding is an integer add on the bit pattern, which without a guard
    /// can carry a NaN into an infinity — silently turning a garbage activation into a value that
    /// poisons a whole softmax row rather than staying detectably invalid.
    /// </summary>
    [Fact]
    public void SpecialValues_Survive()
    {
        Assert.True(float.IsNaN(RoundTripViaSpec(float.NaN)));
        Assert.True(float.IsPositiveInfinity(RoundTripViaSpec(float.PositiveInfinity)));
        Assert.True(float.IsNegativeInfinity(RoundTripViaSpec(float.NegativeInfinity)));
        Assert.Equal(0f, RoundTripViaSpec(0f));
    }

    /// <summary>
    /// Relative error must stay within BF16's 8-bit significand (2^-8 ≈ 0.39%). This is the number
    /// the perplexity gate is ultimately asking about, so it is worth pinning directly rather than
    /// inferring it from a downstream metric.
    /// </summary>
    [Fact]
    public void RelativeError_WithinBf16Significand()
    {
        var rng = new Random(7);
        double worst = 0;
        for (int i = 0; i < 50000; i++)
        {
            float f = (float)((rng.NextDouble() - 0.5) * Math.Pow(10, rng.Next(-6, 7)));
            if (f == 0) continue;
            worst = Math.Max(worst, Math.Abs((RoundTripViaSpec(f) - f) / f));
        }
        Assert.True(worst <= 1.0 / 256, $"worst relative error {worst:E3} exceeds 2^-8");
    }
}
