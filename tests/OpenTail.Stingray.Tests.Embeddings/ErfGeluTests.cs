namespace OpenTail.Stingray.Tests.Embeddings;

/// <summary>The AVX2 path of <see cref="ErfGelu.InPlace"/> must agree with the scalar reference form.</summary>
public sealed class ErfGeluTests
{
    [Fact]
    public void VectorPath_MatchesScalarReference()
    {
        var x = Enumerable.Range(0, 4099).Select(i => (i - 2049) / 256f).Append(float.Epsilon).Append(-0f).ToArray();
        var expected = x.Select(ErfGelu.Apply).ToArray();
        ErfGelu.InPlace(x);
        for (int i = 0; i < x.Length; i++)
            Assert.True(MathF.Abs(x[i] - expected[i]) <= 2e-6f * MathF.Max(1f, MathF.Abs(expected[i])), $"index {i}: {x[i]} vs {expected[i]}");
    }
}
