using OpenTail.Stingray.Diffusion.Flux2;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Verifies <see cref="Flux2RoPE.BuildContextFreqsCompact"/> (new, for GPU residency work,
/// docs/091/092) matches <see cref="Flux2RoPE.BuildContextFreqs"/> (existing, CPU-verified) at
/// every pair -- the compact table must be exactly the full table's even-indexed entries.
/// </summary>
public sealed class Flux2RoPeCompactTableTests
{
    [Fact]
    public void CompactTable_MatchesFullTable_AtEveryPair()
    {
        int[] axesDim = [32, 32, 32, 32];
        int headDim = 128;
        int nPairs = headDim / 2;
        int nTokens = 5;

        var rng = new Random(7);
        var positions = new int[nTokens * 4];
        for (int i = 0; i < positions.Length; i++) positions[i] = rng.Next(-10, 10);

        var (fullCos, fullSin) = Flux2RoPE.BuildContextFreqs(positions, nTokens, axesDim);
        var (compactCos, compactSin) = Flux2RoPE.BuildContextFreqsCompact(positions, nTokens, axesDim);

        Assert.Equal(nTokens * nPairs, compactCos.Length);
        Assert.Equal(nTokens * nPairs, compactSin.Length);

        for (int t = 0; t < nTokens; t++)
        {
            for (int j = 0; j < nPairs; j++)
            {
                float fullC = fullCos[t * headDim + j * 2];
                float fullS = fullSin[t * headDim + j * 2];
                float compactC = compactCos[t * nPairs + j];
                float compactS = compactSin[t * nPairs + j];

                Assert.True(MathF.Abs(fullC - compactC) < 1e-6f, $"token {t} pair {j}: cos full={fullC} compact={compactC}");
                Assert.True(MathF.Abs(fullS - compactS) < 1e-6f, $"token {t} pair {j}: sin full={fullS} compact={compactS}");

                // The full table also duplicates the value at the odd slot (2j+1) -- confirm that invariant too.
                Assert.Equal(fullC, fullCos[t * headDim + j * 2 + 1]);
                Assert.Equal(fullS, fullSin[t * headDim + j * 2 + 1]);
            }
        }
    }
}
