using OpenTail.Stingray.Diffusion.Wan;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Decisive isolation test (docs/081, 2026-09-14): a real C++-vs-C# cross-reference bisection
/// traced Wan's remaining accuracy bug precisely to cross-attention's raw attention output
/// (cosine=0.967, ~43% too-large norm) despite Q/K/V going in being essentially bit-perfect
/// (cosine > 0.9999999 each). Cross-attention is the ONLY place in the whole model where
/// `WanAttention.TiledMultiHeadAttention` is called with qSeq != kvSeq (self-attention always has
/// qSeq == kvSeq == numTokens) -- this test isolates that exact asymmetric-length shape
/// (qSeq=64, kvSeq=512, headDim=128, numHeads=12 -- Wan-1.3B's own real dimensions) against a
/// dead-simple naive reference softmax-attention implementation.
/// </summary>
public sealed class WanAttentionAsymmetricSeqIsolationTest
{
    private static float[] NaiveAttention(float[] q, float[] k, float[] v, int qSeq, int kvSeq, int numHeads, int headDim)
    {
        int dim = numHeads * headDim;
        var output = new float[qSeq * dim];
        float scale = 1.0f / MathF.Sqrt(headDim);
        var scores = new double[kvSeq];

        for (int h = 0; h < numHeads; h++)
        {
            for (int qi = 0; qi < qSeq; qi++)
            {
                double maxScore = double.NegativeInfinity;
                for (int ki = 0; ki < kvSeq; ki++)
                {
                    double dot = 0;
                    for (int d = 0; d < headDim; d++)
                    {
                        float qv = q[qi * dim + h * headDim + d];
                        float kv = k[ki * dim + h * headDim + d];
                        dot += (double)qv * kv;
                    }
                    double s = dot * scale;
                    scores[ki] = s;
                    if (s > maxScore) maxScore = s;
                }
                double sumExp = 0;
                for (int ki = 0; ki < kvSeq; ki++)
                {
                    scores[ki] = Math.Exp(scores[ki] - maxScore);
                    sumExp += scores[ki];
                }
                for (int d = 0; d < headDim; d++)
                {
                    double acc = 0;
                    for (int ki = 0; ki < kvSeq; ki++)
                    {
                        acc += scores[ki] / sumExp * v[ki * dim + h * headDim + d];
                    }
                    output[qi * dim + h * headDim + d] = (float)acc;
                }
            }
        }
        return output;
    }

    [Fact]
    public void TiledMultiHeadAttention_MatchesNaiveReference_AtRealCrossAttentionDims()
    {
        const int qSeq = 64, kvSeq = 512, numHeads = 12, headDim = 128;
        int dim = numHeads * headDim;

        var rng = new Random(7);
        var q = new float[qSeq * dim];
        var k = new float[kvSeq * dim];
        var v = new float[kvSeq * dim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < k.Length; i++) k[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < v.Length; i++) v[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var output = new float[qSeq * dim];
        WanAttention.TiledMultiHeadAttention(q, k, v, output, qSeq, kvSeq, numHeads, headDim);

        var reference = NaiveAttention(q, k, v, qSeq, kvSeq, numHeads, headDim);

        double dot = 0, na = 0, nb = 0, maxDiff = 0;
        for (int i = 0; i < output.Length; i++)
        {
            dot += (double)output[i] * reference[i];
            na += (double)output[i] * output[i];
            nb += (double)reference[i] * reference[i];
            double d = Math.Abs(output[i] - reference[i]);
            if (d > maxDiff) maxDiff = d;
        }
        double cos = dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
        Console.WriteLine($"[WanAttnAsymIsolation] cosine={cos:F9} kernelNorm={Math.Sqrt(na):F4} refNorm={Math.Sqrt(nb):F4} maxDiff={maxDiff:F6}");

        Assert.True(cos > 0.999f, $"TiledMultiHeadAttention diverges from naive reference at asymmetric qSeq!=kvSeq: cosine={cos}");
    }

    [Fact]
    public void TiledMultiHeadAttention_MatchesNaiveReference_AtSymmetricSelfAttnDims()
    {
        // Control: the SAME kernel at qSeq==kvSeq==64 (self-attention shape), already proven correct
        // by the real cross-reference run -- this should trivially pass, confirming the test
        // methodology itself is sound before trusting the asymmetric-case result above.
        const int qSeq = 64, kvSeq = 64, numHeads = 12, headDim = 128;
        int dim = numHeads * headDim;

        var rng = new Random(7);
        var q = new float[qSeq * dim];
        var k = new float[kvSeq * dim];
        var v = new float[kvSeq * dim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < k.Length; i++) k[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < v.Length; i++) v[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var output = new float[qSeq * dim];
        WanAttention.TiledMultiHeadAttention(q, k, v, output, qSeq, kvSeq, numHeads, headDim);

        var reference = NaiveAttention(q, k, v, qSeq, kvSeq, numHeads, headDim);

        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < output.Length; i++)
        {
            dot += (double)output[i] * reference[i];
            na += (double)output[i] * output[i];
            nb += (double)reference[i] * reference[i];
        }
        double cos = dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
        Console.WriteLine($"[WanAttnSymControl] cosine={cos:F9} kernelNorm={Math.Sqrt(na):F4} refNorm={Math.Sqrt(nb):F4}");

        Assert.True(cos > 0.999f, $"Control (symmetric) case failed: cosine={cos}");
    }
}
