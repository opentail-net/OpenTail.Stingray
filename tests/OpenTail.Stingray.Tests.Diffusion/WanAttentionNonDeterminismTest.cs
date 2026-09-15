using OpenTail.Stingray.Diffusion.Wan;

namespace OpenTail.Stingray.Tests.Diffusion;

/// <summary>
/// Decisive test (docs/081, 2026-09-14): calls WanAttention.TiledMultiHeadAttention repeatedly with
/// FIXED, IDENTICAL input (simulating the 30-sequential-call pattern of a real Wan forward pass,
/// where ArrayPool-rented buffers get repeatedly rented/returned across many calls) to check whether
/// it produces the SAME output every time. A real cross-reference investigation found that replaying
/// captured Q/K/V through a freshly-called instance of this kernel gave a DIFFERENT result than what
/// the original live 30-block run itself produced with the identical data -- suggesting real
/// non-determinism (likely an ArrayPool reuse / Parallel.For race), not a pure math bug.
/// </summary>
public sealed class WanAttentionNonDeterminismTest
{
    [Fact]
    public void TiledMultiHeadAttention_RepeatedCallsWithIdenticalInput_ProduceIdenticalOutput()
    {
        const int qSeq = 64, kvSeq = 512, numHeads = 12, headDim = 128;
        int dim = numHeads * headDim;

        var rng = new Random(99);
        var q = new float[qSeq * dim];
        var k = new float[kvSeq * dim];
        var v = new float[kvSeq * dim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < k.Length; i++) k[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < v.Length; i++) v[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        float[]? first = null;
        int mismatches = 0;
        for (int call = 0; call < 40; call++)
        {
            // Interleave with OTHER, differently-sized attention calls (self-attn shape) between
            // each cross-attn-shape call, mirroring the real per-block interleaving of self-attn
            // (qSeq==kvSeq==64) and cross-attn (qSeq=64, kvSeq=512) calls across 30 real blocks --
            // exercises the same ArrayPool rent/return churn pattern the real run hits.
            var dummyQ = new float[qSeq * dim];
            var dummyOut = new float[qSeq * dim];
            Array.Copy(q, dummyQ, q.Length);
            WanAttention.TiledMultiHeadAttention(dummyQ, dummyQ, dummyQ, dummyOut, qSeq, qSeq, numHeads, headDim);

            var output = new float[qSeq * dim];
            WanAttention.TiledMultiHeadAttention(q, k, v, output, qSeq, kvSeq, numHeads, headDim);

            if (first is null)
            {
                first = output;
            }
            else
            {
                double dot = 0, na = 0, nb = 0;
                for (int i = 0; i < output.Length; i++)
                {
                    dot += (double)output[i] * first[i];
                    na += (double)output[i] * output[i];
                    nb += (double)first[i] * first[i];
                }
                double cos = dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
                if (cos < 0.999999)
                {
                    mismatches++;
                    Console.WriteLine($"[WanAttnNonDeterminism] call={call} cosine_vs_first={cos:F9} norm={Math.Sqrt(na):F4} (first norm={Math.Sqrt(nb):F4})");
                }
            }
        }

        Console.WriteLine($"[WanAttnNonDeterminism] mismatches out of 39 repeat calls: {mismatches}");
        Assert.True(mismatches == 0, $"TiledMultiHeadAttention produced different output across {mismatches}/39 repeated calls with IDENTICAL input -- real non-determinism found");
    }
}
