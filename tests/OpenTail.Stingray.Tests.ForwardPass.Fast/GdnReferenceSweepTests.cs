namespace OpenTail.Stingray.Tests.ForwardPass.Fast;

/// <summary>
/// The Gated DeltaNet kernels against an INDEPENDENT double-precision implementation of llama.cpp's recurrence (written from
/// delta-net-base.cpp / ops.cpp's gated_delta_net, not copied from the kernel), over odd head dimensions (SIMD tails), many
/// token counts and chunk sizes, and a non-zero incoming state. The existing GdnKernelsTests compare the paths to each other
/// at head dims 2/8/16; this pins them to the spec and to dims that are not multiples of the vector width.
/// </summary>
public sealed class GdnReferenceSweepTests
{
    private sealed record Inputs(float[] Q, float[] K, float[] V, float[] Alpha, float[] Beta, float[] SsmA, float[] DtBias,
                                 float[] NormW, float[] Z, float[] State0);

    private static Inputs MakeInputs(int tokens, int hv, int d, Random rnd)
    {
        float[] Arr(int n, double lo, double hi) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(lo + rnd.NextDouble() * (hi - lo)); return a; }
        // q and k are L2-normalised per head in the real model (a separate kernel); mirror that so the state stays bounded.
        float[] Unit(int count)
        {
            var a = Arr(count * hv * d, -1, 1);
            for (int c = 0; c < count * hv; c++)
            {
                double s = 0; for (int j = 0; j < d; j++) s += (double)a[c * d + j] * a[c * d + j];
                double inv = 1.0 / Math.Max(Math.Sqrt(s), 1e-6);
                for (int j = 0; j < d; j++) a[c * d + j] = (float)(a[c * d + j] * inv);
            }
            return a;
        }
        return new Inputs(Unit(tokens), Unit(tokens), Arr(tokens * hv * d, -1, 1), Arr(tokens * hv, -2, 2), Arr(tokens * hv, -3, 3),
                          Arr(hv, -2, -0.05), Arr(hv, -1, 1), Arr(d, 0.5, 1.5), Arr(tokens * hv * d, -3, 3), Arr(hv * d * d, -0.3, 0.3));
    }

    /// <summary>Token-by-token double reference. state[h][i*d+j] = S_h[i,j]; returns outputs [tokens, hv, d] and the final state.</summary>
    private static (double[] Out, double[] State) Reference(Inputs x, int tokens, int hv, int d, double normEps)
    {
        var state = x.State0.Select(v => (double)v).ToArray();
        var output = new double[tokens * hv * d];
        double readoutScale = 1.0 / Math.Sqrt(d);
        var p = new double[d]; var dvec = new double[d]; var o = new double[d];
        for (int t = 0; t < tokens; t++)
        {
            for (int h = 0; h < hv; h++)
            {
                double alphaX = x.Alpha[t * hv + h] + (double)x.DtBias[h];
                double dt = alphaX >= 20 ? alphaX : Math.Log(1 + Math.Exp(alphaX));
                double decay = Math.Exp(dt * x.SsmA[h]);
                double b = 1.0 / (1.0 + Math.Exp(-(double)x.Beta[t * hv + h]));
                int sOff = h * d * d, qkv = (t * hv + h) * d;

                for (int i = 0; i < d * d; i++) state[sOff + i] *= decay;
                for (int j = 0; j < d; j++) { double s = 0; for (int i = 0; i < d; i++) s += state[sOff + i * d + j] * x.K[qkv + i]; p[j] = s; }
                for (int j = 0; j < d; j++) dvec[j] = b * (x.V[qkv + j] - p[j]);
                for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) state[sOff + i * d + j] += x.K[qkv + i] * dvec[j];
                for (int j = 0; j < d; j++) { double s = 0; for (int i = 0; i < d; i++) s += state[sOff + i * d + j] * x.Q[qkv + i]; o[j] = s * readoutScale; }

                double ss = 0; for (int j = 0; j < d; j++) ss += o[j] * o[j];
                double inv = 1.0 / Math.Sqrt(ss / d + normEps);
                for (int j = 0; j < d; j++)
                {
                    double z = x.Z[qkv + j];
                    output[qkv + j] = o[j] * inv * x.NormW[j] * (z / (1.0 + Math.Exp(-z)));
                }
            }
        }
        return (output, state);
    }

    private static void Compare(string what, double[] expected, float[] actual, double relTol)
    {
        double scale = expected.Select(Math.Abs).DefaultIfEmpty(0).Max() + 1e-6;
        for (int i = 0; i < expected.Length; i++)
            Assert.True(float.IsFinite(actual[i]) && Math.Abs(expected[i] - actual[i]) <= relTol * scale,
                $"{what}[{i}]: expected {expected[i]:G7}, got {actual[i]:G7} (tol {relTol * scale:E2})");
    }

    private const float Eps = 1e-6f;

    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(1, 3, 3)]
    [InlineData(7, 2, 5)]
    [InlineData(5, 2, 7)]
    [InlineData(16, 1, 12)]
    [InlineData(17, 3, 17)]
    [InlineData(65, 2, 33)]
    [InlineData(130, 1, 100)]
    [InlineData(64, 2, 64)]
    public void SequentialAndDecodeAndChunked_MatchTheIndependentReference(int tokens, int hv, int d)
    {
        var rnd = new Random(tokens * 1000 + hv * 100 + d);
        var x = MakeInputs(tokens, hv, d, rnd);
        var (expOut, expState) = Reference(x, tokens, hv, d, Eps);

        // sequential prefill
        var stateSeq = (float[])x.State0.Clone(); var outSeq = new float[tokens * hv * d];
        GdnKernels.GdnRecurrencePrefill(tokens, x.Q, x.K, x.V, x.Alpha, x.Beta, x.SsmA, x.DtBias, x.NormW, x.Z, stateSeq, outSeq, hv, d, Eps);
        Compare($"prefill out (t={tokens} hv={hv} d={d})", expOut, outSeq, 2e-3);
        Compare($"prefill state (t={tokens} hv={hv} d={d})", expState, stateSeq, 2e-3);

        // decode, one token per call, carrying the state
        var stateDec = (float[])x.State0.Clone(); var outDec = new float[tokens * hv * d];
        int per = hv * d;
        for (int t = 0; t < tokens; t++)
            GdnKernels.GdnRecurrenceDecode(x.Q.AsSpan(t * per, per), x.K.AsSpan(t * per, per), x.V.AsSpan(t * per, per),
                x.Alpha.AsSpan(t * hv, hv), x.Beta.AsSpan(t * hv, hv), x.SsmA, x.DtBias, x.NormW, x.Z.AsSpan(t * per, per),
                stateDec, outDec.AsSpan(t * per, per), hv, d, Eps);
        Compare($"decode out (t={tokens} hv={hv} d={d})", expOut, outDec, 2e-3);
        Compare($"decode state (t={tokens} hv={hv} d={d})", expState, stateDec, 2e-3);

        // chunked prefill at several chunk sizes, including ones that leave remainders and a chunk larger than the input
        foreach (int chunk in new[] { 1, 3, 16, 64, 200 })
        {
            var stateChunk = (float[])x.State0.Clone(); var outChunk = new float[tokens * hv * d];
            GdnKernels.GdnRecurrenceChunkedPrefill(tokens, x.Q, x.K, x.V, x.Alpha, x.Beta, x.SsmA, x.DtBias, x.NormW, x.Z, stateChunk, outChunk, hv, d, Eps, chunk);
            Compare($"chunked(C={chunk}) out (t={tokens} hv={hv} d={d})", expOut, outChunk, 5e-3);
            Compare($"chunked(C={chunk}) state (t={tokens} hv={hv} d={d})", expState, stateChunk, 5e-3);
        }
    }

    [Fact]
    public void SplittingAPrefillAcrossCalls_EqualsOneCall()
    {
        // State continuity across call boundaries (what chunked admission does for a long prompt).
        const int tokens = 90, hv = 2, d = 13;
        var x = MakeInputs(tokens, hv, d, new Random(4242));
        var (expOut, expState) = Reference(x, tokens, hv, d, Eps);
        foreach (int split in new[] { 1, 31, 64, 89 })
        {
            var state = (float[])x.State0.Clone(); var output = new float[tokens * hv * d];
            int per = hv * d;
            void Part(int from, int n) => GdnKernels.GdnRecurrenceChunkedPrefill(n,
                x.Q.AsSpan(from * per, n * per), x.K.AsSpan(from * per, n * per), x.V.AsSpan(from * per, n * per),
                x.Alpha.AsSpan(from * hv, n * hv), x.Beta.AsSpan(from * hv, n * hv), x.SsmA, x.DtBias, x.NormW,
                x.Z.AsSpan(from * per, n * per), state, output.AsSpan(from * per, n * per), hv, d, Eps, 16);
            Part(0, split); Part(split, tokens - split);
            Compare($"split@{split} out", expOut, output, 5e-3);
            Compare($"split@{split} state", expState, state, 5e-3);
        }
    }
}
