using OpenTail.Stingray.Mlx;

namespace OpenTail.Stingray.Tests.Cuda;

public sealed class MlxNativeNumericalTests
{
    [Fact]
    public void MatMul_MatchesNonSquareCpuReference()
    {
        RequireMlx();

        float[] actual = MlxOps.MatMul([1, 2, 3, 4, 5, 6], m: 2, k: 3, b: [1, 2, 3, 4, 5, 6], n: 2);

        Assert.Equal(new float[] { 22, 28, 49, 64 }, actual);
    }

    [Fact]
    public void Rope_MatchesCpuReferenceForNonContiguousPositionsAndDecode()
    {
        RequireMlx();
        float[] input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        int[] positions = [0, 2, 5];

        float[] actual = MlxOps.Rope(input, seqLen: 3, headDim: 4, dims: 4, traditional: false, baseValue: 10000, scale: 1, positions);
        AssertClose(ReferenceRope(input, 3, 4, 4, positions), actual);

        float[] decoded = MlxOps.Rope([1, 2, 3, 4], seqLen: 1, headDim: 4, dims: 4, traditional: false, baseValue: 10000, scale: 1, positions: [6]);
        AssertClose(ReferenceRope([1, 2, 3, 4], 1, 4, 4, [6]), decoded);
    }

    [Fact]
    public void Attention_GqaCausalPrefill_MatchesCpuReferenceAndMasksFutureTokens()
    {
        RequireMlx();
        float[] query = [1, 0, 0, 1, 1, 1, 2, -1, 0.5f, 1, -1, 2];
        float[] key = [1, 0, 0, 1, 2, -1];
        float[] value = [1, 2, 3, 4, 9, 10];

        float[] actual = MlxOps.Attention(query, key, value, numQueryHeads: 2, numKvHeads: 1,
            querySeqLen: 3, kvSeqLen: 3, headDim: 2, scale: 1 / MathF.Sqrt(2), causal: true);
        AssertClose(ReferenceAttention(query, key, value, 2, 1, 3, 3, 2, 1 / MathF.Sqrt(2), causal: true), actual);

        float[] changedFutureValue = [1, 2, 3, 4, -99, 100];
        float[] changed = MlxOps.Attention(query, key, changedFutureValue, numQueryHeads: 2, numKvHeads: 1,
            querySeqLen: 3, kvSeqLen: 3, headDim: 2, scale: 1 / MathF.Sqrt(2), causal: true);
        for (int head = 0; head < 2; head++)
            AssertClose(actual.AsSpan(head * 3 * 2, 2), changed.AsSpan(head * 3 * 2, 2));
    }

    [Fact]
    public void DenseForwardPass_PrefillThenDecode_MatchesIndependentResidualReference()
    {
        RequireMlx();
        const int hiddenDim = 2;
        var layer = new MlxDenseLayer
        {
            AttnNormWeight = [1, 1],
            Wq = new float[4],
            Wk = new float[4],
            Wv = new float[4],
            Wo = new float[4],
            FfnNormWeight = [1, 1],
            WGate = new float[4],
            WUp = new float[4],
            WDown = new float[4],
            HiddenDim = hiddenDim,
            NumHeads = 1,
            NumKvHeads = 1,
            HeadDim = 2,
            IntermDim = 2,
        };
        var model = new MlxDenseModel
        {
            TokenEmbedding = [1, 2, 3, 4],
            Layers = [layer],
            OutputNormWeight = [1, 1],
            OutputWeight = [1, 0, 0, 1],
            HiddenDim = hiddenDim,
            VocabSize = 2,
        };
        using var forwardPass = new MlxForwardPass(model, maxSeqLen: 4);

        ReadOnlySpan<float> prefill = forwardPass.Prefill([0, 1]);
        AssertClose(ReferenceRmsNorm([3, 4], 2, 1e-5f), prefill);

        ReadOnlySpan<float> decode = forwardPass.Forward(token: 0, position: 2);
        AssertClose(ReferenceRmsNorm([1, 2], 2, 1e-5f), decode);
    }

    private static void RequireMlx() => Assert.SkipUnless(MlxBackend.IsAvailable(), "Native MLX with a working Metal device is required; this test has not run numerically on this host.");

    private static float[] ReferenceRope(ReadOnlySpan<float> input, int seqLen, int headDim, int dims, ReadOnlySpan<int> positions)
    {
        var result = input.ToArray();
        int half = dims / 2;
        for (int row = 0; row < seqLen; row++)
        {
            for (int i = 0; i < half; i++)
            {
                double frequency = Math.Pow(10000, -(2.0 * i / dims));
                double angle = positions[row] * frequency;
                float cos = (float)Math.Cos(angle);
                float sin = (float)Math.Sin(angle);
                int first = row * headDim + i;
                int second = row * headDim + i + half;
                float a = input[first];
                float b = input[second];
                result[first] = a * cos - b * sin;
                result[second] = b * cos + a * sin;
            }
        }
        return result;
    }

    private static float[] ReferenceAttention(ReadOnlySpan<float> query, ReadOnlySpan<float> key, ReadOnlySpan<float> value,
        int queryHeads, int kvHeads, int queryLength, int kvLength, int headDim, float scale, bool causal)
    {
        var output = new float[query.Length];
        int headsPerKv = queryHeads / kvHeads;
        for (int head = 0; head < queryHeads; head++)
        {
            int kvHead = head / headsPerKv;
            for (int q = 0; q < queryLength; q++)
            {
                int visible = causal ? Math.Min(q + 1, kvLength) : kvLength;
                var scores = new float[visible];
                float max = float.NegativeInfinity;
                for (int k = 0; k < visible; k++)
                {
                    float dot = 0;
                    for (int d = 0; d < headDim; d++)
                        dot += query[head * queryLength * headDim + q * headDim + d] * key[kvHead * kvLength * headDim + k * headDim + d];
                    scores[k] = dot * scale;
                    max = Math.Max(max, scores[k]);
                }
                float denom = 0;
                for (int k = 0; k < visible; k++)
                    denom += scores[k] = MathF.Exp(scores[k] - max);
                for (int d = 0; d < headDim; d++)
                    for (int k = 0; k < visible; k++)
                        output[head * queryLength * headDim + q * headDim + d] += scores[k] / denom * value[kvHead * kvLength * headDim + k * headDim + d];
            }
        }
        return output;
    }

    private static float[] ReferenceRmsNorm(ReadOnlySpan<float> input, int hiddenDim, float eps)
    {
        float meanSquare = 0;
        for (int i = 0; i < hiddenDim; i++) meanSquare += input[i] * input[i];
        meanSquare /= hiddenDim;
        float inverse = 1 / MathF.Sqrt(meanSquare + eps);
        var result = new float[hiddenDim];
        for (int i = 0; i < hiddenDim; i++) result[i] = input[i] * inverse;
        return result;
    }

    private static void AssertClose(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float tolerance = 1e-4f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(MathF.Abs(expected[i] - actual[i]), 0, tolerance);
    }
}