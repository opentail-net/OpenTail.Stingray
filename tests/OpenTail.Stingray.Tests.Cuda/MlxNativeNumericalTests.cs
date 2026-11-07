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

        int[] contiguousPositions = [3, 4, 5];
        float[] contiguous = MlxOps.Rope(input, seqLen: 3, headDim: 4, dims: 4, traditional: false, baseValue: 10000, scale: 1, contiguousPositions);
        AssertClose(ReferenceRope(input, 3, 4, 4, contiguousPositions), contiguous);

        float[] decoded = MlxOps.Rope([1, 2, 3, 4], seqLen: 1, headDim: 4, dims: 4, traditional: false, baseValue: 10000, scale: 1, positions: [6]);
        AssertClose(ReferenceRope([1, 2, 3, 4], 1, 4, 4, [6]), decoded);
    }

    [Fact]
    public void Attention_OffsetCausalMask_MatchesCpuReferenceAndMasksFutureKeys()
    {
        RequireMlx();
        float[] query = [1, 0, 0, 1, 1, 1, 2, -1];
        float[] key = [1, 0, 0, 1, 2, -1, -1, 2, 0.5f, 1];
        float[] value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        float scale = 1 / MathF.Sqrt(2);

        float[] actual = MlxOps.Attention(query, key, value, numQueryHeads: 2, numKvHeads: 1,
            querySeqLen: 2, kvSeqLen: 5, headDim: 2, scale, causal: true, queryOffset: 3);
        AssertClose(ReferenceAttention(query, key, value, 2, 1, 2, 5, 2, scale, causal: true, queryOffset: 3), actual);

        float[] futureChanged = (float[])value.Clone();
        futureChanged[^2] = -100;
        futureChanged[^1] = 100;
        float[] changed = MlxOps.Attention(query, key, futureChanged, numQueryHeads: 2, numKvHeads: 1,
            querySeqLen: 2, kvSeqLen: 5, headDim: 2, scale, causal: true, queryOffset: 3);
        for (int head = 0; head < 2; head++)
            AssertClose(actual.AsSpan(head * 4, 2), changed.AsSpan(head * 4, 2));
    }

    [Fact]
    public void SplitAndMergeHeads_MatchExactNonSymmetricOrdering()
    {
        RequireMlx();
        float[] flat = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

        float[] split = MlxOps.SplitHeads(flat, seqLen: 3, numHeads: 2, headDim: 2);
        float[] merged = MlxOps.MergeHeads(split, numHeads: 2, seqLen: 3, headDim: 2);

        Assert.Equal(new float[] { 1, 2, 5, 6, 9, 10, 3, 4, 7, 8, 11, 12 }, split);
        Assert.Equal(flat, merged);
    }

    [Fact]
    public void Gemma4VisionAttention_MatchesBidirectionalReferenceAcrossBatchesAndHeads()
    {
        RequireMlx();
        const int batches = 2, tokens = 3, heads = 2, headDim = 64;
        int length = batches * tokens * heads * headDim;
        var query = new float[length];
        var key = new float[length];
        var value = new float[length];
        for (int batch = 0; batch < batches; batch++)
            for (int token = 0; token < tokens; token++)
                for (int head = 0; head < heads; head++)
                    for (int d = 0; d < headDim; d++)
                    {
                        int index = ((batch * tokens + token) * heads + head) * headDim + d;
                        query[index] = (batch + 1) * 0.01f + (token + 1) * 0.02f + (head + 1) * 0.005f + (d % 5) * 0.001f;
                        key[index] = (batch + 1) * -0.015f + (token + 1) * 0.01f + (head + 1) * 0.003f - (d % 7) * 0.0005f;
                        value[index] = (batch + 1) * 0.2f + (token + 1) * 0.07f + (head + 1) * 0.03f + (d % 11) * 0.002f;
                    }

        float[] actual = MlxOps.Gemma4VisionSelfAttention(query, key, value, batches, tokens, heads);
        float[] expected = ReferenceVisionAttention(query, key, value, batches, tokens, heads, headDim);

        Assert.Equal(length, actual.Length);
        AssertClose(expected, actual, tolerance: 2e-3f);
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
        const int hiddenDim = 4;
        var layer = new MlxDenseLayer
        {
            AttnNormWeight = [1.1f, 0.9f, 1.2f, 0.8f],
            Wq = [0.2f, -0.1f, 0.05f, 0.3f, -0.15f, 0.25f, 0.2f, -0.05f, 0.1f, 0.2f, -0.3f, 0.15f, 0.05f, -0.2f, 0.25f, 0.1f],
            Wk = [0.1f, 0.2f, -0.1f, 0.05f, 0.2f, -0.15f, 0.1f, 0.25f, -0.05f, 0.1f, 0.2f, -0.1f, 0.15f, 0.05f, -0.2f, 0.1f],
            Wv = [-0.1f, 0.15f, 0.2f, 0.05f, 0.25f, -0.05f, 0.1f, 0.2f, 0.1f, 0.2f, -0.15f, 0.05f, -0.05f, 0.1f, 0.15f, -0.2f],
            Wo = [0.2f, 0.05f, -0.1f, 0.15f, -0.05f, 0.2f, 0.1f, -0.1f, 0.15f, -0.1f, 0.2f, 0.05f, 0.1f, 0.15f, -0.05f, 0.2f],
            FfnNormWeight = [0.95f, 1.05f, 0.9f, 1.1f],
            WGate = [0.1f, -0.1f, 0.15f, 0.05f, -0.05f, 0.2f, 0.1f, -0.15f, 0.2f, 0.05f, -0.1f, 0.1f, 0.05f, 0.15f, -0.05f, 0.2f],
            WUp = [0.2f, 0.05f, -0.1f, 0.1f, 0.1f, -0.05f, 0.2f, 0.05f, -0.15f, 0.1f, 0.05f, 0.2f, 0.05f, 0.15f, -0.1f, 0.1f],
            WDown = [0.1f, -0.05f, 0.2f, 0.05f, -0.1f, 0.15f, 0.05f, 0.1f, 0.2f, 0.05f, -0.1f, 0.15f, 0.05f, 0.2f, 0.1f, -0.05f],
            HiddenDim = hiddenDim,
            NumHeads = 1,
            NumKvHeads = 1,
            HeadDim = 4,
            IntermDim = 4,
        };
        var model = new MlxDenseModel
        {
            TokenEmbedding = [1f, -0.5f, 0.25f, 2f, -1f, 0.75f, 1.5f, -0.25f],
            Layers = [layer],
            OutputNormWeight = [1.05f, 0.95f, 1.1f, 0.9f],
            OutputWeight = [0.4f, -0.2f, 0.15f, 0.3f, -0.1f, 0.25f, 0.2f, -0.35f],
            HiddenDim = hiddenDim,
            VocabSize = 2,
        };
        using var forwardPass = new MlxForwardPass(model, maxSeqLen: 4);

        ReadOnlySpan<float> prefill = forwardPass.Prefill([0, 1]);
        float[] prefillReference = ReferenceDensePass(model, [0, 1], [0, 1], cachedKv: null);
        AssertClose(prefillReference.AsSpan(prefillReference.Length - model.VocabSize), prefill);

        ReadOnlySpan<float> decode = forwardPass.Forward(token: 0, position: 2);
        float[] decodeReference = ReferenceDensePass(model, [0], [2], cachedKv: ReferenceCachesAfterPrefill(model, [0, 1]));
        AssertClose(decodeReference.AsSpan(decodeReference.Length - model.VocabSize), decode);
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

    private static float[] ReferenceDensePass(MlxDenseModel model, ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions,
        (float[] Keys, float[] Values)? cachedKv)
    {
        MlxDenseLayer layer = Assert.Single(model.Layers);
        int hiddenDim = model.HiddenDim;
        int seqLen = tokens.Length;
        var hidden = new float[seqLen * hiddenDim];
        for (int i = 0; i < seqLen; i++)
            model.TokenEmbedding.AsSpan(tokens[i] * hiddenDim, hiddenDim).CopyTo(hidden.AsSpan(i * hiddenDim, hiddenDim));

        float[] normed = ReferenceRmsNormRows(hidden, seqLen, hiddenDim, layer.AttnNormWeight, layer.Eps);
        float[] q = ReferenceMatMul(normed, seqLen, hiddenDim, layer.Wq, layer.NumHeads * layer.HeadDim);
        float[] k = ReferenceMatMul(normed, seqLen, hiddenDim, layer.Wk, layer.NumKvHeads * layer.HeadDim);
        float[] v = ReferenceMatMul(normed, seqLen, hiddenDim, layer.Wv, layer.NumKvHeads * layer.HeadDim);
        q = ReferenceRope(q, seqLen, layer.HeadDim, layer.HeadDim, positions);
        k = ReferenceRope(k, seqLen, layer.HeadDim, layer.HeadDim, positions);

        float[] allKeys = cachedKv is { } past ? [.. past.Keys, .. k] : k;
        float[] allValues = cachedKv is { } pastValues ? [.. pastValues.Values, .. v] : v;
        int kvLength = allKeys.Length / layer.HeadDim;
        float[] attn = ReferenceAttention(q, allKeys, allValues, layer.NumHeads, layer.NumKvHeads,
            seqLen, kvLength, layer.HeadDim, 1 / MathF.Sqrt(layer.HeadDim), causal: true,
            queryOffset: cachedKv?.Keys.Length / layer.HeadDim ?? 0);
        float[] projected = ReferenceMatMul(attn, seqLen, hiddenDim, layer.Wo, hiddenDim);
        float[] afterAttn = AddReference(hidden, projected);

        float[] normedFfn = ReferenceRmsNormRows(afterAttn, seqLen, hiddenDim, layer.FfnNormWeight, layer.Eps);
        float[] gate = ReferenceMatMul(normedFfn, seqLen, hiddenDim, layer.WGate, layer.IntermDim);
        float[] up = ReferenceMatMul(normedFfn, seqLen, hiddenDim, layer.WUp, layer.IntermDim);
        for (int i = 0; i < gate.Length; i++)
            gate[i] = gate[i] / (1 + MathF.Exp(-gate[i])) * up[i];
        float[] down = ReferenceMatMul(gate, seqLen, layer.IntermDim, layer.WDown, hiddenDim);
        float[] output = AddReference(afterAttn, down);

        float[] outputNorm = ReferenceRmsNormRows(output, seqLen, hiddenDim, model.OutputNormWeight, model.Eps);
        return ReferenceMatMul(outputNorm, seqLen, hiddenDim, model.OutputWeight, model.VocabSize);
    }

    private static (float[] Keys, float[] Values) ReferenceCachesAfterPrefill(MlxDenseModel model, ReadOnlySpan<int> tokens)
    {
        MlxDenseLayer layer = Assert.Single(model.Layers);
        int seqLen = tokens.Length;
        var hidden = new float[seqLen * model.HiddenDim];
        for (int i = 0; i < seqLen; i++)
            model.TokenEmbedding.AsSpan(tokens[i] * model.HiddenDim, model.HiddenDim).CopyTo(hidden.AsSpan(i * model.HiddenDim, model.HiddenDim));
        float[] normed = ReferenceRmsNormRows(hidden, seqLen, model.HiddenDim, layer.AttnNormWeight, layer.Eps);
        float[] k = ReferenceMatMul(normed, seqLen, model.HiddenDim, layer.Wk, layer.HeadDim);
        float[] v = ReferenceMatMul(normed, seqLen, model.HiddenDim, layer.Wv, layer.HeadDim);
        int[] positions = Enumerable.Range(0, seqLen).ToArray();
        return (ReferenceRope(k, seqLen, layer.HeadDim, layer.HeadDim, positions), v);
    }

    private static float[] ReferenceRmsNormRows(ReadOnlySpan<float> input, int rows, int cols, ReadOnlySpan<float> weight, float eps)
    {
        var result = new float[input.Length];
        for (int row = 0; row < rows; row++)
        {
            float meanSquare = 0;
            for (int col = 0; col < cols; col++)
            {
                float x = input[row * cols + col];
                meanSquare += x * x;
            }
            float inverse = 1 / MathF.Sqrt(meanSquare / cols + eps);
            for (int col = 0; col < cols; col++)
                result[row * cols + col] = input[row * cols + col] * inverse * weight[col];
        }
        return result;
    }

    private static float[] ReferenceMatMul(ReadOnlySpan<float> input, int rows, int inputDim, ReadOnlySpan<float> weight, int outputDim)
    {
        var output = new float[rows * outputDim];
        for (int row = 0; row < rows; row++)
            for (int column = 0; column < outputDim; column++)
                for (int i = 0; i < inputDim; i++)
                    output[row * outputDim + column] += input[row * inputDim + i] * weight[i * outputDim + column];
        return output;
    }

    private static float[] ReferenceVisionAttention(ReadOnlySpan<float> query, ReadOnlySpan<float> key,
        ReadOnlySpan<float> value, int batches, int tokens, int heads, int headDim)
    {
        var output = new float[query.Length];
        for (int batch = 0; batch < batches; batch++)
            for (int head = 0; head < heads; head++)
                for (int q = 0; q < tokens; q++)
                {
                    var scores = new float[tokens];
                    float max = float.NegativeInfinity;
                    for (int k = 0; k < tokens; k++)
                    {
                        float dot = 0;
                        for (int d = 0; d < headDim; d++)
                        {
                            int qIndex = ((batch * tokens + q) * heads + head) * headDim + d;
                            int kIndex = ((batch * tokens + k) * heads + head) * headDim + d;
                            dot += query[qIndex] * key[kIndex];
                        }
                        scores[k] = dot;
                        max = Math.Max(max, dot);
                    }
                    float denominator = 0;
                    for (int k = 0; k < tokens; k++) denominator += scores[k] = MathF.Exp(scores[k] - max);
                    for (int d = 0; d < headDim; d++)
                        for (int k = 0; k < tokens; k++)
                        {
                            int outputIndex = ((batch * tokens + q) * heads + head) * headDim + d;
                            int valueIndex = ((batch * tokens + k) * heads + head) * headDim + d;
                            output[outputIndex] += scores[k] / denominator * value[valueIndex];
                        }
                }
        return output;
    }

    private static float[] AddReference(ReadOnlySpan<float> lhs, ReadOnlySpan<float> rhs)
    {
        var result = new float[lhs.Length];
        for (int i = 0; i < result.Length; i++) result[i] = lhs[i] + rhs[i];
        return result;
    }

    private static float[] ReferenceAttention(ReadOnlySpan<float> query, ReadOnlySpan<float> key, ReadOnlySpan<float> value,
        int queryHeads, int kvHeads, int queryLength, int kvLength, int headDim, float scale, bool causal, int queryOffset = 0)
    {
        var output = new float[query.Length];
        int headsPerKv = queryHeads / kvHeads;
        for (int head = 0; head < queryHeads; head++)
        {
            int kvHead = head / headsPerKv;
            for (int q = 0; q < queryLength; q++)
            {
                int visible = causal ? Math.Min(queryOffset + q + 1, kvLength) : kvLength;
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