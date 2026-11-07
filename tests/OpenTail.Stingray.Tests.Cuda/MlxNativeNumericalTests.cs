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

    [Fact]
    public void ScatterAddWeightedRows_MatchesReferenceForSparseSortedIndices()
    {
        RequireMlx();
        float[] inPlace = [1, 2, 3, 4, 5, 6, 7, 8]; // seqLen=4, hiddenDim=2
        float[] rows = [10, 20, 30, 40]; // batchSize=2, hiddenDim=2
        int[] indices = [0, 2];
        float[] weights = [0.5f, 2f];

        float[] actual = MlxOps.ScatterAddWeightedRows(inPlace, seqLen: 4, hiddenDim: 2, rows, indices, weights);

        AssertClose(ReferenceScatterAddWeightedRows(inPlace, 4, 2, rows, indices, weights), actual);
    }

    [Fact]
    public void ScatterAddWeightedRows_NoMatchesLeavesInputUnchanged()
    {
        RequireMlx();
        float[] inPlace = [1, 2, 3, 4];
        float[] rows = [100, 200];
        int[] indices = [1];
        float[] weights = [5f];

        float[] actual = MlxOps.ScatterAddWeightedRows(inPlace, seqLen: 2, hiddenDim: 2, rows, indices, weights);

        AssertClose(ReferenceScatterAddWeightedRows(inPlace, 2, 2, rows, indices, weights), actual);
        AssertClose(inPlace.AsSpan(0, 2), actual.AsSpan(0, 2));
    }

    [Fact]
    public void SwiGluOaiGatherBias_MatchesReferenceAcrossClampedAndUnclampedRows()
    {
        RequireMlx();
        float[] gate = [2f, -1f, 0.5f, 3f]; // rows=2, dim=2
        float[] up = [1f, 2f, -0.5f, 0.2f];
        float[] gateBias = [0.1f, -0.1f, 0.2f, 0.3f]; // numExperts=2, dim=2
        float[] upBias = [-0.2f, 0.1f, 0.05f, -0.1f];
        int[] expertIndices = [1, 0];
        const float alpha = 1.702f, limit = 2.5f;

        float[] actual = MlxOps.SwiGluOaiGatherBias(gate, up, rows: 2, dim: 2, gateBias, upBias, numExperts: 2, expertIndices, alpha, limit);

        AssertClose(ReferenceSwiGluOaiGatherBias(gate, up, 2, 2, gateBias, upBias, expertIndices, alpha, limit), actual, tolerance: 2e-3f);
    }

    [Fact]
    public void MoeBiasWeightedSum_MatchesReferenceWithAndWithoutBias()
    {
        RequireMlx();
        const int tokens = 2, expertsPerToken = 2, dim = 2, numExperts = 2;
        float[] downRows = [1, 2, 3, 4, 5, 6, 7, 8]; // pairs=4, dim=2, in sorted order
        float[] downBias = [0.1f, 0.2f, -0.1f, 0.05f]; // numExperts=2, dim=2
        int[] expertsSorted = [0, 0, 1, 1];
        int[] invOrder = [2, 0, 3, 1]; // pair p's sorted row
        float[] routeWeights = [0.6f, 0.4f, 0.3f, 0.7f];

        float[] withBias = MlxOps.MoeBiasWeightedSum(downRows, tokens, expertsPerToken, dim, downBias, numExperts, expertsSorted, invOrder, routeWeights);
        AssertClose(ReferenceMoeBiasWeightedSum(downRows, tokens, expertsPerToken, dim, downBias, numExperts, expertsSorted, invOrder, routeWeights), withBias);

        float[] withoutBias = MlxOps.MoeBiasWeightedSum(downRows, tokens, expertsPerToken, dim, [], numExperts, expertsSorted, invOrder, routeWeights);
        AssertClose(ReferenceMoeBiasWeightedSum(downRows, tokens, expertsPerToken, dim, [], numExperts, expertsSorted, invOrder, routeWeights), withoutBias);
    }

    [Fact]
    public void Gemma4VisionAddPosition2D_MatchesReferenceAcrossPatchGrid()
    {
        RequireMlx();
        const int patchesX = 2, patchesY = 3, channels = 2, maxPositions = 3;
        var hidden = new float[patchesX * patchesY * channels];
        for (int i = 0; i < hidden.Length; i++) hidden[i] = i * 0.1f;
        var positions = new float[2 * maxPositions * channels];
        for (int i = 0; i < positions.Length; i++) positions[i] = (i + 1) * 0.01f;

        float[] actual = MlxOps.Gemma4VisionAddPosition2D(hidden, positions, patchesX, patchesY, channels, maxPositions);

        AssertClose(ReferenceAddPosition2D(hidden, positions, patchesX, patchesY, channels, maxPositions), actual);
    }

    [Fact]
    public void Gemma4VisionPool3x3_CropsIncompleteTrailingBoxesAndAverages()
    {
        RequireMlx();
        const int patchesX = 4, patchesY = 3, channels = 2; // 4/3=1 merged column (trailing column cropped)
        var input = new float[patchesX * patchesY * channels];
        for (int i = 0; i < input.Length; i++) input[i] = i + 1;

        float[] actual = MlxOps.Gemma4VisionPool3x3(input, patchesX, patchesY, channels);

        AssertClose(ReferencePool3x3(input, patchesX, patchesY, channels), actual);
    }

    [Fact]
    public void Gemma4VisionRope2D_RotatesXAndYHalvesIndependently()
    {
        RequireMlx();
        const int tokens = 2, heads = 2, headDim = 8; // halfWidth=4, quarter=2
        var data = new float[tokens * heads * headDim];
        for (int i = 0; i < data.Length; i++) data[i] = i * 0.1f - 1f;
        var cosine = new float[tokens * (headDim / 2)];
        var sine = new float[tokens * (headDim / 2)];
        for (int i = 0; i < cosine.Length; i++)
        {
            double angle = (i + 1) * 0.3;
            cosine[i] = (float)Math.Cos(angle);
            sine[i] = (float)Math.Sin(angle);
        }

        float[] actual = MlxOps.Gemma4VisionRope2D(data, cosine, sine, tokens, heads, headDim);

        AssertClose(ReferenceRope2D(data, cosine, sine, tokens, heads, headDim), actual, tolerance: 2e-3f);
    }

    private static float[] ReferenceScatterAddWeightedRows(ReadOnlySpan<float> inPlace, int seqLen, int hiddenDim,
        ReadOnlySpan<float> rows, ReadOnlySpan<int> indices, ReadOnlySpan<float> weights)
    {
        var result = inPlace.ToArray();
        for (int k = 0; k < indices.Length; k++)
        {
            int row = indices[k];
            for (int c = 0; c < hiddenDim; c++)
                result[row * hiddenDim + c] += weights[k] * rows[k * hiddenDim + c];
        }
        return result;
    }

    private static float[] ReferenceSwiGluOaiGatherBias(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, int rows, int dim,
        ReadOnlySpan<float> gateBias, ReadOnlySpan<float> upBias, ReadOnlySpan<int> expertIndices, float alpha, float limit)
    {
        var result = new float[rows * dim];
        for (int r = 0; r < rows; r++)
        {
            int expert = expertIndices[r];
            for (int c = 0; c < dim; c++)
            {
                float g = gate[r * dim + c] + gateBias[expert * dim + c];
                float u = up[r * dim + c] + upBias[expert * dim + c];
                float gMin = Math.Min(g, limit);
                float uClamped = Math.Clamp(u, -limit, limit);
                result[r * dim + c] = gMin * (1f / (1f + MathF.Exp(-alpha * gMin))) * (uClamped + 1f);
            }
        }
        return result;
    }

    private static float[] ReferenceMoeBiasWeightedSum(ReadOnlySpan<float> downRows, int tokens, int expertsPerToken, int dim,
        ReadOnlySpan<float> downBias, int numExperts, ReadOnlySpan<int> expertsSorted, ReadOnlySpan<int> invOrder, ReadOnlySpan<float> routeWeights)
    {
        var result = new float[tokens * dim];
        bool hasBias = !downBias.IsEmpty;
        for (int n = 0; n < tokens; n++)
            for (int k = 0; k < expertsPerToken; k++)
            {
                int pair = n * expertsPerToken + k;
                int sortedRow = invOrder[pair];
                for (int d = 0; d < dim; d++)
                {
                    float value = downRows[sortedRow * dim + d];
                    if (hasBias) value += downBias[expertsSorted[sortedRow] * dim + d];
                    result[n * dim + d] += routeWeights[pair] * value;
                }
            }
        return result;
    }

    private static float[] ReferenceAddPosition2D(ReadOnlySpan<float> hidden, ReadOnlySpan<float> positions,
        int patchesX, int patchesY, int channels, int maxPositions)
    {
        var result = new float[hidden.Length];
        for (int py = 0; py < patchesY; py++)
            for (int px = 0; px < patchesX; px++)
            {
                int patch = py * patchesX + px;
                for (int c = 0; c < channels; c++)
                {
                    float xv = positions[(0 * maxPositions + px) * channels + c];
                    float yv = positions[(1 * maxPositions + py) * channels + c];
                    result[patch * channels + c] = hidden[patch * channels + c] + xv + yv;
                }
            }
        return result;
    }

    private static float[] ReferencePool3x3(ReadOnlySpan<float> input, int patchesX, int patchesY, int channels)
    {
        int mergedX = patchesX / 3, mergedY = patchesY / 3;
        var result = new float[mergedY * mergedX * channels];
        for (int my = 0; my < mergedY; my++)
            for (int mx = 0; mx < mergedX; mx++)
                for (int c = 0; c < channels; c++)
                {
                    float sum = 0;
                    for (int ky = 0; ky < 3; ky++)
                        for (int kx = 0; kx < 3; kx++)
                            sum += input[((my * 3 + ky) * patchesX + mx * 3 + kx) * channels + c];
                    result[(my * mergedX + mx) * channels + c] = sum / 9f;
                }
        return result;
    }

    private static float[] ReferenceRope2D(ReadOnlySpan<float> data, ReadOnlySpan<float> cosine, ReadOnlySpan<float> sine,
        int tokens, int heads, int headDim)
    {
        var result = data.ToArray();
        int halfWidth = headDim / 2, quarter = headDim / 4;
        for (int token = 0; token < tokens; token++)
            for (int h = 0; h < heads; h++)
                for (int axis = 0; axis < 2; axis++)
                    for (int index = 0; index < quarter; index++)
                    {
                        int channel = axis * quarter + index;
                        long baseIndex = (token * (long)heads + h) * headDim + axis * halfWidth + index;
                        int table = token * halfWidth + channel;
                        float first = data[(int)baseIndex], second = data[(int)(baseIndex + quarter)];
                        float cos = cosine[table], sin = sine[table];
                        result[(int)baseIndex] = first * cos - second * sin;
                        result[(int)(baseIndex + quarter)] = first * sin + second * cos;
                    }
        return result;
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