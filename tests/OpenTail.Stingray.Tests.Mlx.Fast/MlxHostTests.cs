namespace OpenTail.Stingray.Tests.Mlx.Fast;

public sealed class MlxHostTests
{
    [Fact]
    public void AttentionTensorShapes_PreserveBatchHeadSequenceAndFeatureAxes()
    {
        Assert.Equal(new[] { 1, 2, 3, 4 }, MlxOps.AttentionTensorShape(2, 3, 4));
        Assert.Equal(new[] { 1, 1, 3, 5 }, MlxOps.AttentionMaskShape(3, 5));
    }

    [Fact]
    public void RopeTensorShape_IsBatchSequenceFeature()
    {
        Assert.Equal(new[] { 1, 3, 4 }, MlxOps.RopeTensorShape(3, 4));
    }

    [Fact]
    public void SplitHeadOrderingHelper_ExpectedHeadMajorLayout()
    {
        float[] flat = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        float[] expectedHeadMajor = [1, 2, 5, 6, 9, 10, 3, 4, 7, 8, 11, 12];

        Assert.Equal(expectedHeadMajor, MlxHostReference.SplitHeads(flat, seqLen: 3, numHeads: 2, headDim: 2));
        Assert.Equal(flat, MlxHostReference.MergeHeads(expectedHeadMajor, numHeads: 2, seqLen: 3, headDim: 2));
    }

    [Fact]
    public void KvCacheResetClearsDataAndLength()
    {
        var cache = new MlxKvCache(numHeads: 2, maxSeqLen: 2, headDim: 2);
        cache.Append([1, 2, 3, 4], [5, 6, 7, 8], 1);

        cache.Reset();

        Assert.Equal(0, cache.Length);
        Assert.Empty(cache.Keys());
        Assert.Empty(cache.Values());
    }

    [Fact]
    public void EmptyPrefillAndInjectedFailureRecoveryAreHostTestable()
    {
        var model = MakeModel(3);
        model.LayerForwardForTesting = (layerIndex, hidden, seqLen, positions, cache) =>
        {
            cache!.Append(new float[cache.NumHeads * seqLen * cache.HeadDim], new float[cache.NumHeads * seqLen * cache.HeadDim], seqLen);
            if (layerIndex == 1) throw new InvalidOperationException("injected");
            return hidden.ToArray();
        };
        using var pass = new MlxForwardPass(model, 8,
            (tokens, caches) => model.Prefill(tokens.AsSpan(), caches),
            (token, position, caches) => model.Decode(token, position, caches));

        Assert.Throws<ArgumentException>(() => pass.Prefill(Array.Empty<int>()));
        var ex = Assert.Throws<InvalidOperationException>(() => pass.Prefill([0]));

        Assert.Equal("injected", ex.Message);
        Assert.Equal(0, pass.LengthForTesting);
        Assert.Empty(pass.LastLogitsForTesting);
        Assert.All(pass.CachesForTesting, cache => Assert.Equal(0, cache.Length));
    }

    [Fact]
    public void MlxArchitectureAndFeatureRefusalsAreHostTestable()
    {
        string? originalFlag = Environment.GetEnvironmentVariable("STINGRAY_MLX_DENSE");
        try
        {
            Environment.SetEnvironmentVariable("STINGRAY_MLX_DENSE", "1");
            var unsupportedArchitecture = ForwardPassSelection.Select(new ForwardPassRequest
            {
                Backend = ForwardPassBackend.Mlx,
                Architecture = "not-llama",
            });
            Assert.True(unsupportedArchitecture.IsRefused);

            var unsupportedOption = ForwardPassSelection.Select(new ForwardPassRequest
            {
                Backend = ForwardPassBackend.Mlx,
                Architecture = "llama",
                SessionsRequested = true,
            });
            Assert.True(unsupportedOption.IsRefused);
            Assert.Contains("Sessions", unsupportedOption.Refusal);
        }
        finally
        {
            Environment.SetEnvironmentVariable("STINGRAY_MLX_DENSE", originalFlag);
        }
    }

    private static MlxDenseModel MakeModel(int layers)
    {
        var modelLayers = new MlxDenseLayer[layers];
        for (int i = 0; i < layers; i++)
        {
            modelLayers[i] = new MlxDenseLayer
            {
                AttnNormWeight = new float[2], Wq = new float[4], Wk = new float[4], Wv = new float[4], Wo = new float[4],
                FfnNormWeight = new float[2], WGate = new float[4], WUp = new float[4], WDown = new float[4],
                HiddenDim = 2, NumHeads = 1, NumKvHeads = 1, HeadDim = 2, IntermDim = 2,
            };
        }
        return new MlxDenseModel
        {
            TokenEmbedding = [1, 2, 3, 4], Layers = modelLayers,
            OutputNormWeight = [1, 1], OutputWeight = [1, 0, 0, 1], HiddenDim = 2, VocabSize = 2,
        };
    }
}

internal static class MlxHostReference
{
    internal static float[] SplitHeads(ReadOnlySpan<float> flat, int seqLen, int numHeads, int headDim)
    {
        var split = new float[flat.Length];
        for (int head = 0; head < numHeads; head++)
            for (int position = 0; position < seqLen; position++)
                flat.Slice(position * numHeads * headDim + head * headDim, headDim)
                    .CopyTo(split.AsSpan(head * seqLen * headDim + position * headDim, headDim));
        return split;
    }

    internal static float[] MergeHeads(ReadOnlySpan<float> headFirst, int numHeads, int seqLen, int headDim)
    {
        var flat = new float[headFirst.Length];
        for (int head = 0; head < numHeads; head++)
            for (int position = 0; position < seqLen; position++)
                headFirst.Slice(head * seqLen * headDim + position * headDim, headDim)
                    .CopyTo(flat.AsSpan(position * numHeads * headDim + head * headDim, headDim));
        return flat;
    }
}
