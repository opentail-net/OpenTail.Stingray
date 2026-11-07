using OpenTail.Stingray.Mlx;

namespace OpenTail.Stingray.Tests.Cuda;

/// <summary>
/// <see cref="MlxForwardPass"/>: the <c>IForwardPass</c> engine/planner integration point. Most of
/// its control flow (cache reset, rewind-length validation, position/startPos validation) is plain
/// C# with no native call in it, so it is checked for real; only the calls that reach
/// <see cref="MlxDenseModel.Prefill"/>/<see cref="MlxDenseModel.Decode"/> can only be checked as
/// "reaches the first native call and fails predictably" (no Mac here).
/// </summary>
public sealed class MlxForwardPassTests
{
    private static MlxDenseModel MakeModel(int hiddenDim = 4, int numHeads = 2, int headDim = 2, int intermDim = 8, int vocab = 3, int numLayers = 1)
    {
        var layers = new MlxDenseLayer[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            layers[i] = new MlxDenseLayer
            {
                AttnNormWeight = new float[hiddenDim],
                Wq = new float[hiddenDim * numHeads * headDim],
                Wk = new float[hiddenDim * numHeads * headDim],
                Wv = new float[hiddenDim * numHeads * headDim],
                Wo = new float[numHeads * headDim * hiddenDim],
                FfnNormWeight = new float[hiddenDim],
                WGate = new float[hiddenDim * intermDim],
                WUp = new float[hiddenDim * intermDim],
                WDown = new float[intermDim * hiddenDim],
                HiddenDim = hiddenDim,
                NumHeads = numHeads,
                NumKvHeads = numHeads,
                HeadDim = headDim,
                IntermDim = intermDim,
            };
        }
        return new MlxDenseModel
        {
            TokenEmbedding = new float[vocab * hiddenDim],
            Layers = layers,
            OutputNormWeight = new float[hiddenDim],
            OutputWeight = new float[hiddenDim * vocab],
            HiddenDim = hiddenDim,
            VocabSize = vocab,
        };
    }

    [Fact]
    public void VocabSizeAndMaxSeqLen_ReflectConstructorArguments()
    {
        using var pass = new MlxForwardPass(MakeModel(vocab: 7), maxSeqLen: 123);
        Assert.Equal(7, pass.VocabSize);
        Assert.Equal(123, pass.MaxSeqLen);
    }

    [Fact]
    public void Constructor_NonPositiveMaxSeqLen_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new MlxForwardPass(MakeModel(), maxSeqLen: 0));
    }

    [Fact]
    public void TruncateTo_Zero_IsAcceptedAsFullReset()
    {
        using var pass = new MlxForwardPass(MakeModel(), maxSeqLen: 8);
        pass.TruncateTo(0); // must not throw -- a reset from an already-empty cache is a no-op
    }

    [Fact]
    public void Prefill_EmptyTokensOnFreshInstance_Throws()
    {
        using var pass = new MlxForwardPass(MakeModel(), maxSeqLen: 8);

        Assert.Throws<ArgumentException>(() => pass.Prefill(Array.Empty<int>()));
    }

    [Fact]
    public void ResetCache_ClearsLastLogitsAndEmptyPrefillStillThrows()
    {
        using var pass = new MlxForwardPass(
            MakeModel(), 8,
            (_, _) => [1f, 2f, 3f],
            (_, _, _) => [4f, 5f, 6f]);

        Assert.Equal(new float[] { 1f, 2f, 3f }, pass.Prefill([0]).ToArray());
        pass.ResetCache();

        Assert.Throws<ArgumentException>(() => pass.Prefill(Array.Empty<int>()));
        Assert.Throws<NotSupportedException>(() => pass.Forward(0, position: 1));
        Assert.Equal(new float[] { 4f, 5f, 6f }, pass.Forward(0, position: 0).ToArray());
    }

    [Fact]
    public void Prefill_EmptyTokensAfterResetOfPopulatedSequence_Throws()
    {
        using var pass = new MlxForwardPass(
            MakeModel(), 8,
            (_, _) => [1f, 2f, 3f],
            (_, _, _) => [4f, 5f, 6f]);

        Assert.Equal(new float[] { 1f, 2f, 3f }, pass.Prefill([0]).ToArray());
        pass.ResetCache();

        Assert.Throws<ArgumentException>(() => pass.Prefill(Array.Empty<int>()));
        Assert.Equal(new float[] { 1f, 2f, 3f }, pass.Prefill([1]).ToArray());
    }

    [Fact]
    public void TruncateTo_NonZeroWhenEmpty_ThrowsNotSupported()
    {
        using var pass = new MlxForwardPass(MakeModel(), maxSeqLen: 8);
        Assert.Throws<NotSupportedException>(() => pass.TruncateTo(3));
    }

    [Fact]
    public void Prefill_NonZeroStartPosOnFreshCache_ThrowsNotSupported()
    {
        using var pass = new MlxForwardPass(MakeModel(), maxSeqLen: 8);
        Assert.Throws<NotSupportedException>(() => pass.Prefill([0, 1], startPos: 1));
    }

    [Fact]
    public void Forward_PositionNotEqualToCacheLength_ThrowsNotSupported()
    {
        using var pass = new MlxForwardPass(MakeModel(), maxSeqLen: 8);
        // Cache is fresh (length 0); asking for position 1 skips ahead without prefilling position 0.
        Assert.Throws<NotSupportedException>(() => pass.Forward(0, position: 1));
    }

    [Fact]
    public void Prefill_ValidShapes_WithNoNativeLibraryPresent_ReachesFirstNativeCall_FailsPredictably()
    {
        using var pass = new MlxForwardPass(MakeModel(), maxSeqLen: 8);
        Assert.Throws<DllNotFoundException>(() => pass.Prefill([0, 1, 2]));
    }

    [Fact]
    public void Forward_ValidShapes_WithNoNativeLibraryPresent_ReachesFirstNativeCall_FailsPredictably()
    {
        using var pass = new MlxForwardPass(MakeModel(), maxSeqLen: 8);
        Assert.Throws<DllNotFoundException>(() => pass.Forward(0, position: 0));
    }

    [Fact]
    public void Prefill_FailureAfterEarlierLayerUpdated_ResetsAllCachesAndCanRestart()
    {
        Assert.SkipUnless(MlxBackend.IsAvailable(), "Native MLX with a working Metal device is required; this test has not run numerically on this host.");
        var model = MakeModel(numLayers: 3);
        int calls = 0;
        model.LayerForwardForTesting = (layerIndex, hidden, seqLen, positions, cache) =>
        {
            calls++;
            if (layerIndex < 2)
                cache!.Append(new float[cache.NumHeads * seqLen * cache.HeadDim], new float[cache.NumHeads * seqLen * cache.HeadDim], seqLen);
            if (layerIndex == 1 && calls == 2)
                throw new InvalidOperationException("Injected layer failure.");
            return hidden.ToArray();
        };
        using var pass = new MlxForwardPass(model, maxSeqLen: 8);

        var error = Assert.Throws<InvalidOperationException>(() => pass.Prefill([0]));

        Assert.Equal("Injected layer failure.", error.Message);
        Assert.Equal(0, pass.LengthForTesting);
        Assert.All(pass.CachesForTesting, cache => Assert.Equal(0, cache.Length));

        model.LayerForwardForTesting = (layerIndex, hidden, seqLen, positions, cache) =>
        {
            cache!.Append(new float[cache.NumHeads * seqLen * cache.HeadDim], new float[cache.NumHeads * seqLen * cache.HeadDim], seqLen);
            return hidden.ToArray();
        };
        Assert.Equal(3, pass.Prefill([0]).Length);
        Assert.Equal(1, pass.LengthForTesting);
        Assert.All(pass.CachesForTesting, cache => Assert.Equal(1, cache.Length));
    }

    [Fact]
    public void Decode_FailureAfterEarlierLayerUpdated_ResetsAllCachesAndLogits()
    {
        var model = MakeModel(numLayers: 3);
        model.LayerForwardForTesting = (layerIndex, hidden, seqLen, positions, cache) =>
        {
            cache!.Append(new float[cache.NumHeads * seqLen * cache.HeadDim], new float[cache.NumHeads * seqLen * cache.HeadDim], seqLen);
            return hidden.ToArray();
        };
        model.LayerDecodeForTesting = (layerIndex, hidden, position, cache) =>
        {
            if (layerIndex == 0)
                cache.Append(new float[cache.NumHeads * cache.HeadDim], new float[cache.NumHeads * cache.HeadDim], 1);
            if (layerIndex == 1)
                throw new InvalidOperationException("Injected decode failure.");
            return hidden.ToArray();
        };
        using var pass = new MlxForwardPass(model, 8,
            (tokens, caches) =>
            {
                foreach (MlxKvCache cache in caches)
                    cache.Append(new float[cache.NumHeads * tokens.Length * cache.HeadDim], new float[cache.NumHeads * tokens.Length * cache.HeadDim], tokens.Length);
                return [1f, 2f, 3f];
            },
            model.Decode);
        Assert.Equal(3, pass.Prefill([0]).Length);

        var error = Assert.Throws<InvalidOperationException>(() => pass.Forward(0, position: 1));

        Assert.Equal("Injected decode failure.", error.Message);
        Assert.Equal(0, pass.LengthForTesting);
        Assert.Empty(pass.LastLogitsForTesting);
        Assert.All(pass.CachesForTesting, cache => Assert.Equal(0, cache.Length));
    }
}
