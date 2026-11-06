using OpenTail.Stingray.Mlx;

namespace OpenTail.Stingray.Tests.Cuda;

/// <summary>
/// <see cref="MlxDenseModel"/>'s embedding lookup is plain host-side array indexing (no mlx-c call),
/// so the token-range check is checked for real, like <see cref="MlxKvCache"/>'s. <see cref="MlxDenseModel.Prefill"/>/
/// <see cref="MlxDenseModel.Decode"/> themselves need a real layer forward pass, so they can only be
/// checked for "reaches the first native call and fails predictably" (no Mac here).
/// </summary>
public sealed class MlxDenseModelTests
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
    public void Prefill_TokenOutOfRange_Throws()
    {
        var model = MakeModel(vocab: 3);
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Prefill([0, 1, 3]));
    }

    [Fact]
    public void Decode_WrongCacheCount_Throws()
    {
        var model = MakeModel(numLayers: 2);
        var caches = new[] { new MlxKvCache(numHeads: 2, maxSeqLen: 4, headDim: 2) }; // only 1, model has 2 layers
        Assert.Throws<ArgumentException>(() => model.Decode(0, position: 0, caches));
    }

    [Fact]
    public void Prefill_ValidShapes_WithNoNativeLibraryPresent_ReachesFirstNativeCall_FailsPredictably()
    {
        var model = MakeModel();
        Assert.Throws<DllNotFoundException>(() => model.Prefill([0, 1, 2]));
    }

    [Fact]
    public void Decode_ValidShapes_WithNoNativeLibraryPresent_ReachesFirstNativeCall_FailsPredictably()
    {
        var model = MakeModel(numLayers: 1);
        var caches = new[] { new MlxKvCache(numHeads: 2, maxSeqLen: 4, headDim: 2) };
        Assert.Throws<DllNotFoundException>(() => model.Decode(0, position: 0, caches));
    }
}
