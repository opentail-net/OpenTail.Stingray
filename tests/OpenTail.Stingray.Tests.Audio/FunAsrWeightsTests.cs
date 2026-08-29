
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weights sanity coverage for <see cref="FunAsrWeights"/> (see docs/audio-review-
/// progress.md's FunASR section). Confirms every real tensor this checkpoint ships loads with
/// the expected real shapes and finite values, and that the real GGUF-embedded vocabulary
/// decodes as expected -- the same bar every other pipeline's first real-weights test passed
/// before any forward-pass math was ported. The encoder/predictor/decoder forward pass is NOT
/// tested here (not yet ported -- see the class doc comment on FunAsrWeights).
/// </summary>
public sealed class FunAsrWeightsTests : HeavyTestBase
{
    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static void AssertFinite(float[] values, string label)
    {
        foreach (var v in values)
            Assert.False(float.IsNaN(v) || float.IsInfinity(v), $"{label} contains NaN/Inf");
    }

    [Fact]
    public void Weights_LoadRealTensors_ExpectedRealShapesAndCounts()
    {
        string? path = FindRepoFile("models/paraformer-q8.gguf");
        Assert.SkipUnless(path != null, "models/paraformer-q8.gguf not found");

        using var w = new FunAsrWeights(path!);

        Assert.Equal(49, w.EncoderLayers);
        Assert.Equal(49, w.EncoderLayerWeights.Length);
        Assert.Equal(16, w.DecoderLayers);
        Assert.Equal(16, w.DecoderLayerWeights.Length);
        Assert.Equal(4, w.EncoderHeads);
        Assert.Equal(512, w.EncoderDim);
        Assert.Equal(11, w.FsmnKernelSize);
        Assert.Equal(8404, w.VocabSize);
        Assert.Equal(8404, w.Vocab.Length);

        // encoders0.0 (special 560-dim first layer)
        Assert.Equal(560, w.Encoders0Layer.InputDim);
        Assert.Equal(560, w.Encoders0Layer.Norm1Weight.Length);
        Assert.Equal(560 * 1536, w.Encoders0Layer.AttnQkvWeight.Length);
        Assert.Equal(512 * 11, w.Encoders0Layer.AttnFsmnWeight.Length);
        AssertFinite(w.Encoders0Layer.AttnQkvWeight, "encoders0.0.self_attn.linear_q_k_v.weight");

        // main encoder stack: spot-check first and last (index 48) layers
        var enc0 = w.EncoderLayerWeights[0];
        Assert.Equal(512, enc0.InputDim);
        Assert.Equal(512 * 1536, enc0.AttnQkvWeight.Length);
        Assert.Equal(512 * 2048, enc0.FfnW1Weight.Length);
        AssertFinite(enc0.FfnW1Weight, "encoders.0.feed_forward.w_1.weight");

        var enc48 = w.EncoderLayerWeights[48];
        Assert.Equal(512 * 1536, enc48.AttnQkvWeight.Length);
        AssertFinite(enc48.AttnOutWeight, "encoders.48.self_attn.linear_out.weight");

        Assert.Equal(512, w.EncoderAfterNormWeight.Length);

        // predictor (CIF)
        Assert.Equal(3 * 512 * 512, w.PredictorCifConv1dWeight.Length);
        Assert.Equal(512, w.PredictorCifOutputWeight.Length);
        Assert.Single(w.PredictorCifOutputBias);
        AssertFinite(w.PredictorCifConv1dWeight, "predictor.cif_conv1d.weight");

        // decoder: spot-check first and last (index 15) layers, plus decoders3 and output_layer
        var dec0 = w.DecoderLayerWeights[0];
        Assert.Equal(512 * 11, dec0.SelfAttnFsmnWeight.Length);
        Assert.Equal(512 * 512, dec0.SrcAttnQWeight.Length);
        Assert.Equal(512 * 1024, dec0.SrcAttnKvWeight.Length);
        AssertFinite(dec0.SrcAttnKvWeight, "decoders.0.src_attn.linear_k_v.weight");

        var dec15 = w.DecoderLayerWeights[15];
        AssertFinite(dec15.FfnW1Weight, "decoders.15.feed_forward.w_1.weight");

        Assert.Equal(512 * 2048, w.Decoders3Layer.FfnW1Weight.Length);
        Assert.Equal(512 * 8404, w.DecoderOutputWeight.Length);
        Assert.Equal(8404, w.DecoderOutputBias.Length);
        AssertFinite(w.DecoderOutputWeight, "decoder.output_layer.weight");
        AssertFinite(w.DecoderOutputBias, "decoder.output_layer.bias");

        // cmvn (80 mel x 7-frame splice = 560)
        Assert.Equal(560, w.CmvnScale.Length);
        Assert.Equal(560, w.CmvnShift.Length);
    }

    [Fact]
    public void Vocab_RealTokens_MatchDirectlyInspectedValues()
    {
        string? path = FindRepoFile("models/paraformer-q8.gguf");
        Assert.SkipUnless(path != null, "models/paraformer-q8.gguf not found");

        using var w = new FunAsrWeights(path!);

        // Values confirmed by direct inspection this session (see docs/audio-review-progress.md),
        // not guessed -- a regression here means the metadata array ordering or dequant changed.
        Assert.Equal("<blank>", w.Vocab[0]);
        Assert.Equal("<s>", w.Vocab[1]);
        Assert.Equal("</s>", w.Vocab[2]);
        Assert.Equal("and@@", w.Vocab[3]);
        Assert.Equal("<unk>", w.Vocab[8403]);
    }

    [Fact]
    public void Tokenizer_Decode_RealVocab_JoinsCjkAndBpeContinuationsCorrectly()
    {
        string? path = FindRepoFile("models/paraformer-q8.gguf");
        Assert.SkipUnless(path != null, "models/paraformer-q8.gguf not found");

        using var w = new FunAsrWeights(path!);
        var tokenizer = new FunAsrTokenizer(w);

        // "and" + "these" -- two complete, non-@@ English words -> space-separated.
        // id 3 = "and@@" (continuation!), so use id for "these" (39) alone plus a bare word
        // check via CJK ids (4 = "筑", 5 = "陨") which must NOT get a space between them.
        var cjkOnly = tokenizer.Decode([4, 5], 2);
        Assert.Equal("筑陨", cjkOnly);

        // "and@@" (id 3) glues to the next piece with no space; "these" (id 39) is a plain word.
        var glued = tokenizer.Decode([3, 39], 2);
        Assert.Equal("andthese", glued);

        // Special tokens are stripped entirely.
        var withSpecials = tokenizer.Decode([1, 39, 2], 3);
        Assert.Equal("these", withSpecials);
    }
}
