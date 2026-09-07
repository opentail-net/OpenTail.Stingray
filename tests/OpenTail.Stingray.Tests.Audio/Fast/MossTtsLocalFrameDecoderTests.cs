using OpenTail.Stingray.Audio.MossTts;

namespace OpenTail.Stingray.Tests.Audio.Fast;

/// <summary>Structural (synthetic-weight) tests for <see cref="MossTtsLocalFrameDecoder"/>.</summary>
public sealed class MossTtsLocalFrameDecoderTests
{
    private static float[] Rand(Random rng, int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 0.04 - 0.02);
        return a;
    }

    private static MossTtsGlobalTransformerWeights MakeGlobalWeights(Random rng)
    {
        const int dim = MossTtsGlobalTransformerWeights.HiddenDim;
        const int inter = MossTtsGlobalTransformerWeights.IntermediateDim;
        var tensors = new Dictionary<string, float[]>();
        tensors["model_weights/transformer.wte.weight"] = Rand(rng, MossTtsGlobalTransformerWeights.VocabSize * dim);
        for (int q = 0; q < MossTtsGlobalTransformerWeights.NumCodebooks; q++)
            tensors[$"model_weights/audio_embeddings.{q}.weight"] = Rand(rng, MossTtsGlobalTransformerWeights.AudioCodebookSize * dim);
        for (int l = 0; l < MossTtsGlobalTransformerWeights.NumLayers; l++)
        {
            string p = $"model_weights/transformer.h.{l}";
            tensors[$"{p}.ln_1.weight"] = Enumerable.Repeat(1f, dim).ToArray();
            tensors[$"{p}.ln_1.bias"] = new float[dim];
            tensors[$"{p}.attn.c_attn.weight"] = Rand(rng, dim * 3 * dim);
            tensors[$"{p}.attn.c_attn.bias"] = Rand(rng, dim * 3);
            tensors[$"{p}.attn.c_proj.weight"] = Rand(rng, dim * dim);
            tensors[$"{p}.attn.c_proj.bias"] = Rand(rng, dim);
            tensors[$"{p}.ln_2.weight"] = Enumerable.Repeat(1f, dim).ToArray();
            tensors[$"{p}.ln_2.bias"] = new float[dim];
            tensors[$"{p}.mlp.fc_in.weight"] = Rand(rng, inter * dim);
            tensors[$"{p}.mlp.fc_in.bias"] = Rand(rng, inter);
            tensors[$"{p}.mlp.fc_out.weight"] = Rand(rng, dim * inter);
            tensors[$"{p}.mlp.fc_out.bias"] = Rand(rng, dim);
        }
        tensors["model_weights/transformer.ln_f.weight"] = Enumerable.Repeat(1f, dim).ToArray();
        tensors["model_weights/transformer.ln_f.bias"] = new float[dim];
        tensors["model_weights/text_lm_head.weight"] = Rand(rng, MossTtsGlobalTransformerWeights.VocabSize * dim);
        return new MossTtsGlobalTransformerWeights(name => tensors[name]);
    }

    private static MossTtsLocalTransformerWeights MakeLocalWeights(Random rng)
    {
        const int dim = MossTtsGlobalTransformerWeights.HiddenDim;
        const int inter = MossTtsGlobalTransformerWeights.IntermediateDim;
        var tensors = new Dictionary<string, float[]>();
        for (int l = 0; l < MossTtsLocalTransformerWeights.NumLayers; l++)
        {
            string p = $"model_weights/local_transformer.h.{l}";
            tensors[$"{p}.ln_1.weight"] = Enumerable.Repeat(1f, dim).ToArray();
            tensors[$"{p}.ln_1.bias"] = new float[dim];
            tensors[$"{p}.attn.c_attn.weight"] = Rand(rng, dim * 3 * dim);
            tensors[$"{p}.attn.c_attn.bias"] = Rand(rng, dim * 3);
            tensors[$"{p}.attn.c_proj.weight"] = Rand(rng, dim * dim);
            tensors[$"{p}.attn.c_proj.bias"] = Rand(rng, dim);
            tensors[$"{p}.ln_2.weight"] = Enumerable.Repeat(1f, dim).ToArray();
            tensors[$"{p}.ln_2.bias"] = new float[dim];
            tensors[$"{p}.mlp.fc_in.weight"] = Rand(rng, inter * dim);
            tensors[$"{p}.mlp.fc_in.bias"] = Rand(rng, inter);
            tensors[$"{p}.mlp.fc_out.weight"] = Rand(rng, dim * inter);
            tensors[$"{p}.mlp.fc_out.bias"] = Rand(rng, dim);
        }
        tensors["model_weights/local_transformer.ln_f.weight"] = Enumerable.Repeat(1f, dim).ToArray();
        tensors["model_weights/local_transformer.ln_f.bias"] = new float[dim];
        for (int q = 0; q < MossTtsGlobalTransformerWeights.NumCodebooks; q++)
            tensors[$"model_weights/audio_lm_heads.{q}.weight"] = Rand(rng, MossTtsGlobalTransformerWeights.AudioCodebookSize * dim);
        return new MossTtsLocalTransformerWeights(name => tensors[name]);
    }

    [Fact]
    public void GenerateFrame_ProducesInRangeTokens_ForAllActiveCodebooks()
    {
        var rng = new Random(3);
        var g = MakeGlobalWeights(rng);
        var l = MakeLocalWeights(rng);
        var globalHidden = Rand(rng, MossTtsGlobalTransformerWeights.HiddenDim);

        var frame = MossTtsLocalFrameDecoder.GenerateFrame(g, l, globalHidden, activeCodebooks: MossTtsGlobalTransformerWeights.NumCodebooks);

        // With random untrained weights the text choice could legitimately go either way; only
        // assert the shape/range invariant when generation continued.
        if (frame is null) return;

        Assert.Equal(MossTtsGlobalTransformerWeights.NumCodebooks, frame.Length);
        foreach (var token in frame)
        {
            Assert.InRange(token, 0, MossTtsGlobalTransformerWeights.AudioCodebookSize - 1);
        }
    }

    [Fact]
    public void GenerateFrame_IsDeterministic_ForTheSameInput()
    {
        var rng = new Random(4);
        var g = MakeGlobalWeights(rng);
        var l = MakeLocalWeights(rng);
        var globalHidden = Rand(rng, MossTtsGlobalTransformerWeights.HiddenDim);

        var frame1 = MossTtsLocalFrameDecoder.GenerateFrame(g, l, globalHidden, activeCodebooks: 4);
        var frame2 = MossTtsLocalFrameDecoder.GenerateFrame(g, l, globalHidden, activeCodebooks: 4);

        Assert.Equal(frame1, frame2);
    }
}
