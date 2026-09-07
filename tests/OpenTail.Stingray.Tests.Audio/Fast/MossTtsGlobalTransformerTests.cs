using OpenTail.Stingray.Audio.MossTts;

namespace OpenTail.Stingray.Tests.Audio.Fast;

/// <summary>
/// Structural (synthetic-weight) tests for <see cref="MossTtsGlobalTransformer"/> -- confirms
/// shape/finiteness invariants without needing the real checkpoint. See
/// <see cref="MossTtsGlobalTransformerRealWeightsTests"/> for the real-weight smoke test.
/// </summary>
public sealed class MossTtsGlobalTransformerTests
{
    private static MossTtsGlobalTransformerWeights MakeSyntheticWeights(int seed)
    {
        var rng = new Random(seed);
        float[] Rand(int n)
        {
            var a = new float[n];
            for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 0.04 - 0.02);
            return a;
        }

        const int dim = MossTtsGlobalTransformerWeights.HiddenDim;
        const int inter = MossTtsGlobalTransformerWeights.IntermediateDim;

        var layers = new MossTtsGlobalTransformerLayerWeights[MossTtsGlobalTransformerWeights.NumLayers];
        for (int l = 0; l < layers.Length; l++)
        {
            layers[l] = new MossTtsGlobalTransformerLayerWeights
            {
                Ln1Weight = Enumerable.Repeat(1f, dim).ToArray(),
                Ln1Bias = new float[dim],
                CAttnWeight = Rand(dim * 3 * dim),
                CAttnBias = Rand(dim * 3),
                CProjWeight = Rand(dim * dim),
                CProjBias = Rand(dim),
                Ln2Weight = Enumerable.Repeat(1f, dim).ToArray(),
                Ln2Bias = new float[dim],
                FcInWeight = Rand(inter * dim),
                FcInBias = Rand(inter),
                FcOutWeight = Rand(dim * inter),
                FcOutBias = Rand(dim),
            };
        }

        // Use reflection-free manual construction via a small helper record path: since
        // MossTtsGlobalTransformerWeights only has the RvcPackedTensorSource constructor, build one
        // via a fake in-memory tensor source instead of duplicating a second constructor.
        return BuildViaFakeSource(rng);

        MossTtsGlobalTransformerWeights BuildViaFakeSource(Random r)
        {
            var tensors = new Dictionary<string, float[]>();
            tensors["model_weights/transformer.wte.weight"] = Rand(MossTtsGlobalTransformerWeights.VocabSize * dim);
            for (int q = 0; q < MossTtsGlobalTransformerWeights.NumCodebooks; q++)
                tensors[$"model_weights/audio_embeddings.{q}.weight"] = Rand(MossTtsGlobalTransformerWeights.AudioCodebookSize * dim);
            for (int l = 0; l < MossTtsGlobalTransformerWeights.NumLayers; l++)
            {
                string p = $"model_weights/transformer.h.{l}";
                tensors[$"{p}.ln_1.weight"] = layers[l].Ln1Weight;
                tensors[$"{p}.ln_1.bias"] = layers[l].Ln1Bias;
                tensors[$"{p}.attn.c_attn.weight"] = layers[l].CAttnWeight;
                tensors[$"{p}.attn.c_attn.bias"] = layers[l].CAttnBias;
                tensors[$"{p}.attn.c_proj.weight"] = layers[l].CProjWeight;
                tensors[$"{p}.attn.c_proj.bias"] = layers[l].CProjBias;
                tensors[$"{p}.ln_2.weight"] = layers[l].Ln2Weight;
                tensors[$"{p}.ln_2.bias"] = layers[l].Ln2Bias;
                tensors[$"{p}.mlp.fc_in.weight"] = layers[l].FcInWeight;
                tensors[$"{p}.mlp.fc_in.bias"] = layers[l].FcInBias;
                tensors[$"{p}.mlp.fc_out.weight"] = layers[l].FcOutWeight;
                tensors[$"{p}.mlp.fc_out.bias"] = layers[l].FcOutBias;
            }
            tensors["model_weights/transformer.ln_f.weight"] = Enumerable.Repeat(1f, dim).ToArray();
            tensors["model_weights/transformer.ln_f.bias"] = new float[dim];
            tensors["model_weights/text_lm_head.weight"] = Rand(MossTtsGlobalTransformerWeights.VocabSize * dim);
            return new MossTtsGlobalTransformerWeights(name => tensors[name]);
        }
    }

    [Fact]
    public void ForwardAll_ProducesFiniteOutput_ForMixedTextAndAudioRows()
    {
        var w = MakeSyntheticWeights(seed: 1);
        var pad = MossTtsGlobalTransformerWeights.AudioPadTokenId;
        var noAudio = Enumerable.Repeat(pad, MossTtsGlobalTransformerWeights.NumCodebooks).ToArray();
        var withAudio = Enumerable.Range(0, MossTtsGlobalTransformerWeights.NumCodebooks).Select(i => i % 17).ToArray();

        var rows = new List<MossTtsGlobalRow>
        {
            new(MossTtsGlobalTransformerWeights.ImStartTokenId, (int[])noAudio.Clone()),
            new(100, (int[])noAudio.Clone()),
            new(101, (int[])noAudio.Clone()),
            new(MossTtsGlobalTransformerWeights.AudioStartTokenId, (int[])withAudio.Clone()),
            new(MossTtsGlobalTransformerWeights.PadTokenId, (int[])withAudio.Clone()),
        };

        var all = MossTtsGlobalTransformer.ForwardAll(w, rows);
        Assert.Equal(rows.Count, all.Length);
        foreach (var h in all)
        {
            Assert.Equal(MossTtsGlobalTransformerWeights.HiddenDim, h.Length);
            Assert.All(h, v => Assert.True(float.IsFinite(v)));
        }

        var last = MossTtsGlobalTransformer.ForwardLastHidden(w, rows);
        Assert.Equal(all[^1], last);

        var logits = MossTtsGlobalTransformer.TextLogits(w, last);
        Assert.Equal(MossTtsGlobalTransformerWeights.VocabSize, logits.Length);
        Assert.All(logits, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void ForwardAll_IsCausal_LaterRowsDoNotAffectEarlierHiddenStates()
    {
        var w = MakeSyntheticWeights(seed: 2);
        var pad = MossTtsGlobalTransformerWeights.AudioPadTokenId;
        var noAudio = Enumerable.Repeat(pad, MossTtsGlobalTransformerWeights.NumCodebooks).ToArray();

        var shortRows = new List<MossTtsGlobalRow>
        {
            new(10, (int[])noAudio.Clone()),
            new(11, (int[])noAudio.Clone()),
            new(12, (int[])noAudio.Clone()),
        };
        var longerRows = new List<MossTtsGlobalRow>(shortRows) { new(13, (int[])noAudio.Clone()) };

        var shortOut = MossTtsGlobalTransformer.ForwardAll(w, shortRows);
        var longerOut = MossTtsGlobalTransformer.ForwardAll(w, longerRows);

        for (int i = 0; i < shortRows.Count; i++)
        {
            for (int d = 0; d < MossTtsGlobalTransformerWeights.HiddenDim; d++)
                Assert.Equal(shortOut[i][d], longerOut[i][d], precision: 4);
        }
    }
}
