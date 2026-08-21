using System;
using System.IO;
using OpenTail.Stingray.Audio.Kokoro;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

public sealed class KokoroWeightsTests
{
    private static string? FindModelPath(string fileName)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    /// <summary>
    /// Loads every real named tensor in the Kokoro-82M GGUF graph (bert.*, text_enc.*, pred.*,
    /// dec.*) and asserts the shapes match the architecture's own KV metadata. Failing to find
    /// or mis-sizing any tensor here means the C# port's tensor-name map has drifted from the
    /// real model -- see docs/audio-review-progress.md "REBUILD PLAN".
    /// </summary>
    [Fact]
    public void KokoroWeights_RealGguf_LoadsEveryTensorWithExpectedShape()
    {
        string? modelPath = FindModelPath("kokoro-82m-q8_0.gguf");
        if (modelPath is null) return;

        using var w = new KokoroWeights(modelPath);

        Assert.Equal(512, w.HiddenDim);
        Assert.Equal(128, w.StyleDim);
        Assert.Equal(50, w.MaxDur);
        Assert.Equal(178, w.NTokenVocab);
        Assert.Equal(3, w.NStyleLayers);
        Assert.Equal(768, w.BertHiddenSize);
        Assert.Equal(12, w.BertNumHiddenLayers);
        Assert.Equal([10, 6], w.UpsampleRates);
        Assert.Equal([3, 7, 11], w.ResblockKernelSizes);

        Assert.Equal(768 * 768, w.Bert.AttnQWeight.Length);
        Assert.Equal(512 * 178, w.TextEncoder.EmbeddingWeight.Length);
        Assert.Equal(3, w.TextEncoder.ConvWeight.Length);

        Assert.Equal(3, w.Predictor.F0.Length);
        Assert.Equal(3, w.Predictor.N.Length);
        Assert.Equal(3, w.Predictor.DurEncLstm.Length);

        Assert.Equal(DecoderWeights.DecodeStackSize, w.Decoder.Decode.Length);
        Assert.NotNull(w.Decoder.Decode[3].PoolWeight); // only the upsampling (last) decode block has pool.*
        Assert.Null(w.Decoder.Decode[0].PoolWeight);

        Assert.Equal(2, w.Decoder.Generator.UpWeight.Length);
        Assert.Equal(6, w.Decoder.Generator.ResBlocks.Length); // 2 upsample stages * 3 resblock kernel sizes
        Assert.Equal(9, w.Decoder.Generator.MSourceWeight.Length); // harmonic_num=8 -> 9 input channels to nn.Linear(9,1)

        foreach (var rb in w.Decoder.Generator.ResBlocks)
        {
            for (int i = 0; i < 3; i++)
            {
                Assert.All(rb.Alpha1[i], v => Assert.True(float.IsFinite(v)));
                Assert.All(rb.Alpha2[i], v => Assert.True(float.IsFinite(v)));
            }
        }
    }
}
