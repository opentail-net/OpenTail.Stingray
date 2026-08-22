using System;
using System.IO;
using OpenTail.Stingray.Audio.QwenTTS;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Smoke test running the full real 4-block DAC decoder chain end-to-end (pre-conv -&gt; 4
/// DecoderBlocks -&gt; final SnakeBeta -&gt; final conv -&gt; clamp) on real weights, asserting finite,
/// in-range output and the expected 4×5×3=... real cumulative upsample factor (rates 8*5*4*3=480).
/// Block 0's internal math is separately golden-verified in <see cref="QwenTtsCodecDacTests"/>;
/// this test only proves the full chain wires together and produces sane audio-range output.
/// </summary>
public sealed class QwenTtsCodecDacFullChainTests : HeavyTestBase
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

    [Fact]
    public void Forward_FullChain_RealWeights_ProducesFiniteClampedWaveform()
    {
        string? modelPath = FindModelPath("qwen-tokenizer-12hz-Q8_0.gguf");
        Assert.SkipUnless(modelPath != null, "models/qwen-tokenizer-12hz-Q8_0.gguf not found");

        using var model = GgufModel.Open(modelPath!);
        var weights = new QwenTtsCodecDacWeights(model);

        var rnd = new Random(7);
        int t = 4;
        var input = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[1024];
            for (int d = 0; d < 1024; d++) row[d] = (float)(rnd.NextDouble() * 0.6 - 0.3);
            input[i] = row;
        }

        var wav = QwenTtsCodecDac.Forward(weights, input);

        int expectedRate = 8 * 5 * 4 * 3;
        Assert.Equal(t * expectedRate, wav.Length);

        foreach (var s in wav)
        {
            Assert.True(float.IsFinite(s), "DAC output contains non-finite samples");
            Assert.InRange(s, -1f, 1f);
        }
    }
}
