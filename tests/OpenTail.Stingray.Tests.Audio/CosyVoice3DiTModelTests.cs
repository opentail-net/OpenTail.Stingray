using System.IO;
using OpenTail.Stingray.Audio.CosyVoice;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weights sanity coverage for CosyVoice3's DiT backbone port (see docs/audio-review-
/// progress.md's CosyVoice3 section). Deliberately does NOT test a full pipeline forward pass
/// -- the input_embed concatenation formula isn't confirmed yet (see
/// <see cref="CosyVoice3DiTModel"/>'s doc comment), so these tests exercise the confirmed
/// pieces directly: real weight loading, the ConvPositionEmbedding+proj mechanics given an
/// already-formed input, and the 22-layer transformer backbone + final projection given an
/// already-embedded hidden state.
/// </summary>
public sealed class CosyVoice3DiTModelTests : HeavyTestBase
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

    [Fact]
    public void Weights_LoadRealTensors_ExpectedRealShapes()
    {
        string? path = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        Assert.SkipUnless(path != null, "models/cosyvoice3/CosyVoice3-2512_F16.gguf not found");

        using var model = GgufModel.Open(path!);
        var w = new CosyVoice3DiTWeights(model);

        Assert.Equal(22, w.NumLayers);
        Assert.Equal(22, w.Blocks.Length);
        Assert.Equal(1024 * 320, w.InputProjWeight.Length);
        Assert.Equal(80 * 1024, w.ProjOutWeight.Length);

        foreach (var v in w.Blocks[0].ToQWeight) Assert.False(float.IsNaN(v) || float.IsInfinity(v));
        foreach (var v in w.Blocks[21].FfOutWeight) Assert.False(float.IsNaN(v) || float.IsInfinity(v));
    }

    [Fact]
    public void RunBackbone_RealWeights_ProducesFiniteMelDimOutput()
    {
        string? path = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        Assert.SkipUnless(path != null, "models/cosyvoice3/CosyVoice3-2512_F16.gguf not found");

        using var model = GgufModel.Open(path!);
        var w = new CosyVoice3DiTWeights(model);

        int numFrames = 8;
        var h = new float[numFrames * CosyVoice3DiTWeights.HiddenDim];
        for (int i = 0; i < h.Length; i++) h[i] = 0.01f * MathF.Sin(i * 0.05f);

        var velocity = CosyVoice3DiTModel.RunBackbone(w, h, timestep: 0.5f, numFrames);

        Assert.Equal(numFrames * CosyVoice3DiTWeights.MelDim, velocity.Length);
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        foreach (var v in velocity)
        {
            Assert.False(float.IsNaN(v) || float.IsInfinity(v));
            if (v < min) min = v;
            if (v > max) max = v;
        }
        Assert.True(max - min > 1e-6f, $"backbone output looks degenerate: min={min}, max={max}");
    }

    [Fact]
    public void InputEmbed_RealWeights_ProducesFiniteHiddenDimOutput()
    {
        string? path = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        Assert.SkipUnless(path != null, "models/cosyvoice3/CosyVoice3-2512_F16.gguf not found");

        using var model = GgufModel.Open(path!);
        var w = new CosyVoice3DiTWeights(model);

        int numFrames = 8;
        int concatDim = 320; // real input_embed.proj input width, confirmed from the real tensor shape
        var concat = new float[numFrames * concatDim];
        for (int i = 0; i < concat.Length; i++) concat[i] = 0.02f * MathF.Sin(i * 0.03f);

        var h = CosyVoice3DiTModel.InputEmbed(w, concat, concatDim, numFrames);

        Assert.Equal(numFrames * CosyVoice3DiTWeights.HiddenDim, h.Length);
        foreach (var v in h) Assert.False(float.IsNaN(v) || float.IsInfinity(v));
    }
}
