using System;
using System.IO;
using OpenTail.Stingray.Audio.Parler;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// End-to-end wiring smoke test for <see cref="ParlerFullPipeline"/>'s mixed-source constructor:
/// T5 encoder + DAC codec from the real Safetensors checkpoint, decoder from the real community
/// `ecyht2/parler-tts-mini-v1-GGUF` conversion. The decoder's numerical fidelity against the
/// Safetensors path is already separately golden-verified (`ParlerDecoderGgufTests`) -- this test
/// only verifies the PLUMBING that chains a GGUF-sourced decoder into the same real generation
/// loop (delay pattern, KV cache, EOS logits processor, DAC decode) as the all-Safetensors path,
/// matching this project's established end-to-end smoke-test pattern.
/// </summary>
public sealed class ParlerFullPipelineGgufTests : HeavyTestBase
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
    public void Synthesize_MixedGgufSource_ProducesFinitePcm()
    {
        string? stPath = FindModelPath("parler-tts-mini-v1.safetensors");
        string? ggufPath = FindModelPath("parler-tts-mini-v1-Q8_0.gguf");
        string? tokenizerPath = FindRepoFile("scratch-llamacpp-ref/parler-tokenizer/tokenizer.json");
        Assert.SkipUnless(stPath != null && ggufPath != null && tokenizerPath != null,
            "Parler Safetensors/GGUF model files or the real tokenizer.json not found");

        using var loader = SafetensorsLoader.Open(stPath!);
        using var decoderGguf = GgufModel.Open(ggufPath!);
        using var pipeline = new ParlerFullPipeline(tokenizerPath!, loader, decoderGguf);

        var pcm = pipeline.Synthesize("Hello there.", maxNewTokens: 40, minNewTokens: 10);

        Assert.NotEmpty(pcm);
        foreach (var s in pcm)
            Assert.True(float.IsFinite(s), "PCM sample was not finite");

        double sumSq = 0;
        foreach (var s in pcm) sumSq += s * s;
        double rms = Math.Sqrt(sumSq / pcm.Length);
        Assert.True(rms > 1e-6, $"PCM output appears silent (rms={rms})");
    }
}
