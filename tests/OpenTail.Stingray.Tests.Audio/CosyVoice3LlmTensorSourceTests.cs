using System.IO;
using OpenTail.Stingray.Audio.CosyVoice;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Verifies CosyVoice3's LLM backbone (`models/cosyvoice3/CosyVoice3-2512_F16.gguf`, the
/// official pre-converted single-file GGUF) runs through this engine's real `ForwardPass` via
/// <see cref="CosyVoice3LlmTensorSource"/> -- the same architectural pattern now validated a
/// third time (QwenASR, CosyVoice2, CosyVoice3). See docs/audio-review-progress.md's
/// CosyVoice3 section.
/// </summary>
public sealed class CosyVoice3LlmTensorSourceTests : HeavyTestBase
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
    public void Adapter_MapsRealTensors_ArchitectureAndSpeechVocabConfirmed()
    {
        string? path = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        Assert.SkipUnless(path != null, "models/cosyvoice3/CosyVoice3-2512_F16.gguf not found");

        using var inner = GgufModel.Open(path!);
        using var source = new CosyVoice3LlmTensorSource(inner);

        Assert.Equal("qwen2", source.Metadata["general.architecture"]);
        Assert.Equal(24, System.Convert.ToInt32(source.Metadata["qwen2.block_count"]));
        Assert.Equal(896, System.Convert.ToInt32(source.Metadata["qwen2.embedding_length"]));
        Assert.Equal(6761, source.SpeechVocabSize); // real, checkpoint-specific -- different from CosyVoice2's 6564

        Assert.NotNull(source.FindTensor("token_embd.weight"));
        Assert.NotNull(source.FindTensor("output_norm.weight"));
        Assert.NotNull(source.FindTensor("blk.0.attn_q.weight"));
        Assert.NotNull(source.FindTensor("blk.23.ffn_down.weight"));
    }

    [Fact]
    public void SpeechGenerationMode_RunsRealForwardPass_ProducesFiniteSpeechVocabLogits()
    {
        string? path = FindRepoFile("models/cosyvoice3/CosyVoice3-2512_F16.gguf");
        Assert.SkipUnless(path != null, "models/cosyvoice3/CosyVoice3-2512_F16.gguf not found");

        using var inner = GgufModel.Open(path!);
        using var source = new CosyVoice3LlmTensorSource(inner);
        source.EnableSpeechGenerationMode();

        var hp = ModelHyperparams.FromGgufMetadata(source.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(source, backend, hp);

        var prompt = new int[] { 785, 3974, source.SpeechTokenIdOffset + 12 };
        var logits = fwd.Prefill(prompt).ToArray();

        Assert.Equal(source.SpeechVocabSize, logits.Length);
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        foreach (var v in logits)
        {
            Assert.False(float.IsNaN(v) || float.IsInfinity(v), "logit is NaN/Inf");
            if (v < min) min = v;
            if (v > max) max = v;
        }
        Assert.True(max - min > 1.0f, $"logits look degenerate: min={min}, max={max}");
    }
}
