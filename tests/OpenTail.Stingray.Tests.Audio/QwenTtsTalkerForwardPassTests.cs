using System;
using System.IO;
using OpenTail.Stingray.Audio.QwenTTS;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Proves the core hypothesis behind this fire's QwenTTS investigation: that the real Qwen3-TTS
/// Talker (confirmed same shape family as Fish Speech's slow-AR -- GQA, per-head QK-RMSNorm, NEOX
/// RoPE, SwiGLU) can run through this codebase's existing, UNMODIFIED `ForwardPass` engine via a
/// tensor-name-remapping wrapper (<see cref="QwenTtsTalkerTensorSource"/>), the same sanctioned
/// reuse pattern as `FishSpeechTensorSource`. This is a CONSTRUCTION + SHAPE/FINITE smoke test
/// only (matching this project's own documented convention for a first-pass "does it even run"
/// check before real embedding composition and golden verification exist) -- NOT a numerical
/// correctness claim; the real per-timestep embedding composition (text projection, codec
/// embedding) and golden verification against a real oracle are real, separate, not-yet-done
/// follow-up work (see docs/audio-review-progress.md's QwenTTS entries).
/// </summary>
public sealed class QwenTtsTalkerForwardPassTests : HeavyTestBase
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
    public void ForwardPass_ConstructsAndRuns_AgainstRealQwenTtsTalkerGguf()
    {
        string? modelPath = FindModelPath("qwen-talker-0.6b-base-Q8_0.gguf");
        Assert.SkipUnless(modelPath != null, "models/qwen-talker-0.6b-base-Q8_0.gguf not found");

        using var model = GgufModel.Open(modelPath!);
        var tensorSource = new QwenTtsTalkerTensorSource(model, numLayers: 28);
        var hp = ModelHyperparams.FromGgufMetadata(tensorSource.Metadata, tensorSource);

        Assert.Equal(28, hp.NumLayers);
        Assert.Equal(1024, hp.EmbeddingDim);
        Assert.Equal(16, hp.NumHeads);
        Assert.Equal(8, hp.NumKvHeads);
        Assert.Equal(128, hp.HeadDim);
        Assert.True(hp.IsNeoxRope, "Talker RoPE should resolve to NEOX per the real talker-forward.h ggml_rope_ext(GGML_ROPE_TYPE_NEOX) call");

        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(tensorSource, backend, hp, maxContextLength: 64);

        // Real codec_embd row 0 as a stand-in input embedding (a real embedding table row, not
        // random noise) -- full per-timestep composition (text projection + codec embedding sum)
        // is real, separate follow-up work, not attempted here.
        var codecEmbd = tensorSource.FindTensor("token_embd.weight");
        Assert.NotNull(codecEmbd);
        var embdBytes = tensorSource.GetTensorData(codecEmbd!.Value);
        int bytesPerRow = (hp.EmbeddingDim / 32) * 34; // real Q8_0 block size: 34 bytes per 32-element block
        var row0 = new float[hp.EmbeddingDim];
        Dequantize.ToFloat32(embdBytes[..bytesPerRow], row0, codecEmbd.Value.DType, hp.EmbeddingDim);

        var logits = fwd.ForwardEmbedding(row0, position: 0);

        Assert.True(logits.Length > 0, "ForwardPass produced no logits");
        foreach (var v in logits)
            Assert.True(float.IsFinite(v), "Talker logits contained a non-finite value");
    }
}
