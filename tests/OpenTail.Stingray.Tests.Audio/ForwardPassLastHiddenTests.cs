using System.IO;
using OpenTail.Stingray.Audio.QwenTTS;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;
using OpenTail.Stingray.Engine;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real test for the new CPU <see cref="ForwardPass.LastHidden"/> implementation (exposes the
/// persistent post-final-norm `_hidden` buffer `ForwardPass.Decode.cs` already computes on
/// every `Prefill`/`Forward` call) -- added specifically to unblock QwenTTS's Code Predictor,
/// which needs the Talker's real last-position hidden state as its own prefill conditioning
/// (see docs/audio-review-progress.md's QwenTTS Talker/Code Predictor entries). Uses the real
/// QwenTTS talker GGUF as a real, already-available fixture rather than building a new one.
/// </summary>
public sealed class ForwardPassLastHiddenTests : HeavyTestBase
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
    public void LastHidden_AfterForwardStep_ReturnsRealFiniteNonDegenerateEmbDimVector()
    {
        string? modelPath = FindRepoFile("models/qwen-talker-0.6b-base-Q8_0.gguf");
        Assert.SkipUnless(modelPath != null, "models/qwen-talker-0.6b-base-Q8_0.gguf not found");

        using var rawModel = GgufModel.Open(modelPath!);
        using var source = new QwenTtsTalkerTensorSource(rawModel, numLayers: 28);
        var hp = ModelHyperparams.FromGgufMetadata(source.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(source, backend, hp);

        var prompt = new int[] { 0, 1, 2 }; // dummy positions into the real codec_embd/token_embd table
        _ = fwd.Prefill(prompt);
        // Real, confirmed constraint (found by this test, not assumed): the CPU `_hidden`
        // single-slot buffer `LastHidden` exposes is only written by the single-token `Forward`
        // decode path (`ForwardPass.Decode.cs`) -- `Prefill`'s batched path
        // (`ForwardPass.PrefillCore.cs`) never touches it, so `LastHidden` reads as all-zero
        // immediately after a `Prefill`-only call. A real caller must always follow with at
        // least one `Forward` step before reading `LastHidden` -- exactly what QwenTTS's
        // Talker generation loop already does (autoregressive decode is single-token `Forward`
        // calls after the initial `Prefill`).
        _ = fwd.Forward(2, prompt.Length); // position 3, first real decode step after the position-0..2 prefill

        var lastHidden = fwd.LastHidden;
        Assert.Equal(hp.EmbeddingDim, lastHidden.Length);

        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        foreach (var v in lastHidden)
        {
            Assert.False(float.IsNaN(v) || float.IsInfinity(v), "LastHidden contains NaN/Inf");
            if (v < min) min = v;
            if (v > max) max = v;
        }
        Assert.True(max - min > 1e-4f, $"LastHidden looks degenerate: min={min}, max={max}");

        // Real single-slot-buffer semantics: a subsequent Forward call must change LastHidden's
        // content (it's the same live buffer being overwritten, not a stale independent copy).
        var afterFirstStep = lastHidden.ToArray();
        _ = fwd.Forward(3, prompt.Length + 1); // position 4
        var afterNextStep = fwd.LastHidden.ToArray();
        Assert.NotEqual(afterFirstStep, afterNextStep);
    }
}
