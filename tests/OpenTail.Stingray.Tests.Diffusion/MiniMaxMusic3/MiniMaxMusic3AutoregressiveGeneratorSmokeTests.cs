using OpenTail.Stingray.Core;
using OpenTail.Stingray.Diffusion.MiniMaxMusic3;
using Xunit;

namespace OpenTail.Stingray.Tests.Diffusion.MiniMaxMusic3;

/// <summary>
/// Real-weight smoke test for <see cref="MiniMaxMusic3AutoregressiveGenerator"/>: not a golden-
/// parity check (no real tokenizer/prompt yet -- see the class's own doc comment), just confirms
/// the real per-frame CFG loop runs end-to-end against real weights without exceptions and produces
/// structurally valid output (right shapes, codes within their real vocab ranges). A full golden
/// check needs the real Qwen2Tokenizer vocab for this checkpoint plus a real diffusers reference
/// generation run -- not yet built, see docs/066-minimax-music3-future-plan.md.
/// </summary>
public sealed class MiniMaxMusic3AutoregressiveGeneratorSmokeTests
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

    private static string? FindRepoDir(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (Directory.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Generate_RealWeights_ProducesStructurallyValidFrames()
    {
        string? langDir = FindRepoDir("models/minimax-music3/language_model");
        string? depthPath = FindRepoFile("models/minimax-music3/rvq_depth_decoder.safetensors");
        Assert.SkipUnless(langDir != null, "models/minimax-music3/language_model/ not found");
        Assert.SkipUnless(depthPath != null, "models/minimax-music3/rvq_depth_decoder.safetensors not found");

        using var langLoader = SafetensorsLoader.OpenDirectory(langDir!);
        using var globalModel = new MiniMaxMusic3GlobalModel(langLoader);

        using var depthLoader = SafetensorsLoader.Open(depthPath!);
        var depthWeights = MiniMaxMusic3RvqDepthDecoderWeights.Load(depthLoader);

        // Real prompt shape: <|im_start|>...<|im_end|><|audio_start|> -- using placeholder caption
        // tokens (real tokenizer not yet wired up) but the real structural token ids the CFG-null
        // construction depends on (first token kept real, last two kept real).
        int imStart = 100, imEnd = 200, audioStart = 300, capA = 400, capB = 401, capC = 402;
        int[] promptTokens = [imStart, capA, capB, capC, imEnd, audioStart];

        var random = new Random(42);
        var repr = MiniMaxMusic3AutoregressiveGenerator.Generate(globalModel, depthWeights, promptTokens, maxFrames: 3, random);

        Assert.True(repr.FrameCount >= 1, "expected at least one generated frame");
        Assert.Equal(repr.FrameCount, repr.AcousticTokens.Length);
        Assert.Equal(repr.FrameCount, repr.GlobalHiddenStates.Length);
        Assert.Equal(repr.FrameCount, repr.LocalHiddenStates.Length);

        for (int t = 0; t < repr.FrameCount; t++)
        {
            Assert.InRange(repr.SemanticTokens[t], 0, MiniMaxMusic3Config.SemanticVocabSize - 1);
            Assert.Equal(7, repr.AcousticTokens[t].Length);
            foreach (var code in repr.AcousticTokens[t])
                Assert.InRange(code, 0, MiniMaxMusic3Config.RvqDepthDecoderAudioVocabSize - 1);

            Assert.Equal(MiniMaxMusic3Config.LanguageModelHiddenSize, repr.GlobalHiddenStates[t].Length);
            Assert.Equal(7 * MiniMaxMusic3Config.RvqDepthDecoderHiddenSize, repr.LocalHiddenStates[t].Length);

            bool anyNonFinite = false;
            foreach (var v in repr.GlobalHiddenStates[t]) if (!float.IsFinite(v)) anyNonFinite = true;
            foreach (var v in repr.LocalHiddenStates[t]) if (!float.IsFinite(v)) anyNonFinite = true;
            Assert.False(anyNonFinite, $"frame {t} contains NaN/Inf in hidden states");
        }
    }
}
