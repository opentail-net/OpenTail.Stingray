using System.IO;
using OpenTail.Stingray.Audio.QwenTTS;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real structural test for <see cref="QwenTtsTalkerGeneration.GenerateSemanticCodes"/> -- the
/// real Talker prompt composition (text_proj + codec_embd summed streams, real special ids) and
/// autoregressive generation loop over real weights, using the synthetic-embedding-table
/// technique to feed precomputed per-position embeddings through the existing `ForwardPass`
/// (see <see cref="QwenTtsTalkerTensorSource.SetPromptEmbedding"/>'s doc comment for why this
/// was necessary). Not yet golden-verified against a numeric oracle (no local runnable Python
/// QwenTTS reference confirmed) -- checks the full real loop runs to completion on real weights
/// and produces a real, in-range, non-degenerate semantic-codebook token sequence.
/// </summary>
public sealed class QwenTtsTalkerGenerationTests : HeavyTestBase
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
    public void GenerateSemanticCodes_RealWeights_ProducesInRangeNonDegenerateTokenSequence()
    {
        string? modelPath = FindRepoFile("models/qwen-talker-0.6b-base-Q8_0.gguf");
        Assert.SkipUnless(modelPath != null, "models/qwen-talker-0.6b-base-Q8_0.gguf not found");

        using var rawModel = GgufModel.Open(modelPath!);

        var codes = QwenTtsTalkerGeneration.GenerateSemanticCodes(rawModel, "Hello there, this is a test.", numLayers: 28, maxNewTokens: 20);

        Assert.True(codes.Length > 0, "generation produced zero semantic codes");
        Assert.True(codes.Length <= 20);

        foreach (var c in codes)
            Assert.InRange(c, 0, 3071); // real codec vocab size

        Assert.True(new System.Collections.Generic.HashSet<int>(codes).Count > 1 || codes.Length == 1, "generated sequence looks degenerate (all-identical tokens)");
    }
}
