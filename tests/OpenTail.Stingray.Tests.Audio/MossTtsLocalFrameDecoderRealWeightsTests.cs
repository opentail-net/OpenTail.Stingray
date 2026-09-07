using OpenTail.Stingray.Audio.MossTts;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weight smoke test for MOSS-TTS-Nano's local frame decoder: loads the real checkpoint,
/// runs a real global-transformer prefill to get a real hidden state, then decodes a full frame's
/// 16 RVQ codebook tokens through the real local transformer. Not a numeric golden-parity check
/// (no captured reference trace) -- see docs/audio-review-progress.md for that gap.
/// </summary>
public sealed class MossTtsLocalFrameDecoderRealWeightsTests : HeavyTestBase
{
    private static string? FindRepoFile(string relPath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relPath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void GenerateFrame_OnRealCheckpoint_ProducesInRangeTokensOrEndsCleanly()
    {
        string? path = FindRepoFile("models/_models/moss-tts-nano/MOSS-TTS-Nano-100M-GGUF/moss-tts-nano-100m-q8_0.gguf");
        Assert.SkipUnless(path != null, "moss-tts-nano checkpoint not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        var g = new MossTtsGlobalTransformerWeights(source);
        var l = new MossTtsLocalTransformerWeights(source);

        var pad = MossTtsGlobalTransformerWeights.AudioPadTokenId;
        var noAudio = Enumerable.Repeat(pad, MossTtsGlobalTransformerWeights.NumCodebooks).ToArray();
        var rows = new List<MossTtsGlobalRow>
        {
            new(MossTtsGlobalTransformerWeights.ImStartTokenId, (int[])noAudio.Clone()),
            new(100, (int[])noAudio.Clone()),
            new(200, (int[])noAudio.Clone()),
            new(MossTtsGlobalTransformerWeights.AudioStartTokenId, (int[])noAudio.Clone()),
        };
        var globalHidden = MossTtsGlobalTransformer.ForwardLastHidden(g, rows);
        Assert.All(globalHidden, v => Assert.True(float.IsFinite(v)));

        var frame = MossTtsLocalFrameDecoder.GenerateFrame(g, l, globalHidden, activeCodebooks: MossTtsGlobalTransformerWeights.NumCodebooks);

        // Either a clean "end generation" signal, or a real in-range frame -- both are valid,
        // architecture-correct outcomes for an untrained-on-this-prompt real forward pass; the
        // property under test is that nothing crashes/produces NaN/out-of-range tokens.
        if (frame is null) return;
        Assert.Equal(MossTtsGlobalTransformerWeights.NumCodebooks, frame.Length);
        foreach (var token in frame)
            Assert.InRange(token, 0, MossTtsGlobalTransformerWeights.AudioCodebookSize - 1);
    }
}
