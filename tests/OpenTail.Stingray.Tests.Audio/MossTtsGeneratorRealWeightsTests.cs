using OpenTail.Stingray.Audio.MossTts;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real, end-to-end smoke test for MOSS-TTS-Nano's full text-to-token-codes path: real
/// SentencePiece BPE tokenizer -&gt; real zero-shot prompt builder -&gt; real global transformer +
/// local frame decoder generation loop, all against the actual checkpoint. Not a numeric
/// golden-parity check (no captured reference trace, and the generated audio codes are not yet
/// turned into a waveform -- the audio codec is not ported) -- see
/// docs/audio-review-progress.md for that gap. Confirms the whole wiring produces a real,
/// in-range sequence of RVQ codes without crashing.
/// </summary>
public sealed class MossTtsGeneratorRealWeightsTests : HeavyTestBase
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

    private static byte[]? ExtractEmbeddedFile(GgufModel model, string fileName)
    {
        if (!model.Metadata.TryGetValue("audiocpp.embedded_files.names", out var namesObj) || namesObj is not object[] names) return null;
        var offsets = (object[])model.Metadata["audiocpp.embedded_files.offsets"];
        var data = (object[])model.Metadata["audiocpp.embedded_files.data"];
        var bytes = data.Select(o => (byte)Convert.ToInt64(o)).ToArray();
        for (int i = 0; i < names.Length; i++)
        {
            if ((string)names[i] != fileName) continue;
            long start = Convert.ToInt64(offsets[i]);
            long end = i + 1 < offsets.Length ? Convert.ToInt64(offsets[i + 1]) : bytes.Length;
            return bytes[(int)start..(int)end];
        }
        return null;
    }

    [Fact]
    public void Generate_OnRealCheckpoint_ProducesInRangeAudioCodes()
    {
        string? path = FindRepoFile("models/_models/moss-tts-nano/MOSS-TTS-Nano-100M-GGUF/moss-tts-nano-100m-q8_0.gguf");
        Assert.SkipUnless(path != null, "moss-tts-nano checkpoint not found");

        using var model = GgufModel.Open(path!);
        var tokenizerBytes = ExtractEmbeddedFile(model, "tokenizer.model");
        Assert.NotNull(tokenizerBytes);
        var tokenizer = SentencePieceBpeTokenizer.FromModelBytes(tokenizerBytes!);

        var source = new RvcPackedTensorSource(model);
        var g = new MossTtsGlobalTransformerWeights(source);
        var l = new MossTtsLocalTransformerWeights(source);

        var prompt = MossTtsPromptBuilder.BuildZeroShotPrompt(tokenizer, "Hello there.");
        Assert.NotEmpty(prompt);
        Assert.Equal(MossTtsGlobalTransformerWeights.ImStartTokenId, prompt[0].TextId);
        Assert.Equal(MossTtsGlobalTransformerWeights.AudioStartTokenId, prompt[^1].TextId);

        var codes = MossTtsGenerator.Generate(g, l, prompt, activeCodebooks: 4, maxNewFrames: 3);

        Assert.True(codes.Frames >= 1);
        Assert.Equal(MossTtsGlobalTransformerWeights.NumCodebooks, codes.Codebooks);
        Assert.Equal(codes.Frames * codes.Codebooks, codes.TokenIds.Length);
        Assert.All(codes.TokenIds, id => Assert.InRange(id, 0, MossTtsGlobalTransformerWeights.AudioCodebookSize));
    }
}
