using OpenTail.Stingray.Audio.MossTts;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real, full end-to-end smoke test for MOSS-TTS-Nano: real tokenizer -&gt; real prompt -&gt; real
/// global transformer + local frame decoder generation loop -&gt; real audio codec decode -&gt; a
/// real, finite, correctly-shaped 48kHz stereo waveform. This is the first point this session
/// where MOSS-TTS-Nano produces actual audio samples, not just token codes. Not a numeric
/// golden-parity check against a captured reference waveform -- see
/// docs/audio-review-progress.md for that gap (and the windowed-attention gap noted in
/// <see cref="MossTtsAudioCodecDecoder"/>'s doc comment).
/// </summary>
public sealed class MossTtsAudioCodecDecoderRealWeightsTests : HeavyTestBase
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
    public void FullPipeline_OnRealCheckpoint_ProducesFiniteStereoWaveform()
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
        var quantizer = new MossTtsAudioCodecQuantizerWeights(source);
        var decoderWeights = new MossTtsAudioCodecDecoderWeights(source);

        var prompt = MossTtsPromptBuilder.BuildZeroShotPrompt(tokenizer, "Hello there.");
        var codes = MossTtsGenerator.Generate(g, l, prompt, activeCodebooks: MossTtsGlobalTransformerWeights.NumCodebooks, maxNewFrames: 3);

        // Generation may legitimately produce pad-sentinel codes for codebooks the local decoder
        // never overwrites when activeCodebooks < NumCodebooks; here activeCodebooks==NumCodebooks
        // so every code must be a real in-range codebook index (never the pad sentinel).
        Assert.All(codes.TokenIds, id => Assert.InRange(id, 0, MossTtsAudioCodecQuantizerWeights.CodebookSize - 1));

        var codesPerQuantizer = new int[MossTtsAudioCodecQuantizerWeights.NumQuantizers][];
        for (int q = 0; q < codesPerQuantizer.Length; q++)
        {
            var row = new int[codes.Frames];
            for (int f = 0; f < codes.Frames; f++) row[f] = codes.TokenIds[f * codes.Codebooks + q];
            codesPerQuantizer[q] = row;
        }

        var waveform = MossTtsAudioCodecDecoder.Decode(quantizer, decoderWeights, codesPerQuantizer);

        int expectedSamplesPerChannel = codes.Frames * MossTtsAudioCodecDecoderWeights.SamplesPerFrame;
        Assert.Equal(expectedSamplesPerChannel, waveform.Left.Length);
        Assert.Equal(expectedSamplesPerChannel, waveform.Right.Length);
        Assert.All(waveform.Left, s => Assert.True(float.IsFinite(s)));
        Assert.All(waveform.Right, s => Assert.True(float.IsFinite(s)));
    }
}
