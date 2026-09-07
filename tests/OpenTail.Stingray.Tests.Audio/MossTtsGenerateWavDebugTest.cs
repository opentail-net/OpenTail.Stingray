using OpenTail.Stingray.Audio.MossTts;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: generates a real MOSS-TTS-Nano wav end-to-end for informal
/// listening (no CLI wiring yet). Writes to docs/audio-samples (gitignored, local-only per
/// CLAUDE.md). Not a golden/parity test.</summary>
public sealed class MossTtsGenerateWavDebugTest : HeavyTestBase
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
    public void Generate_RealMossTtsWav()
    {
        string? path = FindRepoFile("models/_models/moss-tts-nano/MOSS-TTS-Nano-100M-GGUF/moss-tts-nano-100m-q8_0.gguf");
        Assert.SkipUnless(path != null, "moss-tts-nano checkpoint not found");
        string? repoRoot = Path.GetDirectoryName(FindRepoFile("docs/audio-review-progress.md"));
        Assert.NotNull(repoRoot);

        using var model = GgufModel.Open(path!);
        var tokenizer = SentencePieceBpeTokenizer.FromModelBytes(ExtractEmbeddedFile(model, "tokenizer.model")!);
        var source = new RvcPackedTensorSource(model);
        var g = new MossTtsGlobalTransformerWeights(source);
        var l = new MossTtsLocalTransformerWeights(source);
        var quantizer = new MossTtsAudioCodecQuantizerWeights(source);
        var decoderWeights = new MossTtsAudioCodecDecoderWeights(source);

        var prompt = MossTtsPromptBuilder.BuildZeroShotPrompt(tokenizer, "Hello there, this is a test of speech synthesis.");
        var codes = MossTtsGenerator.Generate(g, l, prompt, activeCodebooks: MossTtsGlobalTransformerWeights.NumCodebooks, maxNewFrames: 40);
        Console.WriteLine($"Generated {codes.Frames} frames (hitMax={codes.HitMaxNewFrames})");

        var codesPerQuantizer = new int[MossTtsAudioCodecQuantizerWeights.NumQuantizers][];
        for (int q = 0; q < codesPerQuantizer.Length; q++)
        {
            var row = new int[codes.Frames];
            for (int f = 0; f < codes.Frames; f++) row[f] = codes.TokenIds[f * codes.Codebooks + q];
            codesPerQuantizer[q] = row;
        }

        var waveform = MossTtsAudioCodecDecoder.Decode(quantizer, decoderWeights, codesPerQuantizer);
        Assert.True(waveform.Left.Length > 0);

        var result = new OpenTail.Stingray.Audio.AudioGenerationResult(waveform.Left, MossTtsAudioCodecDecoderWeights.SamplingRate);
        string outPath = Path.Combine(repoRoot!, "audio-samples", "moss-tts-nano-real-check.wav");
        result.SaveWav(outPath);
        Console.WriteLine($"Wrote {outPath}, {waveform.Left.Length} samples, {waveform.Left.Length / (double)MossTtsAudioCodecDecoderWeights.SamplingRate:F2}s");
    }
}
