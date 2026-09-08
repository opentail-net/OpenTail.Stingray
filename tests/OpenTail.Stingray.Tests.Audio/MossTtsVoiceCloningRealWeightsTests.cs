using OpenTail.Stingray.Audio.MossTts;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real, first-attempt end-to-end voice-cloning smoke test for MOSS-TTS-Nano: encode a real
/// reference waveform into RVQ codes (<see cref="MossTtsAudioCodecEncoder"/>), build the real
/// `BuildVoiceClonePrompt` reference-audio-conditioned prompt, run the real generation loop, and
/// decode the result into a real, finite, correctly-shaped stereo waveform -- the first point this
/// session where MOSS-TTS-Nano's voice-cloning path is wired ALL the way through, not just the
/// encoder in isolation. Not a numeric golden-parity check (no captured reference voice-clone run
/// exists yet).
/// </summary>
public sealed class MossTtsVoiceCloningRealWeightsTests : HeavyTestBase
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
    public void VoiceClonePrompt_EndToEnd_ProducesFiniteStereoWaveform()
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
        var encoderWeights = new MossTtsAudioCodecEncoderWeights(source);

        // Real reference audio for the voice-clone prompt: decode a few frames of real, in-range
        // codes into a real waveform (same "real, not synthetic-noise" input this session's
        // encoder-only test already uses), then re-encode it -- a real roundtrip through both new
        // encoder-side pieces before they're ever exercised by the actual generation loop.
        const int refFrames = 2;
        var rng = new Random(3);
        var refCodesPerQuantizer = new int[MossTtsAudioCodecQuantizerWeights.NumQuantizers][];
        for (int q = 0; q < refCodesPerQuantizer.Length; q++)
        {
            var row = new int[refFrames];
            for (int f = 0; f < refFrames; f++) row[f] = rng.Next(MossTtsAudioCodecQuantizerWeights.CodebookSize);
            refCodesPerQuantizer[q] = row;
        }
        var refWaveform = MossTtsAudioCodecDecoder.Decode(quantizer, decoderWeights, refCodesPerQuantizer);
        var reEncodedCodes = MossTtsAudioCodecEncoder.Encode(encoderWeights, quantizer,
            new MossTtsAudioInput(refWaveform.Left, refWaveform.Right));

        var prompt = MossTtsPromptBuilder.BuildVoiceClonePrompt(tokenizer, "Hello there.", reEncodedCodes);
        var codes = MossTtsGenerator.Generate(g, l, prompt, activeCodebooks: MossTtsGlobalTransformerWeights.NumCodebooks, maxNewFrames: 3);

        Assert.All(codes.TokenIds, id => Assert.InRange(id, 0, MossTtsAudioCodecQuantizerWeights.CodebookSize - 1));

        var codesPerQuantizer = new int[MossTtsAudioCodecQuantizerWeights.NumQuantizers][];
        for (int q = 0; q < codesPerQuantizer.Length; q++)
        {
            var row = new int[codes.Frames];
            for (int f = 0; f < codes.Frames; f++) row[f] = codes.TokenIds[f * codes.Codebooks + q];
            codesPerQuantizer[q] = row;
        }

        var waveform = MossTtsAudioCodecDecoder.Decode(quantizer, decoderWeights, codesPerQuantizer);
        Assert.All(waveform.Left, s => Assert.True(float.IsFinite(s)));
        Assert.All(waveform.Right, s => Assert.True(float.IsFinite(s)));

        string? repoRoot = Path.GetDirectoryName(FindRepoFile("docs/audio-review-progress.md"));
        if (repoRoot != null)
        {
            string outPath = Path.Combine(repoRoot, "audio-samples", "moss-tts-nano-voiceclone-real-check.wav");
            var interleaved = new float[waveform.Left.Length * 2];
            for (int i = 0; i < waveform.Left.Length; i++)
            {
                interleaved[2 * i] = waveform.Left[i];
                interleaved[2 * i + 1] = waveform.Right[i];
            }
            OpenTail.Stingray.Audio.WavWriter.WriteWav(outPath, interleaved, MossTtsAudioCodecDecoderWeights.SamplingRate, channels: 2);
            Console.WriteLine($"Wrote {outPath}, {waveform.Left.Length} samples/channel, {waveform.Left.Length / (double)MossTtsAudioCodecDecoderWeights.SamplingRate:F2}s");
        }
    }
}
