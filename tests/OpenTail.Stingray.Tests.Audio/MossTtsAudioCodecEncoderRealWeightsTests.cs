using OpenTail.Stingray.Audio.MossTts;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real, first-attempt smoke test for MOSS-TTS-Nano's voice-cloning ENCODER
/// (<see cref="MossTtsAudioCodecEncoder"/>): real weights, a real (decoder-generated, not
/// synthetic-noise) stereo waveform, checks the encoder runs end to end and produces
/// finite, in-range, correctly-shaped `[NumQuantizers][frames]` codes. Not a numeric golden-parity
/// check against a captured reference encode (no independent oracle run yet) -- see
/// docs/audio-review-progress.md's MOSS-TTS-Nano voice-cloning entries.
/// </summary>
public sealed class MossTtsAudioCodecEncoderRealWeightsTests : HeavyTestBase
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
    public void Encode_OnRealDecoderOutput_ProducesFiniteInRangeCodes()
    {
        string? path = FindRepoFile("models/_models/moss-tts-nano/MOSS-TTS-Nano-100M-GGUF/moss-tts-nano-100m-q8_0.gguf");
        Assert.SkipUnless(path != null, "moss-tts-nano checkpoint not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        var quantizer = new MossTtsAudioCodecQuantizerWeights(source);
        var decoderWeights = new MossTtsAudioCodecDecoderWeights(source);
        var encoderWeights = new MossTtsAudioCodecEncoderWeights(source);

        // Real (not synthetic-noise) input: decode a few frames of real, in-range codes into a
        // real waveform, matching this session's "check against a real reference" discipline even
        // without an independent Python/C++ oracle for the encoder specifically.
        const int frames = 4;
        var rng = new Random(7);
        var codesPerQuantizer = new int[MossTtsAudioCodecQuantizerWeights.NumQuantizers][];
        for (int q = 0; q < codesPerQuantizer.Length; q++)
        {
            var row = new int[frames];
            for (int f = 0; f < frames; f++) row[f] = rng.Next(MossTtsAudioCodecQuantizerWeights.CodebookSize);
            codesPerQuantizer[q] = row;
        }
        var waveform = MossTtsAudioCodecDecoder.Decode(quantizer, decoderWeights, codesPerQuantizer);

        var codes = MossTtsAudioCodecEncoder.Encode(encoderWeights, quantizer, new MossTtsAudioInput(waveform.Left, waveform.Right));

        Assert.Equal(MossTtsAudioCodecQuantizerWeights.NumQuantizers, codes.Length);
        int expectedFrames = waveform.Left.Length / MossTtsAudioCodecDecoderWeights.SamplesPerFrame;
        Assert.Equal(expectedFrames, codes[0].Length);
        foreach (var row in codes)
        {
            Assert.Equal(expectedFrames, row.Length);
            Assert.All(row, c => Assert.InRange(c, 0, MossTtsAudioCodecQuantizerWeights.CodebookSize - 1));
        }
    }
}
