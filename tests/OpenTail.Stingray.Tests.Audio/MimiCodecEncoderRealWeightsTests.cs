using OpenTail.Stingray.Audio.PersonaPlex;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real, first-attempt smoke test for <see cref="MimiCodecEncoder"/> (PersonaPlex's live-duplex
/// user-audio conditioning): real weights, a real (decoder-generated, not synthetic-noise) mono
/// waveform, checks the encoder runs end to end and produces finite, in-range, correctly-shaped
/// `[frames][8]` codes. Not a numeric golden-parity check against a captured reference encode (no
/// independent oracle run yet) -- see docs/audio-review-progress.md's PersonaPlex live-duplex
/// entries.
/// </summary>
public sealed class MimiCodecEncoderRealWeightsTests : HeavyTestBase
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
        string? path = FindRepoFile("models/_models/personaplex/PersonaPlex-GGUF/personaplex-7b-v1-q8_0.gguf");
        Assert.SkipUnless(path != null, "personaplex-7b-v1-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        var decoderWeights = MimiCodecDecoderWeights.Load(source.GetTensor);
        var encoderWeights = MimiCodecEncoderWeights.Load(source.GetTensor, source.HasTensor, decoderWeights);

        // Real (not synthetic-noise) input: decode a few real, in-range codes into a real waveform
        // first (same "check against something real" discipline this session's other new encoder
        // ports use even without an independent Python/C++ oracle for the encoder specifically).
        var rng = new Random(23);
        const int frames = 8;
        var decodeCodes = new int[frames][];
        for (int t = 0; t < frames; t++)
        {
            decodeCodes[t] = new int[MimiCodecDecoderWeights.ActiveCodebooks];
            for (int cb = 0; cb < MimiCodecDecoderWeights.ActiveCodebooks; cb++)
                decodeCodes[t][cb] = rng.Next(MimiCodecDecoderWeights.CodebookSize);
        }
        var waveform = MimiCodecDecoder.Decode(decoderWeights, decodeCodes);

        var codes = MimiCodecEncoder.Encode(encoderWeights, waveform);

        Assert.True(codes.Length > 0);
        foreach (var frame in codes)
        {
            Assert.Equal(MimiCodecDecoderWeights.ActiveCodebooks, frame.Length);
            Assert.All(frame, c => Assert.InRange(c, 0, MimiCodecDecoderWeights.CodebookSize - 1));
        }
    }
}
