using OpenTail.Stingray.Audio.MusicGen;
using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for MusicGen's EnCodec-32kHz decoder
/// (<see cref="EncodecDecoderKernels.Decode"/> via <see cref="MusicGenEncodecDecoderWeights"/>) --
/// compares against `scratch-llamacpp-ref/musicgen_encodec_decoder_golden.py`, which loads the
/// real, already-local `models/musicgen-small/musicgen-small.safetensors` `audio_encoder.*`
/// tensor tree directly via safetensors and computes the real SEANet-style EnCodec decoder math
/// (weight_norm-folded convs, 2-layer residual LSTM, trimmed transpose-conv upsampling, identity
/// residual blocks) in numpy, transcribed from the real `transformers` `modeling_encodec.py`.
///
/// Closes gap 2/3 of the "not just the decoder transformer" MusicGen numeric-parity closure
/// requested 2026-09-04 -- this is the first real numeric check of the actual codec (RVQ decode
/// -> conv/LSTM/upsample stack -> PCM) rather than by-ear verification only.
/// </summary>
public sealed class MusicGenEncodecDecoderGoldenParityTests : HeavyTestBase
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
    public void Decode_RealWeights_MatchesGoldenOutput()
    {
        string? modelPath = FindRepoFile("models/musicgen-small/musicgen-small.safetensors");
        Assert.SkipUnless(modelPath != null, "models/musicgen-small/musicgen-small.safetensors not found");

        string? codesPath = FindRepoFile("scratch-llamacpp-ref/musicgen_encodec_decoder_golden_codes.txt");
        string? pcmPath = FindRepoFile("scratch-llamacpp-ref/musicgen_encodec_decoder_golden_pcm.txt");
        Assert.SkipUnless(codesPath != null && pcmPath != null,
            "golden MusicGen EnCodec decoder files not found (re-run scratch-llamacpp-ref/musicgen_encodec_decoder_golden.py)");

        var codeLines = File.ReadAllText(codesPath!).Trim().Split('\n');
        var codes = new int[codeLines.Length][];
        for (int i = 0; i < codeLines.Length; i++) codes[i] = Array.ConvertAll(codeLines[i].Split(','), int.Parse);

        var pcmLines = File.ReadAllText(pcmPath!).Split('\n');
        int goldenLen = int.Parse(pcmLines[0].Trim());
        var goldenParts = pcmLines[1].Trim().Split(',');
        Assert.Equal(goldenLen, goldenParts.Length);
        var golden = new float[goldenLen];
        for (int i = 0; i < golden.Length; i++) golden[i] = float.Parse(goldenParts[i]);

        using var loader = SafetensorsLoader.Open(modelPath!);
        var weights = MusicGenEncodecDecoderWeights.Load(loader);

        var pcm = EncodecDecoderKernels.Decode(weights, codes);

        Assert.Equal(goldenLen, pcm.Length);

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < goldenLen; i++)
        {
            float a = pcm[i];
            float b = golden[i];
            dot += a * b;
            normA += a * a;
            normB += b * b;
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.999, $"cosine similarity {cosine} too low vs golden MusicGen EnCodec decoder output");
    }
}
