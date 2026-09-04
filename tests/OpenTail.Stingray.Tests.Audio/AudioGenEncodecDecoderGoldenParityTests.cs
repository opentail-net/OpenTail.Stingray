using OpenTail.Stingray.Audio.AudioGen;
using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real numeric golden verification for AudioGen's 16kHz EnCodec decoder -- calls the real,
/// production <see cref="EncodecDecoderKernels.Decode"/> (via <see cref="AudioGenEncodecDecoderWeights"/>)
/// against a fixed set of codebook token ids, compared to
/// `scratch-llamacpp-ref/audiogen_encodec_decoder_golden.py`, a pure-numpy oracle transcribed from
/// the real AudioCraft/`transformers` SEANet decoder math using the real, already-local
/// `models/audiogen-medium/audiogen-medium-encodec16k.safetensors` checkpoint (native AudioCraft
/// tensor naming: `decoder.model.{i}.conv.conv.weight_g/weight_v`,
/// `decoder.model.{i}.convtr.convtr.*`, `quantizer.vq.layers.{q}._codebook.embed`).
///
/// This is a DIFFERENT, separately-trained codec checkpoint from MusicGen's 32kHz EnCodec despite
/// sharing the identical <see cref="EncodecDecoderKernels"/> layer skeleton (only the upsampling
/// ratios/dims differ) -- closing this test exercises AudioGen's own real weights end to end
/// through the shared kernel (quantizer sum -> init conv -> 2-layer residual LSTM -> 4 real
/// upsample stages with real trimmed transpose convs and residual blocks -> final conv), the last
/// previously-unverified real numeric stage of the AudioGen pipeline besides the text encoder.
/// </summary>
public sealed class AudioGenEncodecDecoderGoldenParityTests : HeavyTestBase
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
        string? modelPath = FindRepoFile("models/audiogen-medium/audiogen-medium-encodec16k.safetensors");
        Assert.SkipUnless(modelPath != null, "models/audiogen-medium/audiogen-medium-encodec16k.safetensors not found");

        string? codesPath = FindRepoFile("scratch-llamacpp-ref/audiogen_encodec_decoder_golden_codes.txt");
        string? pcmPath = FindRepoFile("scratch-llamacpp-ref/audiogen_encodec_decoder_golden_pcm.txt");
        Assert.SkipUnless(codesPath != null && pcmPath != null,
            "golden AudioGen EnCodec decoder files not found (re-run scratch-llamacpp-ref/audiogen_encodec_decoder_golden.py)");

        var codeLines = File.ReadAllText(codesPath!).Trim().Split('\n');
        int numCodebooks = codeLines.Length;
        var codes = new int[numCodebooks][];
        for (int q = 0; q < numCodebooks; q++) codes[q] = Array.ConvertAll(codeLines[q].Split(','), int.Parse);

        var pcmLines = File.ReadAllText(pcmPath!).Split('\n');
        int goldenLen = int.Parse(pcmLines[0].Trim());
        var goldenParts = pcmLines[1].Trim().Split(',');
        Assert.Equal(goldenLen, goldenParts.Length);
        var golden = new float[goldenLen];
        for (int i = 0; i < golden.Length; i++) golden[i] = float.Parse(goldenParts[i]);

        var sw = System.Diagnostics.Stopwatch.StartNew();

        using var loader = SafetensorsLoader.Open(modelPath!);
        var weights = AudioGenEncodecDecoderWeights.Load(loader);
        var pcm = EncodecDecoderKernels.Decode(weights, codes);

        sw.Stop();
        Assert.True(sw.ElapsedMilliseconds > 50, $"suspiciously fast run ({sw.ElapsedMilliseconds}ms) -- did this actually execute against real weights?");

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
        Assert.True(cosine > 0.999, $"cosine similarity {cosine} too low vs golden AudioGen EnCodec decoder PCM");
    }
}
