using OpenTail.Stingray.Audio.NeuTts;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Numeric golden parity test for the NeuTTS FSQ audio codec decoder:
/// Compares C# <see cref="NeuTtsAudioDecoder.Decode"/> against the reference C++
/// <c>NeuTTSCodecDecoderRuntime</c> from audio.cpp on real weights.
/// </summary>
public sealed class NeuTtsAudioDecoderGoldenParityTests : HeavyTestBase
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
    public void FsqAudioDecoder_RealWeights_MatchesReferenceGoldenParity()
    {
        string? modelPath = FindRepoFile("models/_models/neutts/NeuTTS-2E-GGUF/neutts-2e-orig.gguf");
        Assert.SkipUnless(modelPath != null, "neutts-2e-orig.gguf not found");

        string? codesPath = FindRepoFile("scratch-llamacpp-ref/neutts_fsq_golden_codes.txt");
        string? pcmPath = FindRepoFile("scratch-llamacpp-ref/neutts_fsq_golden_pcm.bin");
        Assert.SkipUnless(codesPath != null && pcmPath != null, "golden NeuTTS FSQ files not found");

        var codesStr = File.ReadAllText(codesPath!).Trim().Split(',');
        int[] codes = Array.ConvertAll(codesStr, int.Parse);

        byte[] pcmBytes = File.ReadAllBytes(pcmPath!);
        float[] golden = new float[pcmBytes.Length / sizeof(float)];
        Buffer.BlockCopy(pcmBytes, 0, golden, 0, pcmBytes.Length);

        using var model = GgufModel.Open(modelPath!);
        var source = new RvcPackedTensorSource(model);
        var codecWeights = NeuTtsAudioDecoderWeights.Load(source.GetTensor);

        float[] waveform = NeuTtsAudioDecoder.Decode(codecWeights, codes);

        Assert.Equal(golden.Length, waveform.Length);

        double dot = 0, normA = 0, normB = 0;
        float maxDiff = 0;
        for (int i = 0; i < golden.Length; i++)
        {
            float a = waveform[i];
            float b = golden[i];
            dot += a * b;
            normA += a * a;
            normB += b * b;
            float diff = Math.Abs(a - b);
            if (diff > maxDiff) maxDiff = diff;
        }

        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Console.WriteLine($"[NeuTts FSQ Golden Parity] samples={waveform.Length}, cosine={cosine:F6}, maxDiff={maxDiff:E6}");

        Assert.True(cosine > 0.9999, $"cosine similarity {cosine} too low vs reference C++ FSQ decoder");
        Assert.True(maxDiff < 0.005f, $"max difference {maxDiff} too high vs reference C++ FSQ decoder");
    }
}
