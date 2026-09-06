
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real structural validation for NemotronAsrMelExtractor against the real downloaded
/// checkpoint and real audio -- confirms finite, non-degenerate log-mel output with the expected
/// shape (128 mels) before any encoder forward-pass code consumes it.</summary>
public sealed class NemotronAsrMelExtractorTests : HeavyTestBase
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
    public void ExtractMel_RealCheckpoint_RealAudio_ProducesFiniteNonDegenerateMel()
    {
        string? modelPath = FindRepoFile("models/_models/nemotron-asr/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf");
        Assert.SkipUnless(modelPath != null, "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf not found");
        string? audioPath = FindRepoFile("examples/audio.cpp/assets/resources/a.wav");
        Assert.SkipUnless(audioPath != null, "reference a.wav not found");

        using var w = new OpenTail.Stingray.Audio.NemotronAsr.NemotronAsrWeights(modelPath!);
        var extractor = OpenTail.Stingray.Audio.NemotronAsr.NemotronAsrMelExtractor.FromWeights(w);

        var (samples, sr, _) = WavReader.ReadWav(audioPath!);
        if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);

        var (mel, frames, validFrames) = extractor.ExtractMel(samples);

        Assert.True(frames > 0, "expected at least one mel frame");
        Assert.True(validFrames > 0 && validFrames <= frames);
        Assert.Equal(frames * OpenTail.Stingray.Audio.NemotronAsr.NemotronAsrMelExtractor.NumMels, mel.Length);

        double sum = 0, sumSq = 0;
        int validCount = validFrames * OpenTail.Stingray.Audio.NemotronAsr.NemotronAsrMelExtractor.NumMels;
        for (int i = 0; i < validCount; i++)
        {
            Assert.True(float.IsFinite(mel[i]), "mel contains a non-finite value in the valid region");
            sum += mel[i];
            sumSq += (double)mel[i] * mel[i];
        }
        double mean = sum / validCount;
        double std = Math.Sqrt(Math.Max(0, sumSq / validCount - mean * mean));
        Console.Error.WriteLine($"[NemotronAsrMel] frames={frames} validFrames={validFrames} mean={mean:F5} std={std:F5}");
        Assert.True(std > 1e-6, "mel output looks collapsed/degenerate");

        // Frames past valid_frames must be exactly zeroed (mask_invalid_frames=true).
        for (int f = validFrames; f < frames; f++)
            for (int m = 0; m < OpenTail.Stingray.Audio.NemotronAsr.NemotronAsrMelExtractor.NumMels; m++)
                Assert.Equal(0f, mel[f * OpenTail.Stingray.Audio.NemotronAsr.NemotronAsrMelExtractor.NumMels + m]);
    }
}
