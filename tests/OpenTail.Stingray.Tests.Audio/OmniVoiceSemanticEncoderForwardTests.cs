
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real forward-pass test for OmniVoiceSemanticEncoder against real weights and real
/// audio -- not yet golden-verified against a real reference run (no CLI/warm-bench task wired up
/// for OmniVoice yet), same first-step-only status HuBERT/RMVPE started at for RVC.</summary>
public sealed class OmniVoiceSemanticEncoderForwardTests : HeavyTestBase
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
    public void Forward_RealAudio_RealWeights_ProducesFiniteNonDegenerateHiddenStates()
    {
        string? checkpointPath = FindRepoFile("models/_models/omnivoice/audio_tokenizer/model.safetensors");
        Assert.SkipUnless(checkpointPath != null, "omnivoice audio_tokenizer model.safetensors not found");
        string? audioPath = FindRepoFile("examples/audio.cpp/assets/resources/a.wav");
        Assert.SkipUnless(audioPath != null, "reference a.wav not found");

        using var loader = OpenTail.Stingray.Core.SafetensorsLoader.Open(checkpointPath!);
        var w = new OpenTail.Stingray.Audio.OmniVoice.OmniVoiceSemanticWeights(loader);

        var (samples, sr, _) = WavReader.ReadWav(audioPath!);
        if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);

        var hidden = OpenTail.Stingray.Audio.OmniVoice.OmniVoiceSemanticEncoder.Forward(w, samples);

        Assert.True(hidden.Length > 0);
        double sum = 0, sumSq = 0;
        int n = 0;
        foreach (var row in hidden)
        {
            Assert.Equal(OpenTail.Stingray.Audio.OmniVoice.OmniVoiceSemanticWeights.HiddenDim, row.Length);
            foreach (var v in row)
            {
                Assert.True(float.IsFinite(v));
                sum += v;
                sumSq += (double)v * v;
                n++;
            }
        }
        double mean = sum / n;
        double std = Math.Sqrt(Math.Max(0, sumSq / n - mean * mean));
        Console.Error.WriteLine($"[OmniVoiceSemantic] tokens={hidden.Length} mean={mean:F5} std={std:F5}");
        Assert.True(std > 1e-4, "semantic encoder output has collapsed to near-silence");
    }
}
