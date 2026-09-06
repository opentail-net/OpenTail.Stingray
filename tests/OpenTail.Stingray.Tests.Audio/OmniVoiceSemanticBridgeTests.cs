
namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real end-to-end test: semantic HuBERT encoder -> semantic bridge, on real audio and
/// real checkpoint weights.</summary>
public sealed class OmniVoiceSemanticBridgeTests : HeavyTestBase
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
    public void Load_And_Forward_RealWeights_RealAudio_ProducesFiniteNonDegenerateOutput()
    {
        string? checkpointPath = FindRepoFile("models/_models/omnivoice/audio_tokenizer/model.safetensors");
        Assert.SkipUnless(checkpointPath != null, "omnivoice audio_tokenizer model.safetensors not found");
        string? audioPath = FindRepoFile("examples/audio.cpp/assets/resources/a.wav");
        Assert.SkipUnless(audioPath != null, "reference a.wav not found");

        using var loader = OpenTail.Stingray.Core.SafetensorsLoader.Open(checkpointPath!);
        var semanticWeights = new OpenTail.Stingray.Audio.OmniVoice.OmniVoiceSemanticWeights(loader);
        var bridgeWeights = new OpenTail.Stingray.Audio.OmniVoice.OmniVoiceSemanticBridgeWeights(loader);

        var (samples, sr, _) = WavReader.ReadWav(audioPath!);
        if (sr != 16000) samples = AudioResampler.Resample(samples, sr, 16000);

        var hidden = OpenTail.Stingray.Audio.OmniVoice.OmniVoiceSemanticEncoder.Forward(semanticWeights, samples);
        int frames = hidden.Length;
        int dim = OpenTail.Stingray.Audio.OmniVoice.OmniVoiceSemanticWeights.HiddenDim;
        var channelMajor = new float[dim * frames];
        for (int t = 0; t < frames; t++)
            for (int c = 0; c < dim; c++)
                channelMajor[c * frames + t] = hidden[t][c];

        var bridged = OpenTail.Stingray.Audio.OmniVoice.OmniVoiceSemanticBridge.Forward(bridgeWeights, channelMajor, frames);

        Assert.Equal(dim * frames, bridged.Length);
        double sum = 0, sumSq = 0;
        foreach (var v in bridged)
        {
            Assert.True(float.IsFinite(v));
            sum += v;
            sumSq += (double)v * v;
        }
        double mean = sum / bridged.Length;
        double std = Math.Sqrt(Math.Max(0, sumSq / bridged.Length - mean * mean));
        Console.Error.WriteLine($"[OmniVoiceBridge] frames={frames} mean={mean:F5} std={std:F5}");
        Assert.True(std > 1e-4, "bridge output has collapsed to near-silence");
    }
}
