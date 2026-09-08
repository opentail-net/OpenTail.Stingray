using OpenTail.Stingray.Audio.MarbleNet;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real, first-attempt GOLDEN-PARITY test for the newly-ported MarbleNet VAD (a small NeMo
/// "Jasper"-family depthwise-separable CNN voice-activity detector), against the checkpoint
/// bundled directly in this repo (`examples/audio.cpp/assets/framework/models/marblenet_vad/`,
/// 466KB -- no download needed) on a real LibriSpeech speech clip. Verified against the vendored
/// reference CLI's own real output on the identical file (`audiocpp_cli --task vad --family
/// marblenet_vad`): `{"start_sample":8320,"end_sample":56320,"confidence":0.977519}` -- this
/// port's first real attempt matched exactly (same sample bounds, confidence to 3 decimal places),
/// no debugging needed.
/// </summary>
public sealed class MarbleNetVadRealWeightsTests
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
    public void DetectSpeech_OnRealLibriSpeechClip_FindsASpeechSegment()
    {
        string? checkpointPath = FindRepoFile("examples/audio.cpp/assets/framework/models/marblenet_vad/marblenet_vad.safetensors");
        Assert.SkipUnless(checkpointPath != null, "marblenet_vad.safetensors not found");
        string? wavPath = FindRepoFile("examples/audio.cpp/assets/asr_validation/librispeech/librispeech_test_clean_6930-75918-0000.wav");
        Assert.SkipUnless(wavPath != null, "librispeech reference clip not found");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var loader = SafetensorsLoader.Open(checkpointPath!);
        var weights = MarbleNetVadWeights.Load(loader);

        var (rawSamples, rawRate, rawChannels) = WavReader.ReadWav(wavPath!);
        Assert.Equal(1, rawChannels);
        var waveform = rawRate == MarbleNetVadWeights.SampleRate
            ? rawSamples
            : AudioResampler.Resample(rawSamples, rawRate, MarbleNetVadWeights.SampleRate, channels: 1, ResampleQuality.BestQuality);

        var extractor = new MarbleNetVadMelExtractor(weights);
        var melFlat = extractor.ExtractMel(waveform);
        int frames = melFlat.Length / MarbleNetVadWeights.NMels;
        Assert.True(frames > 0);

        var melChannelMajor = new float[MarbleNetVadWeights.NMels][];
        for (int m = 0; m < MarbleNetVadWeights.NMels; m++)
        {
            var row = new float[frames];
            for (int f = 0; f < frames; f++) row[f] = melFlat[f * MarbleNetVadWeights.NMels + m];
            melChannelMajor[m] = row;
        }

        var logits = MarbleNetVad.Forward(weights, melChannelMajor);
        Assert.All(logits, row => Assert.All(row, v => Assert.True(float.IsFinite(v))));

        var segments = MarbleNetVad.DecodeSegments(weights, logits, rawRate);
        sw.Stop();
        Console.WriteLine($"[MarbleNetVad] {sw.ElapsedMilliseconds}ms, {logits.Length} output frames, {segments.Count} segment(s):");
        foreach (var seg in segments)
            Console.WriteLine($"  [{seg.StartSample}-{seg.EndSample}] conf={seg.Confidence:F3} ({(seg.EndSample - seg.StartSample) / (double)rawRate:F2}s)");

        // Real golden-parity check: the vendored reference CLI's own output on this identical file
        // (`audiocpp_cli --task vad --family marblenet_vad --model marblenet_vad.safetensors
        // --audio <this clip>`) is `{"start_sample":8320,"end_sample":56320,"confidence":0.977519}`.
        var segment = Assert.Single(segments);
        Assert.Equal(8320, segment.StartSample);
        Assert.Equal(56320, segment.EndSample);
        Assert.Equal(0.977519f, segment.Confidence, precision: 3);
    }
}
