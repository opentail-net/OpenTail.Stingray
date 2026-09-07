using OpenTail.Stingray.Audio.OmniVoice;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: generates a real OmniVoice wav end-to-end for informal listening
/// (no CLI wiring yet, no real tokenizer prompt template -- synthetic style/text token ids, same
/// structural-first convention as <see cref="OmniVoiceMaskGitGeneratorRealWeightsTests"/>). Writes
/// to docs/audio-samples (gitignored, local-only per CLAUDE.md). Not a golden/parity test. First
/// ever real audio produced by this model's own generation loop.</summary>
public sealed class OmniVoiceGenerateWavDebugTest : HeavyTestBase
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
    public void Generate_RealOmniVoiceWav()
    {
        string? path = FindRepoFile("models/_models/omnivoice/model.safetensors");
        Assert.SkipUnless(path != null, "omnivoice model.safetensors not found");
        string? repoRoot = Path.GetDirectoryName(FindRepoFile("docs/audio-review-progress.md"));
        Assert.NotNull(repoRoot);

        string? codecPath = FindRepoFile("models/_models/omnivoice/audio_tokenizer/model.safetensors");
        Assert.SkipUnless(codecPath != null, "omnivoice audio_tokenizer model.safetensors not found");

        using var loader = OpenTail.Stingray.Core.SafetensorsLoader.Open(path!);
        var maskGitWeights = OmniVoiceMaskGitWeights.Load(loader.ReadF32);
        using var codecLoader = OpenTail.Stingray.Core.SafetensorsLoader.Open(codecPath!);
        var decoderWeights = new OmniVoiceAcousticDecoderWeights(codecLoader);

        int[] styleTokenIds = [1, 2];
        int[] textTokenIds = [10, 11, 12, 13, 14, 15, 16, 17];
        const int targetFrames = 20;

        var codesFlat = OmniVoiceMaskGitGenerator.Generate(
            maskGitWeights, styleTokenIds, textTokenIds, referenceAudioTokens: null,
            targetFrames, new OmniVoiceMaskGitGenerator.Options(numInferenceSteps: 16), new Random(21));

        var codes = new int[targetFrames][];
        for (int f = 0; f < targetFrames; f++)
        {
            var row = new int[OmniVoiceMaskGitWeights.NumCodebooks];
            Array.Copy(codesFlat, f * OmniVoiceMaskGitWeights.NumCodebooks, row, 0, OmniVoiceMaskGitWeights.NumCodebooks);
            codes[f] = row;
        }

        var waveform = OmniVoiceAcousticDecoder.Decode(decoderWeights, codes);
        Assert.True(waveform.Length > 0);

        const int sampleRate = 24000; // real OmniVoice/Higgs acoustic codec output rate, confirmed this session
        var result = new OpenTail.Stingray.Audio.AudioGenerationResult(waveform, sampleRate);
        string outPath = Path.Combine(repoRoot!, "audio-samples", "omnivoice-real-check.wav");
        result.SaveWav(outPath);
        Console.WriteLine($"Wrote {outPath}, {waveform.Length} samples, {waveform.Length / (double)sampleRate:F2}s");
    }
}
