using OpenTail.Stingray.Audio.OmniVoice;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real, live end-to-end smoke test for OmniVoice's own MaskGIT generation loop
/// (<see cref="OmniVoiceMaskGitGenerator"/>) -- the first real run of this model's own audio
/// generation algorithm (previously entirely unimplemented, structurally verified against real
/// weights per this session's established convention: finite/in-range output, not a full
/// tokenizer-driven production pipeline).
/// </summary>
public sealed class OmniVoiceMaskGitGeneratorRealWeightsTests : HeavyTestBase
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
    public void Generate_OnRealCheckpoint_ProducesInRangeCodes()
    {
        string? path = FindRepoFile("models/_models/omnivoice/model.safetensors");
        Assert.SkipUnless(path != null, "omnivoice model.safetensors not found");

        using var loader = OpenTail.Stingray.Core.SafetensorsLoader.Open(path!);
        var weights = OmniVoiceMaskGitWeights.Load(loader.ReadF32);

        // Small synthetic real-vocab-range token ids -- structural verification (this session's
        // established first-pass convention), not a real tokenizer-driven prompt.
        int[] styleTokenIds = [1, 2];
        int[] textTokenIds = [10, 11, 12, 13, 14];
        const int targetFrames = 4;

        var codes = OmniVoiceMaskGitGenerator.Generate(
            weights, styleTokenIds, textTokenIds, referenceAudioTokens: null,
            targetFrames, new OmniVoiceMaskGitGenerator.Options(numInferenceSteps: 8), new Random(13));

        Assert.Equal(targetFrames * OmniVoiceMaskGitWeights.NumCodebooks, codes.Length);
        Assert.All(codes, c => Assert.InRange(c, 0, OmniVoiceMaskGitWeights.AudioVocabSize - 2)); // excludes the real mask id
    }
}
