using OpenTail.Stingray.Audio.PersonaPlex;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real-weight smoke test for <see cref="PersonaPlexDepformer"/>.</summary>
public sealed class PersonaPlexDepformerRealWeightsTests : HeavyTestBase
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

    private const int TextVocabSize = 32000, AudioCodebookSize = 2048;

    [Fact]
    public void GenerateFrame_OnRealCheckpoint_ProducesInRangeCodes()
    {
        string? path = FindRepoFile("models/_models/personaplex/PersonaPlex-GGUF/personaplex-7b-v1-q8_0.gguf");
        Assert.SkipUnless(path != null, "personaplex-7b-v1-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        var weights = PersonaPlexDepformerWeights.Load(TextVocabSize, AudioCodebookSize, source.GetTensor);
        var depformer = new PersonaPlexDepformer(weights);

        var rng = new Random(41);
        var temporalLmHidden = Enumerable.Range(0, PersonaPlexDepformerWeights.LmHiddenDim)
            .Select(_ => (float)(rng.NextDouble() * 0.2 - 0.1)).ToArray();

        var codes = depformer.GenerateFrame(temporalLmHidden, promptTextToken: 100, AudioCodebookSize);

        Assert.Equal(PersonaPlexDepformerWeights.NumSteps, codes.Length);
        Assert.All(codes, c => Assert.InRange(c, 0, AudioCodebookSize - 1));
    }
}
