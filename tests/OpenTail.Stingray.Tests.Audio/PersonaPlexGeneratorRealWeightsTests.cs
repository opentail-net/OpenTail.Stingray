using OpenTail.Stingray.Audio.PersonaPlex;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real, live end-to-end smoke test wiring PersonaPlex's temporal LM to its Depformer:
/// each frame's temporal-LM step conditions the Depformer's 16-codebook generation, and the
/// Depformer's output feeds back into the NEXT frame's temporal-LM step -- chaining both real
/// autoregressive decoders (each independently real-weight verified earlier this session) into
/// one live run.</summary>
public sealed class PersonaPlexGeneratorRealWeightsTests : HeavyTestBase
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

    private const int NumLayers = 32, HiddenDim = 4096, NumHeads = 32, HeadDim = 128;
    private const int FfDim = 11264, TextVocabSize = 32000, LmCodebooks = 16, AudioCodebookSize = 2048;
    private const float RopeTheta = 10000f, RmsNormEps = 1e-8f;

    [Fact]
    public void Generate_OnRealCheckpoint_ProducesInRangeFrames()
    {
        string? path = FindRepoFile("models/_models/personaplex/PersonaPlex-GGUF/personaplex-7b-v1-q8_0.gguf");
        Assert.SkipUnless(path != null, "personaplex-7b-v1-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);

        using var llm = new PersonaPlexLmTensorSource(source, NumLayers, HiddenDim, NumHeads, HeadDim, FfDim, TextVocabSize, LmCodebooks, AudioCodebookSize, RopeTheta, RmsNormEps);
        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        var depformerWeights = PersonaPlexDepformerWeights.Load(TextVocabSize, AudioCodebookSize, source.GetTensor);
        var depformer = new PersonaPlexDepformer(depformerWeights);

        var frames = PersonaPlexGenerator.Generate(fwd, llm, depformer, numFrames: 3, TextVocabSize, AudioCodebookSize);

        Assert.Equal(3, frames.Length);
        foreach (var frame in frames)
        {
            Assert.InRange(frame.TextToken, 0, TextVocabSize - 1);
            Assert.Equal(LmCodebooks, frame.AudioCodes.Length);
            Assert.All(frame.AudioCodes, c => Assert.InRange(c, 0, AudioCodebookSize - 1));
        }
    }
}
