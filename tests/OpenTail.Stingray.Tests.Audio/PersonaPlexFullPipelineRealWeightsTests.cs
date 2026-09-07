using OpenTail.Stingray.Audio.PersonaPlex;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real, live end-to-end smoke test chaining EVERY PersonaPlex piece real-weight
/// verified this session: the temporal LM generates frames (each conditioning a Depformer
/// 16-codebook step), and the first 8 of each frame's 16 codes are fed into the real Mimi codec
/// decoder to produce actual audio -- the first full text-to-waveform run for this model.</summary>
public sealed class PersonaPlexFullPipelineRealWeightsTests : HeavyTestBase
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
    public void Generate_ThenDecode_OnRealCheckpoint_ProducesFiniteWaveform()
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

        var frames = PersonaPlexGenerator.Generate(fwd, llm, depformer, numFrames: 4, TextVocabSize, AudioCodebookSize);
        Assert.Equal(4, frames.Length);

        // Real mapping of PersonaPlex's 16 lm_codebooks to Mimi's 8 active codebooks is not
        // independently confirmed this session (see the temporal-LM+Depformer wiring entry's own
        // flagged "two 8-codebook streams?" open question) -- using the first 8 as a documented,
        // reasonable assumption for this structural smoke test, and clamping PersonaPlex's
        // audioCodebookSize=2048 range directly onto Mimi's SAME real codebookSize=2048 (an exact
        // match, not a coincidence -- both real configs share this value).
        Assert.Equal(MimiCodecDecoderWeights.CodebookSize, AudioCodebookSize);
        var mimiCodes = new int[frames.Length][];
        for (int t = 0; t < frames.Length; t++)
        {
            mimiCodes[t] = new int[MimiCodecDecoderWeights.ActiveCodebooks];
            Array.Copy(frames[t].AudioCodes, mimiCodes[t], MimiCodecDecoderWeights.ActiveCodebooks);
        }

        var mimiWeights = MimiCodecDecoderWeights.Load(source.GetTensor);
        var waveform = MimiCodecDecoder.Decode(mimiWeights, mimiCodes);

        Assert.True(waveform.Length > 0);
        Assert.All(waveform, v => Assert.True(float.IsFinite(v)));
        Assert.All(waveform, v => Assert.InRange(v, -1f, 1f));
        Assert.Contains(waveform, v => v != 0f);
    }
}
