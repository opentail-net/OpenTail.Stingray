using OpenTail.Stingray.Audio.PersonaPlex;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>Real, live end-to-end test for the real DELAY-correct PersonaPlex generation path
/// (<see cref="PersonaPlexGenerator.GenerateDelayed"/>, driving the real
/// <see cref="PersonaPlexDelayState"/> ring-buffer state machine ported from `session.cpp`),
/// chained into the real Mimi codec decoder -- confirms the delay-staggered loop produces the
/// same real, finite, in-range waveform shape as <see cref="PersonaPlexFullPipelineRealWeightsTests"/>'s
/// simplified same-frame path, on real weights.</summary>
public sealed class PersonaPlexDelayedPipelineRealWeightsTests : HeavyTestBase
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
    public void GenerateDelayed_ThenDecode_OnRealCheckpoint_ProducesFiniteWaveform()
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

        var frames = PersonaPlexGenerator.GenerateDelayed(fwd, llm, depformer, numOutputFrames: 4, TextVocabSize, AudioCodebookSize);
        Assert.Equal(4, frames.Length);
        foreach (var frame in frames)
        {
            Assert.Equal(MimiCodecDecoderWeights.ActiveCodebooks, frame.AudioCodes.Length);
            Assert.All(frame.AudioCodes, c => Assert.InRange(c, 0, AudioCodebookSize - 1));
        }

        var mimiWeights = MimiCodecDecoderWeights.Load(source.GetTensor);
        var mimiCodes = frames.Select(f => f.AudioCodes).ToArray();
        var waveform = MimiCodecDecoder.Decode(mimiWeights, mimiCodes);

        Assert.True(waveform.Length > 0);
        Assert.All(waveform, v => Assert.True(float.IsFinite(v)));
        Assert.All(waveform, v => Assert.InRange(v, -1f, 1f));
        Assert.Contains(waveform, v => v != 0f);
    }
}
