using OpenTail.Stingray.Audio.PersonaPlex;
using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real, live end-to-end smoke test for PersonaPlex's real voice-id-conditioned bootstrap
/// (<see cref="PersonaPlexGenerator.GenerateWithVoicePrompt"/>): extracts one of the 18 real
/// embedded per-voice-id `.safetensors` assets, replays its real embeddings + delay-cache
/// snapshot, pads with real silence frames, then continues into the ordinary generation loop --
/// the first real voice-prompt-bootstrapped run for this model.
/// </summary>
public sealed class PersonaPlexVoicePromptRealWeightsTests : HeavyTestBase
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
    private const float MimiFrameRate = 12.5f; // real default, assets.h's PersonaPlexMimiConfig::frame_rate

    private static string ExtractVoicePrompt(GgufModel model)
    {
        if (!model.Metadata.TryGetValue("audiocpp.embedded_files.names", out var namesObj) || namesObj is not object[] names)
            throw new InvalidOperationException("PersonaPlex packed GGUF has no embedded_files metadata.");
        var offsets = (object[])model.Metadata["audiocpp.embedded_files.offsets"];
        var data = (object[])model.Metadata["audiocpp.embedded_files.data"];
        var bytes = data.Select(o => (byte)Convert.ToInt64(o)).ToArray();

        const string wanted = "voices_safetensors/NATF0.safetensors";
        for (int i = 0; i < names.Length; i++)
        {
            if ((string)names[i] != wanted) continue;
            long start = Convert.ToInt64(offsets[i]);
            long end = i + 1 < offsets.Length ? Convert.ToInt64(offsets[i + 1]) : bytes.Length;
            string path = Path.Combine(Path.GetTempPath(), "stingray-personaplex-voice-natf0.safetensors");
            File.WriteAllBytes(path, bytes[(int)start..(int)end]);
            return path;
        }
        throw new InvalidOperationException($"PersonaPlex packed GGUF is missing embedded file '{wanted}'.");
    }

    [Fact]
    public void GenerateWithVoicePrompt_OnRealCheckpoint_ProducesFiniteWaveform()
    {
        string? path = FindRepoFile("models/_models/personaplex/PersonaPlex-GGUF/personaplex-7b-v1-q8_0.gguf");
        Assert.SkipUnless(path != null, "personaplex-7b-v1-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);

        string voicePath = ExtractVoicePrompt(model);
        var voicePrompt = PersonaPlexVoicePrompt.Load(voicePath, HiddenDim);
        Assert.True(voicePrompt.Frames > 0);
        Assert.Equal(PersonaPlexDelayState.NumStreams * PersonaPlexDelayState.DelayCacheSteps, voicePrompt.Cache.Length);

        using var llm = new PersonaPlexLmTensorSource(source, NumLayers, HiddenDim, NumHeads, HeadDim, FfDim, TextVocabSize, LmCodebooks, AudioCodebookSize, RopeTheta, RmsNormEps);
        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        var depformerWeights = PersonaPlexDepformerWeights.Load(TextVocabSize, AudioCodebookSize, source.GetTensor);
        var depformer = new PersonaPlexDepformer(depformerWeights);

        var frames = PersonaPlexGenerator.GenerateWithVoicePrompt(
            fwd, llm, depformer, voicePrompt, MimiFrameRate, systemPrompt: "",
            numOutputFrames: 4, TextVocabSize, AudioCodebookSize);

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
