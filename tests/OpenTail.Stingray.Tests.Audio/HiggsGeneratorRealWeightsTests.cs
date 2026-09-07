using OpenTail.Stingray.Audio.HiggsAudio;
using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real, live end-to-end smoke test for Higgs Audio TTS's full zero-shot text-to-waveform
/// generation loop (<see cref="HiggsGenerator"/>) -- the first real full-generation run for this
/// model, chaining prompt building, prefill, the real AR decode loop (delay-masked sampling, real
/// EOC-driven stop state machine), the real delay-pattern reverse, and real codec decode.
/// </summary>
public sealed class HiggsGeneratorRealWeightsTests : HeavyTestBase
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

    private const int NumLayers = 36, HiddenDim = 2560, NumHeads = 32, NumKvHeads = 8, HeadDim = 128;
    private const int FfDim = 9728, VocabSize = 151936, NumCodebooks = 8, AudioVocabSize = 1026;
    private const float RopeTheta = 1_000_000f, RmsNormEps = 1e-6f;

    [Fact]
    public void Generate_OnRealCheckpoint_ProducesFiniteNonSilentAudio()
    {
        string? path = FindRepoFile("models/_models/higgs_audio_tts/Higgs-Audio-v3-TTS-4B-GGUF/higgs-audio-v3-tts-4b-q8_0.gguf");
        Assert.SkipUnless(path != null, "higgs-audio-v3-tts-4b-q8_0.gguf not found");

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        using var llm = new HiggsLlmTensorSource(source, NumLayers, HiddenDim, NumHeads, NumKvHeads, HeadDim, FfDim, VocabSize, RopeTheta, RmsNormEps, NumCodebooks, AudioVocabSize);

        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        var tokenizer = HiggsTtsTextTokenizer.LoadFromPackedGguf(model, audioTokenId: -100);
        var codecWeights = HiggsCodecDecoderWeights.Load(source.GetTensor);

        var result = HiggsGenerator.Generate(
            fwd, llm, tokenizer, codecWeights,
            "Hi.", NumCodebooks, AudioVocabSize,
            maxTokens: 200);

        Assert.NotEmpty(result.RawCodes);
        Assert.All(result.RawCodes, frame =>
        {
            Assert.Equal(NumCodebooks, frame.Length);
            Assert.All(frame, c => Assert.InRange(c, 0, HiggsCodecDecoderWeights.CodebookSize - 1));
        });
        Assert.NotEmpty(result.AudioSamples);
        Assert.All(result.AudioSamples, v => Assert.True(float.IsFinite(v)));
        Assert.Contains(result.AudioSamples, v => v != 0f);
    }
}
