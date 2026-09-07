using OpenTail.Stingray.Audio.HiggsAudio;
using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: generates a real Higgs Audio TTS wav end-to-end for informal
/// listening (no CLI wiring yet). Writes to docs/audio-samples (gitignored, local-only per
/// CLAUDE.md). Not a golden/parity test.</summary>
public sealed class HiggsAudioTtsGenerateWavDebugTest : HeavyTestBase
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
    public void Generate_RealHiggsAudioTtsWav()
    {
        string? path = FindRepoFile("models/_models/higgs_audio_tts/Higgs-Audio-v3-TTS-4B-GGUF/higgs-audio-v3-tts-4b-q8_0.gguf");
        Assert.SkipUnless(path != null, "higgs-audio-v3-tts-4b-q8_0.gguf not found");
        string? repoRoot = Path.GetDirectoryName(FindRepoFile("docs/audio-review-progress.md"));
        Assert.NotNull(repoRoot);

        using var model = GgufModel.Open(path!);
        var source = new RvcPackedTensorSource(model);
        using var llm = new HiggsLlmTensorSource(source, NumLayers, HiddenDim, NumHeads, NumKvHeads, HeadDim, FfDim, VocabSize, RopeTheta, RmsNormEps, NumCodebooks, AudioVocabSize);

        var hp = ModelHyperparams.FromGgufMetadata(llm.Metadata);
        using var backend = new CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);

        var tokenizer = HiggsTtsTextTokenizer.LoadFromPackedGguf(model, audioTokenId: -100);
        var codecWeights = HiggsCodecDecoderWeights.Load(source.GetTensor);

        // Real, confirmed finding (HiggsGeneratorEocTraceDebugTest): this model's real EOC
        // stopping is genuinely TEXT-LENGTH-DEPENDENT -- a short one-sentence prompt causes real
        // early EOC (a very short clip, not a bug), while a longer prompt keeps generating for
        // many more real steps. Use a longer prompt here so the sample is actually listenable.
        var options = new SamplingParams { Temperature = 0.7f, TopK = 50, TopP = 0.95f };
        var result = HiggsGenerator.Generate(
            fwd, llm, tokenizer, codecWeights,
            "Hello there, this is a real end to end test of the Higgs Audio text to speech system. " +
            "I am speaking several sentences in a row so that the generated sample is long enough to " +
            "actually listen to, rather than stopping after only a fraction of a second.",
            NumCodebooks, AudioVocabSize, maxTokens: 800, options, new Random(7));

        Console.WriteLine($"Generated {result.RawCodes.Length} raw frames");
        Assert.True(result.AudioSamples.Length > 0);

        var wavResult = new OpenTail.Stingray.Audio.AudioGenerationResult(result.AudioSamples, HiggsCodecDecoderWeights.OutputSampleRate);
        string outPath = Path.Combine(repoRoot!, "audio-samples", "higgs-audio-tts-real-check.wav");
        wavResult.SaveWav(outPath);
        Console.WriteLine($"Wrote {outPath}, {result.AudioSamples.Length} samples, {result.AudioSamples.Length / (double)HiggsCodecDecoderWeights.OutputSampleRate:F2}s");
    }
}
