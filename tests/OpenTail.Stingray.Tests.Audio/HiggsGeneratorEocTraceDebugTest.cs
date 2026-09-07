using OpenTail.Stingray.Audio.HiggsAudio;
using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>ONE-OFF DEBUG: traces raw per-step codebook-0 sampled ids to understand why
/// generation stops after only ~8 raw frames (real symptom found via
/// HiggsAudioTtsGenerateWavDebugTest -- an unnaturally short "whap" sound).</summary>
public sealed class HiggsGeneratorEocTraceDebugTest : HeavyTestBase
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
    public void Trace_RawCodebook0_AcrossSteps()
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
        var prompt = tokenizer.EncodePrompt("Hello there, this is a much longer real end to end test of speech synthesis. I am saying many more words now, to see whether the model keeps generating audio for a proportionally longer amount of time, or whether it stops almost immediately regardless of how long the input text actually is.", "", 0);
        Console.WriteLine($"Prompt length: {prompt.TokenIds.Length}, last 5 ids: {string.Join(",", prompt.TokenIds[^5..])}");
        fwd.Prefill(prompt.TokenIds);

        var sampler = new HiggsCodebookSampler(NumCodebooks);
        var first = HiggsArStepper.SampleFromHidden(fwd.LastHidden, llm, NumCodebooks, AudioVocabSize);
        Console.WriteLine($"step0 raw: [{string.Join(",", first)}]");
        var masked = sampler.Step(first);
        Console.WriteLine($"step0 masked: [{string.Join(",", masked)}] done={sampler.GenerationDone}");

        int position = prompt.TokenIds.Length;
        for (int step = 1; step < 60 && !sampler.GenerationDone; step++)
        {
            var raw = HiggsArStepper.Step(fwd, llm, sampler.LastCodes, position, NumCodebooks, AudioVocabSize);
            position++;
            Console.WriteLine($"step{step} raw: [{string.Join(",", raw)}]");
            var m = sampler.Step(raw);
            Console.WriteLine($"step{step} masked: [{string.Join(",", m)}] done={sampler.GenerationDone}");
        }
    }
}
