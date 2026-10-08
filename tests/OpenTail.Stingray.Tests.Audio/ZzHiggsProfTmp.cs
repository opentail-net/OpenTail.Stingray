using OpenTail.Stingray.Audio.HiggsAudio;
using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Audio;

// Scratch harness (untracked): Higgs Audio TTS multi-seed samples. Env ZZ_HIGGS=1, ZZ_OUT, ZZ_TEXT.
public sealed class ZzHiggsProfTmp
{
    private const int NumLayers = 36, HiddenDim = 2560, NumHeads = 32, NumKvHeads = 8, HeadDim = 128;
    private const int FfDim = 9728, VocabSize = 151936, NumCodebooks = 8, AudioVocabSize = 1026;
    private const float RopeTheta = 1_000_000f, RmsNormEps = 1e-6f;

    private readonly ITestOutputHelper _out;
    public ZzHiggsProfTmp(ITestOutputHelper o) => _out = o;
    private void Log(string s) { _out.WriteLine(s); Console.WriteLine(s); }

    [Fact]
    public void HiggsMultiSeed()
    {
        if (Environment.GetEnvironmentVariable("ZZ_HIGGS") != "1") return;
        string outDir = Environment.GetEnvironmentVariable("ZZ_OUT") ?? Path.GetTempPath();
        string text = Environment.GetEnvironmentVariable("ZZ_TEXT") ?? "Hello there, this is a test of speech synthesis.";
        using var model = GgufModel.Open(@"C:\Git-Public\OpenTail.Stingray\models\_models\higgs_audio_tts\Higgs-Audio-v3-TTS-4B-GGUF\higgs-audio-v3-tts-4b-q8_0.gguf");
        var source = new RvcPackedTensorSource(model);
        using var llm = new HiggsLlmTensorSource(source, NumLayers, HiddenDim, NumHeads, NumKvHeads, HeadDim, FfDim, VocabSize, RopeTheta, RmsNormEps, NumCodebooks, AudioVocabSize);
        var hp = OpenTail.Stingray.Engine.ArchitectureModelResolver.ResolveHyperparams(llm.Metadata);
        using var backend = new OpenTail.Stingray.Cpu.CpuBackend();
        using var fwd = new ForwardPass(llm, backend, hp);
        var tokenizer = HiggsTtsTextTokenizer.LoadFromPackedGguf(model, audioTokenId: -100);
        var codec = HiggsCodecDecoderWeights.Load(source.GetTensor);
        var options = new SamplingParams { Temperature = 0.7f, TopK = 50, TopP = 0.95f };
        foreach (int seed in new[] { 1, 2, 3, 4 })
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var r = HiggsGenerator.Generate(fwd, llm, tokenizer, codec, text, NumCodebooks, AudioVocabSize, maxTokens: 400, options, new Random(seed));
            double ss = 0; foreach (var v in r.AudioSamples) ss += v * v;
            double secs = r.AudioSamples.Length / (double)HiggsCodecDecoderWeights.OutputSampleRate;
            Log($"[ZZ higgs ours] seed {seed}: {r.RawCodes.Length} frames, {secs:F2}s audio in {sw.Elapsed.TotalSeconds:F1}s, RMS {Math.Sqrt(ss / Math.Max(1, r.AudioSamples.Length)):F4}");
            new OpenTail.Stingray.Audio.AudioGenerationResult(r.AudioSamples, HiggsCodecDecoderWeights.OutputSampleRate).SaveWav(Path.Combine(outDir, $"higgs_ours_s{seed}.wav"));
        }
    }
}
