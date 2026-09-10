using OpenTail.Stingray.Audio.HiggsAudio;
using OpenTail.Stingray.Audio.Rvc;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// TEMPORARY, throwaway perf timing bench for Higgs Audio TTS (4B) for PerformanceLeague.md
/// backfill, using the standard "Hello, I will make some lunch, darling!" prompt. Not part of the
/// permanent suite, delete after use.
/// </summary>
public sealed class HiggsAudioPerfBaselineDebugTest : HeavyTestBase
{
    private const int NumLayers = 36, HiddenDim = 2560, NumHeads = 32, NumKvHeads = 8, HeadDim = 128;
    private const int FfDim = 9728, VocabSize = 151936, NumCodebooks = 8, AudioVocabSize = 1026;
    private const float RopeTheta = 1_000_000f, RmsNormEps = 1e-6f;
    private const int Runs = 3;

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
    public void Bench_Generate()
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

        const string prompt = "Hello, I will make some lunch, darling!";

        var warm = HiggsGenerator.Generate(fwd, llm, tokenizer, codecWeights, prompt, NumCodebooks, AudioVocabSize, maxTokens: 200);
        Assert.NotEmpty(warm.AudioSamples);

        double[] elapsedSec = new double[Runs];
        int sampleCount = 0;
        for (int i = 0; i < Runs; i++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var result = HiggsGenerator.Generate(fwd, llm, tokenizer, codecWeights, prompt, NumCodebooks, AudioVocabSize, maxTokens: 200);
            sw.Stop();
            elapsedSec[i] = sw.Elapsed.TotalSeconds;
            sampleCount = result.AudioSamples.Length;
        }

        const int sampleRate = 24000; // HiggsCodecDecoder native sample rate
        double audioSec = sampleCount / (double)sampleRate;
        double meanSec = elapsedSec.Average();
        double rtf = audioSec > 0 ? meanSec / audioSec : 0;
        string msg = $"[HiggsAudio-TTS] prompt=\"{prompt}\" audio={audioSec:F2}s samples={sampleCount}\n" +
                     $"[HiggsAudio-TTS] runs(s)=[{string.Join(", ", elapsedSec.Select(x => x.ToString("F3")))}] mean={meanSec:F3}s RTF={rtf:F3} (lower=faster; 1.0=realtime)";
        Console.Error.WriteLine(msg);
        File.AppendAllText(Path.Combine(FindRepoFile("docs") ?? ".", "tts-benchmark-log.txt"), msg + "\n\n");
    }
}
