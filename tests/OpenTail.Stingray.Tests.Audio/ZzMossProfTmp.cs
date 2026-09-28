using OpenTail.Stingray.Audio.MossTts;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

// Scratch harness (untracked): MOSS-TTS-Nano multi-seed loudness stats. Env ZZ_MOSS=1, ZZ_OUT dir.
public sealed class ZzMossProfTmp
{
    private readonly ITestOutputHelper _out;
    public ZzMossProfTmp(ITestOutputHelper o) => _out = o;
    private void Log(string s) { _out.WriteLine(s); Console.WriteLine(s); }

    [Fact]
    public void MossMultiSeed()
    {
        if (Environment.GetEnvironmentVariable("ZZ_MOSS") != "1") return;
        string outDir = Environment.GetEnvironmentVariable("ZZ_OUT") ?? Path.GetTempPath();
        string text = Environment.GetEnvironmentVariable("ZZ_TEXT") ?? "Hello there, this is a test of speech synthesis.";
        using var model = GgufModel.Open(@"C:\Git-Public\OpenTail.Stingray\models\_models\moss-tts-nano\MOSS-TTS-Nano-100M-GGUF\moss-tts-nano-100m-q8_0.gguf");
        var tokBytes = MossTtsGenerateWavDebugTest_Extract(model, "tokenizer.model")!;
        var tokenizer = SentencePieceBpeTokenizer.FromModelBytes(tokBytes);
        var source = new OpenTail.Stingray.Audio.Rvc.RvcPackedTensorSource(model);
        var g = new MossTtsGlobalTransformerWeights(source);
        var l = new MossTtsLocalTransformerWeights(source);
        var quantizer = new MossTtsAudioCodecQuantizerWeights(source);
        var dec = new MossTtsAudioCodecDecoderWeights(source);
        var prompt = MossTtsPromptBuilder.BuildZeroShotPrompt(tokenizer, text);
        var opts = new SamplingParams { Temperature = 1.7f, TopP = 0.8f, TopK = 25 };
        foreach (int seed in new[] { 1, 2, 3, 4, 5, 6 })
        {
            var codes = MossTtsGenerator.Generate(g, l, prompt, activeCodebooks: MossTtsGlobalTransformerWeights.NumCodebooks, maxNewFrames: 120, opts, new Random(seed));
            var perQ = new int[MossTtsAudioCodecQuantizerWeights.NumQuantizers][];
            for (int q = 0; q < perQ.Length; q++) { perQ[q] = new int[codes.Frames]; for (int f = 0; f < codes.Frames; f++) perQ[q][f] = codes.TokenIds[f * codes.Codebooks + q]; }
            var wav = MossTtsAudioCodecDecoder.Decode(quantizer, dec, perQ);
            double ss = 0; foreach (var v in wav.Left) ss += v * v;
            Log($"[ZZ moss ours] seed {seed}: frames {codes.Frames}, {wav.Left.Length / (double)MossTtsAudioCodecDecoderWeights.SamplingRate:F2}s, RMS {Math.Sqrt(ss / wav.Left.Length):F4}");
            new OpenTail.Stingray.Audio.AudioGenerationResult(wav.Left, MossTtsAudioCodecDecoderWeights.SamplingRate).SaveWav(Path.Combine(outDir, $"moss_ours_s{seed}.wav"));
        }
    }

    private static byte[]? MossTtsGenerateWavDebugTest_Extract(GgufModel model, string fileName)
    {
        if (!model.Metadata.TryGetValue("audiocpp.embedded_files.names", out var namesObj) || namesObj is not object[] names) return null;
        var offsets = (object[])model.Metadata["audiocpp.embedded_files.offsets"];
        var data = (object[])model.Metadata["audiocpp.embedded_files.data"];
        var bytes = data.Select(o => (byte)Convert.ToInt64(o)).ToArray();
        for (int i = 0; i < names.Length; i++)
        {
            if ((string)names[i] != fileName) continue;
            long start = Convert.ToInt64(offsets[i]);
            long end = i + 1 < offsets.Length ? Convert.ToInt64(offsets[i + 1]) : bytes.Length;
            return bytes[(int)start..(int)end];
        }
        return null;
    }
}
