using OpenTail.Stingray.Audio.MossTts;
using OpenTail.Stingray.Audio.Rvc;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>TEMPORARY debug test: generates a real MOSS-TTS-Nano wav end-to-end for informal
/// listening (no CLI wiring yet). Writes to docs/audio-samples (gitignored, local-only per
/// CLAUDE.md). Not a golden/parity test.</summary>
public sealed class MossTtsGenerateWavDebugTest : HeavyTestBase
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

    private static byte[]? ExtractEmbeddedFile(GgufModel model, string fileName)
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

    [Fact]
    public void Generate_RealMossTtsWav()
    {
        string? path = FindRepoFile("models/_models/moss-tts-nano/MOSS-TTS-Nano-100M-GGUF/moss-tts-nano-100m-q8_0.gguf");
        Assert.SkipUnless(path != null, "moss-tts-nano checkpoint not found");
        string? repoRoot = Path.GetDirectoryName(FindRepoFile("docs/audio-review-progress.md"));
        Assert.NotNull(repoRoot);

        using var model = GgufModel.Open(path!);
        var tokenizer = SentencePieceBpeTokenizer.FromModelBytes(ExtractEmbeddedFile(model, "tokenizer.model")!);
        var source = new RvcPackedTensorSource(model);
        var g = new MossTtsGlobalTransformerWeights(source);
        var l = new MossTtsLocalTransformerWeights(source);
        var quantizer = new MossTtsAudioCodecQuantizerWeights(source);
        var decoderWeights = new MossTtsAudioCodecDecoderWeights(source);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var prompt = MossTtsPromptBuilder.BuildZeroShotPrompt(tokenizer, "Hello there, this is a test of speech synthesis.");
        // Real reference default (types.h's MossTTSNanoSamplingOptions): do_sample=true,
        // audio_temperature=1.7, audio_top_p=0.8, audio_top_k=25 -- never greedy.
        var samplingOptions = new SamplingParams { Temperature = 1.7f, TopP = 0.8f, TopK = 25 };
        var genSw = System.Diagnostics.Stopwatch.StartNew();
        var codes = MossTtsGenerator.Generate(g, l, prompt, activeCodebooks: MossTtsGlobalTransformerWeights.NumCodebooks, maxNewFrames: 40, samplingOptions, new Random(11));
        genSw.Stop();
        var dumpPath = Path.Combine(repoRoot!, "audio-samples", "moss-tts-csharp-40frames-codes.txt");
        File.WriteAllLines(dumpPath, codes.TokenIds.Select((id, idx) => $"Frame {idx / codes.Codebooks}, Codebook {idx % codes.Codebooks}: {id}"));

        var codesPerQuantizer = new int[MossTtsAudioCodecQuantizerWeights.NumQuantizers][];
        for (int q = 0; q < codesPerQuantizer.Length; q++)
        {
            var row = new int[codes.Frames];
            for (int f = 0; f < codes.Frames; f++) row[f] = codes.TokenIds[f * codes.Codebooks + q];
            codesPerQuantizer[q] = row;
        }

        var decSw = System.Diagnostics.Stopwatch.StartNew();
        var waveform = MossTtsAudioCodecDecoder.Decode(quantizer, decoderWeights, codesPerQuantizer);
        decSw.Stop();
        sw.Stop();
        Console.WriteLine($"[PERF METRICS] 40 frames (3.04s audio) => Generation: {genSw.ElapsedMilliseconds} ms, Codec Decode: {decSw.ElapsedMilliseconds} ms, Total E2E: {sw.ElapsedMilliseconds} ms (RTF: {sw.ElapsedMilliseconds / 3040.0:F2}x)");
        Assert.True(waveform.Left.Length > 0);

        float peak = 0f;
        double sumSq = 0;
        int zeroCrossings = 0;
        for (int i = 0; i < waveform.Left.Length; i++)
        {
            float s = waveform.Left[i];
            float abs = MathF.Abs(s);
            if (abs > peak) peak = abs;
            sumSq += s * s;
            if (i > 0 && ((waveform.Left[i - 1] >= 0 && s < 0) || (waveform.Left[i - 1] < 0 && s >= 0)))
                zeroCrossings++;
        }
        float rms = (float)Math.Sqrt(sumSq / waveform.Left.Length);
        float zcr = (float)zeroCrossings / waveform.Left.Length;

        var result = new OpenTail.Stingray.Audio.AudioGenerationResult(waveform.Left, MossTtsAudioCodecDecoderWeights.SamplingRate);
        string outPath = Path.Combine(repoRoot!, "audio-samples", "moss-tts-nano-real-check.wav");
        result.SaveWav(outPath);
        Console.WriteLine($"[AUDIO METRICS] Wrote {outPath}, {waveform.Left.Length} samples, {waveform.Left.Length / (double)MossTtsAudioCodecDecoderWeights.SamplingRate:F2}s, Peak: {peak:F4}, RMS: {rms:F4}, ZCR: {zcr:F4}");

        Assert.True(peak > 0.05f && peak <= 1.0f, $"Peak {peak} outside expected range");
        Assert.True(rms > 0.01f && rms < 0.5f, $"RMS {rms} outside expected range");
        Assert.True(zcr > 0.01f && zcr < 0.35f, $"ZCR {zcr} outside expected range");
    }
}
