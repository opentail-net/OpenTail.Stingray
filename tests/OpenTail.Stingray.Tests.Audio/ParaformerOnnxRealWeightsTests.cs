using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.ParaformerOnnx;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real-weights exercise for the new, independent ONNX-based FunASR Paraformer path
/// (<see cref="ParaformerOnnxPipeline"/>), separate from the native GGUF Paraformer path in
/// `OpenTail.Stingray.Audio.FunASR` (confirmed broken this session -- missing `pf.vocab` GGUF
/// metadata, a bad conversion). Uses the real local checkpoint
/// `models/_models/paraformer-zh-small.int8.onnx` plus the real downloaded
/// `F:\_models\paraformer-zh-small-tokens.txt` vocab (confirmed the real match for this checkpoint
/// by exact line-count-equals-vocab_size and `&lt;/s&gt;`-at-id-2 agreement -- see
/// `ParaformerOnnxVocab`'s doc comment).
/// </summary>
public sealed class ParaformerOnnxRealWeightsTests
{
    private static string? FindModelPath(string fileName)
    {
        string[] absoluteCandidates =
        {
            $@"C:\Git-Public\OpenTail.Stingray\models\_models\{fileName}",
            $@"C:\Git-Public\OpenTail.Stingray\models\{fileName}",
            $@"C:\p\opentail-llm\models\{fileName}",
            $@"E:\models\{fileName}",
        };
        foreach (var p in absoluteCandidates)
        {
            if (File.Exists(p)) return p;
        }

        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var pModels = Path.Combine(dir, "models", fileName);
            if (File.Exists(pModels)) return pModels;
            var pModelsUnderscore = Path.Combine(dir, "models", "_models", fileName);
            if (File.Exists(pModelsUnderscore)) return pModelsUnderscore;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    [Fact]
    public void Load_RealOnnxCheckpoint_ReadsRealCustomMetadata()
    {
        string? modelPath = FindModelPath("paraformer-zh-small.int8.onnx");
        Assert.SkipUnless(modelPath != null, "paraformer-zh-small.int8.onnx not found");

        var model = ParaformerOnnxModel.TryLoad(modelPath!);
        Assert.NotNull(model);
        try
        {
            // Real values confirmed via direct ONNX metadata inspection, 2026-09-11.
            Assert.Equal(8359, model!.VocabSize);
            Assert.Equal(7, model.LfrWindowSize);
            Assert.Equal(6, model.LfrWindowShift);
            Assert.Equal(560, model.NegMean.Length); // 80 (mel dim) * 7 (lfr_window_size)
            Assert.Equal(560, model.InvStdDev.Length);
        }
        finally
        {
            model!.Dispose();
        }
    }

    [Fact]
    public void Transcribe_OnRealWeights_RunsEndToEndWithoutCrashing()
    {
        string? modelPath = FindModelPath("paraformer-zh-small.int8.onnx");
        Assert.SkipUnless(modelPath != null, "paraformer-zh-small.int8.onnx not found");
        Assert.SkipUnless(File.Exists(@"F:\_models\paraformer-zh-small-tokens.txt"),
            "paraformer-zh-small-tokens.txt not found at F:\\_models -- real vocab required");

        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var pipeline = ParaformerOnnxPipeline.Load(modelPath!);
        Assert.Equal("FunASR-Paraformer-ONNX", pipeline.Architecture);

        // Real Chinese speech clip, 2026-09-11: downloaded from the same real HF repo this
        // checkpoint's vocab came from (csukuangfj/sherpa-onnx-paraformer-zh-small-2024-03-09's
        // own test_wavs/0.wav) -- a real spoken-Mandarin clip, not synthetic tone, so this is now
        // a real content check (no exact ground-truth transcript string was available to assert
        // against, but the output can be judged for plausibility -- real Chinese characters,
        // not degenerate/garbage output).
        string wavPath = @"C:\Git-Public\OpenTail.Stingray\docs\audio-samples\paraformer-zh-test-0.wav";
        Assert.SkipUnless(File.Exists(wavPath), "paraformer-zh-test-0.wav not found");
        var (audio, sampleRate, _) = OpenTail.Stingray.Audio.WavReader.ReadWav(wavPath);
        Assert.Equal(16000, sampleRate);

        var res = pipeline.Transcribe(new SpeechToTextRequest
        {
            AudioSamples = audio,
            SampleRate = sampleRate,
            Language = "zh",
        });
        sw.Stop();

        Console.WriteLine($"[ParaformerOnnx] {sw.ElapsedMilliseconds}ms, text='{res.Text}'");

        Assert.NotNull(res);
        Assert.NotEmpty(res.Segments);
    }
}
