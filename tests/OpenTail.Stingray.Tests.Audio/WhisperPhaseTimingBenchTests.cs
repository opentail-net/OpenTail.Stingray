using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Whisper;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// TEMPORARY, throwaway phase-level timing bench -- "Experiment 1" from this session's ChatGPT-
/// advised methodology (see docs/audio-review-progress.md): before trying another kernel-level
/// optimization, find out which PHASE (mel extraction / encoder / decoder autoregressive loop)
/// actually dominates wall time at each model size, since two independent "obviously good" kernel
/// changes (batched GEMM, Q8_0 weight quantization) both measured as regressions without this
/// information. Manually drives the same stages WhisperPipeline.ProcessAudioChunk does, but with
/// a Stopwatch around each one, instead of modifying any production code path. Not part of the
/// permanent suite, delete after use.
/// </summary>
public sealed class WhisperPhaseTimingBenchTests : HeavyTestBase
{
    private static readonly (string Label, string FileName)[] Models =
    [
        ("Tiny", "ggml-tiny.bin"),
        ("Base", "ggml-base.bin"),
        ("Small", "ggml-small.bin"),
        ("Medium", "ggml-medium.bin"),
        ("Large-v3", "ggml-large-v3.bin"),
    ];

    [Fact]
    public void PhaseTiming_MelVsEncoderVsDecoder_AcrossModelSizes()
    {
        var report = new System.Text.StringBuilder();
        report.AppendLine($"ProcessorCount={Environment.ProcessorCount}");

        const int seconds = 12;
        int numSamples = 16000 * seconds;
        float[] audio = new float[numSamples];
        var rng = new Random(42);
        for (int i = 0; i < numSamples; i++)
            audio[i] = MathF.Sin(2.0f * MathF.PI * 220.0f * i / 16000.0f) * 0.3f
                     + MathF.Sin(2.0f * MathF.PI * 880.0f * i / 16000.0f) * 0.1f
                     + (float)(rng.NextDouble() - 0.5) * 0.02f;

        bool anyRan = false;

        foreach (var (label, fileName) in Models)
        {
            string? modelPath = FindModelPath(fileName);
            if (modelPath is null)
            {
                report.AppendLine($"{label}: SKIPPED (models/{fileName} not found)");
                continue;
            }
            anyRan = true;

            var ggml = WhisperGgmlModel.Load(modelPath);
            var config = ggml.ToConfig();
            var melExtractor = new WhisperMelExtractor(config.NumMels);
            var tokenizer = config.IsV3 ? WhisperTokenizer.CreateV3() : WhisperTokenizer.FromGgml(ggml);
            var encoderWeights = new WhisperEncoderWeights(ggml);
            var decoderWeights = new WhisperDecoderWeights(ggml);
            var encoder = new WhisperEncoder(config, encoderWeights);
            var decoder = new WhisperDecoder(config, decoderWeights);

            // Warmup run (JIT, weight prefault) -- not timed.
            RunOnce(audio, config, melExtractor, tokenizer, encoder, decoder, out _, out _, out _, out _, out _);

            const int n = 3;
            var melMs = new double[n];
            var encMs = new double[n];
            var primeMs = new double[n];
            var prefillMs = new double[n];
            var decodeMs = new double[n];
            int tokensGenerated = 0;

            for (int i = 0; i < n; i++)
            {
                RunOnce(audio, config, melExtractor, tokenizer, encoder, decoder,
                    out melMs[i], out encMs[i], out primeMs[i], out prefillMs[i], out decodeMs[i], ref tokensGenerated);
            }

            double Mean(double[] a) { double s = 0; foreach (var v in a) s += v; return s / a.Length; }
            double mel = Mean(melMs), enc = Mean(encMs), prime = Mean(primeMs), prefill = Mean(prefillMs), decode = Mean(decodeMs);
            double total = mel + enc + prime + prefill + decode;

            report.AppendLine(
                $"{label} ({fileName}): total={total:F0}ms | mel={mel:F0}ms({Pct(mel, total)}) " +
                $"encoder={enc:F0}ms({Pct(enc, total)}) primeCrossAttn={prime:F0}ms({Pct(prime, total)}) " +
                $"promptPrefill={prefill:F0}ms({Pct(prefill, total)}) decodeLoop={decode:F0}ms({Pct(decode, total)}) " +
                $"tokensGenerated~{tokensGenerated} msPerToken~{(tokensGenerated > 0 ? decode / tokensGenerated : 0):F1}");
        }

        Assert.SkipUnless(anyRan, "No Whisper ggml models found under models/");

        var reportText = report.ToString();
        Console.Error.WriteLine(reportText);
        File.WriteAllText(Path.Combine(Path.GetTempPath(), "whisper_phase_timing_result.txt"), reportText);
    }

    private static string Pct(double part, double total) => total > 0 ? $"{100.0 * part / total:F0}%" : "0%";

    private static void RunOnce(
        float[] audio, WhisperConfig config, WhisperMelExtractor melExtractor, WhisperTokenizer tokenizer,
        WhisperEncoder encoder, WhisperDecoder decoder,
        out double melMs, out double encMs, out double primeMs, out double prefillMs, out double decodeMs)
    {
        int dummy = 0;
        RunOnce(audio, config, melExtractor, tokenizer, encoder, decoder, out melMs, out encMs, out primeMs, out prefillMs, out decodeMs, ref dummy);
    }

    private static void RunOnce(
        float[] audio, WhisperConfig config, WhisperMelExtractor melExtractor, WhisperTokenizer tokenizer,
        WhisperEncoder encoder, WhisperDecoder decoder,
        out double melMs, out double encMs, out double primeMs, out double prefillMs, out double decodeMs,
        ref int tokensGenerated)
    {
        var sw = Stopwatch.StartNew();

        sw.Restart();
        float[] mel = melExtractor.ExtractMel(audio, padTo30Seconds: true);
        int numFrames = mel.Length / config.NumMels;
        melMs = sw.Elapsed.TotalMilliseconds;

        sw.Restart();
        float[] audioFeatures = encoder.Forward(mel, numFrames);
        int audioFrames = audioFeatures.Length / config.AudioState;
        encMs = sw.Elapsed.TotalMilliseconds;

        int[] initialPrompt = tokenizer.BuildInitialPrompt("en", SpeechTask.Transcribe, enableTimestamps: true);
        var generatedTokens = new List<int>(initialPrompt);
        var kvCache = new WhisperKvCache(config.TextLayer, config.TextCtx, config.TextState);

        sw.Restart();
        decoder.PrimeCrossAttention(kvCache, audioFeatures, audioFrames);
        primeMs = sw.Elapsed.TotalMilliseconds;

        int currentPos = 0;
        sw.Restart();
        for (int i = 0; i < initialPrompt.Length - 1; i++)
            decoder.ForwardStep(initialPrompt[i], currentPos++, kvCache, audioFeatures, audioFrames);
        prefillMs = sw.Elapsed.TotalMilliseconds;

        int lastToken = initialPrompt[^1];
        int maxNewTokens = Math.Min(256, config.TextCtx - initialPrompt.Length);

        sw.Restart();
        int generated = 0;
        for (int step = 0; step < maxNewTokens; step++)
        {
            float[] logits = decoder.ForwardStep(lastToken, currentPos++, kvCache, audioFeatures, audioFrames);

            // Greedy argmax, no sampling-filter overhead (kept minimal -- this bench isolates
            // matrix-math phases, not tokenizer post-processing).
            int bestIdx = 0;
            float maxVal = logits[0];
            for (int j = 1; j < logits.Length; j++)
            {
                if (logits[j] > maxVal) { maxVal = logits[j]; bestIdx = j; }
            }
            generatedTokens.Add(bestIdx);
            lastToken = bestIdx;
            generated++;

            if (bestIdx == WhisperTokenizer.EndOfText || bestIdx == 0) break;
        }
        decodeMs = sw.Elapsed.TotalMilliseconds;
        tokensGenerated = generated;
    }

    private static string? FindModelPath(string fileName)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }
}
