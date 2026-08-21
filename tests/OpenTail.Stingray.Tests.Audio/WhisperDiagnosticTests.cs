using System;
using System.IO;
using System.Linq;
using System.Text;
using OpenTail.Stingray.Audio;
using OpenTail.Stingray.Audio.Whisper;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Throwaway diagnostic for the known bug (2026-08-21): "stingray stt" with real weights
/// transcribes the JFK reference clip as "[Music]" instead of the real speech. Dumps
/// intermediate tensor stats to localize where the numeric chain goes wrong, since no
/// whisper.cpp reference binary is available locally to diff against.
/// See docs/audio-review-progress.md.
/// </summary>
public sealed class WhisperDiagnosticTests
{
    private sealed class Output
    {
        private readonly StringBuilder _sb = new();
        public void WriteLine(string s) { _sb.AppendLine(s); }
        public override string ToString() => _sb.ToString();
    }

    private static string? FindModelPath(string fileName)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, "models", fileName);
            if (File.Exists(p)) return p;
            var jfk = Path.Combine(dir, "examples", "whisper.cpp", "samples", "jfk.wav");
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static string? FindFile(string relativeFromRepoRoot)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativeFromRepoRoot);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static void Stats(string label, ReadOnlySpan<float> data, Output output)
    {
        float min = float.MaxValue, max = float.MinValue, sum = 0, sumAbs = 0;
        int nanCount = 0;
        for (int i = 0; i < data.Length; i++)
        {
            float v = data[i];
            if (float.IsNaN(v) || float.IsInfinity(v)) { nanCount++; continue; }
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
            sumAbs += MathF.Abs(v);
        }
        output.WriteLine($"{label}: n={data.Length} min={min:F4} max={max:F4} mean={sum / data.Length:F4} meanAbs={sumAbs / data.Length:F4} nanInf={nanCount}");
    }

    /// <summary>
    /// Passes normally in CI. Set STINGRAY_AUDIO_DIAGNOSTIC_DUMP=1 to force it to fail with
    /// the full stats dump in its output -- xUnit only surfaces console/output-helper text
    /// for failed tests, so this is the only way to see the dump from a test run. Run
    /// directly for interactive use instead:
    /// tests/OpenTail.Stingray.Tests.Audio/bin/Debug/net10.0/OpenTail.Stingray.Tests.Audio.exe
    /// -class OpenTail.Stingray.Tests.Audio.WhisperDiagnosticTests -verbose
    /// </summary>
    [Fact]
    public void DumpIntermediateStats_Tiny_JfkSample()
    {
        var output = new Output();
        DumpFilterbank(output);
        RunDiagnostic(output);

        if (output.ToString().Length > 0 && Environment.GetEnvironmentVariable("STINGRAY_AUDIO_DIAGNOSTIC_DUMP") == "1")
        {
            Assert.Fail(output.ToString());
        }
    }

    private static void DumpFilterbank(Output output)
    {
        float[] filt = WhisperMelExtractor.CreateSlaneyMelFilterbank(80, 400, 16000, 0f, 8000f);
        int half = 201;
        output.WriteLine("Filter[0][0:10]: " + string.Join(",", filt.AsSpan(0 * half, 10).ToArray()));
        output.WriteLine("Filter[1][0:10]: " + string.Join(",", filt.AsSpan(1 * half, 10).ToArray()));
        output.WriteLine("Filter[40][0:10]: " + string.Join(",", filt.AsSpan(40 * half, 10).ToArray()));
        double[] rowSums = new double[5];
        for (int m = 0; m < 5; m++)
        {
            double sum = 0;
            for (int k = 0; k < half; k++) sum += filt[m * half + k];
            rowSums[m] = sum;
        }
        output.WriteLine("Filter row sums (first 5): " + string.Join(",", rowSums));
    }

    private static void RunDiagnostic(Output output)
    {
        string? modelPath = FindModelPath("ggml-tiny.bin");
        string? wavPath = FindFile("examples/whisper.cpp/samples/jfk.wav");
        if (modelPath is null || wavPath is null)
        {
            output.WriteLine($"SKIP: model={modelPath} wav={wavPath}");
            return;
        }

        var (samples, sampleRate, _) = WavReader.ReadWav(wavPath);
        output.WriteLine($"Loaded {samples.Length} samples @ {sampleRate}Hz");

        var ggml = WhisperGgmlModel.Load(modelPath);
        var config = ggml.ToConfig();
        output.WriteLine($"Config: AudioState={config.AudioState} AudioLayer={config.AudioLayer} AudioHead={config.AudioHead} TextState={config.TextState} TextLayer={config.TextLayer} NumMels={config.NumMels} VocabSize={config.VocabSize}");

        var melExtractor = new WhisperMelExtractor(config.NumMels);
        var tokenizer = WhisperTokenizer.FromGgml(ggml);
        var encoderWeights = new WhisperEncoderWeights(ggml);
        var decoderWeights = new WhisperDecoderWeights(ggml);
        var encoder = new WhisperEncoder(config, encoderWeights);
        var decoder = new WhisperDecoder(config, decoderWeights);

        float[] mel = melExtractor.ExtractMel(samples.AsSpan(0, Math.Min(samples.Length, 16000 * 11)), padTo30Seconds: true);
        int numFrames = mel.Length / config.NumMels;
        output.WriteLine($"Mel frames={numFrames}");
        Stats("Mel", mel, output);
        // First few mel values for the first frame
        output.WriteLine("Mel[frame0, mel0..9]: " + string.Join(",", mel[..Math.Min(10, mel.Length)]));

        float[] audioFeatures = encoder.Forward(mel, numFrames);
        int audioFrames = audioFeatures.Length / config.AudioState;
        output.WriteLine($"Encoder output frames={audioFrames}");
        Stats("EncoderOutput", audioFeatures, output);

        int[] initialPrompt = tokenizer.BuildInitialPrompt("en", SpeechTask.Transcribe, true);
        output.WriteLine("InitialPrompt tokens: " + string.Join(",", initialPrompt));

        var kvCache = new WhisperKvCache(config.TextLayer, config.TextCtx, config.TextState);
        decoder.PrimeCrossAttention(kvCache, audioFeatures, audioFrames);

        int currentPos = 0;
        for (int i = 0; i < initialPrompt.Length - 1; i++)
        {
            decoder.ForwardStep(initialPrompt[i], currentPos++, kvCache, audioFeatures, audioFrames);
        }

        int lastToken = initialPrompt[^1];
        float[] logits = decoder.ForwardStep(lastToken, currentPos++, kvCache, audioFeatures, audioFrames);
        Stats("Logits(step0)", logits, output);

        // Top 10 tokens by logit
        var idx = new int[logits.Length];
        for (int i = 0; i < idx.Length; i++) idx[i] = i;
        Array.Sort(idx, (a, b) => logits[b].CompareTo(logits[a]));
        output.WriteLine("Top 10 tokens (id, logit) [REAL AUDIO]:");
        for (int i = 0; i < 10; i++)
        {
            output.WriteLine($"  {idx[i]}: {logits[idx[i]]:F4}");
        }

        // Cross-attention-sensitivity check: rerun with silence instead of real audio.
        // If the top tokens/logits barely change, cross-attention isn't meaningfully
        // conditioning generation on the audio content.
        float[] silence = new float[16000 * 11];
        float[] silenceMel = melExtractor.ExtractMel(silence, padTo30Seconds: true);
        float[] silenceFeatures = encoder.Forward(silenceMel, silenceMel.Length / config.NumMels);
        Stats("EncoderOutput[SILENCE]", silenceFeatures, output);

        var kvCache2 = new WhisperKvCache(config.TextLayer, config.TextCtx, config.TextState);
        decoder.PrimeCrossAttention(kvCache2, silenceFeatures, audioFrames);
        int pos2 = 0;
        for (int i = 0; i < initialPrompt.Length - 1; i++)
        {
            decoder.ForwardStep(initialPrompt[i], pos2++, kvCache2, silenceFeatures, audioFrames);
        }
        float[] silenceLogits = decoder.ForwardStep(lastToken, pos2++, kvCache2, silenceFeatures, audioFrames);
        Stats("Logits(step0)[SILENCE]", silenceLogits, output);

        var idx2 = new int[silenceLogits.Length];
        for (int i = 0; i < idx2.Length; i++) idx2[i] = i;
        Array.Sort(idx2, (a, b) => silenceLogits[b].CompareTo(silenceLogits[a]));
        output.WriteLine("Top 10 tokens (id, logit) [SILENCE]:");
        for (int i = 0; i < 10; i++)
        {
            output.WriteLine($"  {idx2[i]}: {silenceLogits[idx2[i]]:F4}");
        }

        double diffSum = 0;
        for (int i = 0; i < logits.Length; i++) diffSum += Math.Abs(logits[i] - silenceLogits[i]);
        output.WriteLine($"Mean abs logit diff (real vs silence): {diffSum / logits.Length:F6}");

        double encDiffSum = 0;
        for (int i = 0; i < Math.Min(audioFeatures.Length, silenceFeatures.Length); i++)
            encDiffSum += Math.Abs(audioFeatures[i] - silenceFeatures[i]);
        output.WriteLine($"Mean abs encoder-output diff (real vs silence): {encDiffSum / audioFeatures.Length:F6}");

        // Force past the special/timestamp token cluster on the REAL-audio path and see
        // what surfaces underneath, greedily, for a few steps.
        output.WriteLine("--- Forced-past-special-tokens greedy decode (real audio) ---");
        var kvCache3 = new WhisperKvCache(config.TextLayer, config.TextCtx, config.TextState);
        decoder.PrimeCrossAttention(kvCache3, audioFeatures, audioFrames);
        int pos3 = 0;
        for (int i = 0; i < initialPrompt.Length - 1; i++)
        {
            decoder.ForwardStep(initialPrompt[i], pos3++, kvCache3, audioFeatures, audioFrames);
        }
        int tok3 = lastToken;
        var sb = new StringBuilder();
        for (int step = 0; step < 20; step++)
        {
            float[] stepLogits = decoder.ForwardStep(tok3, pos3++, kvCache3, audioFeatures, audioFrames);
            // Suppress endoftext, startoftranscript, and everything from notimestamps (50363) upward.
            stepLogits[WhisperTokenizer.EndOfText] = float.NegativeInfinity;
            stepLogits[WhisperTokenizer.StartOfTranscript] = float.NegativeInfinity;
            for (int t = tokenizer.NoTimestampsToken; t <= tokenizer.TimestampEnd && t < stepLogits.Length; t++)
                stepLogits[t] = float.NegativeInfinity;

            int best = 0; float bestV = float.NegativeInfinity;
            for (int i = 0; i < stepLogits.Length; i++) if (stepLogits[i] > bestV) { bestV = stepLogits[i]; best = i; }
            sb.Append($"[{best}:{bestV:F2}] ");
            tok3 = best;
        }
        output.WriteLine("Forced tokens: " + sb);
        var (forcedText, _) = tokenizer.DecodeWithTimestamps(
            System.Linq.Enumerable.ToArray(sb.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries)
                .Select(s => int.Parse(s[1..s.IndexOf(':')]))),
            TimeSpan.Zero);
        output.WriteLine("Forced decoded text: " + forcedText);
    }
}
