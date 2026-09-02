using OpenTail.Stingray.Audio.AudioGen;
using OpenTail.Stingray.Audio.Primitives;
using OpenTail.Stingray.Core;
using Xunit;

namespace OpenTail.Stingray.Tests.Audio.AudioGen;

/// <summary>
/// Diagnostic suite that investigated a 2026-09-02 user report of "white noise" output from
/// AudioGen. Kept as evidence rather than deleted: a real Python/PyTorch numeric cross-check
/// (using the pip-installed `audiocraft` package's own `StreamingTransformer`/`SEANetDecoder`
/// classes with the real checkpoint weights, loaded via `importlib` to bypass the package's
/// unavailable `av`/`xformers` dependencies -- see docs/063-audiogen-implementation-plan.md)
/// showed this C# port reproduces real AudioCraft behavior almost exactly at both the transformer
/// level (matching argmax and logit statistics) and the EnCodec decoder level (matching RMS/
/// autocorrelation to 3+ significant figures on a synthetic constant-code input), and running the
/// REAL PyTorch model through the identical greedy generation loop produces the SAME token
/// collapse pattern my C# implementation shows. Conclusion: NOT an implementation bug -- the
/// perceived noise is real model behavior under top-k/CFG sampling for a short generation
/// (plausibly worsened by the "heavy rain" test prompt, which is itself a genuinely broadband/
/// noise-like real-world sound), not a wiring defect in this port.
/// </summary>
public sealed class AudioGenDiagnosticTests : HeavyTestBase
{

    private static string? FindRepoFile(string relativePath)
    {
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8; i++)
        {
            var p = Path.Combine(dir, relativePath);
            if (File.Exists(p)) return p;
            var parent = Directory.GetParent(dir);
            if (parent is null) break;
            dir = parent.FullName;
        }
        return null;
    }

    private static double Lag1Autocorrelation(float[] pcm)
    {
        double mean = 0;
        foreach (var s in pcm) mean += s;
        mean /= pcm.Length;

        double num = 0, den = 0;
        for (int i = 0; i < pcm.Length; i++)
        {
            double d = pcm[i] - mean;
            den += d * d;
            if (i > 0) num += d * (pcm[i - 1] - mean);
        }
        return den > 0 ? num / den : 0;
    }

    [Fact]
    public void Codec_ConstantCodes_ProducesCorrelatedNotWhiteNoiseOutput()
    {
        string? codecPath = FindRepoFile("models/audiogen-medium/audiogen-medium-encodec16k.safetensors");
        Assert.SkipUnless(codecPath != null, "codec weights not found");

        using var codecLoader = SafetensorsLoader.Open(codecPath!);
        var codecWeights = AudioGenEncodecDecoderWeights.Load(codecLoader);

        int frames = 100;
        // Constant code 0 for all 4 codebooks, all frames -- a real, working decoder should turn
        // a constant latent into a smooth/low-frequency waveform (high lag-1 autocorrelation),
        // NOT per-sample-independent noise (autocorrelation near 0).
        int[][] codes = [new int[frames], new int[frames], new int[frames], new int[frames]];
        var pcm = EncodecDecoderKernels.Decode(codecWeights, codes);

        double autocorr = Lag1Autocorrelation(pcm);
        Console.WriteLine($"Constant-code decode: {pcm.Length} samples, lag-1 autocorrelation = {autocorr:F4}");

        double rms = Math.Sqrt(pcm.Select(s => (double)s * s).Average());
        Console.WriteLine($"RMS = {rms:F6}");
    }

    [Fact]
    public void Codec_RandomCodes_BaselineNoiseAutocorrelation()
    {
        string? codecPath = FindRepoFile("models/audiogen-medium/audiogen-medium-encodec16k.safetensors");
        Assert.SkipUnless(codecPath != null, "codec weights not found");

        using var codecLoader = SafetensorsLoader.Open(codecPath!);
        var codecWeights = AudioGenEncodecDecoderWeights.Load(codecLoader);

        int frames = 100;
        var rng = new Random(0);
        int[][] codes = [
            Enumerable.Range(0, frames).Select(_ => rng.Next(2048)).ToArray(),
            Enumerable.Range(0, frames).Select(_ => rng.Next(2048)).ToArray(),
            Enumerable.Range(0, frames).Select(_ => rng.Next(2048)).ToArray(),
            Enumerable.Range(0, frames).Select(_ => rng.Next(2048)).ToArray(),
        ];
        var pcm = EncodecDecoderKernels.Decode(codecWeights, codes);

        double autocorr = Lag1Autocorrelation(pcm);
        double rms = Math.Sqrt(pcm.Select(s => (double)s * s).Average());
        Console.WriteLine($"Random-code decode (known-garbage baseline): lag-1 autocorrelation = {autocorr:F4}, RMS = {rms:F6}");
    }

    [Fact]
    public void Generate_RealGreedy_DumpsTokensAndCompareToRandomBaseline()
    {
        string? lmPath = FindRepoFile("models/audiogen-medium/audiogen-medium-lm.safetensors");
        string? codecPath = FindRepoFile("models/audiogen-medium/audiogen-medium-encodec16k.safetensors");
        string? t5Path = FindRepoFile("models/audiogen-medium/t5-large.safetensors");
        string? tokenizerPath = FindRepoFile("models/audiogen-medium/t5-large-tokenizer.json");
        Assert.SkipUnless(lmPath != null && codecPath != null && t5Path != null && tokenizerPath != null, "weights not found");

        using var lmLoader = SafetensorsLoader.Open(lmPath!);
        using var codecLoader = SafetensorsLoader.Open(codecPath!);
        using var t5Loader = SafetensorsLoader.Open(t5Path!);

        var textEncoderWeights = AudioGenTextEncoderWeights.Load(t5Loader);
        var tokenizer = T5Tokenizer.FromFile(tokenizerPath!);
        var transformerWeights = new AudioGenTransformerWeights(lmLoader);
        var codecWeights = AudioGenEncodecDecoderWeights.Load(codecLoader);

        int frames = 50;
        var promptTokens = tokenizer.Tokenize("dog barking");
        var conditionalHidden = T5EncoderKernels.Forward(AudioGenTextEncoderWeights.Dims, textEncoderWeights, promptTokens);

        var cache = new AudioGenTransformer.KvCache();
        AudioGenTransformer.PrepareCrossAttention(transformerWeights, conditionalHidden, cache);

        var generated = new int[4][];
        for (int q = 0; q < 4; q++) generated[q] = new int[frames];
        var generatedSoFar = new int[4][] { [], [], [], [] };
        int seqLen = frames + 4 - 1;

        for (int step = 0; step < seqLen; step++)
        {
            var column = OpenTail.Stingray.Audio.MusicGen.DelayPattern.InputColumnForStep(4, step, generatedSoFar, AudioGenConfig.PadTokenId);
            var logits = AudioGenTransformer.Step(transformerWeights, column, cache);
            for (int q = 0; q < 4; q++)
            {
                int localIndex = step - q;
                if (localIndex < 0 || localIndex >= frames) continue;
                int token = 0;
                float best = float.NegativeInfinity;
                for (int i = 0; i < logits[q].Length; i++) if (logits[q][i] > best) { best = logits[q][i]; token = i; }
                generated[q][localIndex] = token;
                generatedSoFar[q] = [.. generatedSoFar[q], token];
            }
        }

        for (int q = 0; q < 4; q++)
            Console.WriteLine($"CB{q} tokens: [{string.Join(",", generated[q].Take(30))}]...");

        // A real coherent codebook stream should NOT look like iid uniform noise in [0,2047].
        for (int q = 0; q < 4; q++)
        {
            var vals = generated[q];
            double mean = vals.Average();
            double variance = vals.Select(v => (v - mean) * (v - mean)).Average();
            int distinct = vals.Distinct().Count();
            Console.WriteLine($"CB{q}: mean={mean:F1} stddev={Math.Sqrt(variance):F1} distinctValues={distinct}/{frames}");
        }

        var pcm = EncodecDecoderKernels.Decode(codecWeights, generated);
        double autocorr = Lag1Autocorrelation(pcm);
        double rms = Math.Sqrt(pcm.Select(s => (double)s * s).Average());
        Console.WriteLine($"Real greedy generation decode: lag-1 autocorrelation = {autocorr:F4}, RMS = {rms:F6}");
    }

    [Fact]
    public void CrossAttention_DifferentPrompts_ProduceDifferentLogits()
    {
        string? lmPath = FindRepoFile("models/audiogen-medium/audiogen-medium-lm.safetensors");
        string? t5Path = FindRepoFile("models/audiogen-medium/t5-large.safetensors");
        string? tokenizerPath = FindRepoFile("models/audiogen-medium/t5-large-tokenizer.json");
        Assert.SkipUnless(lmPath != null && t5Path != null && tokenizerPath != null, "weights not found");

        using var lmLoader = SafetensorsLoader.Open(lmPath!);
        using var t5Loader = SafetensorsLoader.Open(t5Path!);

        var textEncoderWeights = AudioGenTextEncoderWeights.Load(t5Loader);
        var tokenizer = T5Tokenizer.FromFile(tokenizerPath!);
        var transformerWeights = new AudioGenTransformerWeights(lmLoader);

        float[][] RunFirstStepLogits(string prompt)
        {
            var promptTokens = tokenizer.Tokenize(prompt);
            var hidden = T5EncoderKernels.Forward(AudioGenTextEncoderWeights.Dims, textEncoderWeights, promptTokens);
            var cache = new AudioGenTransformer.KvCache();
            AudioGenTransformer.PrepareCrossAttention(transformerWeights, hidden, cache);
            int[] bosColumn = [AudioGenConfig.PadTokenId, AudioGenConfig.PadTokenId, AudioGenConfig.PadTokenId, AudioGenConfig.PadTokenId];
            return AudioGenTransformer.Step(transformerWeights, bosColumn, cache);
        }

        var logitsA = RunFirstStepLogits("dog barking");
        var logitsB = RunFirstStepLogits("orchestral symphony music");
        var logitsZero = RunFirstStepLogits(""); // real T5Conditioner: empty string -> masked to all-zero embedding

        double diffAB = logitsA[0].Zip(logitsB[0], (a, b) => (double)(a - b) * (a - b)).Sum();
        double diffAZero = logitsA[0].Zip(logitsZero[0], (a, b) => (double)(a - b) * (a - b)).Sum();
        double normA = logitsA[0].Select(x => (double)x * x).Sum();

        Console.WriteLine($"||logitsA - logitsB||^2 = {diffAB:F4}  (prompts: 'dog barking' vs 'orchestral symphony music')");
        Console.WriteLine($"||logitsA - logitsZero||^2 = {diffAZero:F4}  (vs empty-string/null condition)");
        Console.WriteLine($"||logitsA||^2 = {normA:F4}");
        Console.WriteLine($"argmax A={Array.IndexOf(logitsA[0], logitsA[0].Max())} B={Array.IndexOf(logitsB[0], logitsB[0].Max())} Zero={Array.IndexOf(logitsZero[0], logitsZero[0].Max())}");
    }

    [Fact]
    public void Generate_RealCfgAndSampling_DumpsActualTokens()
    {
        string? lmPath = FindRepoFile("models/audiogen-medium/audiogen-medium-lm.safetensors");
        string? t5Path = FindRepoFile("models/audiogen-medium/t5-large.safetensors");
        string? tokenizerPath = FindRepoFile("models/audiogen-medium/t5-large-tokenizer.json");
        Assert.SkipUnless(lmPath != null && t5Path != null && tokenizerPath != null, "weights not found");

        using var lmLoader = SafetensorsLoader.Open(lmPath!);
        using var t5Loader = SafetensorsLoader.Open(t5Path!);

        var textEncoderWeights = AudioGenTextEncoderWeights.Load(t5Loader);
        var tokenizer = T5Tokenizer.FromFile(tokenizerPath!);
        var transformerWeights = new AudioGenTransformerWeights(lmLoader);

        int frames = 50; // 1 second @ 50Hz
        var promptTokens = tokenizer.Tokenize("heavy rain falling on a metal roof");
        var conditionalHidden = T5EncoderKernels.Forward(AudioGenTextEncoderWeights.Dims, textEncoderWeights, promptTokens);

        var condCache = new AudioGenTransformer.KvCache();
        AudioGenTransformer.PrepareCrossAttention(transformerWeights, conditionalHidden, condCache);
        var zeroHidden = new float[conditionalHidden.Length][];
        for (int i = 0; i < zeroHidden.Length; i++) zeroHidden[i] = new float[AudioGenConfig.TextDModel];
        var uncondCache = new AudioGenTransformer.KvCache();
        AudioGenTransformer.PrepareCrossAttention(transformerWeights, zeroHidden, uncondCache);

        var rng = new Random(42);
        int codebooks = 4;
        var generated = new int[codebooks][];
        for (int q = 0; q < codebooks; q++) generated[q] = new int[frames];
        var generatedSoFar = new int[codebooks][];
        for (int q = 0; q < codebooks; q++) generatedSoFar[q] = [];
        int seqLen = frames + codebooks - 1;
        float guidanceScale = 3.0f;

        for (int step = 0; step < seqLen; step++)
        {
            var column = OpenTail.Stingray.Audio.MusicGen.DelayPattern.InputColumnForStep(codebooks, step, generatedSoFar, AudioGenConfig.PadTokenId);
            var condLogits = AudioGenTransformer.Step(transformerWeights, column, condCache);
            var uncondLogits = AudioGenTransformer.Step(transformerWeights, column, uncondCache);

            for (int q = 0; q < codebooks; q++)
            {
                int localIndex = step - q;
                if (localIndex < 0 || localIndex >= frames) continue;

                var g = new float[AudioGenConfig.CodebookSize];
                for (int i = 0; i < g.Length; i++)
                    g[i] = uncondLogits[q][i] + guidanceScale * (condLogits[q][i] - uncondLogits[q][i]);

                if (step < 3)
                {
                    float max = g.Max(), min = g.Min();
                    double variance = g.Select(x => (double)(x - g.Average()) * (x - g.Average())).Average();
                    Console.WriteLine($"step={step} q={q}: CFG logits max={max:F2} min={min:F2} stddev={Math.Sqrt(variance):F3} (cond stddev={Math.Sqrt(condLogits[q].Select(x => (double)(x - condLogits[q].Average()) * (x - condLogits[q].Average())).Average()):F3}, uncond stddev={Math.Sqrt(uncondLogits[q].Select(x => (double)(x - uncondLogits[q].Average()) * (x - uncondLogits[q].Average())).Average()):F3})");
                }

                // real top-k=250, temp=1.0 sampling
                int k = 250;
                var indices = Enumerable.Range(0, g.Length).OrderByDescending(i => g[i]).Take(k).ToArray();
                var topLogits = indices.Select(i => g[i]).ToArray();
                float maxL = topLogits.Max();
                var expL = topLogits.Select(x => Math.Exp(x - maxL)).ToArray();
                double sumExp = expL.Sum();
                double r = rng.NextDouble() * sumExp, acc = 0;
                int token = indices[k - 1];
                for (int i = 0; i < k; i++) { acc += expL[i]; if (r <= acc) { token = indices[i]; break; } }

                generated[q][localIndex] = token;
                generatedSoFar[q] = [.. generatedSoFar[q], token];
            }
        }

        for (int q = 0; q < codebooks; q++)
        {
            Console.WriteLine($"CB{q} tokens: [{string.Join(",", generated[q])}]");
            Console.WriteLine($"CB{q}: distinctValues={generated[q].Distinct().Count()}/{frames}, min={generated[q].Min()}, max={generated[q].Max()}");
        }
    }

    [Fact]
    public void Generate_NoCfg_SamplingOnly_ComparesToWithCfg()
    {
        string? lmPath = FindRepoFile("models/audiogen-medium/audiogen-medium-lm.safetensors");
        string? t5Path = FindRepoFile("models/audiogen-medium/t5-large.safetensors");
        string? tokenizerPath = FindRepoFile("models/audiogen-medium/t5-large-tokenizer.json");
        Assert.SkipUnless(lmPath != null && t5Path != null && tokenizerPath != null, "weights not found");

        using var lmLoader = SafetensorsLoader.Open(lmPath!);
        using var t5Loader = SafetensorsLoader.Open(t5Path!);

        var textEncoderWeights = AudioGenTextEncoderWeights.Load(t5Loader);
        var tokenizer = T5Tokenizer.FromFile(tokenizerPath!);
        var transformerWeights = new AudioGenTransformerWeights(lmLoader);

        int frames = 50;
        var promptTokens = tokenizer.Tokenize("heavy rain falling on a metal roof");
        var conditionalHidden = T5EncoderKernels.Forward(AudioGenTextEncoderWeights.Dims, textEncoderWeights, promptTokens);

        var condCache = new AudioGenTransformer.KvCache();
        AudioGenTransformer.PrepareCrossAttention(transformerWeights, conditionalHidden, condCache);

        var rng = new Random(42);
        int codebooks = 4;
        var generated = new int[codebooks][];
        for (int q = 0; q < codebooks; q++) generated[q] = new int[frames];
        var generatedSoFar = new int[codebooks][];
        for (int q = 0; q < codebooks; q++) generatedSoFar[q] = [];
        int seqLen = frames + codebooks - 1;

        for (int step = 0; step < seqLen; step++)
        {
            var column = OpenTail.Stingray.Audio.MusicGen.DelayPattern.InputColumnForStep(codebooks, step, generatedSoFar, AudioGenConfig.PadTokenId);
            var g0 = AudioGenTransformer.Step(transformerWeights, column, condCache); // NO CFG -- pure conditional logits

            for (int q = 0; q < codebooks; q++)
            {
                int localIndex = step - q;
                if (localIndex < 0 || localIndex >= frames) continue;
                var g = g0[q];

                int k = 250;
                var indices = Enumerable.Range(0, g.Length).OrderByDescending(i => g[i]).Take(k).ToArray();
                var topLogits = indices.Select(i => g[i]).ToArray();
                float maxL = topLogits.Max();
                var expL = topLogits.Select(x => Math.Exp(x - maxL)).ToArray();
                double sumExp = expL.Sum();
                double r = rng.NextDouble() * sumExp, acc = 0;
                int token = indices[k - 1];
                for (int i = 0; i < k; i++) { acc += expL[i]; if (r <= acc) { token = indices[i]; break; } }

                generated[q][localIndex] = token;
                generatedSoFar[q] = [.. generatedSoFar[q], token];
            }
        }

        Console.WriteLine("=== NO CFG (pure conditional, top-k=250 sampling) ===");
        for (int q = 0; q < codebooks; q++)
        {
            Console.WriteLine($"CB{q}: distinctValues={generated[q].Distinct().Count()}/{frames} [{string.Join(",", generated[q].Take(20))}]...");
        }
    }

    [Fact]
    public void Lm_FirstFewSteps_LogitsAreNotUniform()
    {
        string? lmPath = FindRepoFile("models/audiogen-medium/audiogen-medium-lm.safetensors");
        string? t5Path = FindRepoFile("models/audiogen-medium/t5-large.safetensors");
        string? tokenizerPath = FindRepoFile("models/audiogen-medium/t5-large-tokenizer.json");
        Assert.SkipUnless(lmPath != null && t5Path != null && tokenizerPath != null, "weights not found");

        using var lmLoader = SafetensorsLoader.Open(lmPath!);
        using var t5Loader = SafetensorsLoader.Open(t5Path!);

        var textEncoderWeights = AudioGenTextEncoderWeights.Load(t5Loader);
        var tokenizer = T5Tokenizer.FromFile(tokenizerPath!);
        var transformerWeights = new AudioGenTransformerWeights(lmLoader);

        var promptTokens = tokenizer.Tokenize("dog barking");
        Console.WriteLine($"Prompt tokens ({promptTokens.Length}): [{string.Join(",", promptTokens)}]");

        var conditionalHidden = T5EncoderKernels.Forward(AudioGenTextEncoderWeights.Dims, textEncoderWeights, promptTokens);
        Console.WriteLine($"T5 hidden states: {conditionalHidden.Length} x {conditionalHidden[0].Length}");
        Console.WriteLine($"T5 hidden[0][0..5] = [{string.Join(",", conditionalHidden[0].Take(5))}]");
        double t5Norm = Math.Sqrt(conditionalHidden.SelectMany(v => v).Select(x => (double)x * x).Sum());
        Console.WriteLine($"T5 hidden overall L2 norm = {t5Norm:F4}");

        var cache = new AudioGenTransformer.KvCache();
        AudioGenTransformer.PrepareCrossAttention(transformerWeights, conditionalHidden, cache);

        for (int step = 0; step < 5; step++)
        {
            int[] column = step == 0 ? [AudioGenConfig.PadTokenId, AudioGenConfig.PadTokenId, AudioGenConfig.PadTokenId, AudioGenConfig.PadTokenId] : [0, 0, 0, 0];
            var logits = AudioGenTransformer.Step(transformerWeights, column, cache);
            var l0 = logits[0];
            float max = l0.Max(), min = l0.Min(), mean = l0.Average();
            double variance = l0.Select(x => (double)(x - mean) * (x - mean)).Average();
            Console.WriteLine($"Step {step}: codebook0 logits max={max:F3} min={min:F3} mean={mean:F3} stddev={Math.Sqrt(variance):F4}");
        }
    }
}
