using OpenTail.Stingray.Audio.AudioGen;
using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Tests.Audio;

/// <summary>
/// Real full-chain numeric golden verification for AudioGen: real T5-large text encoder -&gt; real
/// decoder greedy delayed-multi-codebook generation (a few real autoregressive steps) -&gt; real
/// 16kHz EnCodec decode, all against the SAME real, already-local checkpoints the three per-stage
/// golden tests use (<see cref="AudioGenTextEncoderGoldenParityTests"/>,
/// <see cref="AudioGenDecoderGoldenParityTests"/>, <see cref="AudioGenEncodecDecoderGoldenParityTests"/>).
///
/// This is the composition check those per-component tests cannot catch on their own -- it proves
/// the REAL T5 encoder's own output correctly drives the REAL decoder's `output_proj` cross-attention
/// conditioning, and the REAL decoder's greedy-argmax delayed-pattern codebook grid correctly
/// drives the REAL EnCodec decoder, using the exact same production calls
/// <see cref="AudioGenGenerator.Generate"/> makes internally (this test inlines that loop directly,
/// bypassing only <see cref="T5Tokenizer"/> tokenization itself -- fixed token ids stand in for a
/// real tokenized prompt, since piece 1's own test already independently verifies the
/// tokenizer-adjacent <see cref="T5EncoderKernels"/> path; feeding tokenizer output through that
/// SAME already-verified kernel adds no new numeric surface). CFG is disabled
/// (guidanceScale=1) to keep this test to ONE real forward branch per step -- the CFG uncond
/// branch is just a second call to the identical, already individually-verified Step/
/// PrepareCrossAttention path with an all-zero conditioning vector, so this does not skip any
/// unverified math.
///
/// Compares against `scratch-llamacpp-ref/audiogen_e2e_golden.py`, a pure-numpy oracle that chains
/// the identical three real math stages (reusing the same transcribed math as the other three
/// oracle scripts) end to end.
/// </summary>
public sealed class AudioGenEndToEndGoldenParityTests : HeavyTestBase
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

    [Fact]
    public void FullChain_RealWeights_MatchesGoldenOutput()
    {
        string? t5Path = FindRepoFile("models/audiogen-medium/t5-large.safetensors");
        string? lmPath = FindRepoFile("models/audiogen-medium/audiogen-medium-lm.safetensors");
        string? codecPath = FindRepoFile("models/audiogen-medium/audiogen-medium-encodec16k.safetensors");
        Assert.SkipUnless(t5Path != null && lmPath != null && codecPath != null,
            "one or more real audiogen-medium checkpoints (t5-large / lm / encodec16k) not found");

        string? idsPath = FindRepoFile("scratch-llamacpp-ref/audiogen_e2e_golden_input_ids.txt");
        string? pcmPath = FindRepoFile("scratch-llamacpp-ref/audiogen_e2e_golden_pcm.txt");
        Assert.SkipUnless(idsPath != null && pcmPath != null,
            "golden AudioGen end-to-end files not found (re-run scratch-llamacpp-ref/audiogen_e2e_golden.py)");

        var tokenIds = Array.ConvertAll(File.ReadAllText(idsPath!).Trim().Split(','), int.Parse);

        var pcmLines = File.ReadAllText(pcmPath!).Split('\n');
        int goldenLen = int.Parse(pcmLines[0].Trim());
        var goldenParts = pcmLines[1].Trim().Split(',');
        Assert.Equal(goldenLen, goldenParts.Length);
        var golden = new float[goldenLen];
        for (int i = 0; i < golden.Length; i++) golden[i] = float.Parse(goldenParts[i]);

        var sw = System.Diagnostics.Stopwatch.StartNew();

        // Stage 1: real T5-large text encoder.
        using var t5Loader = SafetensorsLoader.Open(t5Path!);
        var t5Weights = AudioGenTextEncoderWeights.Load(t5Loader);
        var encoderHidden = T5EncoderKernels.Forward(AudioGenTextEncoderWeights.Dims, t5Weights, tokenIds);

        // Stage 2: real decoder, delayed multi-codebook greedy generation (mirrors
        // AudioGenGenerator.Generate's loop with useCfg=false, topK<=1).
        using var lmLoader = SafetensorsLoader.Open(lmPath!);
        var lmWeights = new AudioGenTransformerWeights(lmLoader);

        var condCache = new AudioGenTransformer.KvCache();
        AudioGenTransformer.PrepareCrossAttention(lmWeights, encoderHidden, condCache);

        const int frames = 2;
        const int codebooks = AudioGenConfig.NumCodebooks;
        var generated = new int[codebooks][];
        for (int q = 0; q < codebooks; q++) generated[q] = new int[frames];
        var generatedSoFar = new int[codebooks][];
        for (int q = 0; q < codebooks; q++) generatedSoFar[q] = [];
        int seqLen = frames + codebooks - 1;

        for (int step = 0; step < seqLen; step++)
        {
            var column = OpenTail.Stingray.Audio.MusicGen.DelayPattern.InputColumnForStep(codebooks, step, generatedSoFar, AudioGenConfig.PadTokenId);
            var logits = AudioGenTransformer.Step(lmWeights, column, condCache);

            for (int q = 0; q < codebooks; q++)
            {
                int localIndex = step - q;
                if (localIndex < 0 || localIndex >= frames) continue;
                int token = ArgMax(logits[q]);
                generated[q][localIndex] = token;
                generatedSoFar[q] = [.. generatedSoFar[q], token];
            }
        }

        // Stage 3: real 16kHz EnCodec decode.
        using var codecLoader = SafetensorsLoader.Open(codecPath!);
        var codecWeights = AudioGenEncodecDecoderWeights.Load(codecLoader);
        var pcm = EncodecDecoderKernels.Decode(codecWeights, generated);

        sw.Stop();
        // Full chain: T5-large encoder + 48-layer decoder over 5 steps + EnCodec decode with real
        // weight loading for three separate checkpoints -- must take real, non-trivial wall-clock.
        Assert.True(sw.ElapsedMilliseconds > 200, $"suspiciously fast run ({sw.ElapsedMilliseconds}ms) -- did this actually execute against real weights?");

        Assert.Equal(goldenLen, pcm.Length);

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < goldenLen; i++)
        {
            float a = pcm[i];
            float b = golden[i];
            dot += a * b;
            normA += a * a;
            normB += b * b;
        }
        double cosine = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        Assert.True(cosine > 0.999, $"cosine similarity {cosine} too low vs golden AudioGen end-to-end PCM");
    }

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        float bestVal = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }
}
