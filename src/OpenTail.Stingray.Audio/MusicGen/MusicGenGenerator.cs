
namespace OpenTail.Stingray.Audio.MusicGen;

/// <summary>
/// Top-level MusicGen text-to-music generation: prompt -&gt; T5 conditioning -&gt; autoregressive
/// delayed-pattern decode over 4 EnCodec codebooks (with classifier-free guidance) -&gt; EnCodec
/// decode -&gt; mono 32kHz PCM.
///
/// <para><b>Classifier-free guidance null condition</b>: this implementation uses the real
/// AudioCraft/MusicGen convention of an all-ZERO encoder_hidden_states for the unconditional
/// branch (not a T5 encoding of an empty string) -- confirmed against AudioCraft's
/// `ClassifierFreeGuidanceDropout`/`ConditionFuser` null-conditioning behavior. This is the one
/// piece of this pipeline NOT yet golden-verified against a real independent reference run (see
/// docs/062-musicgen-implementation-plan.md) -- worth double-checking against real HF
/// `MusicgenForConditionalGeneration.generate` output before trusting audio quality closely tied
/// to guidance strength.</para>
/// </summary>
public sealed class MusicGenGenerator
{
    private readonly NonGatedT5EncoderWeights _textEncoderWeights;
    private readonly T5Tokenizer _tokenizer;
    private readonly MusicGenTransformerWeights _transformerWeights;
    private readonly Primitives.EncodecDecoderWeights _codecWeights;

    public MusicGenGenerator(
        NonGatedT5EncoderWeights textEncoderWeights,
        T5Tokenizer tokenizer,
        MusicGenTransformerWeights transformerWeights,
        Primitives.EncodecDecoderWeights codecWeights)
    {
        _textEncoderWeights = textEncoderWeights;
        _tokenizer = tokenizer;
        _transformerWeights = transformerWeights;
        _codecWeights = codecWeights;
    }

    /// <summary>Generates mono 32kHz PCM for `durationSeconds` of audio from a text prompt. `seed` drives sampling; pass `topK &lt;= 1` (or `temperature &lt;= 0`) for deterministic greedy decoding, useful for golden tests.</summary>
    public float[] Generate(string prompt, float durationSeconds, int seed = 0,
        float guidanceScale = MusicGenConfig.DefaultGuidanceScale,
        int topK = MusicGenConfig.DefaultTopK,
        float temperature = MusicGenConfig.DefaultTemperature)
    {
        int frames = Math.Max(1, (int)MathF.Round(durationSeconds * MusicGenConfig.FrameRate));

        var promptTokens = _tokenizer.Tokenize(prompt);
        var conditionalHidden = MusicGenTextEncoder.Forward(_textEncoderWeights, promptTokens);

        var condCache = new MusicGenTransformer.KvCache();
        MusicGenTransformer.PrepareCrossAttention(_transformerWeights, conditionalHidden, condCache);

        MusicGenTransformer.KvCache? uncondCache = null;
        bool useCfg = guidanceScale > 1.0f;
        if (useCfg)
        {
            var zeroHidden = new float[conditionalHidden.Length][];
            for (int i = 0; i < zeroHidden.Length; i++) zeroHidden[i] = new float[MusicGenConfig.TextDModel];
            uncondCache = new MusicGenTransformer.KvCache();
            MusicGenTransformer.PrepareCrossAttention(_transformerWeights, zeroHidden, uncondCache);
        }

        var rng = new Random(seed);
        int codebooks = MusicGenConfig.NumCodebooks;
        var generated = new int[codebooks][];
        for (int q = 0; q < codebooks; q++) generated[q] = new int[frames];

        var generatedSoFar = new int[codebooks][];
        for (int q = 0; q < codebooks; q++) generatedSoFar[q] = [];
        int seqLen = frames + codebooks - 1;

        for (int step = 0; step < seqLen; step++)
        {
            var column = DelayPattern.InputColumnForStep(codebooks, step, generatedSoFar, MusicGenConfig.PadTokenId);

            var condLogits = MusicGenTransformer.Step(_transformerWeights, column, condCache);
            float[][] logits = condLogits;

            if (useCfg)
            {
                var uncondLogits = MusicGenTransformer.Step(_transformerWeights, column, uncondCache!);
                logits = new float[codebooks][];
                for (int q = 0; q < codebooks; q++)
                {
                    var g = new float[MusicGenConfig.CodebookSize];
                    for (int i = 0; i < g.Length; i++)
                        g[i] = uncondLogits[q][i] + guidanceScale * (condLogits[q][i] - uncondLogits[q][i]);
                    logits[q] = g;
                }
            }

            // Only codebooks whose real turn has arrived (step >= q) AND still have frames left
            // produce a real sampled token this step; earlier/later positions stay implicit PAD
            // in `generatedSoFar` per DelayPattern's contract (never read back by NextInputColumn
            // until their real turn).
            for (int q = 0; q < codebooks; q++)
            {
                int localIndex = step - q;
                if (localIndex < 0 || localIndex >= frames) continue;
                int token = topK <= 1 || temperature <= 0f
                    ? ArgMax(logits[q])
                    : SampleTopK(logits[q], topK, temperature, rng);
                generated[q][localIndex] = token;
                generatedSoFar[q] = [.. generatedSoFar[q], token];
            }
        }

        return Primitives.EncodecDecoderKernels.Decode(_codecWeights, generated);
    }

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        float bestVal = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    private static int SampleTopK(float[] logits, int topK, float temperature, Random rng)
    {
        int k = Math.Min(topK, logits.Length);
        var indices = new int[logits.Length];
        for (int i = 0; i < indices.Length; i++) indices[i] = i;
        Array.Sort(indices, (a, b) => logits[b].CompareTo(logits[a]));

        var topLogits = new float[k];
        for (int i = 0; i < k; i++) topLogits[i] = logits[indices[i]] / temperature;

        float max = float.NegativeInfinity;
        for (int i = 0; i < k; i++) if (topLogits[i] > max) max = topLogits[i];
        float sum = 0f;
        for (int i = 0; i < k; i++) { topLogits[i] = MathF.Exp(topLogits[i] - max); sum += topLogits[i]; }

        double r = rng.NextDouble() * sum;
        double acc = 0;
        for (int i = 0; i < k; i++)
        {
            acc += topLogits[i];
            if (r <= acc) return indices[i];
        }
        return indices[k - 1];
    }
}
