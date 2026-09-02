
namespace OpenTail.Stingray.Audio.AudioGen;

/// <summary>
/// Top-level AudioGen text-to-sound generation: prompt -&gt; T5-large text conditioning -&gt;
/// autoregressive delayed-pattern decode over 4 EnCodec codebooks (with classifier-free
/// guidance) -&gt; 16kHz EnCodec decode -&gt; mono PCM.
///
/// <para>Structurally almost identical to <see cref="OpenTail.Stingray.Audio.MusicGen.MusicGenGenerator"/>
/// -- proof that the delayed multi-codebook generation LOOP (not just the low-level math) is a
/// genuinely reusable shape across AudioCraft-family models: same
/// <see cref="OpenTail.Stingray.Audio.MusicGen.DelayPattern"/> (real config confirmed identical
/// `delays: [0,1,2,3]`), same CFG combination formula, same top-k/temperature sampling. What
/// differs is entirely at the per-model layer (dims, tensor format/naming, T5 variant, EnCodec
/// ratios) -- already isolated into <see cref="AudioGenTransformer"/>/<see cref="AudioGenTransformerWeights"/>
/// and the shared <see cref="Primitives.EncodecDecoderKernels"/>/<see cref="Primitives.T5EncoderKernels"/>.
/// A further extraction of this loop itself into a shared `AudioTokenGenerator` is a reasonable
/// next DRY step once a THIRD AudioCraft-family model needs it (see
/// docs/063-audiogen-implementation-plan.md) -- not done speculatively here with only two real
/// callers whose per-model glue (KvCache types, Step signatures) still differs enough that a
/// premature interface would likely need reshaping anyway.</para>
///
/// <para><b>Classifier-free guidance null condition</b>: same all-ZERO `encoder_hidden_states`
/// convention as MusicGen -- for AudioGen this is independently CONFIRMED (not guessed) against
/// the real `audiocraft.modules.conditioners.T5Conditioner.forward` source: the null branch
/// still runs T5 on the (empty-string) input, but then multiplies the output by an
/// attention_mask that is explicitly zeroed for empty-string entries (`mask[empty_idx, :] = 0`
/// in `T5Conditioner.tokenize`), so the real null embedding is always all-zero in practice --
/// same end state as MusicGen's implementation, reached by a different real mechanism.</para>
/// </summary>
public sealed class AudioGenGenerator
{
    private readonly NonGatedT5EncoderWeights _textEncoderWeights;
    private readonly T5Tokenizer _tokenizer;
    private readonly AudioGenTransformerWeights _transformerWeights;
    private readonly Primitives.EncodecDecoderWeights _codecWeights;

    public AudioGenGenerator(
        NonGatedT5EncoderWeights textEncoderWeights,
        T5Tokenizer tokenizer,
        AudioGenTransformerWeights transformerWeights,
        Primitives.EncodecDecoderWeights codecWeights)
    {
        _textEncoderWeights = textEncoderWeights;
        _tokenizer = tokenizer;
        _transformerWeights = transformerWeights;
        _codecWeights = codecWeights;
    }

    /// <summary>Generates mono 16kHz PCM for `durationSeconds` of audio from a text prompt. `seed` drives sampling; pass `topK &lt;= 1` (or `temperature &lt;= 0`) for deterministic greedy decoding.</summary>
    public float[] Generate(string prompt, float durationSeconds, int seed = 0,
        float guidanceScale = AudioGenConfig.DefaultGuidanceScale,
        int topK = AudioGenConfig.DefaultTopK,
        float temperature = AudioGenConfig.DefaultTemperature)
    {
        int frames = Math.Max(1, (int)MathF.Round(durationSeconds * AudioGenConfig.FrameRate));

        var promptTokens = _tokenizer.Tokenize(prompt);
        var conditionalHidden = Primitives.T5EncoderKernels.Forward(AudioGenTextEncoderWeights.Dims, _textEncoderWeights, promptTokens);

        var condCache = new AudioGenTransformer.KvCache();
        AudioGenTransformer.PrepareCrossAttention(_transformerWeights, conditionalHidden, condCache);

        AudioGenTransformer.KvCache? uncondCache = null;
        bool useCfg = guidanceScale > 1.0f;
        if (useCfg)
        {
            var zeroHidden = new float[conditionalHidden.Length][];
            for (int i = 0; i < zeroHidden.Length; i++) zeroHidden[i] = new float[AudioGenConfig.TextDModel];
            uncondCache = new AudioGenTransformer.KvCache();
            AudioGenTransformer.PrepareCrossAttention(_transformerWeights, zeroHidden, uncondCache);
        }

        var rng = new Random(seed);
        int codebooks = AudioGenConfig.NumCodebooks;
        var generated = new int[codebooks][];
        for (int q = 0; q < codebooks; q++) generated[q] = new int[frames];

        var generatedSoFar = new int[codebooks][];
        for (int q = 0; q < codebooks; q++) generatedSoFar[q] = [];
        int seqLen = frames + codebooks - 1;

        for (int step = 0; step < seqLen; step++)
        {
            var column = MusicGen.DelayPattern.InputColumnForStep(codebooks, step, generatedSoFar, AudioGenConfig.PadTokenId);

            var condLogits = AudioGenTransformer.Step(_transformerWeights, column, condCache);
            float[][] logits = condLogits;

            if (useCfg)
            {
                var uncondLogits = AudioGenTransformer.Step(_transformerWeights, column, uncondCache!);
                logits = new float[codebooks][];
                for (int q = 0; q < codebooks; q++)
                {
                    var g = new float[AudioGenConfig.CodebookSize];
                    for (int i = 0; i < g.Length; i++)
                        g[i] = uncondLogits[q][i] + guidanceScale * (condLogits[q][i] - uncondLogits[q][i]);
                    logits[q] = g;
                }
            }

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
