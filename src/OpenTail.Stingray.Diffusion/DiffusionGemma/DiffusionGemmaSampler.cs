using System.Numerics.Tensors;

namespace OpenTail.Stingray.Diffusion.DiffusionGemma;

/// <summary>
/// Sampler state and decision logic for DiffusionGemma.
/// Implements temperature decay, EntropyBound acceptance via multinomial inverse CDF,
/// categorical re-noising, and argmax-stability adaptive stopping.
/// </summary>
public sealed class DiffusionGemmaSampler
{
    private readonly DiffusionGemmaConfig _config;
    private readonly DeterministicRng _rng;
    private int _held;

    public DiffusionGemmaSampler(DiffusionGemmaConfig config, int seed = 42)
    {
        _config = config;
        _rng = new DeterministicRng((ulong)seed);
    }

    public void Reset()
    {
        _held = 0;
    }

    /// <summary>
    /// Computes temperature for denoising step <paramref name="step"/> in [0, maxSteps - 1].
    /// Linearly decays from TemperatureMax (0.8) to TemperatureMin (0.408).
    /// </summary>
    public float GetTemperature(int step)
    {
        int maxSteps = Math.Max(1, _config.MaxDenoisingSteps - 1);
        float progress = Math.Clamp((float)step / maxSteps, 0f, 1f);
        return _config.TemperatureMax - (_config.TemperatureMax - _config.TemperatureMin) * progress;
    }

    /// <summary>
    /// Computes Shannon entropy in nats: H = -sum(p * ln(p)).
    /// </summary>
    public static float ComputeEntropyNats(ReadOnlySpan<float> probs)
    {
        float h = 0f;
        for (int i = 0; i < probs.Length; i++)
        {
            float p = probs[i];
            if (p > 1e-12f)
            {
                h -= p * MathF.Log(p);
            }
        }
        return MathF.Max(0f, h);
    }

    /// <summary>
    /// Evaluates one denoising step across all canvas positions using EntropyBound acceptance.
    /// </summary>
    public (bool[] Accepted, int[] Tokens, float MeanEntropy, bool ShouldStop, int[] ArgmaxTokens) Step(
        int step,
        ReadOnlySpan<float> canvasLogits, // [canvasLength, vocabSize]
        int[] prevArgmaxTokens,           // previous step's highest-probability tokens (NOT the re-noised canvas)
        bool[] previouslyAccepted)
    {
        int canvasLen = _config.CanvasLength;
        int vocabSize = _config.VocabSize;
        float temp = GetTemperature(step);
        float tempInv = 1.0f / MathF.Max(1e-4f, temp);

        var currentTokens = new int[canvasLen];
        var argmaxTokens = new int[canvasLen];
        var denoiserTokens = new int[canvasLen];
        var entropies = new float[canvasLen];
        var u = new float[canvasLen];
        var renoise = new int[canvasLen];

        // Draw deterministic step randomness
        for (int pos = 0; pos < canvasLen; pos++)
        {
            u[pos] = _rng.NextFloat();
            renoise[pos] = _rng.NextInt(vocabSize);
        }

        float totalEntropy = 0f;
        var s = new float[vocabSize];
        var e = new float[vocabSize];

        for (int pos = 0; pos < canvasLen; pos++)
        {
            var row = canvasLogits.Slice(pos * vocabSize, vocabSize);
            TensorPrimitives.Multiply(row, tempInv, s);
            int amax = TensorPrimitives.IndexOfMax<float>(s);
            float m = s[amax];

            TensorPrimitives.Subtract(s, m, e);
            TensorPrimitives.Exp(e, e);
            float z = TensorPrimitives.Sum<float>(e);
            float sz = TensorPrimitives.Dot<float>(e, s);

            // Shannon entropy: H = ln(Z) + m - (sum(e * s)) / Z
            float h = MathF.Log(MathF.Max(1e-12f, z)) + m - (sz / MathF.Max(1e-12f, z));
            entropies[pos] = MathF.Max(0f, h);
            totalEntropy += entropies[pos];

            // Multinomial inverse CDF sample
            float target = u[pos] * z;
            float cum = 0f;
            int sampled = vocabSize - 1;
            for (int v = 0; v < vocabSize; v++)
            {
                cum += e[v];
                if (cum >= target)
                {
                    sampled = v;
                    break;
                }
            }

            argmaxTokens[pos] = amax;
            denoiserTokens[pos] = sampled;
        }

        float meanEntropy = totalEntropy / canvasLen;

        // Rank positions by ascending entropy (lowest entropy = highest confidence)
        var order = new int[canvasLen];
        for (int i = 0; i < canvasLen; i++) order[i] = i;
        Array.Sort(order, (a, b) => entropies[a].CompareTo(entropies[b]));

        // EntropyBound acceptance: sum of strictly earlier accepted entropies <= EntropyBudget
        var accepted = new bool[canvasLen];
        double cumE = 0.0;
        for (int i = 0; i < canvasLen; i++)
        {
            int pos = order[i];
            cumE += entropies[pos];
            if (cumE - entropies[pos] <= _config.EntropyBudgetNats)
            {
                accepted[pos] = true;
            }
        }

        // Re-noise rejected positions with fresh random tokens
        for (int pos = 0; pos < canvasLen; pos++)
        {
            currentTokens[pos] = accepted[pos] ? denoiserTokens[pos] : renoise[pos];
        }

        // Adaptive stopping: argmax canvas stable for at least 1 step AND mean entropy below threshold
        bool same = true;
        for (int pos = 0; pos < canvasLen; pos++)
        {
            if (prevArgmaxTokens == null || argmaxTokens[pos] != prevArgmaxTokens[pos])
            {
                same = false;
                break;
            }
        }

        _held = same ? _held + 1 : 0;
        bool confident = meanEntropy < _config.ConvergenceEntropyThreshold;
        bool shouldStop = (_held >= 1 && confident) || step >= _config.MaxDenoisingSteps - 1;

        return (accepted, currentTokens, meanEntropy, shouldStop, argmaxTokens);
    }

    /// <summary>
    /// Deterministic PRNG (SplitMix64) ensuring exact seed reproducibility.
    /// </summary>
    public sealed class DeterministicRng
    {
        private ulong _state;

        public DeterministicRng(ulong seed)
        {
            _state = seed == 0 ? 0x9E3779B97F4A7C15UL : seed;
        }

        public ulong NextU64()
        {
            ulong z = (_state += 0x9E3779B97F4A7C15UL);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }

        public float NextFloat()
        {
            return (NextU64() >> 40) * (1.0f / 16777216.0f);
        }

        public int NextInt(int bound)
        {
            return (int)(NextU64() % (ulong)bound);
        }
    }
}
