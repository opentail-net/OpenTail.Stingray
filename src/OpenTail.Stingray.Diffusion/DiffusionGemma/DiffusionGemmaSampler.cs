namespace OpenTail.Stingray.Diffusion.DiffusionGemma;

/// <summary>
/// Sampler state and decision logic for DiffusionGemma.
/// Implements temperature decay, entropy budget acceptance, categorical re-noising, and early stopping.
/// </summary>
public sealed class DiffusionGemmaSampler
{
    private readonly DiffusionGemmaConfig _config;
    private readonly Random _rng;

    public DiffusionGemmaSampler(DiffusionGemmaConfig config, int seed = 42)
    {
        _config = config;
        _rng = new Random(seed);
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
    /// Evaluates one denoising step across all canvas positions.
    /// Returns:
    ///  - accepted: boolean mask indicating which positions were accepted into this step
    ///  - acceptedTokens: token IDs for each canvas position
    ///  - meanEntropy: average entropy in nats across canvas
    ///  - shouldStop: true if convergence threshold is met
    /// </summary>
    public (bool[] Accepted, int[] Tokens, float MeanEntropy, bool ShouldStop) Step(
        int step,
        ReadOnlySpan<float> canvasLogits, // [canvasLength, vocabSize]
        int[] prevTokens,
        bool[] previouslyAccepted)
    {
        int canvasLen = _config.CanvasLength;
        int vocabSize = _config.VocabSize;
        float temp = GetTemperature(step);

        var currentTokens = new int[canvasLen];
        var entropies = new float[canvasLen];
        var probs = new float[vocabSize];

        float totalEntropy = 0f;

        for (int pos = 0; pos < canvasLen; pos++)
        {
            var posLogits = canvasLogits.Slice(pos * vocabSize, vocabSize);
            DiffusionGemmaSelfConditioning.ComputeSoftProbabilities(posLogits, temp, probs);

            entropies[pos] = ComputeEntropyNats(probs);
            totalEntropy += entropies[pos];

            // Greedy argmax token prediction
            int bestToken = 0;
            float bestProb = probs[0];
            for (int v = 1; v < vocabSize; v++)
            {
                if (probs[v] > bestProb)
                {
                    bestProb = probs[v];
                    bestToken = v;
                }
            }
            currentTokens[pos] = bestToken;
        }

        float meanEntropy = totalEntropy / canvasLen;

        // Rank positions by ascending entropy (lowest entropy = highest confidence)
        var rankedPositions = new int[canvasLen];
        for (int i = 0; i < canvasLen; i++) rankedPositions[i] = i;
        Array.Sort(rankedPositions, (a, b) => entropies[a].CompareTo(entropies[b]));

        // Greedy acceptance up to cumulative entropy budget (0.1 nats)
        var accepted = new bool[canvasLen];
        float cumEntropy = 0f;
        for (int i = 0; i < canvasLen; i++)
        {
            int pos = rankedPositions[i];
            if (previouslyAccepted[pos])
            {
                accepted[pos] = true;
                continue;
            }

            if (cumEntropy + entropies[pos] <= _config.EntropyBudgetNats)
            {
                accepted[pos] = true;
                cumEntropy += entropies[pos];
            }
        }

        // Categorical re-noising for non-accepted positions
        for (int pos = 0; pos < canvasLen; pos++)
        {
            if (!accepted[pos])
            {
                // Re-sample token from uniform categorical noise
                currentTokens[pos] = _rng.Next(vocabSize);
            }
        }

        // Stability check: ratio of identical tokens between steps
        int stableCount = 0;
        for (int pos = 0; pos < canvasLen; pos++)
        {
            if (currentTokens[pos] == prevTokens[pos]) stableCount++;
        }
        float stability = (float)stableCount / canvasLen;

        bool shouldStop = (meanEntropy < _config.ConvergenceEntropyThreshold && stability >= 0.999f)
            || step >= _config.MaxDenoisingSteps - 1;

        return (accepted, currentTokens, meanEntropy, shouldStop);
    }
}
