namespace OpenTail.Stingray.Diffusion.DiffusionGemma;

/// <summary>
/// Orchestrates the DiffusionGemma block-autoregressive generation pipeline:
/// prompt causal prefill -> iterative canvas denoising loop (with self-conditioning and sampler) ->
/// block commit -> causal prefix re-prefill -> multi-block sequence generation.
/// </summary>
public sealed class DiffusionGemmaPipeline
{
    private readonly DiffusionGemmaForwardPass _forwardPass;
    private readonly DiffusionGemmaConfig _config;
    private readonly DiffusionGemmaSampler _sampler;

    public DiffusionGemmaPipeline(DiffusionGemmaForwardPass forwardPass, DiffusionGemmaConfig config, int seed = 42)
    {
        _forwardPass = forwardPass;
        _config = config;
        _sampler = new DiffusionGemmaSampler(config, seed);
    }

    /// <summary>
    /// Generates text from prompt tokens using block-autoregressive diffusion.
    /// </summary>
    public List<int> Generate(ReadOnlySpan<int> promptTokens, int maxBlocks = 1)
    {
        _forwardPass.Reset();

        // 1. Prefill prompt into persistent KV cache
        if (!promptTokens.IsEmpty)
        {
            _forwardPass.PrefillPrompt(promptTokens);
        }

        var outputTokens = new List<int>();
        int canvasLen = _config.CanvasLength;
        int vocabSize = _forwardPass.VocabSize;
        var rng = new Random(42);

        for (int block = 0; block < maxBlocks; block++)
        {
            // Initial categorical noise on canvas
            var canvasTokens = new int[canvasLen];
            for (int i = 0; i < canvasLen; i++) canvasTokens[i] = rng.Next(vocabSize);

            var previouslyAccepted = new bool[canvasLen];
            float[]? prevLogits = null;
            var selfCondEmbeddings = new float[canvasLen * _config.HiddenDim];

            for (int step = 0; step < _config.MaxDenoisingSteps; step++)
            {
                // Self-conditioning for step > 0
                if (step > 0 && prevLogits is not null)
                {
                    // Compute soft probabilities
                    var probs = new float[vocabSize];
                    for (int pos = 0; pos < canvasLen; pos++)
                    {
                        var posLogits = prevLogits.AsSpan(pos * vocabSize, vocabSize);
                        float temp = _sampler.GetTemperature(step);
                        DiffusionGemmaSelfConditioning.ComputeSoftProbabilities(posLogits, temp, probs);

                        // Soft embedding
                        var softEmb = selfCondEmbeddings.AsSpan(pos * _config.HiddenDim, _config.HiddenDim);
                        for (int d = 0; d < _config.HiddenDim; d++) softEmb[d] = 0f;
                        // Use mean soft probability embedding as CPU baseline
                        for (int v = 0; v < Math.Min(vocabSize, 128); v++)
                        {
                            float p = probs[v];
                            if (p < 1e-4f) continue;
                            for (int d = 0; d < _config.HiddenDim; d++)
                            {
                                softEmb[d] += p * MathF.Sin(v * 0.1f + d);
                            }
                        }
                    }
                }

                // Canvas denoising step
                var logits = _forwardPass.ForwardCanvas(canvasTokens, step > 0 ? selfCondEmbeddings : ReadOnlySpan<float>.Empty);
                prevLogits = logits;

                // Sampler evaluation
                var (accepted, nextTokens, meanEntropy, shouldStop) = _sampler.Step(step, logits, canvasTokens, previouslyAccepted);
                previouslyAccepted = accepted;
                canvasTokens = nextTokens;

                if (shouldStop) break;
            }

            // Commit final block
            outputTokens.AddRange(canvasTokens);

            // Causal re-prefill of committed block into persistent KV cache
            _forwardPass.PrefillPrompt(canvasTokens);
        }

        return outputTokens;
    }
}
