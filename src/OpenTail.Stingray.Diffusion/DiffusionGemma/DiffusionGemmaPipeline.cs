namespace OpenTail.Stingray.Diffusion.DiffusionGemma;

/// <summary>
/// Orchestrates the DiffusionGemma block-autoregressive generation pipeline:
/// prompt causal prefill -> iterative canvas denoising loop (with self-conditioning and EntropyBound sampler) ->
/// block commit -> causal prefix prefill -> multi-block sequence generation.
/// </summary>
public sealed class DiffusionGemmaPipeline
{
    private readonly DiffusionGemmaForwardPass _forwardPass;
    private readonly DiffusionGemmaConfig _config;
    private readonly DiffusionGemmaSampler _sampler;
    private readonly DiffusionGemmaSampler.DeterministicRng _rng;

    public DiffusionGemmaPipeline(DiffusionGemmaForwardPass forwardPass, DiffusionGemmaConfig config, int seed = 42)
    {
        _forwardPass = forwardPass;
        _config = config;
        _sampler = new DiffusionGemmaSampler(config, seed);
        _rng = new DiffusionGemmaSampler.DeterministicRng((ulong)seed);
    }

    /// <summary>
    /// Generates text from prompt tokens using block-autoregressive diffusion.
    /// </summary>
    public List<int> Generate(ReadOnlySpan<int> promptTokens, int maxBlocks = 1)
    {
        _forwardPass.Reset();
        _sampler.Reset();

        // 1. Prefill prompt into persistent KV cache
        if (!promptTokens.IsEmpty)
        {
            _forwardPass.PrefillPrompt(promptTokens);
        }

        var outputTokens = new List<int>();
        int canvasLen = _config.CanvasLength;
        int vocabSize = _forwardPass.VocabSize;

        for (int block = 0; block < maxBlocks; block++)
        {
            // Initial categorical noise on canvas using deterministic RNG
            var canvasTokens = new int[canvasLen];
            for (int i = 0; i < canvasLen; i++) canvasTokens[i] = _rng.NextInt(vocabSize);

            var previouslyAccepted = new bool[canvasLen];
            var prevArgmax = new int[canvasLen];
            Array.Fill(prevArgmax, -1);   // step 0 has no previous prediction

            var argmaxCanvas = new int[canvasLen];
            float[]? prevLogits = null;
            float[] selfCondEmbeddings = [];

            for (int step = 0; step < _config.MaxDenoisingSteps; step++)
            {
                // Self-conditioning from previous step's temperature-shaped distribution
                if (step > 0 && prevLogits is not null)
                {
                    float prevTemp = _sampler.GetTemperature(step - 1);
                    var softEmb = _forwardPass.ComputeSoftEmbeddings(prevLogits, prevTemp, canvasLen);
                    var scSignal = new float[canvasLen * _config.HiddenDim];
                    _forwardPass.ApplySelfCondMlp(softEmb, scSignal, canvasLen);
                    selfCondEmbeddings = scSignal;
                }

                // Canvas denoising step
                var logits = _forwardPass.ForwardCanvas(canvasTokens, step > 0 ? selfCondEmbeddings : ReadOnlySpan<float>.Empty);
                prevLogits = logits;

                // Sampler evaluation (EntropyBound acceptance + adaptive stopping)
                var (accepted, nextTokens, meanEntropy, shouldStop, argmax) = _sampler.Step(step, logits, prevArgmax, previouslyAccepted);
                previouslyAccepted = accepted;
                canvasTokens = nextTokens;
                prevArgmax = argmax;
                argmaxCanvas = argmax;

                if (shouldStop) break;
            }

            // Commit the deterministic argmax prediction (never the re-noised working canvas)
            canvasTokens = (int[])argmaxCanvas.Clone();
            outputTokens.AddRange(canvasTokens);

            // Causal prefill of committed block to extend persistent KV cache for subsequent blocks
            _forwardPass.PrefillPrompt(canvasTokens);
        }

        return outputTokens;
    }
}
