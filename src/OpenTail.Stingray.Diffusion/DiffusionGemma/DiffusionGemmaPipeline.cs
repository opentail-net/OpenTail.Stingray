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
            var prevArgmax = new int[canvasLen];
            Array.Fill(prevArgmax, -1);   // no previous prediction at step 0, so step 0 can never count as stable
            var argmaxCanvas = new int[canvasLen];
            float[]? prevLogits = null;
            float[] selfCondEmbeddings = [];

            for (int step = 0; step < _config.MaxDenoisingSteps; step++)
            {
                // Self-conditioning from the previous step's temperature-shaped distribution (exact sum).
                if (step > 0 && prevLogits is not null)
                {
                    selfCondEmbeddings = _forwardPass.ComputeSoftEmbeddings(prevLogits, _sampler.GetTemperature(step), canvasLen);
                }

                // Canvas denoising step
                var logits = _forwardPass.ForwardCanvas(canvasTokens, step > 0 ? selfCondEmbeddings : ReadOnlySpan<float>.Empty);
                prevLogits = logits;

                // Sampler evaluation
                var (accepted, nextTokens, meanEntropy, shouldStop, argmax) = _sampler.Step(step, logits, prevArgmax, previouslyAccepted);
                previouslyAccepted = accepted;
                canvasTokens = nextTokens;
                prevArgmax = argmax;
                argmaxCanvas = argmax;

                if (shouldStop) break;
            }

            // Commit the highest-probability prediction for every position (the re-noised canvas still holds
            // random tokens at non-accepted positions and must never be committed).
            canvasTokens = argmaxCanvas;
            outputTokens.AddRange(canvasTokens);

            // Causal re-prefill of committed block into persistent KV cache
            _forwardPass.PrefillPrompt(canvasTokens);
        }

        return outputTokens;
    }
}
