
namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// StreamDiffusion real-time pipelining engine with Residual Classifier-Free Guidance (R-CFG).
/// Uses pipelined temporal batch queues (Batch = Time Step) to generate 1 completed output frame per neural forward pass.
/// </summary>
public sealed class StreamBatchPipeline
{
    private readonly int _batchSize;
    private readonly int _latentElements;
    private readonly float[][] _streamQueue;
    private readonly int[] _timesteps;
    private float[]? _cachedUncondResidual;
    private bool _hasWarmup;

    public int BatchSize => _batchSize;
    public bool IsWarmedUp => _hasWarmup;

    /// <summary>
    /// Initializes a StreamDiffusion pipeline with a fixed temporal batch depth.
    /// </summary>
    /// <param name="batchSize">Number of pipelined denoising stages (e.g. 4 for 4-step LCM).</param>
    /// <param name="latentElements">Total number of elements per latent frame (C * H * W).</param>
    /// <param name="timesteps">Precomputed discrete or continuous timesteps, length = batchSize.</param>
    public StreamBatchPipeline(int batchSize, int latentElements, int[] timesteps)
    {
        if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));
        if (timesteps.Length != batchSize)
            throw new ArgumentException("Timesteps array must match batchSize.", nameof(timesteps));

        _batchSize = batchSize;
        _latentElements = latentElements;
        _timesteps = (int[])timesteps.Clone();

        _streamQueue = new float[batchSize][];
        for (int i = 0; i < batchSize; i++)
        {
            _streamQueue[i] = new float[latentElements];
        }
    }

    /// <summary>
    /// Sets or updates the cached unconditional prompt residual tensor for Residual Classifier-Free Guidance (R-CFG).
    /// Eliminates the need to evaluate unconditional negative prompts on every frame.
    /// </summary>
    public void SetUncondResidual(ReadOnlySpan<float> uncondPrediction)
    {
        if (uncondPrediction.Length != _latentElements)
            throw new ArgumentException("Unconditional prediction length must match latent elements.", nameof(uncondPrediction));

        _cachedUncondResidual = new float[_latentElements];
        uncondPrediction.CopyTo(_cachedUncondResidual);
    }

    /// <summary>
    /// Pushes a new input noisy latent into the front of the queue, shifts existing latents forward,
    /// and retrieves the flattened batched input for parallel neural network evaluation.
    /// </summary>
    /// <param name="newNoisyLatent">New incoming latent frame (e.g. from webcam / img2img input).</param>
    /// <param name="batchedInputOut">Destination buffer to receive all batch frames [batchSize * latentElements].</param>
    public void PrepareBatchInput(ReadOnlySpan<float> newNoisyLatent, Span<float> batchedInputOut)
    {
        // 1. Shift queue: slot i becomes slot i+1
        for (int i = _batchSize - 1; i > 0; i--)
        {
            Array.Copy(_streamQueue[i - 1], _streamQueue[i], _latentElements);
        }

        // 2. Insert new noisy frame into slot 0
        newNoisyLatent.CopyTo(_streamQueue[0]);

        // 3. Pack into contiguous batched tensor
        for (int i = 0; i < _batchSize; i++)
        {
            _streamQueue[i].AsSpan().CopyTo(batchedInputOut.Slice(i * _latentElements, _latentElements));
        }
    }

    /// <summary>
    /// Applies predicted noise/velocity updates across all pipelined stages and pops the finished output frame.
    /// </summary>
    /// <param name="batchedPredictions">Predicted outputs from neural network [batchSize * latentElements].</param>
    /// <param name="outputFrame">Destination span to receive the fully denoised frame popped from the end of the pipeline.</param>
    /// <param name="guidanceScale">R-CFG guidance scale.</param>
    public void StepAndPop(
        ReadOnlySpan<float> batchedPredictions,
        Span<float> outputFrame,
        float guidanceScale = 1.0f)
    {
        for (int i = 0; i < _batchSize; i++)
        {
            var predSlice = batchedPredictions.Slice(i * _latentElements, _latentElements);
            var queueSlice = _streamQueue[i].AsSpan();

            // Apply R-CFG if active and guidance > 1.0
            if (guidanceScale > 1.0f && _cachedUncondResidual != null)
            {
                for (int j = 0; j < _latentElements; j++)
                {
                    float uncond = _cachedUncondResidual[j];
                    float cond = predSlice[j];
                    float guided = uncond + guidanceScale * (cond - uncond);
                    queueSlice[j] -= (1.0f / _batchSize) * guided;
                }
            }
            else
            {
                for (int j = 0; j < _latentElements; j++)
                {
                    queueSlice[j] -= (1.0f / _batchSize) * predSlice[j];
                }
            }
        }

        // Pop the oldest / fully denoised frame from slot (batchSize - 1)
        _streamQueue[_batchSize - 1].AsSpan().CopyTo(outputFrame);
        _hasWarmup = true;
    }

    /// <summary>
    /// Clears the pipeline queue.
    /// </summary>
    public void Reset()
    {
        for (int i = 0; i < _batchSize; i++)
        {
            Array.Clear(_streamQueue[i]);
        }
        _hasWarmup = false;
    }
}
