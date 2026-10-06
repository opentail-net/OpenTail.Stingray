#nullable enable

using OpenTail.Stingray.Core;

namespace OpenTail.Stingray;

/// <summary>
/// Concrete loaded model representation holding weights and metadata.
/// Implements <see cref="IModel"/> for lifecycle and context creation.
/// Reusable across multiple execution contexts.
/// </summary>
public sealed class Model : IModel
{
    private readonly IModelTensorSource _tensorSource;
    private readonly GgufModel? _ggufModel;
    private readonly ModelHyperparams _hp;
    private readonly string _arch;
    private readonly HashSet<ModelContext> _activeContexts = [];
    private readonly object _lock = new();
    private bool _disposed;

    /// <inheritdoc/>
    public string ModelPath { get; }

    /// <inheritdoc/>
    public string Architecture => _arch;

    /// <inheritdoc/>
    public int ContextLength => _hp.ContextLength;

    /// <inheritdoc/>
    public int EmbeddingLength => _hp.EmbeddingDim;

    /// <summary>Hyperparameters derived from model metadata.</summary>
    public ModelHyperparams Hyperparams => _hp;

    /// <summary>Model load configuration parameters.</summary>
    public ModelParams Parameters { get; }

    /// <summary>Underlying model tensor source (GGUF or SafeTensors).</summary>
    public IModelTensorSource TensorSource => _tensorSource;

    /// <summary>Underlying memory-mapped GGUF model handle, or throws if loaded from non-GGUF format.</summary>
    public GgufModel Gguf => _ggufModel ?? throw new InvalidOperationException("Model was not loaded from a GGUF file.");

    /// <summary>Whether this model was loaded from a GGUF file.</summary>
    public bool IsGguf => _ggufModel is not null;

    /// <summary>Number of currently active child contexts created from this model.</summary>
    public int ActiveContextCount
    {
        get
        {
            lock (_lock)
            {
                return _activeContexts.Count;
            }
        }
    }

    private Model(string modelPath, IModelTensorSource tensorSource, GgufModel? ggufModel, ModelHyperparams hp, string arch, ModelParams parameters)
    {
        ModelPath = modelPath;
        _tensorSource = tensorSource;
        _ggufModel = ggufModel;
        _hp = hp;
        _arch = arch;
        Parameters = parameters;
    }

    /// <summary>
    /// Loads model weights from the specified file path using default model parameters.
    /// </summary>
    public static Model Load(string modelPath) => Load(new ModelParams(modelPath));

    /// <summary>
    /// Loads model weights according to the specified model parameters.
    /// Supports GGUF files and SafeTensors packages.
    /// </summary>
    public static Model Load(ModelParams parameters)
    {
        ArgumentNullException.ThrowIfNull(parameters);

        if (parameters.UseMemoryLock)
            throw new NotSupportedException("Memory locking (mlock) is not supported in OpenTail.Stingray.");
        if (!parameters.UseMemoryMap)
            throw new NotSupportedException("Disabling memory mapping (no-mmap) is not supported in OpenTail.Stingray.");
        if (parameters.MainGpu != 0)
            throw new NotSupportedException("Multi-GPU device index selection is not supported in OpenTail.Stingray (expected MainGpu = 0).");
        if (parameters.TensorSplit is { Count: > 1 })
            throw new NotSupportedException("Multi-GPU tensor split is not supported in OpenTail.Stingray.");

        bool isDirectory = Directory.Exists(parameters.ModelPath);
        bool isSafeTensors = parameters.ModelPath.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase) || isDirectory;

        if (!File.Exists(parameters.ModelPath) && !isDirectory)
        {
            throw new FileNotFoundException($"Model file not found: {parameters.ModelPath}", parameters.ModelPath);
        }

        IModelTensorSource tensorSource;
        GgufModel? gguf = null;

        if (isSafeTensors)
        {
            var st = SafetensorsTensorSource.Open(parameters.ModelPath);
            tensorSource = st;
        }
        else
        {
            gguf = GgufModel.Open(parameters.ModelPath);
            tensorSource = gguf;
        }

        var hp = ModelHyperparams.FromGgufMetadata(tensorSource.Metadata, tensorSource);
        var arch = tensorSource.Metadata.TryGetValue("general.architecture", out var a) ? (string)a : "unknown";
        return new Model(parameters.ModelPath, tensorSource, gguf, hp, arch, parameters);
    }

    /// <inheritdoc/>
    public IModelContext CreateContext(IContextParams? contextParams = null)
    {
        ThrowIfDisposed();
        var concreteParams = contextParams as ContextParams ?? (contextParams is not null ? new ContextParams
        {
            ContextSize = contextParams.ContextSize,
            BatchSize = contextParams.BatchSize,
            ThreadCount = contextParams.ThreadCount,
            FlashAttention = contextParams.FlashAttention,
            TurboQuantMode = contextParams.TurboQuantMode,
            TurboQuantHeadDim = contextParams.TurboQuantHeadDim,
            RopeFrequencyBase = contextParams.RopeFrequencyBase,
            RopeFrequencyScale = contextParams.RopeFrequencyScale
        } : new ContextParams());

        var context = new ModelContext(this, concreteParams);
        lock (_lock)
        {
            _activeContexts.Add(context);
        }
        return context;
    }

    internal void UnregisterContext(ModelContext context)
    {
        lock (_lock)
        {
            _activeContexts.Remove(context);
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_lock)
        {
            if (_disposed) return;
            var contexts = _activeContexts.ToArray();
            foreach (var ctx in contexts)
            {
                ctx.Dispose();
            }
            _activeContexts.Clear();
            if (_tensorSource is IDisposable disposableSource)
            {
                disposableSource.Dispose();
            }
            _disposed = true;
        }
    }
}
