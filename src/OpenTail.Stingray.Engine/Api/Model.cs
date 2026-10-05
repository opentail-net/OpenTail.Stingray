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
    private readonly GgufModel _ggufModel;
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

    /// <summary>Underlying memory-mapped GGUF model handle.</summary>
    public GgufModel Gguf => _ggufModel;

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

    private Model(string modelPath, GgufModel ggufModel, ModelHyperparams hp, string arch, ModelParams parameters)
    {
        ModelPath = modelPath;
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
    /// </summary>
    public static Model Load(ModelParams parameters)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        if (!File.Exists(parameters.ModelPath))
        {
            throw new FileNotFoundException($"Model file not found: {parameters.ModelPath}", parameters.ModelPath);
        }

        var gguf = GgufModel.Open(parameters.ModelPath);
        var hp = ModelHyperparams.FromGgufMetadata(gguf.Metadata, gguf);
        var arch = gguf.Metadata.TryGetValue("general.architecture", out var a) ? (string)a : "unknown";
        return new Model(parameters.ModelPath, gguf, hp, arch, parameters);
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
            RopeFrequencyScale = contextParams.RopeFrequencyScale,
            Embeddings = contextParams.Embeddings
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
            _ggufModel.Dispose();
            _disposed = true;
        }
    }
}
