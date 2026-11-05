using System.Collections.Concurrent;
using OpenTail.Stingray.Audio.Whisper;
using OpenTail.Stingray.Core.Catalog;

namespace OpenTail.Stingray.Server;

/// <summary>Supplies the Whisper pipeline for one speech-to-text request.</summary>
internal interface IWhisperPipelineProvider
{
    /// <summary>
    /// A pipeline over real weights for <paramref name="model"/>: null or empty, <c>whisper-1</c> or <c>transcribe</c> mean the catalog's default
    /// transcribe model, anything else is a transcribe entry's id. The caller owns the pipeline and disposes it.
    /// </summary>
    /// <exception cref="TranscriptionModelUnavailableException">The model is known but its files are not installed (or are damaged).</exception>
    WhisperPipeline Create(string? model);
}

/// <summary>A valid transcription model whose weights are not available on this machine. The route answers 503, not a made-up transcript.</summary>
internal sealed class TranscriptionModelUnavailableException(string modelId, string message) : Exception(message)
{
    public string ModelId { get; } = modelId;
}

/// <summary>
/// Loads the installed catalog model (the same files <c>stingray setup transcribe</c> installs and <c>stingray transcribe</c> runs). The parsed weights are
/// cached per file because they are immutable and large; each request gets its own pipeline over them, since the encoder keeps per-instance work buffers
/// and is not safe to share between concurrent requests.
/// </summary>
internal sealed class InstalledWhisperPipelineProvider : IWhisperPipelineProvider
{
    private readonly Func<ModelHome> _home;
    private readonly ConcurrentDictionary<string, Lazy<WhisperGgmlModel>> _weights = new(StringComparer.OrdinalIgnoreCase);

    public InstalledWhisperPipelineProvider(Func<ModelHome>? home = null) => _home = home ?? ModelHome.Default;

    public WhisperPipeline Create(string? model)
    {
        var entry = Resolve(model);
        var home = _home();
        var state = home.StateOf(entry);
        if (state != InstallState.Installed)
            throw new TranscriptionModelUnavailableException(entry.Id,
                $"The transcription model '{entry.Id}' is {(state == InstallState.Partial ? "incomplete or damaged" : "not installed")} on this machine. " +
                "Run: stingray setup transcribe");

        string path = home.PathOf(entry.MainFile);
        var weights = _weights.GetOrAdd(path, p => new Lazy<WhisperGgmlModel>(
            () => p.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) ? WhisperGgmlModel.LoadFromGguf(p) : WhisperGgmlModel.Load(p),
            LazyThreadSafetyMode.ExecutionAndPublication));
        return WhisperPipeline.FromGgmlModel(weights.Value);
    }

    private static CatalogEntry Resolve(string? model)
    {
        if (string.IsNullOrWhiteSpace(model) || model.Equals("whisper-1", StringComparison.OrdinalIgnoreCase)
            || model.Equals("transcribe", StringComparison.OrdinalIgnoreCase))
            return ModelCatalog.DefaultFor("transcribe")
                ?? throw new InvalidOperationException("The model catalog has no transcribe entry.");
        return ModelCatalog.Entries.FirstOrDefault(e => e.Task.Equals("transcribe", StringComparison.OrdinalIgnoreCase)
                   && e.Id.Equals(model, StringComparison.OrdinalIgnoreCase))
               ?? throw new ArgumentException($"'{model}' is not a transcribe model.", nameof(model));
    }
}
