#nullable enable

using System.Text;
using OpenTail.Stingray.Core.Catalog;
using OpenTail.Stingray.Executors;

namespace OpenTail.Stingray;

/// <summary>
/// One-call entry point for the common chat case: resolves <c>modelIdOrPath</c> — an existing file
/// path, a name already present in <see cref="ModelHome"/>, or a <see cref="ModelCatalog"/> id/task
/// (e.g. <c>"qwen2.5-0.5b"</c> or <c>"chat"</c>) — downloading and SHA-256-verifying it via
/// <see cref="ModelInstaller"/> when missing (the same path <c>stingray setup chat</c> uses), loads
/// it, and returns a ready <see cref="StingraySession"/>.
///
/// <para>This is thin convenience over <see cref="Model"/>, <see cref="ModelContext"/>,
/// <see cref="InteractiveExecutor"/> and <see cref="ChatSession"/> — nothing here changes how those
/// work. Reach for them directly for anything this doesn't cover: vision, speech, embeddings,
/// continuous batching, multiple sessions over one loaded model, or custom sampling/execution
/// plans.</para>
/// </summary>
public static class Stingray
{
    /// <summary>
    /// Resolves and loads a chat model, auto-downloading it through the model catalog if it isn't
    /// already present, and returns a ready-to-use <see cref="StingraySession"/>.
    /// </summary>
    /// <param name="modelIdOrPath">
    /// An existing <c>.gguf</c>/<c>.safetensors</c> path, a filename already in <see cref="ModelHome"/>,
    /// or a <see cref="ModelCatalog"/> id (e.g. <c>"qwen2.5-0.5b"</c>) or task (e.g. <c>"chat"</c>).
    /// </param>
    /// <param name="options">Tuning knobs; defaults cover the common case.</param>
    /// <param name="cancellationToken">Cancels a model download in progress.</param>
    /// <exception cref="FileNotFoundException">
    /// <paramref name="modelIdOrPath"/> is not an existing file, a model-home entry, or a known
    /// catalog id/task.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// The resolved catalog entry's licence needs explicit consent
    /// (<see cref="StingrayOptions.AcceptLicence"/>) before it can be downloaded.
    /// </exception>
    public static async Task<StingraySession> CreateAsync(
        string modelIdOrPath,
        StingrayOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrPath);
        options ??= new StingrayOptions();

        string path = await ResolveOrDownloadAsync(modelIdOrPath, options, cancellationToken).ConfigureAwait(false);

        var model = Model.Load(path);
        IModelContext? context = null;
        try
        {
            context = model.CreateContext(new ContextParams { ContextSize = options.ContextSize });
            var executor = new InteractiveExecutor(context);
            var session = new ChatSession(executor);
            return new StingraySession(model, context, session);
        }
        catch
        {
            // Nothing else holds a reference to context/model yet on this failure path.
            context?.Dispose();
            model.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Mirrors what <see cref="ModelHome.ResolveModelPath"/> and <c>stingray setup</c> each do on
    /// their own: try an existing/already-installed path first, then fall back to the catalog
    /// (by id, else by task) and download through <see cref="ModelInstaller"/> if needed.
    /// </summary>
    private static async Task<string> ResolveOrDownloadAsync(
        string modelIdOrPath, StingrayOptions options, CancellationToken cancellationToken)
    {
        if (ModelHome.TryResolveModelPath(modelIdOrPath, out string? resolvedPath))
            return resolvedPath!;

        var home = ModelHome.Default();
        var entry = ModelCatalog.Find(modelIdOrPath) ?? ModelCatalog.DefaultFor(modelIdOrPath);
        if (entry is null)
        {
            throw new FileNotFoundException(
                $"'{modelIdOrPath}' is neither an existing file, a name already in the model home " +
                $"({home.Root}), nor a known catalog model or task. Known tasks: {string.Join(", ", ModelCatalog.Tasks)}. " +
                "Pass an explicit model file path, or run `stingray pull -r <repo>` first.");
        }

        if (entry.LicenceNeedsConsent && !options.AcceptLicence)
        {
            throw new InvalidOperationException(
                $"Model '{entry.Id}' requires accepting its licence before it can be downloaded: {entry.Licence} " +
                $"Set {nameof(StingrayOptions)}.{nameof(StingrayOptions.AcceptLicence)} = true to proceed.");
        }

        using var http = ModelDownloader.CreateClient("OpenTail.Stingray");
        ModelInstaller.FileProgress? progress = options.DownloadProgress is null
            ? null
            : (_, bytes, total) => options.DownloadProgress.Report((bytes, total));

        return await ModelInstaller.EnsureAsync(entry, home, http, onFileStart: null, progress, cancellationToken)
            .ConfigureAwait(false);
    }
}

/// <summary>Tuning knobs for <see cref="Stingray.CreateAsync"/>. Defaults cover the common case.</summary>
public sealed class StingrayOptions
{
    /// <summary>Maximum context length in tokens. <c>0</c> (the default) selects the model's own default.</summary>
    public uint ContextSize { get; init; } = 0;

    /// <summary>
    /// Must be <c>true</c> to download a catalog entry whose licence is not plainly permissive
    /// (mirrors <c>stingray setup --accept-licence</c>). Most catalog entries don't need this;
    /// <see cref="Stingray.CreateAsync"/> throws <see cref="InvalidOperationException"/> naming the
    /// licence when one does and this is left <c>false</c>.
    /// </summary>
    public bool AcceptLicence { get; init; } = false;

    /// <summary>Optional progress reporter while a model is being downloaded: (bytes so far, total bytes if known).</summary>
    public IProgress<(long Bytes, long? Total)>? DownloadProgress { get; init; }
}

/// <summary>
/// A loaded chat model plus a ready <see cref="ChatSession"/>, as returned by
/// <see cref="Stingray.CreateAsync"/>. Owns the underlying <see cref="Model"/> and
/// <see cref="IModelContext"/>; dispose it (or <c>await using</c> it) when done.
/// </summary>
public sealed class StingraySession : IDisposable, IAsyncDisposable
{
    private readonly Model _model;
    private readonly ChatSession _session;
    private bool _disposed;

    /// <summary>
    /// Escape hatch to the underlying context — its <see cref="IModelContext.Engine"/>,
    /// <see cref="IModelContext.Tokenizer"/>, etc. — for anything <see cref="StingraySession"/>
    /// doesn't expose directly.
    /// </summary>
    public IModelContext Context { get; }

    /// <summary>The underlying loaded model.</summary>
    public Model Model => _model;

    /// <summary>This session's conversation history.</summary>
    public ChatHistory History => _session.History;

    /// <summary>The underlying <see cref="ChatSession"/>, for access beyond <see cref="ChatAsync"/>/<see cref="AskAsync"/>.</summary>
    public ChatSession Session => _session;

    internal StingraySession(Model model, IModelContext context, ChatSession session)
    {
        _model = model;
        Context = context;
        _session = session;
    }

    /// <summary>
    /// Sends a user message and streams the reply as it's generated. The primary chat API —
    /// see <see cref="AskAsync"/> for a buffered convenience over the same underlying stream.
    /// </summary>
    public IAsyncEnumerable<string> ChatAsync(string message, CancellationToken cancellationToken = default) =>
        _session.ChatAsync(message, cancellationToken: cancellationToken);

    /// <summary>
    /// Sends a user message and returns the complete reply once generation finishes. Convenience
    /// over <see cref="ChatAsync"/> for callers who don't need to stream — no separate inference
    /// path, just buffers the same chunks.
    /// </summary>
    public async Task<string> AskAsync(string message, CancellationToken cancellationToken = default)
    {
        var reply = new StringBuilder();
        await foreach (var chunk in ChatAsync(message, cancellationToken).ConfigureAwait(false))
            reply.Append(chunk);
        return reply.ToString();
    }

    /// <summary>Disposes the underlying context, then the underlying model. Idempotent.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Context.Dispose();
        _model.Dispose();
    }

    /// <summary>
    /// Delegates to <see cref="Dispose"/>. <see cref="ModelContext"/> and <see cref="OpenTail.Stingray.Engine.Runtime.RuntimeInstance"/>
    /// only expose synchronous disposal today — the one genuinely asynchronous drain path in this
    /// stack lives on <see cref="OpenTail.Stingray.Engine.InferenceEngine.DisposeAsync"/>, one level
    /// below what this call chain reaches. This overload exists for <c>await using</c> convenience,
    /// not because teardown is non-blocking: it still synchronously cancels and waits (with a
    /// timeout) for any in-flight generation to drain before returning.
    /// </summary>
    public ValueTask DisposeAsync()
    {
        Dispose();
        return ValueTask.CompletedTask;
    }
}
