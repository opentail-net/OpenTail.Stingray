using Microsoft.Extensions.Options;

namespace OpenTail.Stingray.Server;

/// <summary>
/// OpenTail-facing multi-model entry point (docs/032-multi-model-inference-runtime-plan.md
/// §"OpenTail-facing API", Phase 7). Deliberately narrower than the plan's own
/// <c>IInferenceService.GenerateAsync(InferenceRequest, ct)</c> sketch: an endpoint needs the
/// acquired <see cref="ModelRuntime.Loaded"/> bundle (chat template, tokenizer, grammar,
/// tool-boundary tokens) to build its prompt <em>before</em> it can generate at all — that's
/// already how every chat-style endpoint works today, just against a single injected
/// <see cref="IInferenceEngine"/>. Wrapping generation itself behind a narrower
/// request/response-shaped API would mean either duplicating prompt-building inside this
/// interface (moving business logic that belongs to the endpoints) or inventing a callback shape
/// just to hand the bundle back out — <see cref="AcquireAsync"/> handing back the real
/// <see cref="ModelRuntimeHandle"/> is the shape that actually fits: endpoints keep using
/// <c>handle.Runtime.Loaded.Engine.GenerateChunksAsync(...)</c> exactly as they use the injected
/// singleton today, just resolved per-request instead of once at startup.
/// </summary>
public interface IInferenceService
{
    /// <summary>
    /// Resolves a client-supplied model name (the OpenAI/Anthropic/Responses request's
    /// <c>model</c> field) to a canonical <see cref="ModelId"/>. In single-model mode
    /// (<see cref="OpenTailStingrayServerOptions.Models"/> empty), <paramref name="requestedModel"/>
    /// is accepted but ignored — every request resolves to the one configured model, exactly like
    /// today (the response's <c>model</c> field has always reported the loaded engine's own
    /// <c>ModelId</c>, never echoed the request). In multi-model mode, <paramref name="requestedModel"/>
    /// is matched case-insensitively against each <see cref="NamedModelOptions.Alias"/>;
    /// null/empty resolves to the first configured model (the deployment's de facto default); an
    /// unrecognized name throws <see cref="ModelNotFoundException"/>.
    /// </summary>
    ModelId ResolveModel(string? requestedModel);

    /// <summary>Acquires a lease on <paramref name="model"/> — thin passthrough to
    /// <see cref="IModelRuntimeManager.AcquireAsync"/>. Caller disposes when done; see
    /// <see cref="ModelRuntimeHandle"/>'s own contract (scope the handle to the operation being
    /// served, not retained as a "currently selected model" marker).</summary>
    ValueTask<ModelRuntimeHandle> AcquireAsync(ModelId model, CancellationToken ct = default);

    /// <summary>
    /// Client-facing model names. In single-model mode this is the configured model's
    /// canonicalized <see cref="ModelId"/> — <b>not</b> the same string as the loaded engine's own
    /// <c>IInferenceEngine.ModelId</c> (a bare filename), so <c>/v1/models</c> must keep using the
    /// injected engine directly for that mode rather than this property, to stay byte-for-byte
    /// unchanged (see <see cref="IsMultiModel"/>). In multi-model mode this returns every
    /// configured <see cref="NamedModelOptions.Alias"/>, in configuration order — the shape
    /// <c>/v1/models</c> should use there.
    /// </summary>
    IReadOnlyList<string> AvailableModelAliases { get; }

    /// <summary>
    /// True when <see cref="OpenTailStingrayServerOptions.Models"/> is non-empty. Endpoints that
    /// need to keep single-model behavior byte-for-byte unchanged (e.g. <c>/v1/models</c>, which
    /// reports the injected <see cref="IInferenceEngine"/>'s own <c>ModelId</c> rather than
    /// <see cref="AvailableModelAliases"/> in that mode) branch on this rather than inferring it
    /// from <c>AvailableModelAliases.Count == 1</c>, which multi-model mode could also produce.
    /// </summary>
    bool IsMultiModel { get; }
}

/// <summary>Thrown by <see cref="IInferenceService.ResolveModel"/> when a multi-model deployment
/// receives a <c>model</c> field that doesn't match any configured <see cref="NamedModelOptions.Alias"/>.
/// Endpoints should map this to an OpenAI/Anthropic-shaped "model not found" error response rather
/// than letting it surface as a generic 500.</summary>
public sealed class ModelNotFoundException(string requestedModel, IReadOnlyList<string> availableAliases)
    : InvalidOperationException(
        $"Unknown model '{requestedModel}'. Available: {string.Join(", ", availableAliases)}.")
{
    public string RequestedModel { get; } = requestedModel;
    public IReadOnlyList<string> AvailableAliases { get; } = availableAliases;
}

/// <inheritdoc cref="IInferenceService"/>
internal sealed class InferenceService : IInferenceService
{
    private readonly IModelRuntimeManager _manager;

    // Exactly one of these two is non-null, decided once at construction from
    // OpenTailStingrayServerOptions.Models — mirrors the same empty-means-unchanged bimodal split
    // ServiceCollectionExtensions already uses for the IModelRuntimeManager loader closure.
    private readonly ModelId? _singleModelId;
    private readonly Dictionary<string, ModelId>? _aliasToId;
    private readonly IReadOnlyList<string> _aliases;

    public InferenceService(IModelRuntimeManager manager, IOptions<OpenTailStingrayServerOptions> options)
    {
        _manager = manager;
        var opts = options.Value;

        if (opts.Models.Count == 0)
        {
            var id = ModelId.Canonicalize(opts.ModelPath ?? "engine-factory-model");
            _singleModelId = id;
            _aliases = [id.Value];
            return;
        }

        var map = new Dictionary<string, ModelId>(StringComparer.OrdinalIgnoreCase);
        var aliases = new List<string>(opts.Models.Count);
        foreach (var named in opts.Models)
        {
            if (!map.TryAdd(named.Alias, ModelId.Canonicalize(named.ModelPath)))
            {
                throw new InvalidOperationException(
                    $"Duplicate model alias '{named.Alias}' in OpenTailStingrayServerOptions.Models " +
                    "— aliases must be unique.");
            }
            aliases.Add(named.Alias);
        }
        _aliasToId = map;
        _aliases = aliases;
    }

    public IReadOnlyList<string> AvailableModelAliases => _aliases;

    public bool IsMultiModel => _aliasToId is not null;

    public ModelId ResolveModel(string? requestedModel)
    {
        if (_singleModelId is { } single) return single;

        if (string.IsNullOrWhiteSpace(requestedModel))
            return _aliasToId![_aliases[0]];

        if (_aliasToId!.TryGetValue(requestedModel, out var id)) return id;

        throw new ModelNotFoundException(requestedModel, _aliases);
    }

    public ValueTask<ModelRuntimeHandle> AcquireAsync(ModelId model, CancellationToken ct = default) =>
        _manager.AcquireAsync(model, ct);
}
