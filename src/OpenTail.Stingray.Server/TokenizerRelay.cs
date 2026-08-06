using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Server;

/// <summary>
/// Bridges the lazy engine load (which produces a tokenizer as a side-effect) to DI consumers
/// such as the <c>/tokenize</c> and <c>/detokenize</c> endpoints that need an
/// <see cref="ITokenizer"/> without re-opening the model.
///
/// <para>The relay is registered as a singleton before <see cref="Engine.IInferenceEngine"/>.
/// When <c>IInferenceEngine</c> is first resolved, <see cref="ServiceCollectionExtensions"/>
/// calls <see cref="Set"/> to populate the relay. Any endpoint that resolves
/// <c>ITokenizer</c> will trigger engine load first (because the engine factory calls
/// <see cref="Set"/>), so the tokenizer is always available by the time it is requested.</para>
/// </summary>
public sealed class TokenizerRelay
{
    private ITokenizer? _tokenizer;

    /// <summary>The loaded tokenizer, or <c>null</c> before the engine has been resolved.</summary>
    public ITokenizer? Tokenizer => _tokenizer;

    /// <summary>Called once by <see cref="ServiceCollectionExtensions"/> after model load.</summary>
    internal void Set(ITokenizer tokenizer) =>
        Interlocked.CompareExchange(ref _tokenizer, tokenizer, null);
}
