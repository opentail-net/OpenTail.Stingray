namespace OpenTail.Stingray.Core.Grammar;

/// <summary>
/// Forces generation to BEGIN a tool call, for OpenAI <c>tool_choice:"required"</c>.
///
/// <para>This is the mirror image of the argument constraints. Those <i>arm on</i> the model's
/// open marker and shape what follows it; none of them can compel the model to emit one, so a
/// request that asked for a guaranteed call could previously come back as prose. This constraint
/// masks the vocabulary down to the single open-marker token until that token has been emitted,
/// then goes permanently inert and hands over to the argument grammar composed alongside it.</para>
///
/// <para><b>It deliberately forces the marker as the FIRST generated token</b>, which means no
/// preamble and no reasoning block before the call. That is the strict reading of "required": the
/// response must be a tool call, not prose that happens to contain one. On a thinking model this
/// suppresses the think block for that turn, which is a real behavioural cost and the reason the
/// caller must opt in per request rather than it being a server default.</para>
///
/// <para>Constructed only via <see cref="IToolCallAdapter.BuildForcedCallConstraint"/>, which
/// returns null when the family's open marker is not a single vocabulary token. A null result means
/// "this model cannot be forced" and the caller must refuse the request rather than silently
/// degrade to unforced generation — the whole point of the feature is that the client can rely on
/// getting a call.</para>
/// </summary>
public sealed class ForcedToolCallConstraint : ITokenConstraint
{
    private readonly int _openMarkerId;
    private readonly int _vocabSize;
    private float[]? _mask;
    private bool _satisfied;

    /// <param name="vocab">Vocabulary the mask is sized from.</param>
    /// <param name="openMarkerId">
    /// Token id of the family's tool-call open marker. Must be a valid id; callers obtain it from
    /// <see cref="GrammarVocabulary.TryGetSpecialToken"/> and skip constructing this type when the
    /// lookup fails.
    /// </param>
    public ForcedToolCallConstraint(GrammarVocabulary vocab, int openMarkerId)
    {
        ArgumentNullException.ThrowIfNull(vocab);
        if (openMarkerId <= 0 || openMarkerId >= vocab.VocabSize)
            throw new ArgumentOutOfRangeException(nameof(openMarkerId),
                $"open-marker token id {openMarkerId} is outside the vocabulary (size {vocab.VocabSize}).");
        _vocabSize = vocab.VocabSize;
        _openMarkerId = openMarkerId;
    }

    /// <summary>True until the open marker has been emitted; false forever afterwards.</summary>
    public bool IsConstraining => !_satisfied;

    public ReadOnlySpan<float> Filter(ReadOnlySpan<float> logits)
    {
        if (_satisfied) return logits;

        // Allocated on first constrained step, then reused — the interface requires the per-token
        // path to be allocation-free after warm-up.
        _mask ??= new float[_vocabSize];
        int n = Math.Min(logits.Length, _mask.Length);
        _mask.AsSpan(0, n).Fill(float.NegativeInfinity);
        if (_openMarkerId < n) _mask[_openMarkerId] = logits[_openMarkerId];
        return _mask.AsSpan(0, n);
    }

    public void Accept(int token)
    {
        if (token == _openMarkerId) _satisfied = true;
    }

    public void Reset() => _satisfied = false;
}
