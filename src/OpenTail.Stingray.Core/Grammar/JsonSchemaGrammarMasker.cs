using System;
using System.Buffers;
using System.Text;

namespace OpenTail.Stingray.Core.Grammar;

/// <summary>
/// Token-aware decoding constraint enforcing JSON Schema grammar rules during logit sampling (<see cref="ITokenConstraint"/>).
/// Enforces valid JSON formatting and property/type boundaries without building a monolithic schema engine.
///
/// <para>
/// This constraint backs <c>session.GenerateAsync&lt;T&gt;()</c>'s always-valid-JSON contract, so
/// unlike the tool-argument constraints' "silently degrade to unconstrained" dead-state convention
/// (a best-effort nicety layered over otherwise free-form chat), it fails closed on both ends of the
/// lifecycle: <see cref="Filter"/> throws <see cref="JsonGrammarDeadStateException"/> if the grammar
/// reaches a state with no legal next token, and once the root value closes it keeps constraining --
/// forcing an end-of-generation token -- instead of handing sampling back to the unconstrained path.
/// </para>
/// </summary>
public sealed class JsonSchemaGrammarMasker : ITokenConstraint
{
    private readonly GrammarVocabulary _vocab;
    private readonly GrammarStateMachine _stateMachine;
    private readonly GrammarStateMachine _scratchStateMachine;
    private readonly float[] _maskedLogits;
    private readonly StringBuilder _accumulatedText = new();
    private readonly byte[] _pendingBytes = new byte[4];
    private int _pendingCount;

    public JsonSchemaGrammarMasker(GrammarVocabulary vocab, GrammarStateMachine stateMachine)
    {
        _vocab = vocab ?? throw new ArgumentNullException(nameof(vocab));
        _stateMachine = stateMachine ?? throw new ArgumentNullException(nameof(stateMachine));
        // One reused "scratch" instance for candidate-token validation in Filter() below, instead
        // of allocating a fresh GrammarStateMachine.Clone() per vocabulary entry per decode step
        // (previously 100k+ allocations/step -- see docs/bugstofix.md). Must share the same root
        // properties as _stateMachine; GrammarStateMachine.CopyFrom resets it without allocating.
        _scratchStateMachine = new GrammarStateMachine(_stateMachine.Properties);
        _maskedLogits = new float[vocab.VocabSize];
    }

    /// <summary>
    /// Always true: this constraint governs the entire call, including the post-completion tail,
    /// where it forces an end-of-generation token instead of releasing control (see <see cref="Filter"/>).
    /// </summary>
    public bool IsConstraining => true;

    public GrammarStateMachine StateMachine => _stateMachine;

    /// <exception cref="JsonGrammarDeadStateException">
    /// The grammar reached a state with no legal next token (e.g. a required literal the bound
    /// tokenizer's vocabulary cannot express from here). Fails closed rather than returning the
    /// input logits unmasked, per this class's always-valid-JSON contract.
    /// </exception>
    public ReadOnlySpan<float> Filter(ReadOnlySpan<float> logits)
    {
        if (logits.Length == 0)
        {
            return logits;
        }

        if (_stateMachine.IsTerminal)
        {
            return FilterEogOnly(logits);
        }

        int count = Math.Min(logits.Length, _maskedLogits.Length);
        logits[..count].CopyTo(_maskedLogits);

        bool anyAllowed = false;
        for (int i = 0; i < count; i++)
        {
            var bytes = _vocab.TokenBytes(i);
            if (bytes.Length == 0)
            {
                // Empty-byte tokens (EOG/control) never advance the JSON structure -- forbidding
                // them keeps generation from truncating mid-object instead of merely going unmasked.
                _maskedLogits[i] = float.NegativeInfinity;
                continue;
            }

            // BPE tokens commonly contain more than one JSON character or partial UTF-8 bytes.
            // Validate combined pending + candidate token bytes against an isolated state snapshot
            // -- reusing one scratch instance (reset per candidate, not reallocated) rather than
            // cloning a whole new GrammarStateMachine for every vocabulary entry.
            _scratchStateMachine.CopyFrom(_stateMachine);
            if (EvaluateCandidateToken(bytes, _scratchStateMachine))
            {
                anyAllowed = true;
            }
            else
            {
                _maskedLogits[i] = float.NegativeInfinity;
            }
        }

        if (!anyAllowed)
        {
            throw new JsonGrammarDeadStateException(_stateMachine.CurrentState, _accumulatedText.ToString());
        }

        return _maskedLogits;
    }

    /// <summary>
    /// Terminal state: only an end-of-generation token is a legal continuation once the root value
    /// has closed, so this forces the sampler to stop instead of letting generation run on unmasked.
    /// </summary>
    private ReadOnlySpan<float> FilterEogOnly(ReadOnlySpan<float> logits)
    {
        int count = Math.Min(logits.Length, _maskedLogits.Length);
        Array.Fill(_maskedLogits, float.NegativeInfinity, 0, count);

        bool anyEog = false;
        foreach (int id in _vocab.EogTokenIds)
        {
            if ((uint)id < (uint)count)
            {
                _maskedLogits[id] = logits[id];
                anyEog = true;
            }
        }

        // No EOG id resolvable in this vocabulary/index range: never wedge post-completion.
        return anyEog ? _maskedLogits : logits;
    }

    private bool EvaluateCandidateToken(ReadOnlySpan<byte> tokenBytes, GrammarStateMachine candidateState)
    {
        if (_pendingCount > 0)
        {
            // If partial UTF-8 bytes are pending from previous accepted tokens, candidate token
            // must start with a UTF-8 continuation byte (10xxxxxx) to complete the code point.
            if ((tokenBytes[0] & 0xC0) != 0x80)
            {
                return false;
            }
        }

        int totalLen = _pendingCount + tokenBytes.Length;
        Span<byte> combined = totalLen <= 256 ? stackalloc byte[totalLen] : new byte[totalLen];
        if (_pendingCount > 0)
        {
            _pendingBytes.AsSpan(0, _pendingCount).CopyTo(combined);
        }
        tokenBytes.CopyTo(combined[_pendingCount..]);

        ReadOnlySpan<byte> span = combined;
        Span<char> charBuf = stackalloc char[2];

        while (!span.IsEmpty)
        {
            var status = Rune.DecodeFromUtf8(span, out Rune rune, out int bytesConsumed);
            if (status == OperationStatus.Done)
            {
                int charsWritten = rune.EncodeToUtf16(charBuf);
                for (int i = 0; i < charsWritten; i++)
                {
                    if (!candidateState.TryAcceptChar(charBuf[i]))
                    {
                        return false;
                    }
                }
                span = span[bytesConsumed..];
            }
            else if (status == OperationStatus.NeedMoreData)
            {
                // Incomplete trailing UTF-8 multi-byte sequence is only valid inside JSON string literals.
                return candidateState.CurrentState is JsonLexicalState.StringValue or JsonLexicalState.ObjectKeyContent;
            }
            else // InvalidData
            {
                if (candidateState.CurrentState is JsonLexicalState.StringValue or JsonLexicalState.ObjectKeyContent)
                {
                    if (!candidateState.TryAcceptChar('\uFFFD'))
                    {
                        return false;
                    }
                    span = span[1..];
                }
                else
                {
                    return false;
                }
            }
        }

        return true;
    }

    public void Accept(int token)
    {
        if (_stateMachine.IsTerminal)
        {
            // Filter() only ever offers EOG-only logits once terminal, so there's nothing further
            // to track regardless of what was actually sampled.
            return;
        }

        var bytes = _vocab.TokenBytes(token);
        if (bytes.Length == 0) return;

        int totalLen = _pendingCount + bytes.Length;
        Span<byte> combined = totalLen <= 256 ? stackalloc byte[totalLen] : new byte[totalLen];
        if (_pendingCount > 0)
        {
            _pendingBytes.AsSpan(0, _pendingCount).CopyTo(combined);
        }
        bytes.CopyTo(combined[_pendingCount..]);

        ReadOnlySpan<byte> span = combined;
        Span<char> charBuf = stackalloc char[2];

        while (!span.IsEmpty)
        {
            var status = Rune.DecodeFromUtf8(span, out Rune rune, out int bytesConsumed);
            if (status == OperationStatus.Done)
            {
                int charsWritten = rune.EncodeToUtf16(charBuf);
                for (int i = 0; i < charsWritten; i++)
                {
                    char c = charBuf[i];
                    _accumulatedText.Append(c);
                    if (!_stateMachine.TryAcceptChar(c))
                    {
                        throw new InvalidOperationException("Accepted token violates the JSON grammar state.");
                    }
                }
                span = span[bytesConsumed..];
            }
            else if (status == OperationStatus.NeedMoreData)
            {
                span.CopyTo(_pendingBytes);
                _pendingCount = span.Length;
                return;
            }
            else // InvalidData
            {
                char replacement = '\uFFFD';
                _accumulatedText.Append(replacement);
                if (!_stateMachine.TryAcceptChar(replacement))
                {
                    throw new InvalidOperationException("Accepted token violates the JSON grammar state.");
                }
                span = span[1..];
            }
        }

        _pendingCount = 0;
    }

    public void Reset()
    {
        _pendingCount = 0;
        _accumulatedText.Clear();
        _stateMachine.Reset();
    }
}

/// <summary>
/// Thrown by <see cref="JsonSchemaGrammarMasker.Filter"/> when the compiled grammar reaches a state
/// with no legal next token -- e.g. a required literal (property name, enum value) the bound
/// tokenizer's vocabulary cannot express from the current position. <see cref="JsonSchemaGrammarMasker"/>
/// backs <c>session.GenerateAsync&lt;T&gt;()</c>'s always-valid-JSON contract, so a dead state is a
/// hard failure the caller must see rather than a silent slide into unconstrained generation.
/// </summary>
public sealed class JsonGrammarDeadStateException : InvalidOperationException
{
    /// <summary>The grammar state active when no candidate token validated.</summary>
    public JsonLexicalState State { get; }

    /// <summary>The JSON text accumulated so far, for diagnostics.</summary>
    public string AccumulatedText { get; }

    public JsonGrammarDeadStateException(JsonLexicalState state, string accumulatedText)
        : base($"JSON schema grammar reached a dead state (no legal next token) at {state} after "
            + $"generating '{accumulatedText}'.")
    {
        State = state;
        AccumulatedText = accumulatedText;
    }
}

