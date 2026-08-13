using System;
using System.Collections.Generic;

namespace OpenTail.Stingray.Engine;

/// <summary>
/// Model-free draft source interface for prompt-lookup speculative decoding (<c>Plan 009</c> specification
/// in <c>docs/009 - Prompt Lookup Speculation (PLD).md</c>). Proposes continuation draft tokens by matching
/// tail n-gram history without model weights or extra forward passes.
/// </summary>
public interface IPromptLookupDecoder
{
    /// <summary>Largest tail n-gram length tried for a match.</summary>
    int NgramMax { get; }

    /// <summary>Smallest tail n-gram length tried before giving up.</summary>
    int NgramMin { get; }

    /// <summary>Tokens observed so far in the current session history.</summary>
    int Count { get; }

    /// <summary>Reset the lookup history to a new prompt.</summary>
    void Reset(IReadOnlyList<int> promptTokens);

    /// <summary>Record an emitted token into history.</summary>
    void Append(int token);

    /// <summary>Propose continuation draft tokens for the current history.</summary>
    int[] Propose(int maxTokens);

    /// <summary>Propose continuation draft tokens for an explicit token history slice.</summary>
    int[] Propose(ReadOnlySpan<int> tokenHistory, int currentPosition, int maxDraftTokens);
}
