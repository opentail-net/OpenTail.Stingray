namespace OpenTail.Stingray.Engine;

/// <summary>
/// Model-free draft source for speculative decoding (issue #207, llama.cpp lookup-decoding
/// analog): proposes continuation tokens by matching the tail n-gram of the generated
/// context against the prompt + everything generated so far. Proposals cost no forward
/// passes, so on copy-heavy workloads (RAG quotation, summarization, code edits,
/// self-repetitive output) the speculative step gets its draft for free; when nothing
/// matches it proposes nothing and the step degrades to a plain single-token decode.
///
/// Matching: longest tail n-gram first (<see cref="NgramMax"/> down to
/// <see cref="NgramMin"/>), most recent occurrence wins (generated text repeats locally).
/// <see cref="NgramMin"/> defaults to 2 — 1-gram matches fire constantly and mostly
/// propose junk, which makes every step pay a wider verify batch for nothing.
/// </summary>
public sealed class PromptLookupDraft : IPromptLookupDecoder
{
    private readonly List<int> _history = new();

    /// <summary>Largest tail n-gram length tried for a match.</summary>
    public int NgramMax { get; }

    /// <summary>Smallest tail n-gram length tried before giving up (no proposal).</summary>
    public int NgramMin { get; }

    public PromptLookupDraft(int ngramMax = 3, int ngramMin = 2)
    {
        if (ngramMin < 1) throw new ArgumentOutOfRangeException(nameof(ngramMin));
        if (ngramMax < ngramMin) throw new ArgumentOutOfRangeException(nameof(ngramMax));
        NgramMax = ngramMax;
        NgramMin = ngramMin;
    }

    /// <summary>Tokens observed so far (prompt + emitted).</summary>
    public int Count => _history.Count;

    /// <summary>Reset the history to a new prompt (start of a generation).</summary>
    public void Reset(IReadOnlyList<int> promptTokens)
    {
        _history.Clear();
        if (promptTokens is not null) _history.AddRange(promptTokens);
    }

    /// <summary>Record an emitted token so future proposals can match it.</summary>
    public void Append(int token) => _history.Add(token);

    /// <summary>
    /// Propose up to <paramref name="maxTokens"/> continuation tokens for the current history.
    /// </summary>
    public int[] Propose(int maxTokens)
    {
        var span = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_history);
        return Propose(span, _history.Count, maxTokens);
    }

    /// <summary>
    /// Propose continuation tokens by matching the tail n-gram of <paramref name="tokenHistory"/> up to <paramref name="currentPosition"/>.
    /// </summary>
    public int[] Propose(ReadOnlySpan<int> tokenHistory, int currentPosition, int maxDraftTokens)
    {
        if (maxDraftTokens <= 0 || tokenHistory.IsEmpty || currentPosition <= 0) return Array.Empty<int>();

        int len = Math.Min(currentPosition, tokenHistory.Length);

        for (int n = NgramMax; n >= NgramMin; n--)
        {
            if (len < n + 1) continue;

            for (int i = len - n - 1; i >= 0; i--)
            {
                bool match = true;
                for (int j = 0; j < n; j++)
                {
                    if (tokenHistory[i + j] != tokenHistory[len - n + j])
                    {
                        match = false;
                        break;
                    }
                }
                if (!match) continue;

                int start = i + n;
                int available = len - start;
                int count = Math.Min(maxDraftTokens, available);
                if (count <= 0) continue;

                var proposal = new int[count];
                tokenHistory.Slice(start, count).CopyTo(proposal);
                return proposal;
            }
        }
        return Array.Empty<int>();
    }
}
