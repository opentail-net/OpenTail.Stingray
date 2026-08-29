
namespace OpenTail.Stingray.Engine;

/// <summary>
/// Token-prefix trie and state tracking for <see cref="SamplingParams.AllowedChoices"/>.
/// </summary>
public sealed class TokenChoiceTrie
{
    public sealed class TrieNode
    {
        public Dictionary<int, TrieNode> Children { get; } = new();
        public bool IsTerminal { get; set; }
    }

    private readonly TrieNode _root = new();

    private TokenChoiceTrie() { }

    public static TokenChoiceTrie Build(IEnumerable<string> choices, ITokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(choices);
        ArgumentNullException.ThrowIfNull(tokenizer);

        var trie = new TokenChoiceTrie();

        foreach (string rawChoice in choices.Distinct())
        {
            if (string.IsNullOrEmpty(rawChoice))
            {
                continue;
            }

            var tokens = tokenizer.Encode(rawChoice);
            int[] arr = tokens is int[] a ? a : tokens.ToArray();

            if (arr.Length == 0)
            {
                continue;
            }

            TrieNode current = trie._root;
            foreach (int tokenId in arr)
            {
                if (!current.Children.TryGetValue(tokenId, out var nextNode))
                {
                    nextNode = new TrieNode();
                    current.Children[tokenId] = nextNode;
                }
                current = nextNode;
            }
            current.IsTerminal = true;
        }

        return trie;
    }

    public ChoiceState CreateState() => new ChoiceState(_root);

    public sealed class ChoiceState
    {
        private TrieNode _currentNode;

        internal ChoiceState(TrieNode root)
        {
            _currentNode = root;
        }

        // A terminal node is always a valid stopping point, even when it also has children (one
        // allowed choice is a token-prefix of another, e.g. "APPROVED" / "APPROVED_WITH_CHANGES").
        // Requiring zero children here made the shorter choice permanently unreachable: the
        // generation loop checks IsComplete before sampling each next token, so once this node is
        // reached the loop stops immediately, before MaskLogits could ever force-continue toward
        // the longer choice. Dropping the children check means the shortest matching choice wins
        // when choices share a prefix -- deterministic and predictable, versus the prior behavior
        // of the shorter choice being silently impossible to produce at all.
        public bool IsComplete => _currentNode.IsTerminal;
        public bool IsTerminal => _currentNode.IsTerminal;
        public bool HasChildren => _currentNode.Children.Count > 0;

        public void MaskLogits(Span<float> logits)
        {
            var allowed = _currentNode.Children.Keys;
            if (allowed.Count == 0)
            {
                if (_currentNode.IsTerminal)
                {
                    // Choice complete
                    return;
                }
                throw new InvalidOperationException("Constrained choice state cannot be satisfied: no valid continuation tokens remaining.");
            }

            // Mask out all token IDs not in allowed children set
            for (int i = 0; i < logits.Length; i++)
            {
                if (!allowed.Contains(i))
                {
                    logits[i] = float.NegativeInfinity;
                }
            }
        }

        public bool Advance(int tokenId)
        {
            if (_currentNode.Children.TryGetValue(tokenId, out var nextNode))
            {
                _currentNode = nextNode;
                return true;
            }
            return false;
        }
    }
}
