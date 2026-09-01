
namespace OpenTail.Stingray.Tests.Core;

/// <summary>
/// Parity tests for <see cref="GgufTokenizer.SpmMergePiecesByScore"/> -- the real SentencePiece
/// BPE merge (vocabulary-membership-gated, score-priority-ordered) that replaced using the
/// merges-rank-table algorithm for genuine SPM tokenization. See <see cref="GgufTokenizer.
/// SpmMergePiecesByScore"/>'s own doc comment for why the two algorithms are not the same thing.
/// These run with no model file: synthetic vocab/score tables, checked against a naive O(n^2)
/// reference implementation of llama.cpp's real <c>llm_tokenizer_spm_session::tokenize</c>.
/// </summary>
public sealed class SpmMergeByScoreTests
{
    /// <summary>
    /// Reference (oracle): repeatedly scan every adjacent pair, keep only those whose
    /// concatenation is in the vocab, and apply the highest-score one (leftmost on a tie) --
    /// exactly <c>llm_tokenizer_spm_session::tokenize</c>'s real behaviour, just O(n^2).
    /// </summary>
    private static List<string> NaiveMergeByScore(
        List<string> input, IReadOnlyDictionary<string, int> vocab, float[]? scores)
    {
        float Score(int id) => scores is not null && (uint)id < (uint)scores.Length ? scores[id] : 0f;

        var pieces = new List<string>(input);
        while (true)
        {
            int bestIdx = -1;
            float bestScore = float.NegativeInfinity;
            for (int i = 0; i < pieces.Count - 1; i++)
            {
                string merged = pieces[i] + pieces[i + 1];
                if (!vocab.TryGetValue(merged, out int id)) continue;
                float s = Score(id);
                if (s > bestScore)
                {
                    bestScore = s;
                    bestIdx = i;
                }
            }
            if (bestIdx < 0) break;
            pieces[bestIdx] += pieces[bestIdx + 1];
            pieces.RemoveAt(bestIdx + 1);
        }
        return pieces;
    }

    /// <summary>
    /// Real bug found re-checking `ernie4_5` (2026-09-01): after merging, a piece with no direct
    /// vocab entry fell straight to a single UnknownTokenId for the WHOLE piece instead of real
    /// llama.cpp's per-UTF8-BYTE `&lt;0xXX&gt;` fallback (`llm_tokenizer_spm_session::resegment`'s
    /// "output any symbols that did not form tokens as bytes" branch, confirmed against
    /// `llama_vocab::byte_to_token`). A newline mid-prompt was the real-world trigger: no direct
    /// "\n" vocab entry, but a real "&lt;0x0A&gt;" byte-fallback entry existed and was never tried.
    /// </summary>
    [Fact]
    public void EncodeSpm_PieceWithNoDirectVocabEntry_UsesByteFallbackToken_NotUnk()
    {
        var source = new TokenizerSource
        {
            ModelFamily = "llama",
            Tokens = ["<unk>", "<s>", "</s>", "A", "<0x0A>", "B"],
            Scores = [0f, 0f, 0f, 0f, 0f, 0f],
            UnknownTokenId = 0,
            BosTokenId = 1,
            EosTokenId = 2,
        };
        var tokenizer = GgufTokenizer.FromSource(source);

        // "A\nB" -> "A" and "B" are direct vocab hits; "\n" has no direct entry but does have a
        // real "<0x0A>" byte-fallback entry (id 4) -- must resolve to that, not to UnknownTokenId.
        var ids = tokenizer.Encode("A\nB");
        Assert.Contains(4, ids);
        Assert.DoesNotContain(0, ids);
    }

    [Fact]
    public void EmptyInput_ReturnsEmpty()
    {
        Assert.Empty(GgufTokenizer.SpmMergePiecesByScore([], new Dictionary<string, int>(), null));
    }

    [Fact]
    public void SingleSymbol_Unchanged()
    {
        Assert.Equal(["a"], GgufTokenizer.SpmMergePiecesByScore(["a"], new Dictionary<string, int>(), null));
    }

    [Fact]
    public void NoVocabMatch_Unchanged()
    {
        // "ab"/"bc" not in vocab at all -> no merge is even a candidate, regardless of score.
        var vocab = new Dictionary<string, int> { ["xy"] = 0 };
        Assert.Equal(["a", "b", "c"], GgufTokenizer.SpmMergePiecesByScore(["a", "b", "c"], vocab, null));
    }

    [Fact]
    public void NoScoresArray_StillMergesOnVocabMembershipAlone()
    {
        // The exact xverse-shaped case: no scores array at all (null), but the vocab still
        // contains the merged forms -- must still merge, matching llama.cpp's real fallback
        // (every score defaults to 0.0f, but merging is gated on vocab membership, not score).
        var vocab = new Dictionary<string, int> { ["a"] = 0, ["b"] = 1, ["c"] = 2, ["ab"] = 3, ["abc"] = 4 };
        Assert.Equal(["abc"], GgufTokenizer.SpmMergePiecesByScore(["a", "b", "c"], vocab, null));
    }

    [Fact]
    public void HighestScoreMergeAppliedFirst_ThenCascades()
    {
        // "bc" has the higher score, so it merges before "ab" gets a chance; "a"+"bc" isn't in
        // vocab, so the result stops at ["a","bc"], not ["ab","c"] or "abc".
        var vocab = new Dictionary<string, int> { ["a"] = 0, ["b"] = 1, ["c"] = 2, ["ab"] = 3, ["bc"] = 4 };
        var scores = new float[5];
        scores[3] = 1.0f; // "ab"
        scores[4] = 5.0f; // "bc" -- higher score, wins
        Assert.Equal(["a", "bc"], GgufTokenizer.SpmMergePiecesByScore(["a", "b", "c"], vocab, scores));
    }

    [Fact]
    public void EqualScore_LeftmostWins()
    {
        // Two (a,a) candidates at equal score 0; leftmost must win (matches
        // llm_bigram_spm::comparator's l.left>r.left tie-break under a max-heap).
        var vocab = new Dictionary<string, int> { ["a"] = 0, ["aa"] = 1, ["aaa"] = 2 };
        var scores = new float[3]; // all zero -- pure tie
        Assert.Equal(["aaa"], GgufTokenizer.SpmMergePiecesByScore(["a", "a", "a"], vocab, scores));
    }

    [Fact]
    public void MultiCharOperandMerges_Cascade()
    {
        var vocab = new Dictionary<string, int> { ["a"] = 0, ["b"] = 1, ["c"] = 2, ["ab"] = 3, ["abc"] = 4 };
        var scores = new float[5];
        scores[3] = 1.0f;
        scores[4] = 2.0f;
        Assert.Equal(["abc"], GgufTokenizer.SpmMergePiecesByScore(["a", "b", "c"], vocab, scores));
    }

    /// <summary>
    /// Fuzz: thousands of random (input, vocab, score) triples over a tiny alphabet -- including
    /// multi-char operands, cascading merges, ties, and negative scores -- must match the naive
    /// O(n^2) reference exactly.
    /// </summary>
    [Fact]
    public void FastPath_MatchesNaive_AcrossRandomInputs()
    {
        var rng = new Random(20260902);
        string[] alphabet = ["a", "b", "c"];

        var operands = new List<string>(alphabet);
        foreach (var x in alphabet)
            foreach (var y in alphabet)
            {
                operands.Add(x + y);
                foreach (var z in alphabet) operands.Add(x + y + z);
            }

        for (int iter = 0; iter < 5000; iter++)
        {
            // Random vocab: a random subset of operands, each given a random score (including
            // negative and duplicate/tied values -- exercises the leftmost tie-break).
            int vocabCount = rng.Next(1, operands.Count);
            var chosen = operands.OrderBy(_ => rng.Next()).Take(vocabCount).ToList();
            var vocab = new Dictionary<string, int>();
            var scores = new float[chosen.Count];
            for (int i = 0; i < chosen.Count; i++)
            {
                vocab[chosen[i]] = i;
                scores[i] = rng.Next(-3, 4); // small integer range -> frequent ties on purpose
            }

            int len = rng.Next(0, 30);
            var input = new List<string>(len);
            for (int k = 0; k < len; k++) input.Add(alphabet[rng.Next(alphabet.Length)]);

            var expected = NaiveMergeByScore(input, vocab, scores);
            var actual = GgufTokenizer.SpmMergePiecesByScore(new List<string>(input), vocab, scores);

            Assert.True(expected.SequenceEqual(actual),
                $"mismatch on input=[{string.Join(",", input)}] " +
                $"expected=[{string.Join(",", expected)}] actual=[{string.Join(",", actual)}]");
        }
    }
}
