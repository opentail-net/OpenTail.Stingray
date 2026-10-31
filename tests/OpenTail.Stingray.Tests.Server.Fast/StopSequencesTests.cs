using System.Runtime.CompilerServices;
using System.Text;

namespace OpenTail.Stingray.Tests.Server.Fast;

/// <summary>
/// Plan 2 P7c: <see cref="StopSequences"/> turns client stop strings into a cut of the engine's chunk stream. In-memory only: the
/// source is a scripted async enumerable that records whether it was cancelled and how far it was read.
/// </summary>
public sealed class StopSequencesTests
{
    private sealed class Script
    {
        public List<GenerateChunk> Chunks { get; } = [];
        public int Yielded;
        public bool TokenCancelled;
        public bool Disposed;

        public IAsyncEnumerable<GenerateChunk> Source(CancellationToken ct) => Run(ct);

        private async IAsyncEnumerable<GenerateChunk> Run([EnumeratorCancellation] CancellationToken ct)
        {
            try
            {
                foreach (var c in Chunks)
                {
                    ct.ThrowIfCancellationRequested();
                    Yielded++;
                    yield return c;
                    await Task.Yield();
                }
            }
            finally
            {
                TokenCancelled = ct.IsCancellationRequested;
                Disposed = true;
            }
        }
    }

    private static Script Text(params string[] pieces)
    {
        var s = new Script();
        foreach (var p in pieces) s.Chunks.Add(new GenerateChunk(GenerateChunkKind.Text, p));
        s.Chunks.Add(new GenerateChunk(GenerateChunkKind.Stop, ""));
        return s;
    }

    private static async Task<(List<GenerateChunk> Out, string Joined)> Collect(
        Script script, string[] stops, StopSequenceMatch match, CancellationToken ct = default)
    {
        var all = new List<GenerateChunk>();
        var sb = new StringBuilder();
        await foreach (var c in StopSequences.Apply(stops, script.Source, match, ct).WithCancellation(ct))
        {
            all.Add(c);
            if (c.Kind == GenerateChunkKind.Text) sb.Append(c.Text);
        }
        return (all, sb.ToString());
    }

    [Fact]
    public async Task NoStops_IsAPassThrough_OfTheSameSourceStream()
    {
        var script = Text("Hello ", "world");
        var match = new StopSequenceMatch();

        var (all, joined) = await Collect(script, [], match);

        Assert.Equal("Hello world", joined);
        Assert.Null(match.Matched);
        Assert.Equal(3, all.Count);   // two text chunks and the engine's stop chunk, untouched
        Assert.False(script.TokenCancelled);
    }

    [Fact]
    public async Task StopInsideOneChunk_CutsBeforeIt_AndCancelsTheEngine()
    {
        var script = Text("Hello ", "wor", "ld! END more", "never read", "nor this");
        var match = new StopSequenceMatch();

        var (_, joined) = await Collect(script, ["END"], match);

        Assert.Equal("Hello world! ", joined);
        Assert.Equal("END", match.Matched);
        Assert.True(script.TokenCancelled, "the engine's token was not cancelled, so generation would have run on");
        Assert.True(script.Disposed);
        Assert.True(script.Yielded <= 3, $"the source was read {script.Yielded} chunks; it should stop after the matching chunk");
    }

    [Fact]
    public async Task StopSplitAcrossChunks_IsFound_AndNoPartOfItIsEmitted()
    {
        var script = Text("abc E", "N", "D tail");
        var match = new StopSequenceMatch();

        var (_, joined) = await Collect(script, ["END"], match);

        Assert.Equal("abc ", joined);
        Assert.Equal("END", match.Matched);
    }

    [Fact]
    public async Task APartialStopThatNeverCompletes_IsReleasedAtTheEnd()
    {
        var script = Text("abc E", "N");
        var match = new StopSequenceMatch();

        var (_, joined) = await Collect(script, ["END"], match);

        Assert.Equal("abc EN", joined);
        Assert.Null(match.Matched);
        Assert.False(script.TokenCancelled);
    }

    [Fact]
    public async Task TheEnginesStopChunk_ReleasesHeldTextBeforeItself()
    {
        var script = Text("ab");   // "b" could start the stop "bX", so it is held until the stop chunk arrives
        var match = new StopSequenceMatch();

        var (all, joined) = await Collect(script, ["bX"], match);

        Assert.Equal("ab", joined);
        Assert.Equal(GenerateChunkKind.Stop, all[^1].Kind);
        Assert.Equal(GenerateChunkKind.Text, all[^2].Kind);
    }

    [Fact]
    public async Task OfSeveralStops_TheOneFoundFirstInTheTextWins()
    {
        var script = Text("a b c d");
        var match = new StopSequenceMatch();

        var (_, joined) = await Collect(script, ["d", "c", "b"], match);

        Assert.Equal("a ", joined);
        Assert.Equal("b", match.Matched);
    }

    [Fact]
    public async Task StopAtTheVeryStart_EmitsNoText()
    {
        var script = Text("END", "more");
        var match = new StopSequenceMatch();

        var (all, joined) = await Collect(script, ["END"], match);

        Assert.Equal("", joined);
        Assert.Equal("END", match.Matched);
        Assert.DoesNotContain(all, c => c.Kind == GenerateChunkKind.Text);
    }

    [Fact]
    public async Task Matching_IsCaseSensitive()
    {
        var script = Text("the end");
        var match = new StopSequenceMatch();

        var (_, joined) = await Collect(script, ["END"], match);

        Assert.Equal("the end", joined);
        Assert.Null(match.Matched);
    }

    [Fact]
    public async Task OverlappingPrefix_DoesNotHideALaterMatch()
    {
        var script = Text("aaab");
        var match = new StopSequenceMatch();

        var (_, joined) = await Collect(script, ["aab"], match);

        Assert.Equal("a", joined);
        Assert.Equal("aab", match.Matched);
    }

    [Fact]
    public async Task ThinkingUsageAndToolCallChunks_PassThroughUnchanged_AndAreNotSearched()
    {
        var script = new Script();
        script.Chunks.Add(new GenerateChunk(GenerateChunkKind.Usage, "", PromptTokens: 7));
        script.Chunks.Add(new GenerateChunk(GenerateChunkKind.Thinking, "thinking END thinking"));
        script.Chunks.Add(new GenerateChunk(GenerateChunkKind.Text, "answer"));
        script.Chunks.Add(new GenerateChunk(GenerateChunkKind.Stop, ""));
        var match = new StopSequenceMatch();

        var (all, joined) = await Collect(script, ["END"], match);

        Assert.Equal("answer", joined);
        Assert.Null(match.Matched);
        Assert.Contains(all, c => c.Kind == GenerateChunkKind.Usage && c.PromptTokens == 7);
        Assert.Contains(all, c => c.Kind == GenerateChunkKind.Thinking && c.Text == "thinking END thinking");
    }

    [Fact]
    public async Task CallerCancellation_StillSurfacesToTheCaller()
    {
        var script = Text("a", "b", "c");
        using var cts = new CancellationTokenSource();
        var match = new StopSequenceMatch();

        var enumerated = 0;
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var _ in StopSequences.Apply(["zzz"], script.Source, match, cts.Token).WithCancellation(cts.Token))
            {
                enumerated++;
                if (enumerated == 1) cts.Cancel();
            }
        });
        Assert.True(script.Disposed);
    }

    [Theory]
    [InlineData("abc", new[] { "abcd" }, 3)]
    [InlineData("abcX", new[] { "abcd" }, 0)]
    [InlineData("xab", new[] { "abc", "b" }, 2)]
    [InlineData("a", new[] { "a" }, 0)]       // a one-character stop has no proper prefix to hold back
    [InlineData("", new[] { "abc" }, 0)]
    [InlineData("aab", new[] { "abc" }, 2)]
    public void LongestStopPrefixAtEnd_HoldsBackExactlyTheAmbiguousTail(string text, string[] stops, int expected)
        => Assert.Equal(expected, StopSequences.LongestStopPrefixAtEnd(text, stops));

    // ── property tests against a plain reference ──

    private static (string Out, string? Matched) CharReference(string full, string[] stops)
    {
        // One character at a time: stop at the first position where the text so far ends with a stop string. If several stops end
        // there, take the longest (the one that started earliest), so no part of a matched stop string is left in the output.
        for (int end = 1; end <= full.Length; end++)
        {
            string? best = null;
            foreach (string stop in stops)
                if (end >= stop.Length && string.CompareOrdinal(full, end - stop.Length, stop, 0, stop.Length) == 0
                    && (best is null || stop.Length > best.Length))
                    best = stop;
            if (best is not null) return (full[..(end - best.Length)], best);
        }
        return (full, null);
    }

    [Fact]
    public async Task OneCharacterChunks_MatchTheCharacterReference_OnManyRandomInputs()
    {
        var rng = new Random(20261031);
        for (int iteration = 0; iteration < 400; iteration++)
        {
            string full = new(Enumerable.Range(0, rng.Next(0, 24)).Select(_ => "ab"[rng.Next(2)]).ToArray());
            var stops = Enumerable.Range(0, rng.Next(1, 4))
                .Select(_ => new string(Enumerable.Range(0, rng.Next(1, 4)).Select(_ => "ab"[rng.Next(2)]).ToArray()))
                .Distinct().ToArray();
            var script = Text([.. full.Select(ch => ch.ToString())]);
            var match = new StopSequenceMatch();

            var (_, joined) = await Collect(script, stops, match);

            var (expectedOut, expectedMatch) = CharReference(full, stops);
            Assert.True(expectedOut == joined && expectedMatch == match.Matched,
                $"text '{full}', stops [{string.Join(",", stops)}]: got '{joined}'/{match.Matched ?? "none"}, expected '{expectedOut}'/{expectedMatch ?? "none"}");
        }
    }

    [Fact]
    public async Task AnyChunking_NeverLeaksAStop_AndOnlyEverCutsAtAnOccurrence()
    {
        var rng = new Random(31102026);
        for (int iteration = 0; iteration < 400; iteration++)
        {
            string full = new(Enumerable.Range(0, rng.Next(0, 30)).Select(_ => "abc"[rng.Next(3)]).ToArray());
            var stops = Enumerable.Range(0, rng.Next(1, 4))
                .Select(_ => new string(Enumerable.Range(0, rng.Next(1, 4)).Select(_ => "abc"[rng.Next(3)]).ToArray()))
                .Distinct().ToArray();
            var pieces = new List<string>();
            for (int i = 0; i < full.Length;)
            {
                int n = rng.Next(1, 6);
                pieces.Add(full.Substring(i, Math.Min(n, full.Length - i)));
                i += n;
            }
            var script = Text([.. pieces]);
            var match = new StopSequenceMatch();

            var (_, joined) = await Collect(script, stops, match);

            string where = $"text '{full}', stops [{string.Join(",", stops)}], chunks [{string.Join("|", pieces)}], got '{joined}'/{match.Matched ?? "none"}";
            Assert.True(full.StartsWith(joined, StringComparison.Ordinal), "output is not a prefix of the text: " + where);
            foreach (string stop in stops)
                Assert.False(joined.Contains(stop, StringComparison.Ordinal), "a stop string leaked into the output: " + where);
            if (match.Matched is null)
                Assert.True(joined == full, "no match but text was lost: " + where);
            else
                Assert.True(full.StartsWith(joined + match.Matched, StringComparison.Ordinal), "the matched stop does not follow the output: " + where);
        }
    }
}
