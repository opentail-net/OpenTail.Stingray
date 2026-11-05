using System.Runtime.CompilerServices;
using System.Text;

namespace OpenTail.Stingray.Server.Endpoints;

/// <summary>Which stop sequence ended a generation, filled in by <see cref="StopSequences.Apply"/>.</summary>
internal sealed class StopSequenceMatch
{
    /// <summary>The stop string that matched, or null when generation ended some other way.</summary>
    public string? Matched { get; internal set; }
}

/// <summary>
/// Client stop strings (<c>stop</c> on the OpenAI endpoint, <c>stop_sequences</c> on the Anthropic one) as a wrapper around the
/// engine's chunk stream, so all four handlers share one implementation.
/// <para>Only answer text (<see cref="GenerateChunkKind.Text"/>) is searched; thinking, usage, tool-call and stop chunks pass through.
/// Text that could still turn out to be the start of a stop string is held back until the next chunk settles it, so a stop string
/// split across chunks is found and none of it is ever emitted. The earliest match wins, and the text before it is emitted. On a
/// match the engine's generation is cancelled (not just hidden), and <see cref="StopSequenceMatch.Matched"/> says which string
/// matched. Matching is case-sensitive, as in the OpenAI and Anthropic APIs.</para>
/// <para>The engine counts one token per chunk it produced; because held-back text is re-emitted in different pieces, a caller that
/// counts the chunks it receives gets an approximate token count for a response cut by a stop string.</para>
/// </summary>
internal static class StopSequences
{
    public static IAsyncEnumerable<GenerateChunk> Apply(
        IReadOnlyList<string>? stops,
        Func<CancellationToken, IAsyncEnumerable<GenerateChunk>> source,
        StopSequenceMatch match,
        CancellationToken cancellationToken)
        => stops is { Count: > 0 }
            ? Run(stops, source, match, cancellationToken)
            : source(cancellationToken);

    private static async IAsyncEnumerable<GenerateChunk> Run(
        IReadOnlyList<string> stops,
        Func<CancellationToken, IAsyncEnumerable<GenerateChunk>> source,
        StopSequenceMatch match,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var pending = new StringBuilder();
        var enumerator = source(linked.Token).GetAsyncEnumerator(linked.Token);
        try
        {
            while (true)
            {
                GenerateChunk chunk;
                try
                {
                    if (!await enumerator.MoveNextAsync()) break;
                    chunk = enumerator.Current;
                }
                catch (OperationCanceledException) when (linked.IsCancellationRequested && !cancellationToken.IsCancellationRequested)
                {
                    break;   // we cancelled the engine ourselves after a match
                }

                if (chunk.Kind != GenerateChunkKind.Text)
                {
                    // The engine's end-of-generation chunk means no more text is coming: release anything held back first.
                    if (chunk.Kind == GenerateChunkKind.Stop && pending.Length > 0)
                    {
                        yield return new GenerateChunk(GenerateChunkKind.Text, pending.ToString());
                        pending.Clear();
                    }
                    yield return chunk;
                    continue;
                }

                pending.Append(chunk.Text);
                string text = pending.ToString();

                int at = -1;
                string? hit = null;
                foreach (string stop in stops)
                {
                    int i = text.IndexOf(stop, StringComparison.Ordinal);
                    if (i >= 0 && (at < 0 || i < at)) { at = i; hit = stop; }
                }

                if (at >= 0)
                {
                    match.Matched = hit;
                    if (at > 0) yield return chunk with { Text = text[..at] };
                    linked.Cancel();
                    yield break;
                }

                int emit = text.Length - LongestStopPrefixAtEnd(text, stops);
                if (emit > 0)
                {
                    yield return chunk with { Text = text[..emit] };
                    pending.Remove(0, emit);
                }
            }

            if (pending.Length > 0)
                yield return new GenerateChunk(GenerateChunkKind.Text, pending.ToString());
        }
        finally
        {
            try { await enumerator.DisposeAsync(); }
            catch (OperationCanceledException) { /* the generation we just cancelled */ }
        }
    }

    /// <summary>Length of the longest suffix of <paramref name="text"/> that is a proper prefix of some stop string.</summary>
    internal static int LongestStopPrefixAtEnd(string text, IReadOnlyList<string> stops)
    {
        int best = 0;
        foreach (string stop in stops)
        {
            int max = Math.Min(stop.Length - 1, text.Length);
            for (int k = max; k > best; k--)
            {
                if (string.CompareOrdinal(text, text.Length - k, stop, 0, k) == 0) { best = k; break; }
            }
        }
        return best;
    }
}
