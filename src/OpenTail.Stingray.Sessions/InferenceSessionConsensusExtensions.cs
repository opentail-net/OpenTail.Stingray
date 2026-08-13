using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Extension methods providing parallel branch consensus and majority voting over <see cref="IInferenceSession"/>.
/// </summary>
public static class InferenceSessionConsensusExtensions
{
    /// <summary>
    /// Spawns <paramref name="branchCount"/> zero-copy branches from <paramref name="session"/>, generates candidate outputs in parallel,
    /// performs majority voting over the normalized answers, retains the winning branch, and disposes the losing branches.
    /// </summary>
    public static async Task<BranchVoteResult> ForkAndVoteAsync(
        this IInferenceSession session,
        SamplingParams samplingParams,
        int branchCount,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(samplingParams);

        if (branchCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(branchCount), branchCount, "Branch count must be greater than or equal to 1.");
        }

        if (branchCount == 1)
        {
            var textBuilder = new System.Text.StringBuilder();
            await foreach (var chunk in session.GenerateAsync(samplingParams, cancellationToken).ConfigureAwait(false))
            {
                textBuilder.Append(chunk.Text);
            }
            string text = textBuilder.ToString();
            string normalized = NormalizeAnswer(text);
            var singleVote = new BranchVote(session.Id, text, normalized, IsWinner: true);
            return new BranchVoteResult(session, text, new[] { singleVote });
        }

        var branches = session.ForkMany(branchCount);
        IInferenceSession? winningBranch = null;

        try
        {
            var tasks = new Task<(IInferenceSession Branch, string Text)>[branchCount];

            for (int i = 0; i < branchCount; i++)
            {
                int branchIndex = i;
                var branch = branches[branchIndex];
                var branchParams = samplingParams;

                tasks[branchIndex] = Task.Run(async () =>
                {
                    var sb = new System.Text.StringBuilder();
                    await foreach (var chunk in branch.GenerateAsync(branchParams, cancellationToken).ConfigureAwait(false))
                    {
                        sb.Append(chunk.Text);
                    }
                    return (branch, sb.ToString());
                }, cancellationToken);
            }

            var branchResults = await Task.WhenAll(tasks).ConfigureAwait(false);

            var frequencyMap = new Dictionary<string, int>(StringComparer.Ordinal);
            var answerToFirstBranchMap = new Dictionary<string, (IInferenceSession Branch, string Text, int FirstIndex)>(StringComparer.Ordinal);

            for (int i = 0; i < branchResults.Length; i++)
            {
                var (b, text) = branchResults[i];
                string normalized = NormalizeAnswer(text);

                frequencyMap[normalized] = frequencyMap.TryGetValue(normalized, out int count) ? count + 1 : 1;

                if (!answerToFirstBranchMap.ContainsKey(normalized))
                {
                    answerToFirstBranchMap[normalized] = (b, text, i);
                }
            }

            // Find highest vote count
            int maxVotes = frequencyMap.Values.Max();

            // Find winning answer (tie break by lowest first index)
            string winningAnswer = frequencyMap
                .Where(kv => kv.Value == maxVotes)
                .Select(kv => kv.Key)
                .OrderBy(ans => answerToFirstBranchMap[ans].FirstIndex)
                .First();

            var (winningB, winningT, _) = answerToFirstBranchMap[winningAnswer];
            winningBranch = winningB;

            var votes = new List<BranchVote>(branchCount);
            for (int i = 0; i < branchResults.Length; i++)
            {
                var (b, text) = branchResults[i];
                string normalized = NormalizeAnswer(text);
                bool isWinner = ReferenceEquals(b, winningBranch);
                votes.Add(new BranchVote(b.Id, text, normalized, isWinner));
            }

            // Dispose losing branches asynchronously
            foreach (var b in branches)
            {
                if (!ReferenceEquals(b, winningBranch))
                {
                    await b.DisposeAsync().ConfigureAwait(false);
                }
            }

            return new BranchVoteResult(winningBranch, winningT, votes);
        }
        catch
        {
            // Transactional cleanup: dispose all spawned branches on cancellation or exception
            foreach (var b in branches)
            {
                if (!ReferenceEquals(b, winningBranch))
                {
                    try { await b.DisposeAsync().ConfigureAwait(false); } catch { }
                }
            }
            throw;
        }
    }

    private static string NormalizeAnswer(string raw)
    {
        if (string.IsNullOrEmpty(raw)) return string.Empty;
        return raw.Trim();
    }
}
