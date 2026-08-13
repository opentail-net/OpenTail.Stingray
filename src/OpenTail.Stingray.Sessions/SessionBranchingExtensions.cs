using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Engine;

namespace OpenTail.Stingray.Sessions;

/// <summary>
/// Result of a single completion branch generated during zero-copy session multiverse branching.
/// </summary>
public sealed record InferenceBranchResult(
    int BranchIndex,
    IInferenceSession Session,
    IReadOnlyList<int> GeneratedTokens,
    string Text);

/// <summary>
/// Extension methods for native zero-copy session branching and multiverse candidate generation
/// (<c>Plan 008</c> specification in <c>docs/008-Implementation Plan — Zero-Copy Session Multiverse  Parallel Branching.md</c>).
/// </summary>
public static class SessionBranchingExtensions
{
    /// <summary>
    /// Spawns <paramref name="count"/> zero-copy child sessions sharing the parent's physical KV page table,
    /// and generates independent candidate completions across all branches simultaneously without prompt re-prefill.
    /// </summary>
    /// <param name="session">Parent session containing prompt history and KV state.</param>
    /// <param name="count">Number of independent candidate branches to spawn (must be >= 1).</param>
    /// <param name="sampling">Sampling parameters for branch generation. Derived RNG seeds are assigned per branch.</param>
    /// <param name="cancellationToken">Cancellation token for controlling branch generation.</param>
    public static async Task<IReadOnlyList<InferenceBranchResult>> GenerateBranchesAsync(
        this IInferenceSession session,
        int count,
        SamplingParams sampling,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(sampling);

        var branches = session.ForkMany(count);

        var tasks = branches.Select(async (branch, index) =>
        {
            // GenerateChunk carries decoded TEXT, not a per-chunk token id -- parsing chunk.Text
            // as an integer only ever succeeds by coincidence (real generated text isn't numeric),
            // so GeneratedTokens was silently empty for every real generation. Recover the actual
            // generated token ids from the session's own TokenHistory (before/after this branch's
            // generation) instead of trying to reconstruct them from the text stream.
            int tokensBefore = (int)branch.TokenCount;
            var textBuilder = new StringBuilder();

            await foreach (var chunk in branch.GenerateAsync(sampling, cancellationToken).ConfigureAwait(false))
            {
                textBuilder.Append(chunk.Text);
            }

            var generatedTokens = branch.TokenHistory.Skip(tokensBefore).ToList();

            return new InferenceBranchResult(
                BranchIndex: index,
                Session: branch,
                GeneratedTokens: generatedTokens,
                Text: textBuilder.ToString());
        });

        return await Task.WhenAll(tasks).ConfigureAwait(false);
    }
}
